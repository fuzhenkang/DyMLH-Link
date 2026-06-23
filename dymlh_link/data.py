from dataclasses import dataclass

import torch


@dataclass
class Snapshot:
    graph: object
    features: dict
    global_nids: dict
    name: str


@dataclass
class DynamicLinkData:
    snapshots: list
    target_graph: object
    target_global_nids: dict
    input_dims: dict
    num_global_nodes: dict
    target_etype: tuple
    train_pos_edges: torch.Tensor
    valid_pos_edges: torch.Tensor
    test_pos_edges: torch.Tensor
    train_neg_edges: torch.Tensor
    valid_neg_edges: torch.Tensor
    test_neg_edges: torch.Tensor


def _import_dgl():
    try:
        import dgl
    except ImportError as exc:
        raise ImportError("DGL is required. Please install dgl>=2.1.0 for your CUDA/PyTorch environment.") from exc
    return dgl


def _split_paths(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_one_graph(path, graph_index):
    dgl = _import_dgl()
    graphs, metadata = dgl.load_graphs(path)
    if graph_index >= len(graphs):
        raise IndexError("{} contains {} graph(s), but graph_index={} was requested".format(path, len(graphs), graph_index))
    return graphs[graph_index], metadata


def _resolve_target_etype(graph, spec):
    if spec is None:
        raise ValueError("--target-etype is required, for example author:coauthor:author or coauthor")
    canonical = tuple(part.strip() for part in spec.split(":"))
    if len(canonical) == 3:
        if canonical not in graph.canonical_etypes:
            raise ValueError("--target-etype {} is not in graph canonical_etypes: {}".format(canonical, graph.canonical_etypes))
        return canonical
    matches = [etype for etype in graph.canonical_etypes if etype[1] == spec]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("--target-etype '{}' is not in graph canonical_etypes: {}".format(spec, graph.canonical_etypes))
    raise ValueError("--target-etype '{}' is ambiguous. Use src:rel:dst. Matches: {}".format(spec, matches))


def _node_global_ids(graph, ntype, global_id_key):
    ndata = graph.nodes[ntype].data
    if global_id_key in ndata:
        return ndata[global_id_key].long().cpu()
    return torch.arange(graph.num_nodes(ntype), dtype=torch.long)


def _node_features(graph, ntype, feat_key, fallback):
    ndata = graph.nodes[ntype].data
    if feat_key in ndata:
        return ndata[feat_key].float().cpu()
    if fallback == "degree":
        indeg = torch.zeros(graph.num_nodes(ntype), dtype=torch.float32)
        outdeg = torch.zeros(graph.num_nodes(ntype), dtype=torch.float32)
        for etype in graph.canonical_etypes:
            if etype[2] == ntype:
                indeg += graph.in_degrees(etype=etype).float().cpu()
            if etype[0] == ntype:
                outdeg += graph.out_degrees(etype=etype).float().cpu()
        return torch.stack([indeg, outdeg], dim=1)
    raise KeyError("Feature key '{}' was not found for node type '{}'".format(feat_key, ntype))


def _edge_mask(graph, etype, split):
    aliases = {
        "train": ["train_mask"],
        "valid": ["valid_mask", "val_mask"],
        "test": ["test_mask"],
    }
    edata = graph.edges[etype].data
    for key in aliases[split]:
        if key in edata:
            return edata[key].bool().cpu()
    raise KeyError("Target etype {} does not contain a {} mask".format(etype, split))


def _remap_tensor(values, raw_to_compact):
    return torch.tensor([raw_to_compact[int(value)] for value in values.tolist()], dtype=torch.long)


def _snapshot_from_graph(path, graph, feat_key, global_id_key, feature_fallback):
    features = {}
    global_nids = {}
    for ntype in graph.ntypes:
        features[ntype] = _node_features(graph, ntype, feat_key, feature_fallback)
        global_nids[ntype] = _node_global_ids(graph, ntype, global_id_key)
    return Snapshot(graph=graph, features=features, global_nids=global_nids, name=path)


def _masked_edges(graph, etype, compact_global_nids, mask):
    src_type, _rel, dst_type = etype
    src, dst = graph.edges(etype=etype)
    src_compact = compact_global_nids[src_type][src.cpu()][mask]
    dst_compact = compact_global_nids[dst_type][dst.cpu()][mask]
    return torch.stack([src_compact, dst_compact], dim=0)


def _stored_negative_edges(metadata, split, target_etype, target_global_nids):
    src_key = "{}_neg_src".format(split)
    dst_key = "{}_neg_dst".format(split)
    if src_key not in metadata or dst_key not in metadata:
        raise KeyError(
            "Target bin metadata must contain '{}' and '{}'. Regenerate the bin file with fixed negatives.".format(
                src_key, dst_key
            )
        )
    src_type, _relation, dst_type = target_etype
    src_local = metadata[src_key].long().view(-1).cpu()
    dst_local = metadata[dst_key].long().view(-1).cpu()
    if src_local.numel() != dst_local.numel():
        raise ValueError("{} and {} must have the same length".format(src_key, dst_key))
    if src_local.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if int(src_local.min()) < 0 or int(src_local.max()) >= target_global_nids[src_type].numel():
        raise ValueError("{} contains node ids outside target node type '{}'".format(src_key, src_type))
    if int(dst_local.min()) < 0 or int(dst_local.max()) >= target_global_nids[dst_type].numel():
        raise ValueError("{} contains node ids outside target node type '{}'".format(dst_key, dst_type))
    return torch.stack([
        target_global_nids[src_type][src_local],
        target_global_nids[dst_type][dst_local],
    ], dim=0)


def _compact_global_ids(raw_snapshots, target_snapshot):
    by_type = {ntype: [] for ntype in target_snapshot.graph.ntypes}
    for snapshot in raw_snapshots + [target_snapshot]:
        for ntype, ids in snapshot.global_nids.items():
            by_type.setdefault(ntype, []).append(ids)
    maps = {}
    num_global_nodes = {}
    for ntype, chunks in by_type.items():
        unique_ids = torch.unique(torch.cat(chunks)).tolist()
        maps[ntype] = {int(raw_id): idx for idx, raw_id in enumerate(unique_ids)}
        num_global_nodes[ntype] = len(unique_ids)
    return maps, num_global_nodes


def _apply_global_maps(snapshot, raw_to_compact):
    return Snapshot(
        graph=snapshot.graph,
        features=snapshot.features,
        global_nids={
            ntype: _remap_tensor(ids, raw_to_compact[ntype])
            for ntype, ids in snapshot.global_nids.items()
        },
        name=snapshot.name,
    )


def load_dynamic_link_data(args):
    snapshot_paths = _split_paths(args.snapshot_bins)
    if not snapshot_paths:
        raise ValueError("--snapshot-bins must contain at least one path")

    raw_graphs = [_load_one_graph(path, args.graph_index)[0] for path in snapshot_paths]
    target_graph, target_metadata = _load_one_graph(args.target_bin, args.target_graph_index)
    target_spec = getattr(args, "target_etype", None) or getattr(args, "target_layer", None)
    target_etype = _resolve_target_etype(target_graph, target_spec)

    raw_snapshots = [
        _snapshot_from_graph(path, graph, args.feat_key, args.global_id_key, args.feature_fallback)
        for path, graph in zip(snapshot_paths, raw_graphs)
    ]
    target_snapshot = _snapshot_from_graph(args.target_bin, target_graph, args.feat_key, args.global_id_key, args.feature_fallback)

    input_dims = {ntype: feat.shape[1] for ntype, feat in target_snapshot.features.items()}
    for snapshot in raw_snapshots:
        for ntype, feat in snapshot.features.items():
            if ntype not in input_dims:
                input_dims[ntype] = feat.shape[1]
            elif feat.shape[1] != input_dims[ntype]:
                raise ValueError("Feature dimension mismatch for node type '{}'.".format(ntype))

    raw_to_compact, num_global_nodes = _compact_global_ids(raw_snapshots, target_snapshot)
    snapshots = [_apply_global_maps(snapshot, raw_to_compact) for snapshot in raw_snapshots]
    target_global_nids = {
        ntype: _remap_tensor(ids, raw_to_compact[ntype])
        for ntype, ids in target_snapshot.global_nids.items()
    }

    train_edges = _masked_edges(target_graph, target_etype, target_global_nids, _edge_mask(target_graph, target_etype, "train"))
    valid_edges = _masked_edges(target_graph, target_etype, target_global_nids, _edge_mask(target_graph, target_etype, "valid"))
    test_edges = _masked_edges(target_graph, target_etype, target_global_nids, _edge_mask(target_graph, target_etype, "test"))
    train_neg_edges = _stored_negative_edges(target_metadata, "train", target_etype, target_global_nids)
    valid_neg_edges = _stored_negative_edges(target_metadata, "valid", target_etype, target_global_nids)
    test_neg_edges = _stored_negative_edges(target_metadata, "test", target_etype, target_global_nids)

    return DynamicLinkData(
        snapshots=snapshots,
        target_graph=target_graph,
        target_global_nids=target_global_nids,
        input_dims=input_dims,
        num_global_nodes=num_global_nodes,
        target_etype=target_etype,
        train_pos_edges=train_edges,
        valid_pos_edges=valid_edges,
        test_pos_edges=test_edges,
        train_neg_edges=train_neg_edges,
        valid_neg_edges=valid_neg_edges,
        test_neg_edges=test_neg_edges,
    )


def move_snapshots_to_device(snapshots, device):
    output = []
    for item in snapshots:
        output.append(
            Snapshot(
                graph=item.graph.to(device),
                features={ntype: features.to(device) for ntype, features in item.features.items()},
                global_nids={ntype: nids.to(device) for ntype, nids in item.global_nids.items()},
                name=item.name,
            )
        )
    return output
