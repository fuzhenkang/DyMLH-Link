from dataclasses import dataclass

import torch


@dataclass
class Snapshot:
    graph: object
    features: torch.Tensor
    global_nids: torch.Tensor
    name: str


@dataclass
class DynamicLinkData:
    snapshots: list
    target_graph: object
    target_global_nids: torch.Tensor
    input_dim: int
    num_global_nodes: int
    node_type: str
    target_etype: tuple
    layer_etypes: list
    train_pos_edges: torch.Tensor
    valid_pos_edges: torch.Tensor
    test_pos_edges: torch.Tensor
    all_positive_edges: set
    undirected: bool


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
    graphs, _metadata = dgl.load_graphs(path)
    if graph_index >= len(graphs):
        raise IndexError("{} contains {} graph(s), but --graph-index={} was requested".format(path, len(graphs), graph_index))
    return graphs[graph_index]


def _resolve_node_type(graph, node_type=None):
    is_homogeneous = getattr(graph, "is_homogeneous", False)
    if callable(is_homogeneous):
        is_homogeneous = is_homogeneous()
    if is_homogeneous:
        return None
    if node_type is not None:
        if node_type not in graph.ntypes:
            raise ValueError("--node-type '{}' is not in graph.ntypes: {}".format(node_type, graph.ntypes))
        return node_type
    if len(graph.ntypes) == 1:
        return graph.ntypes[0]
    raise ValueError("--node-type is required when a graph has multiple node types: {}".format(graph.ntypes))


def _node_data(graph, node_type):
    return graph.ndata if node_type is None else graph.nodes[node_type].data


def _num_nodes(graph, node_type):
    return graph.num_nodes() if node_type is None else graph.num_nodes(node_type)


def _node_global_ids(graph, node_type, global_id_key):
    ndata = _node_data(graph, node_type)
    if global_id_key in ndata:
        return ndata[global_id_key].long().cpu()
    return torch.arange(_num_nodes(graph, node_type), dtype=torch.long)


def _node_features(graph, node_type, feat_key, fallback):
    ndata = _node_data(graph, node_type)
    if feat_key in ndata:
        return ndata[feat_key].float().cpu()
    if fallback == "degree":
        if node_type is None:
            indeg = graph.in_degrees().float().cpu().unsqueeze(1)
            outdeg = graph.out_degrees().float().cpu().unsqueeze(1)
        else:
            indeg = torch.zeros(graph.num_nodes(node_type), dtype=torch.float32)
            outdeg = torch.zeros(graph.num_nodes(node_type), dtype=torch.float32)
            for etype in graph.canonical_etypes:
                if etype[2] == node_type:
                    indeg += graph.in_degrees(etype=etype).float().cpu()
                if etype[0] == node_type:
                    outdeg += graph.out_degrees(etype=etype).float().cpu()
            indeg = indeg.unsqueeze(1)
            outdeg = outdeg.unsqueeze(1)
        return torch.cat([indeg, outdeg], dim=1)
    raise KeyError("Feature key '{}' was not found in graph node data".format(feat_key))


def _resolve_target_etype(graph, node_type, target_layer):
    if node_type is None:
        return None
    canonical = tuple(part.strip() for part in target_layer.split(":"))
    if len(canonical) == 3 and canonical in graph.canonical_etypes:
        return canonical
    matches = [etype for etype in graph.canonical_etypes if etype[1] == target_layer]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("--target-layer '{}' is not in graph canonical etypes: {}".format(target_layer, graph.canonical_etypes))
    raise ValueError("--target-layer '{}' is ambiguous. Use src:rel:dst. Matches: {}".format(target_layer, matches))


def _parse_layer_etypes(graph, node_type, use_layers):
    if node_type is None:
        return [None]
    candidates = [etype for etype in graph.canonical_etypes if etype[0] == node_type and etype[2] == node_type]
    if not use_layers:
        return candidates
    relation_to_etypes = {}
    for etype in candidates:
        relation_to_etypes.setdefault(etype[1], []).append(etype)
    selected = []
    for raw in use_layers.split(","):
        token = raw.strip()
        if not token:
            continue
        canonical = tuple(part.strip() for part in token.split(":"))
        if len(canonical) == 3:
            if canonical not in candidates:
                raise ValueError("--use-layers contains unknown canonical etype {}".format(canonical))
            selected.append(canonical)
        elif token in relation_to_etypes:
            selected.extend(relation_to_etypes[token])
        else:
            raise ValueError("--use-layers contains unknown layer '{}'".format(token))
    return sorted(set(selected), key=candidates.index)


def _edge_mask(graph, etype, split):
    aliases = {"train": ["train_mask"], "valid": ["valid_mask", "val_mask"], "test": ["test_mask"]}
    edata = graph.edata if etype is None else graph.edges[etype].data
    for key in aliases[split]:
        if key in edata:
            return edata[key].bool().cpu()
    layer_name = "homogeneous graph" if etype is None else ":".join(etype)
    raise KeyError("Target layer '{}' does not contain a {} mask".format(layer_name, split))


def _remap_tensor(values, raw_to_compact):
    return torch.tensor([raw_to_compact[int(value)] for value in values.tolist()], dtype=torch.long)


def _edges(graph, etype=None):
    return graph.edges() if etype is None else graph.edges(etype=etype)


def _masked_edges(graph, etype, compact_global_nids, mask):
    src, dst = _edges(graph, etype)
    return torch.stack([compact_global_nids[src.cpu()][mask], compact_global_nids[dst.cpu()][mask]], dim=0)


def _edge_set(graph, etype, compact_global_nids):
    src, dst = _edges(graph, etype)
    src_global = compact_global_nids[src.cpu()]
    dst_global = compact_global_nids[dst.cpu()]
    return set(zip(src_global.tolist(), dst_global.tolist()))


def _positive_exclusion_set(target_graph, target_etype, snapshots, layer_etypes, target_global_nids, mode):
    all_positive = set()
    all_positive |= _edge_set(target_graph, target_etype, target_global_nids)
    if mode == "all":
        for etype in layer_etypes:
            all_positive |= _edge_set(target_graph, etype, target_global_nids)
    for snapshot in snapshots:
        if mode == "target":
            if target_etype in snapshot.graph.canonical_etypes:
                all_positive |= _edge_set(snapshot.graph, target_etype, snapshot.global_nids)
        else:
            for etype in layer_etypes:
                if etype in snapshot.graph.canonical_etypes:
                    all_positive |= _edge_set(snapshot.graph, etype, snapshot.global_nids)
    return all_positive


def load_dynamic_link_data(args):
    snapshot_paths = _split_paths(args.snapshot_bins)
    if not snapshot_paths:
        raise ValueError("--snapshot-bins must contain at least one path")

    raw_graphs = [_load_one_graph(path, args.graph_index) for path in snapshot_paths]
    target_graph = _load_one_graph(args.target_bin, args.target_graph_index)
    node_type = _resolve_node_type(target_graph, args.node_type)
    target_etype = _resolve_target_etype(target_graph, node_type, args.target_layer) if node_type is not None else None
    layer_etypes = _parse_layer_etypes(target_graph, node_type, args.use_layers)
    if node_type is not None and target_etype not in layer_etypes:
        layer_etypes = layer_etypes + [target_etype]

    raw_snapshots = []
    all_global_ids = []
    input_dim = None
    for path, graph in zip(snapshot_paths, raw_graphs):
        snapshot_node_type = _resolve_node_type(graph, node_type)
        global_nids = _node_global_ids(graph, snapshot_node_type, args.global_id_key)
        features = _node_features(graph, snapshot_node_type, args.feat_key, args.feature_fallback)
        if input_dim is None:
            input_dim = features.shape[1]
        elif features.shape[1] != input_dim:
            raise ValueError("All snapshots must have the same feature dimension.")
        raw_snapshots.append(Snapshot(graph=graph, features=features, global_nids=global_nids, name=path))
        all_global_ids.append(global_nids)

    target_global_nids = _node_global_ids(target_graph, node_type, args.global_id_key)
    all_global_ids.append(target_global_nids)
    unique_ids = torch.unique(torch.cat(all_global_ids)).tolist()
    raw_to_compact = {int(raw_id): idx for idx, raw_id in enumerate(unique_ids)}

    snapshots = [Snapshot(item.graph, item.features, _remap_tensor(item.global_nids, raw_to_compact), item.name) for item in raw_snapshots]
    target_global_nids = _remap_tensor(target_global_nids, raw_to_compact)

    train_edges = _masked_edges(target_graph, target_etype, target_global_nids, _edge_mask(target_graph, target_etype, "train"))
    valid_edges = _masked_edges(target_graph, target_etype, target_global_nids, _edge_mask(target_graph, target_etype, "valid"))
    test_edges = _masked_edges(target_graph, target_etype, target_global_nids, _edge_mask(target_graph, target_etype, "test"))

    all_positive = _positive_exclusion_set(target_graph, target_etype, snapshots, layer_etypes, target_global_nids, args.negative_exclude_layers)
    if args.undirected:
        all_positive |= set((dst, src) for src, dst in all_positive)

    return DynamicLinkData(snapshots, target_graph, target_global_nids, input_dim, len(unique_ids), node_type, target_etype, layer_etypes, train_edges, valid_edges, test_edges, all_positive, args.undirected)


def move_snapshots_to_device(snapshots, device):
    return [Snapshot(item.graph.to(device), item.features.to(device), item.global_nids.to(device), item.name) for item in snapshots]


def sample_negative_edges(pos_edges, num_nodes, positive_edge_set, negative_ratio=1.0, undirected=False, device=None):
    num_pos = pos_edges.shape[1]
    num_neg = max(1, int(num_pos * negative_ratio))
    src_pos = pos_edges[0].detach().cpu()
    neg_src = []
    neg_dst = []
    while len(neg_src) < num_neg:
        batch_size = max(num_neg - len(neg_src), 1024)
        src = src_pos[torch.randint(0, num_pos, (batch_size,))]
        dst = torch.randint(0, num_nodes, (batch_size,))
        for s, d in zip(src.tolist(), dst.tolist()):
            if s == d:
                continue
            if (s, d) in positive_edge_set:
                continue
            if undirected and (d, s) in positive_edge_set:
                continue
            neg_src.append(s)
            neg_dst.append(d)
            if len(neg_src) >= num_neg:
                break
    edges = torch.tensor([neg_src, neg_dst], dtype=torch.long)
    return edges.to(device) if device is not None else edges
