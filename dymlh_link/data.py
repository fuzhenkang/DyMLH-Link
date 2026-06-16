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


def _get_ndata(graph):
    if hasattr(graph, "ndata"):
        return graph.ndata
    raise TypeError("This project expects homogeneous DGLGraph snapshots.")


def _node_global_ids(graph, global_id_key):
    ndata = _get_ndata(graph)
    if global_id_key in ndata:
        return ndata[global_id_key].long().cpu()
    return torch.arange(graph.num_nodes(), dtype=torch.long)


def _node_features(graph, feat_key, fallback):
    ndata = _get_ndata(graph)
    if feat_key in ndata:
        return ndata[feat_key].float().cpu()
    if fallback == "degree":
        indeg = graph.in_degrees().float().cpu().unsqueeze(1)
        outdeg = graph.out_degrees().float().cpu().unsqueeze(1)
        return torch.cat([indeg, outdeg], dim=1)
    raise KeyError("Feature key '{}' was not found in graph.ndata".format(feat_key))


def _load_one_graph(path, graph_index):
    dgl = _import_dgl()
    graphs, _metadata = dgl.load_graphs(path)
    if graph_index >= len(graphs):
        raise IndexError("{} contains {} graph(s), but --graph-index={} was requested".format(path, len(graphs), graph_index))
    return graphs[graph_index]


def _edge_mask(graph, split):
    aliases = {
        "train": ["train_mask"],
        "valid": ["valid_mask", "val_mask"],
        "test": ["test_mask"],
    }
    for key in aliases[split]:
        if key in graph.edata:
            return graph.edata[key].bool().cpu()
    raise KeyError("Target graph edata does not contain a {} mask".format(split))


def _remap_tensor(values, raw_to_compact):
    return torch.tensor([raw_to_compact[int(value)] for value in values.tolist()], dtype=torch.long)


def _target_edges(graph, target_global_nids, mask):
    src, dst = graph.edges()
    src_compact = target_global_nids[src.cpu()][mask]
    dst_compact = target_global_nids[dst.cpu()][mask]
    return torch.stack([src_compact, dst_compact], dim=0)


def _edge_set(graph, compact_global_nids):
    src, dst = graph.edges()
    src_global = compact_global_nids[src.cpu()]
    dst_global = compact_global_nids[dst.cpu()]
    return set(zip(src_global.tolist(), dst_global.tolist()))


def load_dynamic_link_data(args):
    snapshot_paths = _split_paths(args.snapshot_bins)
    if not snapshot_paths:
        raise ValueError("--snapshot-bins must contain at least one path")

    raw_snapshots = []
    all_global_ids = []
    input_dim = None
    for path in snapshot_paths:
        graph = _load_one_graph(path, args.graph_index)
        global_nids = _node_global_ids(graph, args.global_id_key)
        features = _node_features(graph, args.feat_key, args.feature_fallback)
        if input_dim is None:
            input_dim = features.shape[1]
        elif features.shape[1] != input_dim:
            raise ValueError("All snapshots must have the same feature dimension.")
        raw_snapshots.append(Snapshot(graph=graph, features=features, global_nids=global_nids, name=path))
        all_global_ids.append(global_nids)

    target_graph = _load_one_graph(args.target_bin, args.target_graph_index)
    target_global_nids = _node_global_ids(target_graph, args.global_id_key)
    all_global_ids.append(target_global_nids)
    unique_ids = torch.unique(torch.cat(all_global_ids)).tolist()
    raw_to_compact = {int(raw_id): idx for idx, raw_id in enumerate(unique_ids)}

    snapshots = [
        Snapshot(
            graph=item.graph,
            features=item.features,
            global_nids=_remap_tensor(item.global_nids, raw_to_compact),
            name=item.name,
        )
        for item in raw_snapshots
    ]
    target_global_nids = _remap_tensor(target_global_nids, raw_to_compact)

    train_edges = _target_edges(target_graph, target_global_nids, _edge_mask(target_graph, "train"))
    valid_edges = _target_edges(target_graph, target_global_nids, _edge_mask(target_graph, "valid"))
    test_edges = _target_edges(target_graph, target_global_nids, _edge_mask(target_graph, "test"))

    all_positive = _edge_set(target_graph, target_global_nids)
    for snapshot in snapshots:
        all_positive |= _edge_set(snapshot.graph, snapshot.global_nids)
    if args.undirected:
        all_positive |= set((dst, src) for src, dst in all_positive)

    return DynamicLinkData(
        snapshots=snapshots,
        target_graph=target_graph,
        target_global_nids=target_global_nids,
        input_dim=input_dim,
        num_global_nodes=len(unique_ids),
        train_pos_edges=train_edges,
        valid_pos_edges=valid_edges,
        test_pos_edges=test_edges,
        all_positive_edges=all_positive,
        undirected=args.undirected,
    )


def move_snapshots_to_device(snapshots, device):
    output = []
    for item in snapshots:
        output.append(
            Snapshot(
                graph=item.graph.to(device),
                features=item.features.to(device),
                global_nids=item.global_nids.to(device),
                name=item.name,
            )
        )
    return output


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
