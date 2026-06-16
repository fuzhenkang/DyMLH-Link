import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from dymlh_link.data import load_dynamic_link_data, move_snapshots_to_device, sample_negative_edges
from dymlh_link.metrics import compute_metrics, link_loss
from dymlh_link.model import DynamicMCCELinkPredictor


def parse_metapaths(spec, graph):
    if not spec:
        return None
    relation_to_etypes = {}
    for etype in graph.canonical_etypes:
        relation_to_etypes.setdefault(etype[1], []).append(etype)
    metapaths = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = []
        for token in raw.split(">"):
            token = token.strip()
            canonical = tuple(part.strip() for part in token.split(":"))
            if len(canonical) == 3:
                if canonical not in graph.canonical_etypes:
                    raise ValueError("Metapath etype {} is not in graph".format(canonical))
                path.append(canonical)
            else:
                matches = relation_to_etypes.get(token, [])
                if len(matches) != 1:
                    raise ValueError("Relation '{}' in metapath '{}' is not unique. Use src:rel:dst.".format(token, raw))
                path.append(matches[0])
        if path:
            metapaths.append(tuple(path))
    return metapaths or None


def enumerate_metapaths(graph, max_length, closure="both"):
    if max_length < 1:
        raise ValueError("--metapath-length must be >= 1")
    results = []

    def walk(path, current_type, depth):
        if depth >= max_length:
            return
        for etype in graph.canonical_etypes:
            if etype[0] != current_type:
                continue
            next_path = path + [etype]
            is_closed = next_path[0][0] == etype[2]
            if closure == "both" or (closure == "closed" and is_closed) or (closure == "open" and not is_closed):
                results.append(tuple(next_path))
            walk(next_path, etype[2], depth + 1)

    for ntype in graph.ntypes:
        walk([], ntype, 0)
    unique = []
    seen = set()
    for path in results:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def format_metapath(path):
    return ">".join("{}:{}:{}".format(*etype) for etype in path)


def build_parser():
    parser = argparse.ArgumentParser(description="Dynamic MCCE-MHGNN for multilayer heterogeneous target-link prediction.")
    parser.add_argument("--snapshot-bins", type=str, required=True, help="Comma-separated historical DGL .bin snapshots, e.g. 2015.bin,2016.bin,...,2019.bin.")
    parser.add_argument("--target-bin", type=str, required=True, help="Prediction target DGL .bin graph, e.g. 2020.bin. Target edge masks must be stored here.")
    parser.add_argument("--target-etype", type=str, default=None, help="Target canonical edge type, e.g. author:coauthor:author. A unique relation name is also accepted.")
    parser.add_argument("--target-layer", type=str, default=None, help="Backward-compatible alias of --target-etype.")
    parser.add_argument("--graph-index", type=int, default=0, help="Graph index for each historical snapshot bin.")
    parser.add_argument("--target-graph-index", type=int, default=0, help="Graph index for the target bin.")
    parser.add_argument("--feat-key", type=str, default="feat")
    parser.add_argument("--global-id-key", type=str, default="global_id")
    parser.add_argument("--feature-fallback", type=str, default="degree", choices=["degree", "none"])
    parser.add_argument("--undirected", action="store_true", default=False, help="Treat homogeneous positive target edges as undirected during negative sampling.")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2, help="Intra-layer GraphSAGE layer count.")
    parser.add_argument("--sage-aggregator-type", type=str, default="mean", choices=["mean", "pool", "lstm", "gcn"])
    parser.add_argument("--sage-normalize", action="store_true", default=False)
    parser.add_argument("--metapaths", type=str, default=None, help="Comma-separated MCCE metapaths; use rel>rel or src:rel:dst>src:rel:dst.")
    parser.add_argument("--metapath-length", type=int, default=3)
    parser.add_argument("--metapath-closure", type=str, default="both", choices=["closed", "open", "both"])
    parser.add_argument("--metapath-fusion", type=str, default="conv", choices=["mean", "weight", "conv", "cat"])
    parser.add_argument("--fusion-mode", type=str, default="both", choices=["intra", "context", "both"])
    parser.add_argument("--context-encoder", type=str, default="gcn", choices=["gcn", "conv", "mean", "attention"])
    parser.add_argument("--context-use-v", action="store_true", default=False)
    parser.add_argument("--context-heads", type=int, default=8)
    parser.add_argument("--number-layers", type=int, default=1, help="Cross-layer MCCE context layer count.")
    parser.add_argument("--no-gate", action="store_true", default=False, help="Use linear fusion instead of gate fusion between intra and cross-layer embeddings.")
    parser.add_argument("--temporal-model", type=str, default="gru", choices=["gru", "lstm", "transformer", "attention"])
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--predictor", type=str, default="distmult", choices=["dot", "distmult", "mlp"])
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--eval-negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-exclude-layers", type=str, default="target", choices=["target", "all"], help="Exclude positives from target relation only or all relations when sampling negatives.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "pr_auc", "f1"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def make_batch(pos_edges, data, ratio, device):
    dst_type = data.target_etype[2]
    neg_edges = sample_negative_edges(
        pos_edges,
        data.num_global_nodes[dst_type],
        data.all_positive_edges,
        data.target_etype,
        negative_ratio=ratio,
        undirected=data.undirected,
        device=device,
    )
    pos_edges = pos_edges.to(device)
    edges = torch.cat([pos_edges, neg_edges], dim=1)
    labels = torch.cat([
        torch.ones(pos_edges.shape[1], device=device),
        torch.zeros(neg_edges.shape[1], device=device),
    ])
    return edges, labels


def run_split(model, snapshots, data, pos_edges, ratio, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    edges, labels = make_batch(pos_edges, data, ratio, device)
    with torch.set_grad_enabled(is_train):
        scores = model(snapshots, edges)
        loss = link_loss(scores, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    metrics = compute_metrics(scores, labels)
    metrics["loss"] = float(loss.detach().cpu())
    return metrics


def metric_line(epoch, split, metrics):
    return {
        "epoch": epoch,
        "split": split,
        "loss": metrics["loss"],
        "auc": metrics["auc"],
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
    }


def save_outputs(args, records, best_valid, test_metrics):
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or "{}_dynamic_mcce_{}".format(time.strftime("%Y%m%d_%H%M%S"), args.temporal_model)
    csv_path = os.path.join(args.output_dir, "{}_metrics.csv".format(run_name))
    json_path = os.path.join(args.output_dir, "{}_summary.json".format(run_name))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "split", "loss", "auc", "pr_auc", "f1"])
        writer.writeheader()
        writer.writerows(records)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"best_valid": best_valid, "test": test_metrics, "args": vars(args)}, handle, indent=2)
    return csv_path, json_path


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    data = load_dynamic_link_data(args)
    metapaths = parse_metapaths(args.metapaths, data.target_graph) if args.metapaths else enumerate_metapaths(
        data.target_graph, args.metapath_length, args.metapath_closure
    )
    context_metapaths = [mp for mp in metapaths if all(etype[0] != etype[2] for etype in mp)]
    if args.fusion_mode != "intra" and not context_metapaths:
        raise ValueError("No cross-layer heterogeneous metapaths were found. Provide --metapaths or increase --metapath-length.")

    snapshots = move_snapshots_to_device(data.snapshots, device)
    model = DynamicMCCELinkPredictor(
        graph=data.target_graph,
        input_dims=data.input_dims,
        hidden_dim=args.hidden_dim,
        num_global_nodes=data.num_global_nodes,
        target_etype=data.target_etype,
        metapaths=metapaths,
        gnn_layers=args.gnn_layers,
        sage_aggregator_type=args.sage_aggregator_type,
        sage_normalize=args.sage_normalize,
        metapath_fusion=args.metapath_fusion,
        context_encoder=args.context_encoder,
        context_use_v=args.context_use_v,
        context_heads=args.context_heads,
        number_layers=args.number_layers,
        fusion_mode=args.fusion_mode,
        use_gate=not args.no_gate,
        temporal_model=args.temporal_model,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        dropout=args.dropout,
        predictor=args.predictor,
        predictor_hidden_dim=args.predictor_hidden_dim,
        max_snapshots=len(snapshots),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Historical snapshots: {}".format(len(snapshots)))
    print("Target graph edges use train/valid/test masks from: {}".format(args.target_bin))
    print("Target etype: {}".format(data.target_etype))
    print("Node types: {}".format(data.target_graph.ntypes))
    print("Global nodes by type: {}".format(data.num_global_nodes))
    print("MCCE metapaths: {}".format(", ".join(format_metapath(path) for path in metapaths)))
    print("Train/valid/test positives: {}/{}/{}".format(
        data.train_pos_edges.shape[1], data.valid_pos_edges.shape[1], data.test_pos_edges.shape[1]
    ))
    print("Temporal model: {}".format(args.temporal_model))

    records = []
    best_score = -float("inf")
    best_valid = None
    best_test = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_split(model, snapshots, data, data.train_pos_edges, args.negative_ratio, device, optimizer)
        valid_metrics = run_split(model, snapshots, data, data.valid_pos_edges, args.eval_negative_ratio, device)
        test_metrics = run_split(model, snapshots, data, data.test_pos_edges, args.eval_negative_ratio, device)
        records.extend([
            metric_line(epoch, "train", train_metrics),
            metric_line(epoch, "valid", valid_metrics),
            metric_line(epoch, "test", test_metrics),
        ])

        score = valid_metrics[args.early_stop_metric]
        if score > best_score:
            best_score = score
            best_valid = valid_metrics
            best_test = test_metrics
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                "Epoch {:04d} | train loss {:.4f} auc {:.4f} | valid loss {:.4f} auc {:.4f} pr_auc {:.4f} f1 {:.4f} | test auc {:.4f}".format(
                    epoch,
                    train_metrics["loss"],
                    train_metrics["auc"],
                    valid_metrics["loss"],
                    valid_metrics["auc"],
                    valid_metrics["pr_auc"],
                    valid_metrics["f1"],
                    test_metrics["auc"],
                )
            )
        if args.patience > 0 and bad_epochs >= args.patience:
            print("Early stopping at epoch {}.".format(epoch))
            break

    csv_path, json_path = save_outputs(args, records, best_valid, best_test)
    print("Best valid: {}".format(best_valid))
    print("Test at best valid: {}".format(best_test))
    print("Saved metrics to {}".format(csv_path))
    print("Saved summary to {}".format(json_path))


if __name__ == "__main__":
    main()
