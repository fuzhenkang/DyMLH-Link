import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dymlh_link.data import load_dynamic_link_data, move_snapshots_to_device, sample_negative_edges
from dymlh_link.metrics import binary_metrics
from dymlh_link.model import DynamicHomogeneousLinkPredictor


def build_parser():
    parser = argparse.ArgumentParser(description="Snapshot-based dynamic homogeneous link prediction.")
    parser.add_argument("--snapshot-bins", type=str, required=True, help="Comma-separated historical DGL .bin snapshots, e.g. 2015.bin,2016.bin,...,2019.bin.")
    parser.add_argument("--target-bin", type=str, required=True, help="Prediction target DGL .bin graph, e.g. 2020.bin. Edge masks must be stored here.")
    parser.add_argument("--graph-index", type=int, default=0, help="Graph index for each historical snapshot bin.")
    parser.add_argument("--target-graph-index", type=int, default=0, help="Graph index for the target bin.")
    parser.add_argument("--feat-key", type=str, default="feat")
    parser.add_argument("--global-id-key", type=str, default="global_id")
    parser.add_argument("--feature-fallback", type=str, default="degree", choices=["degree", "none"])
    parser.add_argument("--undirected", action="store_true", default=False, help="Treat positive target edges as undirected during negative sampling.")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--sage-aggregator-type", type=str, default="mean", choices=["mean", "pool", "lstm", "gcn"])
    parser.add_argument("--temporal-model", type=str, default="gru", choices=["gru", "lstm", "transformer", "attention"])
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--predictor", type=str, default="dot", choices=["dot", "distmult", "mlp"])
    parser.add_argument("--predictor-hidden-dim", type=int, default=None)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--eval-negative-ratio", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "pr_auc", "f1"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def make_batch(pos_edges, data, ratio, device):
    neg_edges = sample_negative_edges(
        pos_edges,
        data.num_global_nodes,
        data.all_positive_edges,
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
        logits = model(snapshots, edges)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    metrics = binary_metrics(logits, labels)
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
    run_name = args.run_name or "{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), args.temporal_model)
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
    snapshots = move_snapshots_to_device(data.snapshots, device)
    model = DynamicHomogeneousLinkPredictor(
        input_dim=data.input_dim,
        hidden_dim=args.hidden_dim,
        num_global_nodes=data.num_global_nodes,
        gnn_layers=args.gnn_layers,
        sage_aggregator_type=args.sage_aggregator_type,
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
    print("Global nodes: {}".format(data.num_global_nodes))
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
