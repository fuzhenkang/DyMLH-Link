# DyMLH-Link

Snapshot-based dynamic homogeneous link prediction.

The training logic follows this setting:

```text
Input snapshots: 2015, 2016, 2017, 2018, 2019
Prediction target: 2020 links
```

The historical snapshots are used only as temporal context. The target graph, for example `2020.bin`, stores the target positive edges and their `train_mask`, `valid_mask`/`val_mask`, and `test_mask`.

## Data Format

Each snapshot is a homogeneous DGL graph saved by `dgl.save_graphs`.

Historical snapshots:

```python
g.ndata["feat"]       # optional node features
g.ndata["global_id"]  # optional stable node id across years
```

Target graph:

```python
g.ndata["feat"]          # optional
g.ndata["global_id"]     # optional but recommended
g.edata["train_mask"]
g.edata["valid_mask"]    # or g.edata["val_mask"]
g.edata["test_mask"]
```

If `global_id` is not present, local node ids are used. If `feat` is not present, the loader uses `[in_degree, out_degree]` as fallback features by default.

## Model

Each historical snapshot is encoded by a shared GraphSAGE encoder. Node embeddings are aligned by `global_id`, then fused across time by one temporal module:

```text
GraphSAGE(G_2015), ..., GraphSAGE(G_2019)
        -> global node alignment
        -> GRU / LSTM / Transformer / temporal attention
        -> dot / DistMult / MLP link predictor for 2020
```

The temporal module is controlled by:

```text
--temporal-model gru
--temporal-model lstm
--temporal-model transformer
--temporal-model attention
```

## Training Command

```bash
python -u Link_Prediction.py \
  --snapshot-bins data/graph_2015.bin,data/graph_2016.bin,data/graph_2017.bin,data/graph_2018.bin,data/graph_2019.bin \
  --target-bin data/graph_2020.bin \
  --feat-key feat \
  --global-id-key global_id \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --sage-aggregator-type mean \
  --temporal-model gru \
  --temporal-layers 1 \
  --predictor dot \
  --negative-ratio 1.0 \
  --eval-negative-ratio 1.0 \
  --epochs 500 \
  --patience 50 \
  --early-stop-metric auc \
  --log-every 10 \
  --output-dir outputs
```

For ablation experiments, change only the temporal module:

```bash
--temporal-model lstm
--temporal-model transformer --temporal-heads 4 --temporal-layers 2
--temporal-model attention
```

## Outputs

Every run saves:

```text
outputs/<run_name>_metrics.csv
outputs/<run_name>_summary.json
```

The CSV contains train, validation, and test `loss`, `auc`, `pr_auc`, and `f1` for each epoch.
