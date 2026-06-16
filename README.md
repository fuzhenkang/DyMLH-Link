# DyMLH-Link

Snapshot-based dynamic multilayer link prediction.

The intended task is:

```text
2015, 2016, 2017, 2018, 2019 multilayer network snapshots
        -> predict links in one target layer of the 2020 multilayer network
```

Each snapshot is a DGL heterograph with one node type and multiple edge types. Each edge type represents one layer of the multilayer network:

```python
('node', 'layer_0', 'node')
('node', 'layer_1', 'node')
('node', 'layer_2', 'node')
```

Only the prediction target graph, for example `graph_2020.bin`, needs edge masks on the target layer:

```python
target_etype = ('node', 'layer_0', 'node')
g.edges[target_etype].data['train_mask']
g.edges[target_etype].data['valid_mask']
g.edges[target_etype].data['test_mask']
```

## Data Format

Historical and target snapshots should be saved with `dgl.save_graphs`.

Node data:

```python
g.nodes['node'].data['feat']
g.nodes['node'].data['global_id']
```

The `global_id` field aligns the same real node across years. Nodes may appear or disappear across snapshots.

## Model

For every historical snapshot:

```text
each layer edge type -> relation-specific GraphSAGE encoder
all layer embeddings -> layer fusion
yearly embedding sequence -> GRU / LSTM / Transformer / temporal attention
final node embedding -> target-layer link predictor
```

Layer fusion is controlled by:

```text
--layer-fusion mean
--layer-fusion attention
--layer-fusion weight
--layer-fusion cat
```

Temporal fusion is controlled by:

```text
--temporal-model gru
--temporal-model lstm
--temporal-model transformer
--temporal-model attention
```

## Training Command

```bash
python -u Link_Prediction.py \
  --snapshot-bins data/simulated_multilayer/graph_2015.bin,data/simulated_multilayer/graph_2016.bin,data/simulated_multilayer/graph_2017.bin,data/simulated_multilayer/graph_2018.bin,data/simulated_multilayer/graph_2019.bin \
  --target-bin data/simulated_multilayer/graph_2020.bin \
  --node-type node \
  --target-layer layer_0 \
  --use-layers layer_0,layer_1,layer_2 \
  --feat-key feat \
  --global-id-key global_id \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --sage-aggregator-type mean \
  --layer-fusion attention \
  --temporal-model gru \
  --temporal-layers 1 \
  --predictor dot \
  --negative-ratio 1.0 \
  --eval-negative-ratio 1.0 \
  --negative-exclude-layers target \
  --epochs 500 \
  --patience 50 \
  --early-stop-metric auc \
  --log-every 10 \
  --output-dir outputs \
  --undirected
```

`--negative-exclude-layers target` excludes target-layer positive edges from all years during negative sampling. Use `--negative-exclude-layers all` if any edge in any used layer should prevent a node pair from being sampled as negative.

## Simulated Data

Run:

```text
generate_simulated_data.ipynb
```

It creates dynamic multilayer heterographs under:

```text
data/simulated_multilayer/
```

## Outputs

Every run saves:

```text
outputs/<run_name>_metrics.csv
outputs/<run_name>_summary.json
```

The CSV contains train, validation, and test `loss`, `auc`, `pr_auc`, and `f1` for each epoch.
