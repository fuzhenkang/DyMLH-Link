# DyMLH-Link

Dynamic MCCE-MHGNN for multilayer heterogeneous link prediction.

The task is:

```text
2015-2019 multilayer heterogeneous network snapshots
        -> predict links in one target relation/layer of the 2020 network
```

This project keeps the main spatial architecture close to MCCE-MHGNN. The dynamic part is added after each snapshot has been encoded by the MCCE-style multilayer heterogeneous encoder.

## Data Format

Each historical snapshot and the target graph should be a DGL heterograph saved by `dgl.save_graphs`.

Example schema:

```python
Graph(num_nodes={'author': 5000, 'paper': 17014, 'venue': 8610},
      num_edges={
        ('author', 'coauthor', 'author'): ...,
        ('author', 'author_to_paper', 'paper'): ...,
        ('paper', 'paper_to_author', 'author'): ...,
        ('paper', 'paper_to_venue', 'venue'): ...,
        ('venue', 'venue_to_paper', 'paper'): ...,
        ('venue', 'venue_to_venue', 'venue'): ...
      })
```

Node features and temporal alignment ids are stored by node type:

```python
g.nodes['author'].data['feat']
g.nodes['author'].data['global_id']
g.nodes['paper'].data['feat']
g.nodes['paper'].data['global_id']
g.nodes['venue'].data['feat']
g.nodes['venue'].data['global_id']
```

`global_id` aligns the same real node across years. Different snapshots may contain different node sets.

Only the target-year graph needs split masks on the prediction target relation:

```python
target_etype = ('author', 'coauthor', 'author')
g.edges[target_etype].data['train_mask']
g.edges[target_etype].data['valid_mask']
g.edges[target_etype].data['test_mask']
```

For cross-layer prediction, the target relation can also be heterogeneous:

```python
target_etype = ('author', 'author_to_venue', 'venue')
```

## Model

For each snapshot, the spatial encoder follows MCCE-MHGNN:

```text
node features by type
  -> intra-layer same-type DGL SAGEConv encoder
  -> cross-layer metapath context encoder
  -> metapath context fusion
  -> gate/linear fusion of intra and cross-layer embeddings
```

Then the dynamic module is applied:

```text
snapshot embeddings from 2015, ..., 2019
  -> GRU / LSTM / Transformer / temporal attention
  -> target-year node embeddings
  -> target-relation link predictor
```

The cross-layer part is not simple layer pooling. It uses MCCE-style heterogeneous metapaths such as:

```text
author:author_to_paper:paper>paper:paper_to_author:author
author:author_to_venue:venue>venue:venue_to_author:author
```

Only metapaths made of heterogeneous edges are used for cross-layer context. Same-type relations such as `author:coauthor:author` are handled by the intra-layer encoder implemented with DGL `SAGEConv`. For a node type with multiple same-type relations, the same `SAGEConv` layer is applied to each relation and the relation outputs are averaged. Node types without same-type relations use a learned self-only projection.

## Training Command

```bash
python -u Link_Prediction.py \
  --snapshot-bins data/aminer/graph_2015.bin,data/aminer/graph_2016.bin,data/aminer/graph_2017.bin,data/aminer/graph_2018.bin,data/aminer/graph_2019.bin \
  --target-bin data/aminer/graph_2020.bin \
  --target-etype author:coauthor:author \
  --feat-key feat \
  --global-id-key global_id \
  --hidden-dim 128 \
  --gnn-layers 2 \
  --sage-aggregator-type mean \
  --metapaths "author:author_to_paper:paper>paper:paper_to_author:author,author:author_to_venue:venue>venue:venue_to_author:author" \
  --context-encoder gcn \
  --metapath-fusion conv \
  --fusion-mode both \
  --number-layers 1 \
  --temporal-model gru \
  --temporal-layers 1 \
  --predictor distmult \
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

For cross-layer target prediction:

```bash
--target-etype author:author_to_venue:venue
```

## Important Parameters

`--gnn-layers` controls the number of stacked intra-layer DGL `SAGEConv` layers. `--sage-aggregator-type` selects the official DGL `mean`, `pool`, `lstm`, or `gcn` aggregator.

`--context-encoder` controls the MCCE cross-layer semantic encoder:

```text
gcn / conv / mean / attention
```

`--metapath-fusion` controls fusion across different MCCE metapaths:

```text
mean / weight / conv / cat
```

`--fusion-mode` controls whether the snapshot representation uses:

```text
intra    only same-type intra-layer structure
context  only cross-layer metapath context
both     intra-layer + cross-layer fusion
```

`--no-gate` disables gate fusion and uses linear fusion instead.

`--temporal-model` controls the dynamic module:

```text
gru / lstm / transformer / attention
```

`--predictor` supports:

```text
dot / distmult / mlp
```

## Negative Sampling

`--negative-exclude-layers target` excludes target-relation positive edges from all years.

`--negative-exclude-layers all` excludes positive edges from all relations in all snapshots and the target graph.

## Outputs

Every run saves:

```text
outputs/<run_name>_metrics.csv
outputs/<run_name>_summary.json
```

The CSV contains train, validation, and test `loss`, `auc`, `pr_auc`, and `f1` for each epoch.
