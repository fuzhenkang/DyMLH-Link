import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_sageconv():
    try:
        from dgl.nn import SAGEConv
    except ImportError as exc:
        raise ImportError("DGL with dgl.nn.SAGEConv is required.") from exc
    return SAGEConv


class SnapshotGraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, aggregator_type="mean", dropout=0.5):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        SAGEConv = _import_sageconv()
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type))
        self.dropout = dropout

    def forward(self, graph, features):
        h = features.float()
        for idx, layer in enumerate(self.layers):
            h = layer(graph, h)
            if idx < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
        return h


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        self.dropout = dropout

    def forward(self, sequence, presence_mask):
        scores = self.score(sequence).squeeze(-1)
        scores = scores.masked_fill(~presence_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(presence_mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = F.dropout(weights / denom, self.dropout, training=self.training)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, temporal_model="gru", num_layers=1, num_heads=4, dropout=0.0, max_snapshots=64):
        super().__init__()
        self.temporal_model = temporal_model
        if temporal_model == "gru":
            self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif temporal_model == "lstm":
            self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif temporal_model == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.position = nn.Parameter(torch.zeros(1, max_snapshots, hidden_dim))
            nn.init.normal_(self.position, std=0.02)
        elif temporal_model == "attention":
            self.encoder = TemporalAttention(hidden_dim, dropout)
        else:
            raise ValueError("temporal_model must be gru, lstm, transformer, or attention")

    def _last_valid(self, outputs, presence_mask):
        last_idx = presence_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        return outputs.gather(1, gather_idx).squeeze(1)

    def forward(self, sequence, presence_mask):
        sequence = sequence * presence_mask.unsqueeze(-1).float()
        if self.temporal_model in {"gru", "lstm"}:
            outputs, _state = self.encoder(sequence)
            output = self._last_valid(outputs, presence_mask)
            return torch.where(presence_mask.any(dim=1, keepdim=True), output, torch.zeros_like(output))
        if self.temporal_model == "transformer":
            length = sequence.shape[1]
            padding_mask = ~presence_mask
            all_missing = padding_mask.all(dim=1)
            if all_missing.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_missing, 0] = False
            encoded = self.encoder(sequence + self.position[:, :length], src_key_padding_mask=padding_mask)
            output = self._last_valid(encoded, presence_mask)
            return torch.where(presence_mask.any(dim=1, keepdim=True), output, torch.zeros_like(output))
        return self.encoder(sequence, presence_mask)


class DynamicHomogeneousLinkPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_global_nodes,
        gnn_layers=2,
        sage_aggregator_type="mean",
        temporal_model="gru",
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.5,
        predictor="dot",
        predictor_hidden_dim=None,
        max_snapshots=64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_global_nodes = num_global_nodes
        self.snapshot_encoder = SnapshotGraphSAGEEncoder(input_dim, hidden_dim, gnn_layers, sage_aggregator_type, dropout)
        self.temporal_encoder = TemporalEncoder(hidden_dim, temporal_model, temporal_layers, temporal_heads, dropout, max_snapshots)
        self.predictor = predictor
        if predictor == "distmult":
            self.relation = nn.Parameter(torch.ones(hidden_dim))
        elif predictor == "mlp":
            inner_dim = predictor_hidden_dim or hidden_dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 4, inner_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, 1),
            )
        elif predictor != "dot":
            raise ValueError("predictor must be dot, distmult, or mlp")

    def encode(self, snapshots):
        device = next(self.parameters()).device
        sequence = []
        masks = []
        for snapshot in snapshots:
            local_h = self.snapshot_encoder(snapshot.graph, snapshot.features)
            global_h = local_h.new_zeros((self.num_global_nodes, self.hidden_dim))
            global_h[snapshot.global_nids] = local_h
            mask = torch.zeros(self.num_global_nodes, dtype=torch.bool, device=device)
            mask[snapshot.global_nids] = True
            sequence.append(global_h)
            masks.append(mask)
        sequence = torch.stack(sequence, dim=1)
        presence_mask = torch.stack(masks, dim=1)
        return self.temporal_encoder(sequence, presence_mask)

    def score_edges(self, node_embeddings, edges):
        src_h = node_embeddings[edges[0]]
        dst_h = node_embeddings[edges[1]]
        if self.predictor == "mlp":
            edge_h = torch.cat([src_h, dst_h, src_h * dst_h, torch.abs(src_h - dst_h)], dim=1)
            return self.edge_mlp(edge_h).squeeze(-1)
        if self.predictor == "distmult":
            return torch.sum(src_h * self.relation * dst_h, dim=1)
        return torch.sum(F.normalize(src_h, p=2, dim=1) * F.normalize(dst_h, p=2, dim=1), dim=1)

    def forward(self, snapshots, edges):
        node_embeddings = self.encode(snapshots)
        return self.score_edges(node_embeddings, edges)
