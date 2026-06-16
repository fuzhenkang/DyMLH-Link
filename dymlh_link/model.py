import torch
import torch.nn as nn
import torch.nn.functional as F


def _etype_key(etype):
    return "homogeneous" if etype is None else "__".join(etype)


class RelationGraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator_type="mean", dropout=0.0):
        super().__init__()
        if aggregator_type not in {"mean", "pool", "lstm", "gcn"}:
            raise ValueError("aggregator_type must be mean, pool, lstm, or gcn")
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neigh = nn.Linear(in_dim, out_dim)
        self.fc_gcn = nn.Linear(in_dim, out_dim)
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(in_dim, in_dim)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)

    def _aggregate_one_relation(self, graph, ntype, etype, features, message_features, reducer):
        import dgl.function as fn
        if etype is not None and etype not in graph.canonical_etypes:
            return features.new_zeros(features.shape)
        with graph.local_scope():
            reduce_func = fn.max("m", "h_sage_neigh") if reducer == "max" else fn.mean("m", "h_sage_neigh")
            if ntype is None:
                graph.ndata["h_sage_src"] = message_features
                graph.update_all(fn.copy_u("h_sage_src", "m"), reduce_func)
                return graph.ndata.get("h_sage_neigh", features.new_zeros(features.shape))
            graph.nodes[ntype].data["h_sage_src"] = message_features
            graph.update_all(fn.copy_u("h_sage_src", "m"), reduce_func, etype=etype)
            return graph.nodes[ntype].data.get("h_sage_neigh", features.new_zeros(features.shape))

    def _aggregate(self, graph, ntype, etypes, features):
        if self.aggregator_type == "pool":
            message_features = F.relu(self.fc_pool(features))
            reducer = "max"
        else:
            message_features = features
            reducer = "mean"
        outputs = [self._aggregate_one_relation(graph, ntype, etype, features, message_features, reducer) for etype in etypes]
        if not outputs:
            return features.new_zeros(features.shape)
        if self.aggregator_type == "lstm":
            sequence = torch.stack(outputs, dim=1)
            _out, (hidden, _cell) = self.lstm(sequence)
            return hidden[-1]
        return torch.mean(torch.stack(outputs), dim=0)

    def forward(self, graph, ntype, etypes, features):
        features = features.float()
        neigh = self._aggregate(graph, ntype, etypes, features)
        if self.aggregator_type == "gcn":
            h = self.fc_gcn((features + neigh) * 0.5)
        else:
            h = self.fc_self(features) + self.fc_neigh(neigh)
        return F.dropout(F.relu(h), self.dropout, training=self.training)


class RelationGraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, aggregator_type="mean", dropout=0.5):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.layers.append(RelationGraphSAGELayer(in_dim, hidden_dim, aggregator_type, dropout))

    def forward(self, graph, ntype, etypes, features):
        h = features
        for layer in self.layers:
            h = layer(graph, ntype, etypes, h)
        return h


class LayerFusion(nn.Module):
    def __init__(self, num_layers, hidden_dim, fusion_type="attention", dropout=0.0):
        super().__init__()
        if fusion_type not in {"mean", "attention", "weight", "cat"}:
            raise ValueError("layer_fusion must be mean, attention, weight, or cat")
        self.fusion_type = fusion_type
        if fusion_type == "attention":
            self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        elif fusion_type == "weight":
            self.weight = nn.Parameter(torch.full((num_layers,), 1.0 / max(1, num_layers)))
        elif fusion_type == "cat":
            self.linear = nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, h_list):
        if len(h_list) == 1:
            return h_list[0]
        stacked = torch.stack(h_list, dim=1)
        if self.fusion_type == "mean":
            return torch.mean(stacked, dim=1)
        if self.fusion_type == "weight":
            weight = torch.softmax(self.weight, dim=0)
            return torch.sum(stacked * weight.view(1, -1, 1), dim=1)
        if self.fusion_type == "cat":
            return F.relu(self.linear(torch.flatten(stacked, start_dim=1)))
        alpha = torch.softmax(self.attn(stacked).squeeze(-1), dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        return torch.sum(stacked * alpha.unsqueeze(-1), dim=1)


class MultilayerSnapshotEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_etypes, node_type, num_layers=2, aggregator_type="mean", layer_fusion="attention", dropout=0.5):
        super().__init__()
        self.layer_etypes = layer_etypes
        self.node_type = node_type
        self.encoders = nn.ModuleDict({_etype_key(etype): RelationGraphSAGEEncoder(input_dim, hidden_dim, num_layers, aggregator_type, dropout) for etype in layer_etypes})
        self.fusion = LayerFusion(len(layer_etypes), hidden_dim, layer_fusion, dropout)

    def forward(self, graph, features):
        h_list = [self.encoders[_etype_key(etype)](graph, self.node_type, [etype], features) for etype in self.layer_etypes]
        return self.fusion(h_list)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        self.dropout = dropout

    def forward(self, sequence, presence_mask):
        scores = self.score(sequence).squeeze(-1).masked_fill(~presence_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(presence_mask, weights, torch.zeros_like(weights))
        weights = F.dropout(weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12), self.dropout, training=self.training)
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
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
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


class DynamicMultilayerLinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_global_nodes, layer_etypes, node_type, gnn_layers=2, sage_aggregator_type="mean", layer_fusion="attention", temporal_model="gru", temporal_layers=1, temporal_heads=4, dropout=0.5, predictor="dot", predictor_hidden_dim=None, max_snapshots=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_global_nodes = num_global_nodes
        self.snapshot_encoder = MultilayerSnapshotEncoder(input_dim, hidden_dim, layer_etypes, node_type, gnn_layers, sage_aggregator_type, layer_fusion, dropout)
        self.temporal_encoder = TemporalEncoder(hidden_dim, temporal_model, temporal_layers, temporal_heads, dropout, max_snapshots)
        self.predictor = predictor
        if predictor == "distmult":
            self.relation = nn.Parameter(torch.ones(hidden_dim))
        elif predictor == "mlp":
            inner_dim = predictor_hidden_dim or hidden_dim
            self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 4, inner_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(inner_dim, 1))
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
        return self.temporal_encoder(torch.stack(sequence, dim=1), torch.stack(masks, dim=1))

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
        return self.score_edges(self.encode(snapshots), edges)
