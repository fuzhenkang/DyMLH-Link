import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_dgl():
    if "dgl.graphbolt" not in sys.modules:
        graphbolt = types.ModuleType("dgl.graphbolt")
        graphbolt.__all__ = []
        sys.modules["dgl.graphbolt"] = graphbolt
    try:
        import dgl
    except ImportError as exc:
        raise ImportError("DGL >= 2.1.0 is required.") from exc
    return dgl


def _etype_key(etype):
    return "__".join(etype)


def _metapath_key(metapath):
    return "||".join(_etype_key(etype) for etype in metapath)


class DGLSAGEConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator_type="mean", activation=F.relu, dropout=0.0, normalize=False):
        super().__init__()
        if aggregator_type not in {"mean", "pool", "lstm", "gcn"}:
            raise ValueError("aggregator_type must be one of mean, pool, lstm, or gcn")
        _import_dgl()
        from dgl.nn import SAGEConv

        self.activation = activation
        self.dropout = dropout
        self.normalize = normalize
        self.conv = SAGEConv(
            in_dim,
            out_dim,
            aggregator_type=aggregator_type,
            feat_drop=dropout,
            activation=activation,
        )
        self.self_only = nn.Linear(in_dim, out_dim)

    def forward(self, graph, ntype, etypes, features):
        features = features.float()
        relation_outputs = []
        for etype in etypes:
            if etype not in graph.canonical_etypes:
                continue
            relation_graph = graph[etype]
            relation_outputs.append(self.conv(relation_graph, (features, features)))
        if relation_outputs:
            h = torch.mean(torch.stack(relation_outputs), dim=0)
        else:
            h = self.self_only(F.dropout(features, self.dropout, training=self.training))
            if self.activation is not None:
                h = self.activation(h)
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)
        return h


class MECCHMetapathFusion(nn.Module):
    def __init__(self, n_metapaths, in_dim, out_dim, fusion_type="conv"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "mean":
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "weight":
            self.weight = nn.Parameter(torch.full((n_metapaths,), 1.0 / n_metapaths))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "conv":
            self.conv = nn.Parameter(torch.full((n_metapaths, in_dim), 1.0 / n_metapaths))
            self.linear = nn.Linear(in_dim, out_dim)
        elif fusion_type == "cat":
            self.linear = nn.Linear(n_metapaths * in_dim, out_dim)
        else:
            raise ValueError("Unknown metapath_fusion '{}'".format(fusion_type))

    def forward(self, h_list):
        if self.fusion_type == "mean":
            fused = torch.mean(torch.stack(h_list), dim=0)
        elif self.fusion_type == "weight":
            fused = torch.sum(torch.stack(h_list) * self.weight[:, None, None], dim=0)
        elif self.fusion_type == "conv":
            fused = torch.sum(torch.stack(h_list).transpose(0, 1) * self.conv, dim=1)
        else:
            fused = torch.hstack(h_list)
        return self.linear(fused), fused


class MetapathContextEncoder(nn.Module):
    def __init__(self, in_dim, encoder_type="gcn", use_v=False, n_heads=8):
        super().__init__()
        if encoder_type == "conv":
            encoder_type = "gcn"
        if in_dim % n_heads != 0:
            raise ValueError("in_dim must be divisible by n_heads")
        self.encoder_type = encoder_type
        self.use_v = use_v
        self.n_heads = n_heads
        self.d_k = in_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        if encoder_type == "attention":
            self.q_linear = nn.Linear(in_dim, in_dim, bias=False)
            self.k_linear = nn.Linear(in_dim, in_dim, bias=False)
            if use_v:
                self.v_linear = nn.Linear(in_dim, in_dim, bias=False)
        elif encoder_type == "gcn":
            self.source_linear = nn.Linear(in_dim, in_dim, bias=False)
            self.self_linear = nn.Linear(in_dim, in_dim, bias=True)
        elif encoder_type != "mean":
            raise ValueError("Unknown context encoder '{}'".format(encoder_type))

    def forward(self, target_embedding, embeddings, suffix_graphs):
        if self.encoder_type == "attention":
            return self._attention_forward(target_embedding, embeddings, suffix_graphs)
        if self.encoder_type == "gcn":
            return self._gcn_forward(target_embedding, embeddings, suffix_graphs)
        return self._mean_forward(target_embedding, embeddings, suffix_graphs)

    def _mean_forward(self, target_embedding, embeddings, suffix_graphs):
        import dgl.function as fn

        message_sum = target_embedding.new_zeros(target_embedding.shape)
        degree_sum = target_embedding.new_zeros(target_embedding.shape[0])
        for graph, source_type, target_type in suffix_graphs:
            if source_type not in embeddings or target_type not in embeddings:
                continue
            with graph.local_scope():
                graph.nodes[source_type].data["h_src"] = embeddings[source_type]
                graph.update_all(fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                message_sum = message_sum + graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros(target_embedding.shape))
                degree_sum = degree_sum + graph.in_degrees().to(target_embedding.device).float()
        return (message_sum + target_embedding) / (degree_sum.unsqueeze(-1) + 1.0).clamp_min(1.0)

    def _gcn_forward(self, target_embedding, embeddings, suffix_graphs):
        import dgl.function as fn

        message_sum = target_embedding.new_zeros(target_embedding.shape)
        degree_sum = target_embedding.new_zeros(target_embedding.shape[0])
        for graph, source_type, target_type in suffix_graphs:
            if source_type not in embeddings or target_type not in embeddings:
                continue
            with graph.local_scope():
                graph.nodes[source_type].data["h_src"] = self.source_linear(embeddings[source_type])
                graph.update_all(fn.copy_u("h_src", "m"), fn.sum("m", "h_neigh"))
                message_sum = message_sum + graph.nodes[target_type].data.get("h_neigh", target_embedding.new_zeros(target_embedding.shape))
                degree_sum = degree_sum + graph.in_degrees().to(target_embedding.device).float()
        neigh = message_sum / degree_sum.unsqueeze(-1).clamp_min(1.0)
        return F.relu(self.self_linear(target_embedding) + neigh)

    def _attention_forward(self, target_embedding, embeddings, suffix_graphs):
        import dgl.function as fn
        from dgl.nn.functional import edge_softmax

        n_target = target_embedding.shape[0]
        q = self.q_linear(target_embedding).view(n_target, self.n_heads, self.d_k)
        outputs = []
        for graph, source_type, target_type in suffix_graphs:
            if source_type not in embeddings or target_type not in embeddings:
                continue
            source_embedding = embeddings[source_type]
            with graph.local_scope():
                graph.nodes[source_type].data["k"] = self.k_linear(source_embedding).view(-1, self.n_heads, self.d_k)
                graph.nodes[source_type].data["v"] = (
                    self.v_linear(source_embedding).view(-1, self.n_heads, self.d_k)
                    if self.use_v else source_embedding.view(-1, self.n_heads, self.d_k)
                )
                graph.nodes[target_type].data["q"] = q
                graph.apply_edges(fn.u_dot_v("k", "q", "score"))
                graph.edata["score"] = graph.edata["score"] / self.sqrt_dk
                graph.edata["alpha"] = edge_softmax(graph, graph.edata["score"], norm_by="dst")
                graph.update_all(fn.u_mul_e("v", "alpha", "m"), fn.sum("m", "h_neigh"))
                zero = target_embedding.new_zeros((n_target, self.n_heads, self.d_k))
                outputs.append(graph.nodes[target_type].data.get("h_neigh", zero).reshape(n_target, -1))
        if not outputs:
            return target_embedding
        return torch.mean(torch.stack(outputs + [target_embedding]), dim=0)


class DynamicMCCESnapshotEncoder(nn.Module):
    """MCCE-MHGNN spatial encoder applied independently to one temporal snapshot."""

    def __init__(self, graph, input_dims, hidden_dim, gnn_layers=2, dropout=0.5, use_gate=True,
                 metapaths=None, metapath_fusion="conv", context_encoder="gcn", context_use_v=False,
                 context_heads=8, number_layers=1, fusion_mode="both", sage_aggregator_type="mean",
                 sage_normalize=False):
        super().__init__()
        if not metapaths:
            raise ValueError("Dynamic MCCE requires at least one metapath")
        self.hidden_dim = hidden_dim
        self.ntypes = list(graph.ntypes)
        self.dropout = dropout
        self.use_gate = use_gate
        self.number_layers = number_layers
        self.fusion_mode = fusion_mode
        self.context_metapaths = [mp for mp in metapaths if all(etype[0] != etype[2] for etype in mp)]
        if fusion_mode != "intra" and not self.context_metapaths:
            raise ValueError("MCCE cross-layer context requires at least one metapath made only of heterogeneous edges")
        self.input_projectors = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in self.ntypes})
        self.intra_etypes = {ntype: [etype for etype in graph.canonical_etypes if etype[0] == ntype and etype[2] == ntype] for ntype in self.ntypes}
        self.intra_sage = nn.ModuleDict({
            ntype: nn.ModuleList([
                DGLSAGEConvLayer(
                    hidden_dim,
                    hidden_dim,
                    aggregator_type=sage_aggregator_type,
                    activation=F.relu,
                    dropout=dropout,
                    normalize=sage_normalize,
                )
                for _ in range(gnn_layers)
            ])
            for ntype in self.ntypes
        })
        self.context_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(number_layers)])
        self.context_encoders = nn.ModuleList()
        self.metapath_fuse = nn.ModuleList()
        for _ in range(number_layers):
            encoders = nn.ModuleDict()
            fusers = nn.ModuleDict()
            for target in self.ntypes:
                target_metapaths = [mp for mp in self.context_metapaths if mp[-1][2] == target]
                for mp in target_metapaths:
                    encoders[_metapath_key(mp)] = MetapathContextEncoder(hidden_dim, context_encoder, context_use_v, context_heads)
                if target_metapaths:
                    fusers[target] = MECCHMetapathFusion(len(target_metapaths), hidden_dim, hidden_dim, metapath_fusion)
            self.context_encoders.append(encoders)
            self.metapath_fuse.append(fusers)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self._suffix_cache = {}

    def _encode_intra(self, graph, features):
        embeddings = {ntype: F.relu(self.input_projectors[ntype](features[ntype].float())) for ntype in self.ntypes if ntype in features}
        for ntype in self.ntypes:
            if ntype not in embeddings:
                continue
            for sage in self.intra_sage[ntype]:
                embeddings[ntype] = sage(graph, ntype, self.intra_etypes[ntype], embeddings[ntype])
        return embeddings

    def _prepare_suffix_graphs(self, graph):
        dgl = _import_dgl()
        graph_key = id(graph)
        if graph_key in self._suffix_cache:
            return self._suffix_cache[graph_key]
        cache = {}
        relations = set(graph.canonical_etypes)
        for mp in self.context_metapaths:
            if any(etype not in relations for etype in mp):
                continue
            suffixes = []
            for start in range(0, len(mp)):
                etype_suffix = [etype[1] for etype in mp[start:]]
                try:
                    suffix_graph = dgl.metapath_reachable_graph(graph, etype_suffix)
                except Exception:
                    continue
                suffix_graph = suffix_graph.to(graph.device)
                suffixes.append((suffix_graph, mp[start][0], mp[-1][2]))
            cache[_metapath_key(mp)] = suffixes
        self._suffix_cache[graph_key] = cache
        return cache

    def _context_embedding(self, graph, embeddings, target, layer_idx):
        if target not in embeddings:
            return None
        suffix_cache = self._prepare_suffix_graphs(graph)
        contexts = []
        target_metapaths = [mp for mp in self.context_metapaths if mp[-1][2] == target]
        for mp in target_metapaths:
            key = _metapath_key(mp)
            if key not in suffix_cache or key not in self.context_encoders[layer_idx]:
                continue
            encoder = self.context_encoders[layer_idx][key]
            contexts.append(encoder(embeddings[target], embeddings, suffix_cache[key]))
        if not contexts:
            return embeddings[target].new_zeros(embeddings[target].shape)
        projected, _ = self.metapath_fuse[layer_idx][target](contexts)
        return self.context_norms[layer_idx](projected)

    def forward(self, graph, features):
        structural = self._encode_intra(graph, features)
        if self.fusion_mode == "intra":
            return structural
        context_input = structural
        cross = None
        for layer_idx in range(self.number_layers):
            cross = {ntype: self._context_embedding(graph, context_input, ntype, layer_idx) for ntype in self.ntypes if ntype in context_input}
            if layer_idx < self.number_layers - 1:
                context_input = {ntype: F.dropout(F.relu(h), self.dropout, training=self.training) for ntype, h in cross.items() if h is not None}
        if self.fusion_mode == "context":
            return cross
        output = {}
        for ntype in structural:
            combined = torch.cat((structural[ntype], cross[ntype]), dim=1)
            if self.use_gate:
                gate = torch.sigmoid(self.gate(combined))
                fused = gate * structural[ntype] + (1.0 - gate) * cross[ntype]
            else:
                fused = F.relu(self.fusion(combined))
            output[ntype] = F.dropout(fused, self.dropout, training=self.training)
        return output


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


class DynamicMCCELinkPredictor(nn.Module):
    def __init__(self, graph, input_dims, hidden_dim, num_global_nodes, target_etype, metapaths,
                 gnn_layers=2, sage_aggregator_type="mean", sage_normalize=False,
                 metapath_fusion="conv", context_encoder="gcn", context_use_v=False,
                 context_heads=8, number_layers=1, fusion_mode="both", use_gate=True,
                 temporal_model="gru", temporal_layers=1, temporal_heads=4, dropout=0.5,
                 predictor="dot", predictor_hidden_dim=None, max_snapshots=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_global_nodes = num_global_nodes
        self.target_etype = target_etype
        self.snapshot_encoder = DynamicMCCESnapshotEncoder(
            graph, input_dims, hidden_dim, gnn_layers=gnn_layers, dropout=dropout, use_gate=use_gate,
            metapaths=metapaths, metapath_fusion=metapath_fusion, context_encoder=context_encoder,
            context_use_v=context_use_v, context_heads=context_heads, number_layers=number_layers,
            fusion_mode=fusion_mode, sage_aggregator_type=sage_aggregator_type, sage_normalize=sage_normalize,
        )
        self.temporal_encoders = nn.ModuleDict({
            ntype: TemporalEncoder(hidden_dim, temporal_model, temporal_layers, temporal_heads, dropout, max_snapshots)
            for ntype in graph.ntypes
        })
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
        sequences = {ntype: [] for ntype in self.num_global_nodes}
        masks = {ntype: [] for ntype in self.num_global_nodes}
        for snapshot in snapshots:
            local_embeddings = self.snapshot_encoder(snapshot.graph, snapshot.features)
            for ntype, total_nodes in self.num_global_nodes.items():
                global_h = next(self.parameters()).new_zeros((total_nodes, self.hidden_dim))
                mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
                if ntype in local_embeddings and ntype in snapshot.global_nids:
                    global_nids = snapshot.global_nids[ntype]
                    global_h[global_nids] = local_embeddings[ntype]
                    mask[global_nids] = True
                sequences[ntype].append(global_h)
                masks[ntype].append(mask)
        output = {}
        for ntype in sequences:
            sequence = torch.stack(sequences[ntype], dim=1)
            presence_mask = torch.stack(masks[ntype], dim=1)
            output[ntype] = self.temporal_encoders[ntype](sequence, presence_mask)
        return output

    def score_edges(self, embeddings, edges):
        src_type, _rel_type, dst_type = self.target_etype
        src_h = embeddings[src_type][edges[0]]
        dst_h = embeddings[dst_type][edges[1]]
        if self.predictor == "mlp":
            edge_h = torch.cat([src_h, dst_h, src_h * dst_h, torch.abs(src_h - dst_h)], dim=1)
            return self.edge_mlp(edge_h).squeeze(-1)
        if self.predictor == "distmult":
            return torch.sum(src_h * self.relation * dst_h, dim=1)
        return torch.sum(F.normalize(src_h, p=2, dim=1) * F.normalize(dst_h, p=2, dim=1), dim=1)

    def forward(self, snapshots, edges):
        embeddings = self.encode(snapshots)
        return self.score_edges(embeddings, edges)
