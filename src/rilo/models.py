import torch
from torch import nn
from torch.nn import functional as F


class Operators(nn.Module):
    def __init__(self, dimension, hidden=256, codes=16, continuous=False, mapping=None):
        super().__init__()
        self.metadata = {"dimension": dimension, "hidden": hidden, "codes": codes, "continuous": continuous,
                         "mapping": mapping}
        self.continuous = continuous
        self.encoder = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.transition = nn.GRUCell(hidden, hidden)
        self.residual = nn.Linear(hidden, dimension)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.relation_projector = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.router = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, codes))
        self.codes = nn.Parameter(torch.empty(codes, hidden))
        nn.init.orthogonal_(self.codes)
        self.register_buffer("mapping", torch.tensor(mapping if mapping is not None else [], dtype=torch.long))

    def decode(self, hidden, anchor):
        return F.normalize(anchor + torch.sigmoid(self.gate(hidden)) * self.residual(hidden), dim=-1)

    def step(self, hidden, anchor, code):
        hidden = self.transition(self.codes[code], hidden)
        return hidden, self.decode(hidden, anchor)

    def assignment(self, relation_ids, vectors, temperature=0.2):
        if self.mapping.numel():
            probabilities = F.one_hot(self.mapping[relation_ids], len(self.codes)).float()
        else:
            probabilities = F.softmax(self.router(vectors) / temperature, dim=-1)
        hard = F.one_hot(probabilities.argmax(-1), len(self.codes)).to(probabilities.dtype)
        return hard + probabilities - probabilities.detach(), probabilities

    def forward(self, anchor, relation_ids, lengths, relation_vectors, temperature=0.2):
        hidden = self.encoder(anchor)
        distributions = []
        for depth in range(relation_ids.shape[1]):
            active = lengths > depth
            ids = relation_ids[:, depth].clamp_min(0)
            vectors = relation_vectors[ids]
            if self.continuous:
                action = self.relation_projector(vectors)
            else:
                assigned, probabilities = self.assignment(ids, vectors, temperature)
                action = assigned @ self.codes
                distributions.append(probabilities[active])
            update = self.transition(action, hidden)
            hidden = torch.where(active[:, None], update, hidden)
        extra = anchor.new_zeros(2)
        if distributions:
            probabilities = torch.cat(distributions).clamp_min(1e-8)
            marginal = probabilities.mean(0)
            extra = torch.stack(((marginal * (marginal * len(self.codes)).log()).sum(),
                                 -(probabilities * probabilities.log()).sum(-1).mean()))
        return self.decode(hidden, anchor), extra

    @torch.no_grad()
    def route(self, ids, vectors):
        return self.assignment(ids, vectors)[0].argmax(-1)


class Proposal(nn.Module):
    def __init__(self, dimension, hidden, codes, depth):
        super().__init__()
        self.metadata = {"dimension": dimension, "hidden": hidden, "codes": codes, "depth": depth}
        self.project = nn.Linear(dimension, hidden)
        self.depth = nn.Embedding(depth + 1, hidden)
        self.network = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.GELU(), nn.Linear(hidden, codes))

    def forward(self, state, pivot, endpoint, depth):
        q, memory = state.chunk(2, dim=-1)
        features = [self.project(item) for item in (q, memory, pivot, endpoint)]
        features.append(self.depth(depth))
        return self.network(torch.cat(features, dim=-1))


class ContinuousPlan(nn.Module):
    def __init__(self, dimension, hidden=256, depth=3, codes=16):
        super().__init__()
        self.metadata = {"dimension": dimension, "hidden": hidden, "depth": depth, "codes": codes}
        self.depth = depth
        self.world = Operators(dimension, hidden, codes=codes, continuous=True)
        self.condition = nn.Sequential(nn.Linear(dimension * 3, hidden), nn.GELU())
        self.action = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, state, anchor):
        observed = self.condition(torch.cat((state, anchor), -1))
        hidden = self.world.encoder(anchor)
        for _ in range(self.depth):
            action = self.action(torch.cat((hidden, observed), -1))
            hidden = self.world.transition(action, hidden)
        return self.world.decode(hidden, anchor), hidden


class ValueModel(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.metadata = {"dimension": dimension, "hidden": hidden}
        self.dimension = dimension
        self.project = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.network = nn.Sequential(nn.Linear(hidden * 6 + 6, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, features):
        vectors = features[..., :6 * self.dimension].reshape(*features.shape[:-1], 6, self.dimension)
        projected = self.project(vectors).flatten(-2)
        raw = self.network(torch.cat((projected, features[..., 6 * self.dimension:]), -1)).squeeze(-1)
        return torch.where(features[..., -1] > 0, raw, torch.zeros_like(raw))
