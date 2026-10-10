"""Same-invariant-input controls for the coupled boundary-memory hypothesis.

These are independently implemented family controls. They are not published
baseline reproductions. Every graph encoder uses the same normalized boundary
response and the same branch-excluded observed leaf features.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .boundary_memory import BoundaryResponseGNN, BoundaryMemoryModel
from .kron_interface import LeafInterfaceBank
from .models import log_interval_kernel_mass


class BoundaryFeatureMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32, num_layers=2, **_):
        super().__init__()
        self.num_classes = num_classes
        layers = []
        width = input_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_dim), nn.GELU()))
            width = hidden_dim
        layers.append(nn.Linear(width, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, bank):
        if bank.query_cache is None:
            raise ValueError("Query/reference cache required")
        local = bank.leaf_x[:, bank.query_cache.query_indices]
        return self.head(local[bank.target_graph])


class BoundaryLocalGNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32, num_layers=2, **_):
        super().__init__()
        self.hidden_dim, self.num_classes = hidden_dim, num_classes
        self.encoder = BoundaryResponseGNN(input_dim, hidden_dim, num_layers)
        self.head = nn.Linear(hidden_dim, num_classes)

    def prepare(self, bank):
        if bank.query_cache is None:
            raise ValueError("Query/reference cache required")
        h = self.encoder(bank.leaf_x, bank.query_cache.normalized_response)
        return h[:, bank.query_cache.query_indices], h.mean(1)

    def forward(self, bank):
        local, _ = self.prepare(bank)
        return self.head(local[bank.target_graph])


class BoundaryKernelGNN(BoundaryLocalGNN):
    def __init__(self, *args, initial_tau=.05, **kwargs):
        super().__init__(*args, **kwargs)
        adjusted = initial_tau - 1e-4
        if adjusted <= 0:
            raise ValueError("initial_tau must exceed 1e-4")
        self.raw_tau = nn.Parameter(torch.tensor(adjusted + math.log(-math.expm1(-adjusted))))
        self.head = nn.Linear(2*self.hidden_dim, self.num_classes)

    def forward(self, bank):
        leaf, _ = self.prepare(bank)
        tau = torch.nn.functional.softplus(self.raw_tau) + 1e-4
        weight = log_interval_kernel_mass(bank.positions, bank.intervals,
                    tau.to(bank.intervals.dtype)).softmax(-1).to(leaf.dtype)
        context = torch.einsum("pg,gqh->pqh", weight, leaf)
        return self.head(torch.cat((leaf[bank.target_graph], context), -1))


def _interval_features(bank, dtype):
    left, right = bank.intervals.to(dtype).unbind(-1)
    return torch.stack((left, right, (left+right)/2, right-left), -1)


class BoundaryGRU(BoundaryLocalGNN):
    """Per-query standard bidirectional GRU over graph records in genomic order.

    End-of-record readout matches a common snapshot baseline. Endpoints and span
    are observed inputs. Repeating an interval can change this recurrent state.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = nn.Linear(2*self.hidden_dim + 4, self.hidden_dim)
        self.temporal = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(3*self.hidden_dim, self.num_classes)

    def forward(self, bank):
        local, graph = self.prepare(bank)
        geometry = _interval_features(bank, local.dtype)[:, None].expand(-1, local.shape[1], -1)
        tokens = self.token(torch.cat((local, graph[:, None].expand_as(local), geometry), -1))
        order = torch.argsort(bank.intervals[:, 0])
        memory, _ = self.temporal(tokens[order].transpose(0, 1))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        context = memory[:, inverse[bank.target_graph]].transpose(0, 1)
        return self.head(torch.cat((local[bank.target_graph], context), -1))


class BoundaryTransformer(BoundaryLocalGNN):
    """Independent temporal Transformer with fixed-coordinate attention readout."""
    def __init__(self, *args, temporal_layers=2, num_heads=4, dropout=0., **kwargs):
        super().__init__(*args, **kwargs)
        if self.hidden_dim % num_heads:
            raise ValueError("Hidden width must be divisible by attention heads")
        self.token = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.interval_encoder = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.GELU(),
                                             nn.Linear(self.hidden_dim, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(self.hidden_dim, num_heads, 4*self.hidden_dim,
                    dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, temporal_layers,
                    norm=nn.LayerNorm(self.hidden_dim), enable_nested_tensor=False)
        self.coordinate = nn.Sequential(nn.Linear(3, self.hidden_dim), nn.GELU(),
                                        nn.Linear(self.hidden_dim, self.hidden_dim))
        self.query = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Sequential(nn.Linear(2*self.hidden_dim, self.hidden_dim), nn.GELU(),
                                  nn.Linear(self.hidden_dim, self.num_classes))

    def forward(self, bank):
        local, graph = self.prepare(bank)
        tokens = self.token(torch.cat((local, graph[:, None].expand_as(local)), -1))
        tokens = tokens + self.interval_encoder(_interval_features(bank, local.dtype))[:, None]
        order = torch.argsort(bank.intervals[:, 0])
        memory = self.temporal(tokens[order].transpose(0, 1))
        p = bank.positions.to(local.dtype)
        coordinate = torch.stack((p, torch.sin(2*math.pi*p), torch.cos(2*math.pi*p)), -1)
        leaf = local[bank.target_graph].transpose(0, 1)
        position = self.coordinate(coordinate)[None].expand_as(leaf)
        query = self.query(torch.cat((leaf, position), -1))
        context, _ = self.attention(query, memory, memory, need_weights=False)
        return self.head(torch.cat((leaf, self.norm(query+context)), -1)).transpose(0, 1).contiguous()


def make_boundary_model(name, input_dim, num_classes, settings):
    common = dict(input_dim=input_dim, num_classes=num_classes,
                  hidden_dim=settings["hidden_dim"], num_layers=settings["num_layers"])
    classes = {"boundary_feature": BoundaryFeatureMLP, "boundary_local": BoundaryLocalGNN,
               "boundary_kernel": BoundaryKernelGNN, "boundary_gru": BoundaryGRU,
               "boundary_transformer": BoundaryTransformer}
    if name in classes:
        return classes[name](**common)
    variants = {
        "boundary_full": ("full", "ordered"),
        "boundary_diagonal": ("diagonal", "ordered"),
        "boundary_gain_matched": ("gain_matched_diagonal", "ordered"),
        "boundary_gain_leak": ("gain_leak_diagonal", "ordered"),
        "boundary_bag": ("full", "homogeneous_bag"),
        "boundary_equilibrium": ("full", "equilibrium"),
        "boundary_diagonal_equilibrium": ("diagonal", "equilibrium"),
    }
    if name not in variants:
        raise ValueError(f"Unknown boundary study model: {name}")
    coupling, aggregation = variants[name]
    return BoundaryMemoryModel(**common, latent_dim=settings.get("latent_dim", 8),
        coupling=coupling, aggregation=aggregation,
        bidirectional=settings.get("bidirectional", True),
        symmetric_readout=settings.get("symmetric_readout", True),
        initial_coupling=settings.get("initial_coupling", 20.),
        initial_damping=settings.get("initial_damping", 1.), validate=False)
