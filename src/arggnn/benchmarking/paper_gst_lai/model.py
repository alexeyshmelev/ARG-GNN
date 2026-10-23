"""Unchanged author GraphSetConv with an explicitly new LAI frontend/head."""
from __future__ import annotations
import importlib
import math
from pathlib import Path
import sys
import torch
from torch import nn
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeGNNV1
from .. import european_native_models as io

AUTHOR_ROOT = ("research/reviews/author_repository_deep_audit_20261023_v1/gst/history/"
               "6469d56aa131463325058b897b1e435c6d2edcd2/files/src")
AUTHOR = dict(path=AUTHOR_ROOT+"/graph_set_transformer/models/gst.py",
    sha256="bc08c15d55247b32bda98e4b03fcaee988ceb8f654423240a33ca632ce11ef72")


def native_graphsetconv():
    expected = io._checked(AUTHOR)
    source_root = str(io.ROOT / AUTHOR_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    module = importlib.import_module("graph_set_transformer.models.gst")
    io._need(Path(module.__file__).resolve() == expected,
             "Exact pinned unchanged author GraphSetConv module required")
    return module.GraphSetConv


class ObservedFrontEnd(nn.Module):
    """Same native tuple19 site pooling, known references, ages and node flags.

    Incoming branch geometry/support are exposed as additional observed node
    channels because native GraphSetConv does not accept edge attributes.
    Query ancestry is always unknown. No TreeLoom weights are imported.
    """
    _site_chunk = MutationRoutedWholeTreeGNNV1._site_chunk
    encode_sites = MutationRoutedWholeTreeGNNV1.encode_sites

    def __init__(self, hidden=32):
        super().__init__()
        self.hidden_dim = hidden
        self.site_chunk_size = 256
        self.checkpoint_sites = True
        self.site_mlp = nn.Sequential(nn.Linear(19, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
        self.site_pool_projection = nn.Sequential(nn.Linear(2*hidden+3, hidden), nn.Tanh())
        self.leaf_projection = nn.Sequential(nn.Linear(hidden+6, hidden), nn.Tanh())
        self.node_projection = nn.Sequential(nn.Linear(12, hidden), nn.Tanh())

    def forward(self, bank):
        t, h = bank.sample_index.shape
        site = self.encode_sites(bank)
        labels = bank.reference_class
        io._need(bool((labels[bank.query_column] == -1).all()),
                 "All observed query ancestry classes must remain unknown")
        onehot = torch.nn.functional.one_hot(labels.clamp_min(0), 4).to(site.dtype)
        onehot = onehot * (labels >= 0)[:, None]
        observed = torch.cat((onehot, (labels < 0).to(site.dtype)[:,None],
            (bank.sample_age.log1p()/math.log(10001.))[:,None]),-1)
        leaf = self.leaf_projection(torch.cat((site, observed[None].expand(t,h,6)),-1))
        sampled = bank.node_time.new_zeros(bank.parent.numel())
        sampled.index_fill_(0,bank.sample_index.flatten(),1.)
        raw = bank.branch_support_true
        support = torch.cat((raw[:,:3],raw[:,3:4].log1p()/math.log(100001.),raw[:,4:5]),-1)
        node = torch.cat(((bank.node_time.log1p()/math.log(10001.))[:,None],
            (bank.parent < 0).to(sampled.dtype)[:,None],sampled[:,None],
            bank.edge_geometry,support),-1)
        base = self.node_projection(node)
        return base.index_add(0,bank.sample_index.flatten(),leaf.reshape(-1,self.hidden_dim))


class GstLai8(nn.Module):
    def __init__(self, class_order):
        super().__init__()
        io._need(len(class_order) in (3,4,6) and len(set(class_order)) == len(class_order),
                 "Declared native LAI class order required")
        self.output_classes = tuple(class_order)
        self.frontend = ObservedFrontEnd(32)
        graphsetconv = native_graphsetconv()
        self.blocks = nn.ModuleList([graphsetconv(filters=32,in_channels=32,
            activation="silu",mhsa_dropout=.1,ffn_dropout=.1,pooling="attn",
            use_gating=True,ffn_multiplier=2,num_heads=4,drop_path=0.,
            node_set_mode="cross_attn") for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Dropout(.1),
                                  nn.Linear(32,len(class_order)))
        self.float()

    def forward_nodes(self, features, edge_index, graph_batch, set_batch):
        x = features
        for block in self.blocks:
            x = block(x,edge_index,graph_batch,set_batch)
        return self.head(x)

    def forward(self, window):
        x = self.frontend(window.bank)
        return self.forward_nodes(x,window.edge_index,window.graph_batch,window.set_batch)

    def sampled_logits(self, window):
        logits = self(window)
        nodes = window.bank.sample_index[window.focal_local_hold,window.focal_sample_column]
        return logits[nodes]


def specification():
    return dict(condition="gst_lai_8_native_graphsetconv",author_graphsetconv_ref=AUTHOR,
        source_commit="6469d56aa131463325058b897b1e435c6d2edcd2",
        unchanged_native_GraphSetConv=True,layers=2,hidden=32,heads=4,dropout=.1,
        pooling="attn",node_set_mode="cross_attn",use_gating=True,
        set_construction="disjoint consecutive eight-native-tree sets; final set may be shorter",
        all_native_trees_edges_sites_samples_retained=True,
        edge_direction="both directions of each real parent-child edge",
        observed_frontend="native tuple19 allele/donor pooling plus references/ages/node flags/branch geometry/support",
        LAI_head="new32->32->native_classes node head",
        native_GST_table_reproduction=False,isolated_layer_attribution=False,
        context_difference="GST context is local eight-tree sets; TreeLoom retains full-genome memory",
        tuning_policy="starting condition fixed; tune only on held-out validation if separately authorized",
        fabricated_persistent_identity_or_genomic_history=False,
        frontend_source_ref=io._ref(Path(__file__)),
        tuple19_site_pool_source_ref=io._ref(io.ROOT/"src/arggnn/mutation_routed_whole_tree_gnn_v1.py"),
        native_bank_encoder_ref=io._ref(io.ROOT/"research/reviews/TreeLoom_preparation_reuse_source_20261017_v1/optimized_observations_v1.py"),
        operational_complete_set_failstops=dict(max_nodes=262144,max_sites=16384),
        failstop_never_truncates_or_drops_a_tree=True)
