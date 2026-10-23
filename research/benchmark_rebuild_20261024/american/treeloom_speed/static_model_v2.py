"""Frozen H050-equation TreeLoom on original registered American30Mb parents.

No CLI, data/checkpoint reader, writer, admission, resource policy or launcher.
The caller authenticates source/observations and supplies a complete parent.
No partial-block execution or independent block reset is supported.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
from collections.abc import Mapping
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arggnn.american_c3_common_v1 import require, CLASSES
from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from arggnn.american_c3_gnn_observations_v1 import native_tree, encode_block
from arggnn.american_c3_gnn_targets_v1 import (
    TrainFocalTargetVectorV1, identity_sha)
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from arggnn.anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1
from arggnn.anonymous_clade_flag_control_v1 import RankEightCladeFlagAdapterV1
from arggnn.ordered_clade_episode_gnn_v1 import (
    OrderedCladeEpisodesV1, RankEightCladeEpisodeAdapterV1,
    build_clade_episode_arrays_v1)
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as memory_core

FAMILY = "TreeLoom_frozen_H050_American_full30Mb_C3_v1"
TRAIN = ()  # Bound once by the authenticated static fresh facade.
VAL = ()  # Registered validation and test observations only.
EXPECTED_PARAMETERS = 32132  # Source-derived; constructor checks actual future model.
EXPECTED_TRAINABLE_TENSORS = 49
HAPLOTYPES, QUERIES, FOCALS = 1600, 1000, 10000
MEMORY_ARM = "stable_leaf_memory"
GENEALOGY = "inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"


def _short_ref(value):
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str) and isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"]),
        "Caller-authenticated exact source/observation reference")
    return dict(value)


def _bank_tensors(bank):
    result = []
    for field in fields(bank):
        value = getattr(bank, field.name)
        result.extend(value if isinstance(value, tuple) else (value,))
    require(all(isinstance(x, torch.Tensor) for x in result),
        "Observation-only native tensor fields")
    return tuple(result)


def _episode_tensors(episodes):
    return (episodes.flags.node_flags,) + tuple(getattr(episodes, n) for n in
        ("member_node", "member_hold", "member_episode", "episode_size", "hold_span"))


def _reference_vector(device):
    # Original sample order: AFR, EUR, EAS references, then true queries.
    return torch.tensor([2]*200+[0]*200+[1]*200+[-1]*1000,
        dtype=torch.long, device=device)


@dataclass(frozen=True)
class AmericanTreeLoomWholeParentV1:
    bank: MutationRoutedWholeTreeInputsV1
    episodes: OrderedCladeEpisodesV1
    cut: memory_core.CutPlanV1
    parent_id: str
    split: str
    sequence_length_bp: int
    native_holds: int
    original_sites: int
    focal_hold: torch.Tensor
    focal_sample_column: torch.Tensor
    focal_identity_sha256: str
    observation_ref: dict
    focal_registry_ref: dict
    native_preparation_ref: dict
    bank_stamps: tuple
    episode_stamps: tuple
    focal_stamps: tuple

    def validate(self):
        bank = self.bank
        require(type(bank) is MutationRoutedWholeTreeInputsV1
            and isinstance(self.episodes, OrderedCladeEpisodesV1)
            and isinstance(self.cut, memory_core.CutPlanV1),
            "Exact full-tree observation/structural/memory payload types")
        require((self.split == "train" and self.parent_id in TRAIN
                 and self.sequence_length_bp == 30000000)
            or (self.split == "val_prediction" and self.parent_id in VAL
                 and self.sequence_length_bp == 30000000),
            "Only original20x30Mb TRAIN or5x30Mb VAL prediction parents")
        require(self.bank_stamps == memory_core._stamps(_bank_tensors(bank))
            and self.episode_stamps == memory_core._stamps(_episode_tensors(self.episodes))
            and self.focal_stamps == memory_core._stamps(
                (self.focal_hold, self.focal_sample_column)),
            "Immutable complete observations, topology recipes and focal join")
        t, h = bank.sample_index.shape
        require((t, h) == (self.native_holds, HAPLOTYPES)
            and self.original_sites == len(bank.genotype)
            and bank.genotype.shape == (self.original_sites, HAPLOTYPES)
            and bank.log_donor_allele_frequency.shape[-1] == 6
            and torch.equal(bank.query_column,
                torch.arange(600, 1600, device=bank.parent.device))
            and torch.equal(bank.reference_class, _reference_vector(bank.parent.device))
            and torch.equal(bank.readout_hold,
                torch.arange(t, device=bank.parent.device))
            and float(bank.spans.double().sum()) == self.sequence_length_bp,
            "Complete native parent/all sites/all1600/all1000/exact600 references")
        require(self.episodes.native_holds == t
            and self.episodes.haplotypes == h
            and self.episodes.native_nodes == bank.parent.numel()
            and torch.equal(self.episodes.hold_span, bank.spans),
            "Globally prepared proper-clade episodes on every original native hold")
        self.cut.validate(bank)
        require(self.cut.haplotypes == HAPLOTYPES
            and memory_core.HORIZON == 350.0
            and self.focal_hold.dtype == self.focal_sample_column.dtype == torch.long
            and self.focal_hold.shape == self.focal_sample_column.shape == (FOCALS,)
            and self.focal_hold.device == self.focal_sample_column.device == bank.parent.device
            and not self.focal_hold.requires_grad and not self.focal_sample_column.requires_grad
            and bool(((self.focal_hold >= 0) & (self.focal_hold < t)).all())
            and bool(((self.focal_sample_column >= 600)
                      & (self.focal_sample_column < 1600)).all()),
            "Exact full native focal readout and unchanged350generation cut")
        require(isinstance(self.focal_identity_sha256, str)
            and len(self.focal_identity_sha256) == 64,
            "Original10000focal identity hash")
        for value in (self.observation_ref, self.focal_registry_ref, self.native_preparation_ref):
            _short_ref(value)
        return bank

    def to(self, device):
        self.validate()
        bank = self.bank.to(device)
        episodes = self.episodes.to(device)
        cut = self.cut._to_for_bank(device, self.bank, bank)
        focal_hold = self.focal_hold.to(device)
        focal_sample = self.focal_sample_column.to(device)
        result = type(self)(bank, episodes, cut, self.parent_id, self.split,
            self.sequence_length_bp, self.native_holds, self.original_sites,
            focal_hold, focal_sample, self.focal_identity_sha256,
            dict(self.observation_ref), dict(self.focal_registry_ref),
            dict(self.native_preparation_ref),
            memory_core._stamps(_bank_tensors(bank)),
            memory_core._stamps(_episode_tensors(episodes)),
            memory_core._stamps((focal_hold, focal_sample)))
        result.validate()
        return result


def prepare_from_loaded_observation_v1(state, *, split, observation_ref,
        focal_registry_ref, native_preparation_ref, max_nodes, max_holds,
        max_episode_payload_bytes):
    """Root supplies existing sanitized load_observation output, never truth.

    One full parent is encoded. This does not load files, accept a block range,
    prune, subsample, replay native inference or open a target resolver.
    Bounds reject the whole parent. They never authorize workload reduction.
    """
    for ref in (observation_ref, focal_registry_ref, native_preparation_ref):
        _short_ref(ref)
    require(isinstance(state, Mapping)
        and all(type(x) is int and x > 0 for x in
            (max_nodes, max_holds, max_episode_payload_bytes)),
        "Explicit root resource failstop bounds")
    manifest, ts = state["manifest"], state["ts"]
    parent_id = manifest["parent_id"]
    length = int(ts.sequence_length)
    require(ts.sequence_length == length, "Exact integer native physical support")
    require(manifest["schema"] == "american_c3_observed_v1"
        and manifest["genealogy_kind"] == GENEALOGY
        and manifest["query_ancestry_truth_in_inputs"] is False
        and manifest["class_order"] == list(CLASSES),
        "Same sanitized inferred native observation, no query ancestry in inputs")
    require((split == "train" and parent_id in TRAIN and length == 30000000)
        or (split == "val_prediction" and parent_id in VAL and length == 30000000),
        "Exact independent TRAIN or original held-out VAL parent")
    require(np.array_equal(state["labels"],
            np.array([2]*200+[0]*200+[1]*200+[-1]*1000, dtype=np.int64))
        and np.array_equal(state["query_columns"], np.arange(600, 1600))
        and len(state["nodes"]) == 1600
        and len(state["focals"]["record_ids"]) == FOCALS
        and state["focals"]["parent_id"].item() == parent_id
        and state["focals"]["class_order"].tolist() == list(CLASSES)
        and np.array_equal(state["focals"]["sample_nodes"],
            state["nodes"][state["focals"]["sample_indices"]])
        and np.isin(state["focals"]["sample_indices"], np.arange(600, 1600)).all(),
        "All original reference/query identities and native10000focal join")
    genotype = np.asarray(state["genotype"])
    require(genotype.shape == (ts.num_sites, HAPLOTYPES)
        and (genotype >= 0).all()
        and np.array_equal(state["positions"], ts.tables.sites.position),
        "All original complete observed SNP calls; no new filtering/imputation")
    require(ts.num_trees == len(state["spans"]) <= max_holds,
        "Full-parent native hold limit, never a block truncate")
    trees, intervals, nodes = [], [], 0
    for tree in ts.trees():
        native = native_tree(tree, state["nodes"])
        nodes += len(native.parent)
        require(nodes <= max_nodes, "Full-parent native node limit")
        trees.append(native)
        intervals.append(tuple(tree.interval))
    bank, observation_metadata = encode_block(trees, intervals, genotype,
        state["allele_count"], state["positions"], state["labels"], state["ages"],
        num_classes=3, age_split=350., query_columns=state["query_columns"])
    require(bank.parent.numel() == nodes
        and len(bank.spans) == ts.num_trees
        and len(bank.genotype) == ts.num_sites,
        "No native node, tree or SNP was dropped")
    arrays, episode_metadata = build_clade_episode_arrays_v1(
        bank.parent.numpy(), bank.sample_index.numpy(), bank.spans.numpy(),
        max_payload_bytes=max_episode_payload_bytes)
    episodes = OrderedCladeEpisodesV1.from_verified_arrays(arrays, episode_metadata)
    cut, cut_metadata = memory_core.prepare_cut_plan_v1(bank,
        expected_haplotypes=HAPLOTYPES, max_nodes=max_nodes, max_holds=max_holds)
    positions = state["focals"]["positions_bp0"]
    require(np.isfinite(positions).all() and ((positions >= 0) & (positions < length)).all(),
        "No native focal coordinate extrapolation")
    exact_focal_hold = np.searchsorted(np.asarray(list(ts.breakpoints())),
        positions, side="right")-1
    require(np.array_equal(state["focal_tree"], exact_focal_hold),
        "Same original half-open native focal/tree join")
    focal_hold = torch.as_tensor(exact_focal_hold.copy(), dtype=torch.long)
    focal_sample = torch.as_tensor(
        np.asarray(state["focals"]["sample_indices"]).copy(), dtype=torch.long)
    identity = identity_sha(state["focals"])
    bundle = AmericanTreeLoomWholeParentV1(bank, episodes, cut,
        parent_id, split, length, ts.num_trees, ts.num_sites,
        focal_hold, focal_sample, identity,
        dict(observation_ref), dict(focal_registry_ref), dict(native_preparation_ref),
        memory_core._stamps(_bank_tensors(bank)),
        memory_core._stamps(_episode_tensors(episodes)),
        memory_core._stamps((focal_hold, focal_sample)))
    bundle.validate()
    return bundle, dict(schema="American_TreeLoom_full_observation_preparation_v1",
        parent_id=parent_id, split=split, observation_only=True,
        observation_ref=dict(observation_ref), focal_registry_ref=dict(focal_registry_ref),
        native_preparation_ref=dict(native_preparation_ref),
        focal_identity_sha256=identity, focal_records=FOCALS,
        all_native_trees_nodes_sites_samples_retained=True,
        structural_recipes_global_before_partition=True,
        artificial_block_boundaries=False,
        observation_metadata=observation_metadata,
        episode_metadata=episode_metadata, cut_metadata=cut_metadata,
        private_truth_opened=False, model_executed=False)


class AmericanTreeLoomC3FullV1(AmericanC3WholeTreeGNNV1):
    """All final TreeLoom mechanisms; only fixed C3 observation/class adaptation."""
    sweep_tree_state_v1 = AnonymousCladeTensorGNNV1.sweep_tree_state_v1
    leaf_context_v1 = AnonymousCladeTensorGNNV1.leaf_context_v1

    def __init__(self, *, checkpoint_memory=True):
        super().__init__(hidden_dim=32, class_order=CLASSES,
            expected_haplotypes=HAPLOTYPES, donor_cohorts=2,
            site_chunk_size=256, checkpoint_sites=True)
        self.clade_coupling = RankEightCladeFlagAdapterV1()
        self.episode_coupling = RankEightCladeEpisodeAdapterV1()
        self.memory = memory_core.CutMemoryV1(
            arm=MEMORY_ARM, checkpoint_memory=checkpoint_memory)
        self.float()
        require(sum(p.numel() for p in self.parameters()) == EXPECTED_PARAMETERS
            and len(tuple(self.parameters())) == EXPECTED_TRAINABLE_TENSORS
            and all(p.requires_grad and p.dtype == torch.float32
                    for p in self.parameters()),
            "Actual future C3 final TreeLoom parameter shape/trainability contract")

    def prepare_tree_base_v1(self, bank):
        """Same source equations as H050's H038 body, class width4→3 only."""
        t, h = bank.sample_index.shape
        n, d = len(bank.parent), self.hidden_dim
        site_leaf = self.encode_sites(bank)
        labels = bank.reference_class
        onehot = F.one_hot(labels.clamp_min(0), 3).to(site_leaf.dtype)
        onehot = onehot * (labels >= 0)[:, None]
        unknown = (labels < 0).to(site_leaf.dtype)[:, None]
        age = (bank.sample_age.log1p()/math.log(10001.))[:, None]
        observed_leaf = torch.cat((onehot, unknown, age), -1)[None].expand(t, h, 5)
        leaf_base = self.leaf_projection(torch.cat((site_leaf, observed_leaf), -1))
        sampled = bank.node_time.new_zeros(n)
        sampled.index_fill_(0, bank.sample_index.flatten(), 1.)
        root = (bank.parent < 0).to(sampled.dtype)
        node_age = bank.node_time.log1p()/math.log(10001.)
        base = self.node_projection(torch.stack((node_age, root, sampled), -1))
        base = base.index_add(0, bank.sample_index.flatten(), leaf_base.reshape(-1, d))
        raw = bank.branch_support_true
        support = torch.cat((raw[:, :3],
            raw[:, 3:4].log1p()/math.log(100001.), raw[:, 4:5]), -1)
        gate_input = torch.cat((bank.edge_geometry, support), -1)
        ug, dg = self.up_gate(gate_input).sigmoid(), self.down_gate(gate_input).sigmoid()
        nonroot = bank.parent >= 0
        degree = torch.bincount(bank.parent[nonroot], minlength=n).to(base.dtype).clamp_min(1)
        return base, ug, dg, degree

    def encode_trees(self, bank, *, control="aligned"):
        raise ValueError("Final TreeLoom cannot use the inherited single-sweep encoder")

    def encode_bank(self, bank):
        raise ValueError("Final TreeLoom requires the globally prepared complete-parent bundle")

    def encode_full_parent_v1(self, bundle):
        require(type(self) is AmericanTreeLoomC3FullV1
            and self.memory.arm == MEMORY_ARM
            and type(bundle) is AmericanTreeLoomWholeParentV1,
            "Fixed final stable-leaf model; no independent block or baseline substitution")
        bank = bundle.validate()
        parameter = next(self.parameters())
        require(bank.parent.device == parameter.device
            and all(p.dtype == torch.float32 and p.device == parameter.device
                    for p in self.parameters()),
            "Root-transferred complete FP32 parent and model on one device")
        base, ug, dg, degree = self.prepare_tree_base_v1(bank)
        upward, outside = self.sweep_tree_state_v1(bank, base, ug, dg, degree)
        injection = (self.clade_coupling(upward, bundle.episodes.flags)
            + self.episode_coupling(upward, bundle.episodes, arm="own_state"))
        delta = self.memory(upward, outside, bank.spans, bundle.cut)
        injection = injection.index_add(0, bundle.cut.representatives, delta)
        second_up, second_out = self.sweep_tree_state_v1(
            bank, base + injection, ug, dg, degree)
        return self.leaf_context_v1(bank, second_up, second_out)

    def forward_holds_v1(self, bundle):
        bank = bundle.validate()
        context = self.encode_full_parent_v1(bundle)
        # Original fixed normalization; not rescaled for30Mb length transfer.
        span = bank.spans.log1p()/math.log(2000001.)
        prev = torch.cat((bank.spans.new_zeros(1),
            .5*(bank.spans[1:]+bank.spans[:-1])))
        nxt = torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]),
            bank.spans.new_zeros(1)))
        geometry = torch.stack((span, prev.log1p()/math.log(2000001.),
            nxt.log1p()/math.log(2000001.)), -1)
        forward, _ = self.scan_context(context, geometry, direction="forward")
        reverse, _ = self.scan_context(context, geometry, direction="reverse")
        return self.classify_context(context, forward, reverse)

    def forward_complete(self, block):
        raise ValueError("Use full final TreeLoom bundle; C3 block API is not equivalent")

    def forward(self, bundle):
        logits = self.forward_holds_v1(bundle)
        return logits[bundle.focal_hold, bundle.focal_sample_column]

    def checkpoint_contract(self):
        return dict(schema=FAMILY+"_checkpoint_contract",
            family=FAMILY, class_order=list(CLASSES), hidden_dim=32,
            expected_haplotypes=1600, query_haplotypes=1000,
            donor_cohorts=2, donor_channels=6,
            site_feature_width=15, observed_leaf_width=5,
            shared_complete_tree_sweeps=2, local_flag_rank=8,
            local_own_state_rank=8, memory_arm=MEMORY_ARM,
            memory_rank=8, memory_parameters=513,
            donor_age_cohorts="<350,>=350", cut_generations=350.,
            cut_sample_policy="<=350 real ancestor;>350 actual-sample singleton",
            memory_length_scale_initial_bp=5000.,
            memory_length_scale_bounds_bp=[100., 100000.],
            recurrent_contract="one_layer_bidirectional_equations_as_two_directional_GRUs",
            physical_geometry_normalizer=2000001.,
            all_parent_context_required=True, all_native_trees_nodes_sites_samples=True,
            expected_parameters=EXPECTED_PARAMETERS,
            expected_trainable_tensors=EXPECTED_TRAINABLE_TENSORS,
            legacy_C3_checkpoint_is_final_TreeLoom=False)



def final_from_supplied_C3_own_state_v1(state, metadata, *, initializer_ref):
    """Future authenticated C3 stage3 weights only; never a checkpoint reader.

    Existing29,571parameter C3 saved600 states are rejected. ROOT must establish
    the new C3 own-state stage's genuine scientific/source/training lineage.
    """
    _short_ref(initializer_ref)
    require(isinstance(metadata, Mapping)
        and metadata.get("family") == "American_TreeLoom_C3_own_state_initializer_v1"
        and metadata.get("class_order") == list(CLASSES)
        and metadata.get("trained_parent_ids") == list(TRAIN)
        and metadata.get("stage_epochs") == 30
        and metadata.get("stage_updates") == 600
        and metadata.get("cumulative_updates") == 1800
        and metadata.get("parameter_count") == 31619
        and metadata.get("fixed_boundary_pass") == 30
        and metadata.get("VAL_truth_used_for_initializer_choice") is False
        and metadata.get("initializer_ref") == initializer_ref,
        "Separately authenticated completed C3 own-state initializer, not saved600 backbone")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70001)
        model = AmericanTreeLoomC3FullV1()
    expected = model.state_dict()
    memory_keys = {"memory.U.weight", "memory.O.weight", "memory.ell_theta"}
    require(isinstance(state, Mapping) and set(state) == set(expected)-memory_keys,
        "All inherited C3 tree, structural, own-state and recurrent weights; only memory absent")
    for key, tensor in state.items():
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32 and tensor.shape == expected[key].shape
            and bool(torch.isfinite(tensor).all()), "Exact finite supplied C3 own-state tensor: "+key)
    copied = model.load_state_dict(state, strict=False)
    require(set(copied.missing_keys) == memory_keys and not copied.unexpected_keys
        and not bool(model.memory.O.weight.count_nonzero())
        and all(torch.equal(model.state_dict()[key], value) for key, value in state.items()),
        "Inherited state retained, memory residual initially zero")
    return model

def train_complete_parent_step_v1(model, bundle, targets, optimizer, *, resource_check):
    """One future admitted all-focal mean-CE parent update; no driver or resolver."""
    require(type(model) is AmericanTreeLoomC3FullV1
        and type(bundle) is AmericanTreeLoomWholeParentV1
        and bundle.split == "train" and bundle.parent_id in TRAIN
        and isinstance(targets, TrainFocalTargetVectorV1)
        and targets.parent_id == bundle.parent_id
        and targets.focal_registry_ref == bundle.focal_registry_ref
        and targets.focal_identity_sha256 == bundle.focal_identity_sha256
        and targets.labels_cpu.device.type == "cpu"
        and targets.labels_cpu.dtype == torch.long
        and targets.labels_cpu.shape == (FOCALS,)
        and bool(((targets.labels_cpu >= 0) & (targets.labels_cpu < 3)).all())
        and callable(resource_check), "Separate sealed TRAIN-only native focal target vector")
    require(type(optimizer) is torch.optim.AdamW
        and len(optimizer.param_groups) == 1,
        "Explicit original AdamW over all final TreeLoom parameters")
    group = optimizer.param_groups[0]
    require(group["lr"] == .001 and group["weight_decay"] == .0001
        and group["betas"] == (.9, .999) and group["eps"] == 1e-8
        and group["maximize"] is False and group["amsgrad"] is False
        and all(p.requires_grad for p in model.parameters())
        and len(group["params"]) == EXPECTED_TRAINABLE_TENSORS
        and len({id(p) for p in group["params"]}) == EXPECTED_TRAINABLE_TENSORS
        and {id(p) for p in group["params"]} == {id(p) for p in model.parameters()},
        "No omitted/frozen memory or structural parameters and unchanged optimizer")
    bank = bundle.validate()
    resource_check("TreeLoom_complete_parent_begin")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=bank.parent.device.type, enabled=False):
        logits = model(bundle)
        require(logits.shape == (FOCALS, 3), "All original native focal logits")
        loss = F.cross_entropy(logits, targets.labels_cpu.to(logits.device), reduction="mean")
    require(bool(torch.isfinite(loss)), "Finite full-parent focal CE")
    resource_check("TreeLoom_complete_parent_backward")
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
        and any(bool((g != 0).any()) for g in grads),
        "Finite connected final-mechanism gradients and nonzero global gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    require(bool(torch.isfinite(norm)), "Finite full-parent gradient norm")
    resource_check("TreeLoom_complete_parent_optimizer")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "Finite updated final TreeLoom weights")
    resource_check("TreeLoom_complete_parent_end")
    return dict(family=FAMILY, parent_id=bundle.parent_id,
        native_focals=FOCALS, loss=float(loss.detach()), gradient_norm=float(norm),
        one_complete_parent_optimizer_update=True, detached_history=False,
        artificial_block_boundaries=False, ancestry_accuracy=None)


def predict_focals_v1(model, bundle):
    """Truth-free scientific prediction only; ROOT must separately seal all30cells."""
    require(type(model) is AmericanTreeLoomC3FullV1
        and type(bundle) is AmericanTreeLoomWholeParentV1
        and bundle.split == "val_prediction" and bundle.parent_id in VAL,
        "Original full30Mb VAL predictions, TEST deferred")
    bundle.validate()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=bundle.bank.parent.device.type, enabled=False):
        probability = model(bundle).softmax(-1).cpu()
    require(probability.dtype == torch.float32 and probability.shape == (FOCALS, 3)
        and bool(torch.isfinite(probability).all())
        and bool((probability >= 0).all())
        and torch.allclose(probability.sum(-1), torch.ones(FOCALS), rtol=1e-5, atol=1e-6),
        "Raw FP32 native-focal probabilities")
    return probability
