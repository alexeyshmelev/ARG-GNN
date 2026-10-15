"""Focused CPU/FP32 synthetic checks for registered H045 source, no real inputs."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for these CPU-only checks"
import numpy as np
import torch
from torch_geometric.nn.aggr import SumAggregation
from arggnn.anonymous_clade_flag_control_v1 import (
    AnonymousCladeFlagControlGNNV1, build_clade_flag_arrays_v1,
)
from arggnn.mutation_routed_whole_tree_gnn_v1 import (
    MutationRoutedWholeTreeInputsV1, validate_inputs_v1,
)
from arggnn.ordered_clade_episode_gnn_v1 import (
    ARMS, ARRAY_FIELDS, MODEL_KWARGS, OrderedCladeEpisodesV1,
    OrderedCladeEpisodeGNNV1, RankEightCladeEpisodeAdapterV1,
    build_clade_episode_arrays_v1, build_clade_episodes_v1,
    build_matched_episode_family_v1, load_h042_episode_warmstart_v1,
)

torch.set_num_threads(1)
torch.manual_seed(101)
checks = []


def check(name, ok):
    if not bool(ok):
        raise AssertionError(name)
    checks.append(name)


def rejects(name, fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError):
        checks.append(name)
    else:
        raise AssertionError("Expected rejection: " + name)


def forest(patterns=("A", "A", "A", "B", "A", "A", "star"), h=4):
    parent, samples, reps, unary = [], [], [], []
    for pattern in patterns:
        start = len(parent)
        row = list(range(start, start+h))
        parent.extend([-2]*h)
        if pattern == "star":
            root = len(parent)
            parent.append(-1)
            for node in row:
                parent[node] = root
            reps.append(None); unary.append(None)
        else:
            pair = row[:2] if pattern == "A" else [row[0], row[2]]
            representative = len(parent)
            ancestor = representative+1
            root = representative+2
            parent.extend([ancestor, root, -1])
            for node in row:
                parent[node] = representative if node in pair else root
            reps.append(representative); unary.append(ancestor)
        samples.append(row)
    return np.array(parent, dtype=np.int64), np.array(samples, dtype=np.int64), reps, unary


def prepared(patterns=("A", "A", "A", "B", "A", "A", "star"), h=4):
    parent, sample, reps, unary = forest(patterns, h)
    spans = np.array([1, 2, 5, 3, 7, 11, 13][:len(patterns)], dtype=np.float32)
    arrays, meta = build_clade_episode_arrays_v1(parent, sample, spans)
    return parent, sample, reps, unary, arrays, meta, OrderedCladeEpisodesV1.from_verified_arrays(arrays, meta)


def synthetic_bank(patterns=("A", "A", "B"), h=952):
    parent, sample, _, _ = forest(patterns, h)
    n, (t, _) = len(parent), sample.shape
    children = [[] for _ in range(n)]
    for node, ancestor in enumerate(parent):
        if ancestor >= 0:
            children[ancestor].append(node)
    height = np.zeros(n, dtype=np.int64)
    for node in range(n):
        if children[node]:
            height[node] = max(height[x] for x in children[node])+1
    levels = [np.flatnonzero(height == value) for value in range(int(height.max())+1)]
    up_children, ranks = [], []
    for level in levels:
        ch, rank = [], []
        for i, node in enumerate(level):
            ch.extend(children[node])
            rank.extend([i]*len(children[node]))
        up_children.append(np.array(ch, dtype=np.int64))
        ranks.append(np.array(rank, dtype=np.int64))
    depth = np.zeros(n, dtype=np.int64)
    for node in reversed(range(n)):
        if parent[node] >= 0:
            depth[node] = depth[parent[node]]+1
    down = [np.flatnonzero(depth == value) for value in range(1, int(depth.max())+1)]
    edge = np.zeros((n, 4), dtype=np.float32)
    edge[parent >= 0, 2] = .1
    support = np.zeros((n, 5), dtype=np.float32)
    support[:, 4] = 1
    s = 2*t
    arrays = dict(
        parent=parent, node_time=height.astype(np.float32), sample_index=sample,
        upward_levels=levels, upward_children=up_children, upward_parent_rank=ranks,
        downward_levels=down, edge_geometry=edge,
        branch_support_true=support, branch_support_sham=support.copy(),
        genotype=np.tile(np.arange(h, dtype=np.int16) % 2, (s, 1)),
        log_donor_allele_frequency=np.full((s, 2, 8), np.log(.5), dtype=np.float32),
        donor_call_mask=np.ones((s, 8), dtype=np.bool_),
        panel_allele_frequency=np.full((s, 2), .5, dtype=np.float32),
        site_entropy=np.full(s, .5, dtype=np.float32),
        site_hold=np.repeat(np.arange(t, dtype=np.int64), 2),
        spans=np.array([1, 2, 5][:t], dtype=np.float32),
        readout_hold=np.arange(t, dtype=np.int64),
        query_column=np.arange(686, dtype=np.int64),
        reference_class=np.full(h, -1, dtype=np.int64),
        sample_age=np.zeros(h, dtype=np.float32),
    )
    return MutationRoutedWholeTreeInputsV1.from_arrays(arrays)


parent, sample, reps, unary, arrays, meta, episodes = prepared()
check("all_native_vertices_preserved", episodes.native_nodes == len(parent)
      and sum(meta["native_nodes_per_tree"]) == len(parent))
check("deepest_unary_representatives_only", set(arrays["member_node"]) == {x for x in reps if x is not None}
      and not set(x for x in unary if x is not None) & set(arrays["member_node"]))
check("roots_and_sampled_leaves_excluded", not set(np.flatnonzero(parent < 0)) & set(arrays["member_node"])
      and not set(sample.flatten()) & set(arrays["member_node"]))
check("maximal_consecutive_episode_sizes", sorted(arrays["episode_size"].tolist()) == [1, 2, 3])
ids = {int(node): int(ep) for node, ep in zip(arrays["member_node"], arrays["member_episode"])}
check("absent_tree_splits_reappearance", ids[reps[0]] == ids[reps[2]]
      and ids[reps[4]] == ids[reps[5]] and ids[reps[0]] != ids[reps[4]])
oldflags, _ = build_clade_flag_arrays_v1(parent, sample)
check("H042_flags_exactly_preserved", np.array_equal(oldflags, arrays["node_flags"]))
check("payload_compact_bytes_and_cap", episodes.tensor_bytes() == meta["payload_bytes"]
      and meta["payload_bytes"] <= meta["array_allocation_bound_bytes"]
      == 18*len(parent)+4*len(sample))
check("linear_forest_root_discovery", meta["parent_links_followed"] <= len(parent))
check("no_exact_bitsets_in_payload", set(arrays) == set(ARRAY_FIELDS)
      and all(isinstance(value, np.ndarray) for value in arrays.values())
      and meta["exact_bitsets_retained"] is False)
copied = arrays["hold_span"].copy()
arrays["hold_span"][0] = 20
check("payload_owns_independent_array_storage", episodes.hold_span[0].item() == float(copied[0]))
arrays["hold_span"][:] = copied

adapter = RankEightCladeEpisodeAdapterV1()
up = torch.zeros((len(parent), 32), dtype=torch.float32)
for tree, node in enumerate(reps):
    if node is not None:
        up[node] = float((tree+1)**2)
member, pool, eligible = adapter.pooled_context(up, episodes)
first_rows = [i for i, node in enumerate(episodes.member_node.tolist()) if node in reps[:3]]
# States 1,4,9 and widths 1,2,5 -> 53/7,46/6,9/3.
expected = torch.tensor([53/7, 46/6, 3], dtype=torch.float32)
check("unequal_span_three_member_LOO_by_hand",
      torch.allclose(pool[first_rows, 0], expected, atol=1e-6, rtol=1e-6))
check("singleton_pool_zero_finite", bool(torch.isfinite(pool).all()) and bool((pool[~eligible] == 0).all()))
check("actual_parameter_free_PyG_aggregation", isinstance(adapter.aggregation, SumAggregation)
      and len(list(adapter.aggregation.parameters())) == 0)
check("new_O_exactly_zero", torch.count_nonzero(adapter.O.weight) == 0)
for arm in ARMS:
    check("initial_injection_exactly_zero_"+arm, torch.count_nonzero(adapter(up, episodes, arm=arm)) == 0)
loss = adapter(up.requires_grad_(), episodes, arm="foreign_episode").square().sum()
# A linear loss supplies a nonzero upstream signal at the zero-output start.
loss = adapter(up, episodes, arm="foreign_episode").sum()
loss.backward()
check("first_backward_core_gradients_zero", all(torch.count_nonzero(getattr(adapter, k).weight.grad) == 0
      for k in ("U", "V", "W")))
check("first_backward_O_can_learn", torch.count_nonzero(adapter.O.weight.grad) > 0)
with torch.no_grad():
    for k in ("U", "V", "O"):
        getattr(adapter, k).weight.fill_(.05)
    adapter.W.weight.zero_()
excluded = torch.ones(len(parent), dtype=torch.bool)
excluded[episodes.member_node[eligible].long()] = False
for arm in ARMS:
    injection = adapter(up, episodes, arm=arm)
    check("identical_eligible_node_mask_"+arm,
          torch.count_nonzero(injection[excluded]) == 0
          and bool((injection[~excluded].abs().sum(-1) > 0).all()))
grads = {}
for arm in ARMS:
    states = up.detach().clone().requires_grad_()
    injection = adapter(states, episodes, arm=arm)
    grads[arm] = torch.autograd.grad(injection[reps[0]].sum(), states)[0]
check("foreign_gradient_reaches_other_tree", grads["foreign_episode"][reps[1]].abs().sum() > 0
      and grads["foreign_episode"][reps[2]].abs().sum() > 0)
check("own_state_zero_cross_tree_adapter_gradient", torch.count_nonzero(grads["own_state"][reps[1]:reps[1]+1]) == 0
      and torch.count_nonzero(grads["own_state"][reps[2]:reps[2]+1]) == 0)
check("episode_gap_blocks_gradient", torch.count_nonzero(grads["foreign_episode"][reps[4]]) == 0)
# Internal IDs change, sampled column order does not.
rng = np.random.default_rng(101)
permutation = np.arange(len(parent))
internal = np.setdiff1d(permutation, sample.flatten())
permutation[internal] = rng.permutation(internal)
newparent = np.empty_like(parent)
newparent[permutation] = np.where(parent < 0, -1, permutation[parent.clip(min=0)])
new_arrays, new_meta = build_clade_episode_arrays_v1(newparent, permutation[sample], copied)
renumbered = OrderedCladeEpisodesV1.from_verified_arrays(new_arrays, new_meta)
newup = torch.empty_like(up)
newup[torch.from_numpy(permutation)] = up.detach()
for arm in ARMS:
    original = adapter(up.detach(), episodes, arm=arm)
    renamed = adapter(newup, renumbered, arm=arm)[torch.from_numpy(permutation)]
    check("internal_renumbering_equivariance_"+arm, torch.allclose(original, renamed, atol=1e-6))
check("flag_renumbering_equivariance", np.array_equal(new_arrays["node_flags"][permutation], arrays["node_flags"]))
_, _, _, _, empty_arrays, empty_meta, empty = prepared(("star", "star"))
check("no_proper_clades_empty_payload", empty.member_node.numel() == empty.episode_size.numel() == 0)
for arm in ARMS:
    check("no_proper_clades_zero_finite_"+arm,
          torch.count_nonzero(adapter(torch.ones(empty.native_nodes, 32), empty, arm=arm)) == 0)
for label, spans in (
        ("zero_span", np.zeros(len(sample), dtype=np.float32)),
        ("negative_span", -copied), ("FP64_span", copied.astype(np.float64)),
        ("nonfinite_span", np.full(len(sample), np.inf, dtype=np.float32))):
    rejects(label, lambda spans=spans: build_clade_episode_arrays_v1(parent, sample, spans))
rejects("too_small_explicit_payload_cap", lambda: build_clade_episode_arrays_v1(parent, sample, copied, max_payload_bytes=1))
rejects("cycle_in_unused_component", lambda: build_clade_episode_arrays_v1(
    np.concatenate((parent, [len(parent)+1, len(parent)])), sample, copied))
badarrays = {k: v.copy() for k, v in arrays.items()}
badarrays["member_node"][0] = len(parent)
badmeta = deepcopy(meta)
badmeta["array_sha256"]["member_node"] = hashlib.sha256(badarrays["member_node"].tobytes()).hexdigest()
rejects("out_of_bounds_even_with_updated_hash", lambda: OrderedCladeEpisodesV1.from_verified_arrays(badarrays, badmeta))
badarrays = {k: v.copy() for k, v in arrays.items()}
badarrays["member_hold"][1] = 0
badmeta = deepcopy(meta)
badmeta["array_sha256"]["member_hold"] = hashlib.sha256(badarrays["member_hold"].tobytes()).hexdigest()
rejects("duplicate_episode_tree_even_with_updated_hash", lambda: OrderedCladeEpisodesV1.from_verified_arrays(badarrays, badmeta))
badmeta = deepcopy(meta); badmeta["eligible_member_count"] += 1
rejects("wrong_eligibility_metadata", lambda: OrderedCladeEpisodesV1.from_verified_arrays(arrays, badmeta))
rejects("unregistered_arm", lambda: adapter(up, episodes, arm="other"))

default = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    fpmodel = OrderedCladeEpisodeGNNV1()
finally:
    torch.set_default_dtype(default)
check("model_FP32_under_FP64_default", all(p.dtype == torch.float32 for p in fpmodel.parameters()))
check("registered_total_31812", sum(p.numel() for p in fpmodel.parameters()) == 31812)
check("new_weights_1024", sum(p.numel() for p in fpmodel.episode_coupling.parameters()) == 1024)
base = AnonymousCladeFlagControlGNNV1()
with torch.no_grad():
    base.clade_coupling.O.weight.normal_(0, .1)
payload = dict(
    schema="anonymous_clade_flag_checkpoint_v1", hypothesis_id="H042", phase="train",
    arm="exact_clade_flags", seed=101, pass_index=30, additional_optimizer_steps=960,
    parameter_count=30788, teacher_used=False, model_kwargs=dict(MODEL_KWARGS),
    model_state={k: v.clone() for k, v in base.state_dict().items()},
)
family, optimizers, proof = build_matched_episode_family_v1(payload)
check("warmstart_complete_H042_state_preserved", all(
    torch.equal(model.state_dict()[k], v) for model in family.values()
    for k, v in payload["model_state"].items()))
check("warmstart_learned_flag_O_preserved_nonzero", all(
    torch.count_nonzero(model.clade_coupling.O.weight) > 0 for model in family.values()))
check("matched_all_initial_weights", all(torch.equal(
    family[ARMS[0]].state_dict()[k], family[ARMS[1]].state_dict()[k])
    for k in family[ARMS[0]].state_dict()))
check("matched_distinct_parameter_storage", all(
    a.data_ptr() != b.data_ptr()
    for a, b in zip(family[ARMS[0]].parameters(), family[ARMS[1]].parameters())))
check("independent_empty_AdamW_state", optimizers[ARMS[0]] is not optimizers[ARMS[1]]
      and all(not opt.state for opt in optimizers.values()))
check("registered_AdamW_settings", all(
    opt.defaults["lr"] == .001 and opt.defaults["weight_decay"] == .0001
    and opt.defaults["betas"] == (.9, .999) and opt.defaults["eps"] == 1e-8
    for opt in optimizers.values()))
check("full_body_trainable", all(p.requires_grad for model in family.values() for p in model.parameters()))
bad = dict(payload, pass_index=29)
rejects("nonpass30_warmstart", lambda: load_h042_episode_warmstart_v1(fpmodel, bad))
bad = deepcopy(payload); del bad["model_state"]["clade_coupling.O.weight"]
rejects("missing_learned_H042_flag_weight", lambda: load_h042_episode_warmstart_v1(fpmodel, bad))
bad = deepcopy(payload); key = next(iter(bad["model_state"])); bad["model_state"][key] = bad["model_state"][key].double()
rejects("FP64_warmstart", lambda: load_h042_episode_warmstart_v1(fpmodel, bad))
bank = synthetic_bank()
check("synthetic_full952_bank_valid", validate_inputs_v1(bank, expected_haplotypes=952)["haplotypes"] == 952)
full_episodes, full_meta = build_clade_episodes_v1(bank)
check("source_bank_episode_API", full_episodes.native_holds == 3 and full_episodes.haplotypes == 952)
with torch.no_grad():
    expected_logits = base.forward_holds(bank, flags=full_episodes.flags)
    outputs = {arm: model.forward_holds(bank, episodes=full_episodes) for arm, model in family.items()}
check("zero_new_O_exact_H042_forward_identity", all(torch.equal(v, expected_logits) for v in outputs.values()))
check("matched_initial_arm_logits", torch.equal(outputs[ARMS[0]], outputs[ARMS[1]]))
check("FP32_full952_hold_logits", all(v.shape == (3, 952, 4) and v.dtype == torch.float32
      and bool(torch.isfinite(v).all()) for v in outputs.values()))
model = family["foreign_episode"]
calls = []
hook = model.pyg_upward.register_forward_hook(lambda *args: calls.append("up"))
downcalls = []
hook2 = model.pyg_downward.register_forward_hook(lambda *args: downcalls.append("down"))
with torch.no_grad():
    model.episode_coupling.O.weight.fill_(.01)
logits = model.forward_train_holds(bank, episodes=full_episodes)
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), torch.arange(logits.numel()//4) % 4)
loss.backward()
hook.remove(); hook2.remove()
check("two_complete_actual_PyG_sweeps", len(calls) == 2*sum(bool(x.numel()) for x in bank.upward_children)
      and len(downcalls) == 2*len(bank.downward_levels))
check("full_body_backward_finite", all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
check("SNP_BiGRU_backbone_episode_all_learn", all(
    next(module.parameters()).grad.abs().sum() > 0
    for module in (model.site_mlp, model.genomic_gru, model.up_message, model.clade_coupling, model.episode_coupling)))
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
optimizers["foreign_episode"].step()
check("independent_optimizer_after_one_synthetic_update",
      len(optimizers["foreign_episode"].state) > 0 and not optimizers["own_state"].state)
rejects("mismatched_bank_spans", lambda: family["own_state"].forward_holds(
    replace(bank, spans=bank.spans+1), episodes=full_episodes))
with torch.no_grad():
    selected = family["own_state"](bank, episodes=full_episodes)
check("unchanged_query_readout_shape", selected.shape == (3, 686, 4))
source_path = ROOT / "src/arggnn/ordered_clade_episode_gnn_v1.py"
receipt = dict(
    schema="H045_clade_episode_CPU_synthetic_checks_v1", status="passed",
    check_count=len(checks), checks=checks, device="cpu", dtype="float32",
    torch_version=torch.__version__, numpy_version=np.__version__,
    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    synthetic_only=True, actual_GPU_operations=False, production_data_or_checkpoint_reads=False,
    registered_training_executed=False, source_only=True,
    parameter_count=31812, added_parameter_count=1024,
    full952_synthetic_nodes=len(bank.parent),
    full952_episode_payload_bytes=full_meta["payload_bytes"],
    compact_array_allocation_bound="18*N+4*T bytes; excludes forest scratch, masks and activations",
)
print(json.dumps(receipt, indent=2, sort_keys=True))
if "--write-receipt" in sys.argv:
    Path(__file__).with_name("cpu_synthetic_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)+"\n")
