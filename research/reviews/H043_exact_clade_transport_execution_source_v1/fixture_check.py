"""Handmade CPU API qualification. No real banks, checkpoints, truth or GPU."""
from pathlib import Path
import ast
import copy
from dataclasses import fields, replace
import hashlib
import io
import json
import os
import sys
from types import SimpleNamespace

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/H043_exact_clade_transport_execution_source_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
for name, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                       ("TORCH_HOME", "cache/torch"), ("TORCH_EXTENSIONS_DIR", "cache/extensions")):
    (HERE / relative).mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(HERE / relative)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
from arggnn import train_anonymous_exact_clade_transport_v1 as api
assert "torch" not in sys.modules, "API import must be stdlib-only and inert"
checks = ["stdlib_only_inert_API_import"]
api.science()
np, torch = api.np, api.torch
torch.set_num_threads(1)
import torch_geometric
from arggnn.anonymous_exact_clade_transport_v1 import AnonymousExactCladeTransportGNNV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1


def rejected(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError("Expected rejection")


cfg = dict(schema="anonymous_exact_clade_transport_execution_config_v1", hypothesis_id="H043",
    linked_hypothesis_ids=["H038", "H042"], operator=api.OPERATOR, arms=list(api.ARMS),
    execution_mode="paired_sequential_one_parent", seed=101, dtype="float32", passes=30,
    updates_per_pass=32, updates_per_arm=960, checkpoint_passes=list(api.PASSES),
    TRAIN_parent_ids=list(api.TRAIN), VAL_parent_ids=list(api.VAL), parent_order="shuffle_seed101_plus_pass",
    model=api.MODEL, parameters_per_arm=30788, base_parameters=29764, added_parameters=1024,
    event_block_size=32768, checkpoint_event_blocks=True, learning_rate=.001, weight_decay=.0001,
    AdamW_betas=[.9, .999], AdamW_epsilon=1e-8, gradient_clip_norm=1., early_stopping=False,
    first_step_qualification="within_continuous_training_no_extra_updates",
    VAL_influences_optimization=False, conditioned_on_H040_VAL=False, all_H042_flags_retained=True,
    selection=api.SELECTION, metrics=api.METRICS, dense_bitset_sidecars=False,
    transport_arrays_saved_to_disk=False, dense_prediction_arrays_saved=False,
    warmstart_completion_ref=api.WARM_COMPLETION, warmstart_checkpoint_ref=api.WARM_CHECKPOINT,
    scientific_refs=list(api.SCIENTIFIC_REFS), practical_accuracy_gain_threshold=.001,
    CPU_bank_cap_bytes=10000000, CPU_target_cap_bytes=10000000, transport_cache_cap_bytes=100000,
    transport_payload_cap_bytes=100000, GPU_residency_cap_bytes=10000000,
    GPU_transport_cap_bytes=100000, RAM_prediction_cap_bytes=10000000)
api.validate_protocol(cfg)
for key, value in (("passes", 31), ("seed", 202), ("arms", list(reversed(api.ARMS))),
                   ("practical_accuracy_gain_threshold", .002), ("event_block_size", 1),
                   ("conditioned_on_H040_VAL", True), ("early_stopping", True),
                   ("transport_arrays_saved_to_disk", True)):
    rejected(lambda k=key, v=value: api.validate_protocol(dict(cfg, **{k: v})))
rejected(lambda: api.run_context({}, {}))
for ref in api.SCIENTIFIC_REFS:
    api.shared.bound(ref)
checks.append("frozen_protocol_and_source_pins_reject_drift")

# Literal observed-only 952-leaf star -> cherry+polytomy, no genotype or target use.
n = 952
p0 = np.r_[np.full(n, n, dtype=np.int64), -1]
p1 = np.full(n + 2, n + 1, dtype=np.int64)
p1[:2] = n
p1[n] = n + 1
p1[n + 1] = -1
p1[p1 >= 0] += len(p0)
parent = torch.from_numpy(np.r_[p0, p1])
samples = torch.stack((torch.arange(n), torch.arange(n) + len(p0)))
empty = torch.empty(0)
values = {f.name: empty for f in fields(MutationRoutedWholeTreeInputsV1)}
values.update(parent=parent, sample_index=samples, query_column=torch.arange(686))
bank = MutationRoutedWholeTreeInputsV1(**values)
keys = tuple(("synthetic_%d" % i, 0) for i in range(n))
calls = []
cache = api.TransportCache(cfg, lambda: None, lambda pid, b, lim: calls.append((pid, lim)))
payload = cache.get("fixture", bank, keys)
assert payload.tensor_bytes() == 6 * parent.numel() + 24
assert calls[0][1]["pair_upper_bound"] == 952
assert cache.get("fixture", bank, keys) is payload and cache.hits == 1
rejected(lambda: cache.get("fixture", replace(bank, query_column=torch.arange(686).flip(0)), keys))
rejected(lambda: cache.get("fixture", bank, tuple(reversed(keys))))
payload.weight[0] *= .5
rejected(lambda: cache.get("fixture", bank, keys))
cache.values.clear()
cache.bytes = 0
rebuilt = cache.get("fixture", bank, keys)
assert rebuilt is not payload and cache.builds == 2
assert len(calls) == 2 and cache.proofs["fixture"]["metadata"]["all_H042_flags_retained"] is True
small = api.TransportCache(dict(cfg, transport_payload_cap_bytes=6 * parent.numel() - 1),
                           lambda: None, lambda *args: None)
rejected(lambda: small.get("too_large", bank, keys))
small = api.TransportCache(dict(cfg, transport_cache_cap_bytes=1),
                           lambda: None, lambda *args: None)
rejected(lambda: small.get("too_large", bank, keys))
assert not small.values and small.bytes == 0
checks.append("actual_complete_payload_cache_recipe_cap_rebuild_and_bank_binding")

# Synthetic FP32 warm schema made from initialized source weights, never H038 bytes.
base = AnonymousExactCladeTransportGNNV1(**api.MODEL)
warm = dict(arm="aligned", seed=101, teacher_used=False,
    model_state={k: v.detach().clone() for k, v in base.state_dict().items()
                 if not k.startswith("clade_coupling.")})
family, optimizers, proofs, hashes = api.build_family_v1(warm, torch.device("cpu"), lambda: None)
assert len(set(hashes.values())) == 1
assert all(sum(p.numel() for p in m.parameters()) == 30788 for m in family.values())
assert all(not o.state for o in optimizers.values())
assert not ({p.data_ptr() for p in family["true_partner"].parameters()}
            & {p.data_ptr() for p in family["own_state"].parameters()})
checks.append("two_actual_PyG_models_matched_states_and_independent_fresh_AdamW")
del family, optimizers, base, warm

outstanding, seen, fail_arm = 0, [], None
shared_bank = SimpleNamespace(parent=torch.tensor([-1]))
shared_transport = object()
mass = torch.zeros(2, 686, 4, dtype=torch.float32)
mass[..., 0] = 1


class Lifetime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        global outstanding
        assert outstanding == 0
        outstanding += 1
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        global outstanding
        assert outstanding == 1
        outstanding -= 1
        return grad


class Toy(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.weight = torch.nn.Parameter(torch.zeros(686, 4))

    def forward_train_holds(self, b, *, transport, control):
        assert b is shared_bank and transport is shared_transport and control == "aligned"
        assert outstanding == 0
        seen.append(self.arm)
        if self.arm == fail_arm:
            raise RuntimeError("literal fixture second-arm failure before optimizer")
        return Lifetime.apply(self.weight).unsqueeze(0).expand(2, -1, -1)


def toy_pair():
    pair = {a: Toy(a) for a in api.ARMS}
    opts = {a: torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001,
            betas=(.9, .999), eps=1e-8) for a, m in pair.items()}
    return pair, opts


pair, opts = toy_pair()
steps = {a: 0 for a in api.ARMS}
def applied(arm):
    steps[arm] += 1
def guard():
    assert all(p.grad is None for m in pair.values() for p in m.parameters())
rows = api.train_parent_pair_v1(pair, opts, shared_bank, mass, shared_transport, guard,
                               qualify=True, after_optimizer_step=applied)
assert seen == list(api.ARMS) and outstanding == 0 and steps == {a: 1 for a in api.ARMS}
assert all(row["finite_FP32"] and row["memory"]["real_GPU_qualification"] is False for row in rows.values())
assert all(isinstance(v, (bool, float, dict)) for row in rows.values() for v in row.values())
assert all(all(p.grad is None for p in m.parameters()) for m in pair.values())
assert all(float(next(iter(o.state.values()))["step"]) == 1 for o in opts.values())
assert not (next(iter(opts["true_partner"].state.values()))["exp_avg"].data_ptr()
            == next(iter(opts["own_state"].state.values()))["exp_avg"].data_ptr())
checks.append("two_sequential_real_FP32_AdamW_updates_one_shared_payload_no_concurrent_graphs")
fail_arm = "own_state"
pair, opts = toy_pair()
steps = {a: 0 for a in api.ARMS}
rejected(lambda: api.train_parent_pair_v1(pair, opts, shared_bank, mass, shared_transport, guard,
                                       after_optimizer_step=applied), RuntimeError)
assert steps == {"true_partner": 1, "own_state": 0} and outstanding == 0
assert all(p.grad is None for m in pair.values() for p in m.parameters())
checks.append("partial_pair_preserves_first_applied_update_count_and_clears_gradients")
fail_arm = None
bad_mass = mass.clone()
bad_mass[0, 0, 0] = float("nan")
pair, opts = toy_pair()
steps = {a: 0 for a in api.ARMS}
rejected(lambda: api.train_parent_pair_v1(pair, opts, shared_bank, bad_mass, shared_transport, guard,
                                       after_optimizer_step=applied))
# Failed graph is abandoned and execution stops, so it never reaches a second arm.
assert steps == {a: 0 for a in api.ARMS}
assert all(p.grad is None for m in pair.values() for p in m.parameters())
outstanding = 0
checks.append("invalid_target_rejected_before_any_optimizer_step")

# Every durable-looking seal and truth binding below exists only in fixture RAM.
class MemoryIO:
    root = ROOT
    def __init__(self):
        self.data, self.truth_reads = {}, 0
    def put(self, name, data):
        ref = dict(path="fixture/" + name, sha256=hashlib.sha256(data).hexdigest())
        self.data[ref["path"]] = data
        return ref
    def save(self, name, value, kind=None):
        if kind == "torch":
            stream = io.BytesIO()
            torch.save(value, stream)
            data = stream.getvalue()
        else:
            data = json.dumps(value, sort_keys=True).encode()
        return self.put(name, data)
    def read(self, ref, **kwargs):
        data = self.data[ref["path"]]
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
        if "/truth/" in ref["path"]:
            self.truth_reads += 1
        return data
    def decode_json(self, ref):
        return json.loads(self.read(ref))


memory = MemoryIO()
parents, banks, bindings = {}, {}, {}
cohorts = np.array(["present_day"] * 182 + ["Bronze_Age"] * 162 + ["BAA"] * 14
                   + ["Neo"] * 302 + ["Yam"] * 26)
truth_stream = io.BytesIO()
np.savez(truth_stream, intervals_bp=np.array([[0., 2000000.]]),
         query_raw_node_ids=np.arange(686), native_paths_1based=np.ones((1, 686), dtype=np.int64),
         source_classes_0based=np.zeros((1, 686), dtype=np.int64))
for pid in api.VAL:
    parents[pid] = dict(parent_id=pid, split="val", positions=np.array([50.]), edges=np.array([0., 100.]),
        retained_indexes=np.array([0]), query_slots=np.arange(686), raw_query_nodes=np.arange(686),
        cohorts=cohorts.copy(), coverage_original=[0., 100.], samples=[dict(sample_id="diploid_%d" % (i // 2)) for i in range(686)])
    banks[pid] = SimpleNamespace(spans=torch.tensor([100.]), readout_hold=torch.tensor([0]))
    ref = memory.put(pid + "/truth/full_truth_intervals.npz", truth_stream.getvalue())
    bindings[pid] = dict(split="val", truth_npz=ref)
reader = api.PhaseReader(memory, "evaluate", bindings)
ram = api.TransportRAMMatrix(memory, reader, parents, banks, cfg)
reader.ram = ram
rejected(lambda: reader.read(bindings[api.VAL[0]]["truth_npz"]))
probability = np.full((1, 686, 4), .25, dtype=np.float32)
for at in api.PASSES:
    for arm in api.ARMS:
        for pid in api.VAL:
            if (at, arm, pid) == (30, "own_state", api.VAL[-1]):
                continue
            ram.seal(at, arm, pid, probability, dict(path="fixture/synthetic.pt", sha256="0" * 64))
rejected(ram.verify)
assert not ram.verified and not reader.truth_gate and memory.truth_reads == 0
rejected(lambda: ram.seal(5, "true_partner", api.VAL[0], probability, {}))
ram.seal(30, "own_state", api.VAL[-1], probability, dict(path="fixture/synthetic.pt", sha256="0" * 64))
ram.verify()
assert len(ram.values) == 96 and ram.verified and memory.truth_reads == 0
assert all(not value.flags.writeable for value in ram.values.values())
reader.truth_gate = True
assert reader.current_probability_seal_v1(ram.records[5, "true_partner", api.VAL[0]],
                                         parents[api.VAL[0]]).shape == (1, 686, 4)
rejected(lambda: ram.grid(ram.records[5, "true_partner", api.VAL[0]], parents[api.VAL[1]]))
value = ram.values[5, "true_partner", api.VAL[0]]
value.setflags(write=True)
rejected(lambda: ram.grid(ram.records[5, "true_partner", api.VAL[0]], parents[api.VAL[0]]))
value.setflags(write=False)
checks.append("all96_H043_seals_before_truth_immutable_current_grid_and_parent_join")

# Exact twelve-checkpoint provenance parser, synthetic source states in RAM only.
synthetic_model = AnonymousExactCladeTransportGNNV1(**api.MODEL)
synthetic_state = {k: v.detach().clone() for k, v in synthetic_model.state_dict().items()}
config_ref = memory.save("synthetic_config.json", cfg)
input_ref = dict(path="fixture/synthetic_input.json", sha256="1" * 64)
observation_ref = dict(path="fixture/synthetic_observations.json", sha256="2" * 64)
initial_hash = api.shared.state_identity(synthetic_state)
train = dict(schema="H043_transport_train_completion_v1", status="completed_fixed_30_pass_paired_TRAIN",
    hypothesis_id="H043", arms=list(api.ARMS), steps={a: 960 for a in api.ARMS},
    outer_truth_read=False, VAL_truth_read=False, conditioned_on_H040_VAL=False,
    scientific_refs=list(api.SCIENTIFIC_REFS), initial_state_sha256={a: initial_hash for a in api.ARMS},
    config_ref=config_ref, input_ref=input_ref, observations_seal=observation_ref, checkpoint_groups={})
for at in api.PASSES:
    train["checkpoint_groups"][str(at)] = {}
    for arm in api.ARMS:
        cp = dict(schema="H043_transport_checkpoint_v1", hypothesis_id="H043", phase="train", arm=arm,
            pass_index=at, additional_optimizer_steps=32 * at, seed=101, parameter_count=30788,
            model_kwargs=api.MODEL, teacher_used=False, optimizer_state_reset=True,
            scientific_refs=list(api.SCIENTIFIC_REFS), warmstart_checkpoint_ref=api.WARM_CHECKPOINT,
            initial_state_sha256=initial_hash, input_ref=input_ref, config_ref=config_ref,
            observation_seal=observation_ref, conditioned_on_H040_VAL=False, model_state=synthetic_state)
        train["checkpoint_groups"][str(at)][arm] = memory.save(
            "%s.synthetic.pass_%d.pt" % (arm, at), cp, "torch")
states, refs = api.load_checkpoints_v1(memory, train, cfg)
assert len(states) == len(refs) == 12
bad = copy.deepcopy(train)
bad["checkpoint_groups"]["10"]["true_partner"] = bad["checkpoint_groups"]["10"]["own_state"]
rejected(lambda: api.load_checkpoints_v1(memory, bad, cfg))
bad = copy.deepcopy(train)
bad["checkpoint_groups"]["5"]["true_partner"] = bad["checkpoint_groups"]["10"]["true_partner"]
rejected(lambda: api.load_checkpoints_v1(memory, bad, cfg))
bad = copy.deepcopy(train)
bad["initial_state_sha256"]["own_state"] = "3" * 64
rejected(lambda: api.load_checkpoints_v1(memory, bad, cfg))
cap_only_states, _ = api.load_checkpoints_v1(memory, train, dict(cfg, RAM_prediction_cap_bytes=1))
assert len(cap_only_states) == 12
del cap_only_states
rejected(lambda: api.load_checkpoints_v1(memory, train, dict(cfg, practical_accuracy_gain_threshold=.002)))
assert memory.truth_reads == 0
checks.append("twelve_synthetic_checkpoint_cells_exact_arm_pass_config_and_matched_initial_provenance")
del synthetic_model, synthetic_state, states

selected, full = {}, {}
for arm in api.ARMS:
    validation = FullPanelPrimaryValidationV1(parents, bindings, reader, api.fit.metric_helpers_v1(), cfg)
    rows = []
    for at in api.PASSES:
        cells, means = validation.score([(pid, ram.records[at, arm, pid]) for pid in api.VAL])
        rows.append(dict(pass_index=at, cells=cells, equal_parent_metrics=means))
    best = min(rows, key=lambda row: api.h038.selection_key_v1(row["equal_parent_metrics"], row["pass_index"]))
    assert best["pass_index"] == 5
    assert best["equal_parent_metrics"]["primary_present_day_182"]["accuracy"] == 1.
    cells, means = validation.full_final([(pid, ram.records[5, arm, pid]) for pid in api.VAL])
    selected[arm] = best
    full[arm] = dict(cells=cells, equal_parent_metrics=means)
    rejected(lambda: validation.full_final([(pid, ram.records[5, arm, pid]) for pid in api.VAL]))
checks.append("unchanged_original_primary_scorer_all_selected_populations_and_earliest_tie_selection")
effects = api.paired_parent_differences_v1(selected, .001)
assert effects["paired_parent_count"] == 8 and effects["sample_SD_paired_accuracy_delta"] == 0
assert effects["accuracy_effect_sign_counts"] == dict(positive=0, zero=8, negative=0)
populations = set(full["true_partner"]["cells"][0]["groups"])
assert len(populations) == 7
for group in populations:
    report = api.paired_parent_differences_v1(full, .001, group=group)
    assert report["paired_parent_count"] == 8 and report["sample_SD_paired_accuracy_delta"] == 0
    assert all(row["sample_SD"] == 0 for row in report["metric_effect_summaries"].values())
negative = copy.deepcopy(selected)
for i, cell in enumerate(negative["true_partner"]["cells"]):
    cell["groups"]["primary_present_day_182"]["accuracy"] = .5 if i % 2 else 1.
negative_effects = api.paired_parent_differences_v1(negative, .001)
assert negative_effects["accuracy_effect_sign_counts"] == dict(positive=0, zero=4, negative=4)
assert negative_effects["mean_paired_accuracy_delta_true_minus_own"] == -.25
assert negative_effects["sample_SD_paired_accuracy_delta"] > 0
assert not negative_effects["reaches_practical_accuracy_threshold"]
checks.append("eight_paired_parent_effects_SD_negative_outcomes_and_all_seven_populations")

tree = ast.parse((ROOT / api.SOURCE).read_text())
names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
assert "main" not in names
assert "run_train" in names and "run_evaluate" in names
assert "_memory_start" in names and "train_parent_pair_v1" in names
checks.append("no_CLI_admission_launcher_or_accounting_entry_point")
receipt = dict(schema="H043_execution_source_CPU_fixture_receipt_v1", status="passed",
    checks=checks, check_count=len(checks), torch=torch.__version__, PyG=torch_geometric.__version__,
    cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], CPU_threads=torch.get_num_threads(),
    actual_PyG_full_model_constructors=True, actual_whole_parent_model_forward_or_training=False,
    tiny_toy_sequential_training_only=True, original_scorer_on_synthetic_in_RAM_truth=True,
    real_data_or_checkpoint_or_H040_VAL_read=False, GPU_or_admission_or_journal_actions=False,
    source_ref=dict(path=api.SOURCE, sha256=hashlib.sha256((ROOT / api.SOURCE).read_bytes()).hexdigest()))
(HERE / "fixture_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
