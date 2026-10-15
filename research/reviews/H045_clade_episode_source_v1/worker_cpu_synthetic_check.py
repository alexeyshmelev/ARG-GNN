"""CPU-only H045 worker integration smoke: synthetic tensors and in-memory I/O."""
from pathlib import Path
from copy import deepcopy
import ast
import hashlib
import io
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
from arggnn import train_ordered_clade_episode_v1 as worker
assert "torch" not in worker.__dict__, "Worker import must stay inert"
import numpy as np
import torch
from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from arggnn.ordered_clade_episode_gnn_v1 import MODEL_KWARGS
torch.set_num_threads(1)
torch.manual_seed(101)

checks = ["worker_import_inert_without_science_or_I_O"]


def check(name, ok):
    if not bool(ok):
        raise AssertionError(name)
    checks.append(name)


def rejects(name, fn):
    try:
        fn()
    except (ValueError, KeyError, TypeError):
        checks.append(name)
    else:
        raise AssertionError("Expected rejection: " + name)


# Reuse only fixture definitions, without rerunning the separate model suite.
tree = ast.parse(Path(__file__).with_name("cpu_synthetic_check.py").read_text())
fixture = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
    and node.name in ("forest", "synthetic_bank")], type_ignores=[])
exec(compile(fixture, "<H045 synthetic fixtures>", "exec"), globals())
bank = synthetic_bank()
cfg = dict(
    schema="ordered_clade_episode_execution_config_v1", hypothesis_id="H045",
    linked_hypothesis_ids=["H038", "H042"], operator=worker.OPERATOR, arms=list(worker.ARMS),
    execution_mode="paired_sequential_one_parent", seed=101, dtype="float32",
    passes=30, updates_per_pass=32, updates_per_arm=960, checkpoint_passes=list(worker.PASSES),
    TRAIN_parent_ids=list(worker.TRAIN), VAL_parent_ids=list(worker.VAL),
    parent_order="shuffle_seed101_plus_pass", model=worker.MODEL,
    parameters_per_arm=31812, base_parameters=30788, added_parameters=1024,
    learning_rate=.001, weight_decay=.0001, AdamW_betas=[.9, .999], AdamW_epsilon=1e-8,
    gradient_clip_norm=1., early_stopping=False,
    first_step_qualification="within_continuous_training_no_extra_updates",
    VAL_influences_optimization=False, conditioned_on_H040_VAL=False,
    all_H042_flags_retained=True, same_VAL_is_development_evidence=True,
    practical_boundary_F1_loss_tolerance=.01, practical_accuracy_gain_threshold=.001,
    selection=worker.SELECTION, metrics=worker.METRICS, dense_bitset_sidecars=False,
    episode_arrays_saved_to_disk=False, dense_prediction_arrays_saved=False,
    warmstart_completion_ref=worker.WARM_COMPLETION, warmstart_checkpoint_ref=worker.WARM_CHECKPOINT,
    scientific_refs=list(worker.SCIENTIFIC_REFS),
    **{name: 10000000 for name in worker.RESOURCE_CAPS},
)
worker.validate_protocol(cfg)
checks.append("registered_protocol_accepts_exact_H045_settings")
for key, value in (("parameters_per_arm", 30788), ("practical_boundary_F1_loss_tolerance", .02),
                   ("warmstart_checkpoint_ref", {}), ("arms", ["foreign_episode"])):
    bad = dict(cfg); bad[key] = value
    rejects("reject_protocol_"+key, lambda bad=bad: worker.validate_protocol(bad))
worker.science()
for ref in worker.SCIENTIFIC_REFS:
    check("frozen_source_pin_"+ref["path"],
          hashlib.sha256((ROOT/ref["path"]).read_bytes()).hexdigest() == ref["sha256"])
bounds = worker.episode_upper_bounds_v1(bank)
check("worker_payload_bound_exact", bounds["compact_payload_upper_bound_bytes"] == 18*len(bank.parent)+4*len(bank.spans))
guard_calls, preparations = [], []
guard = lambda: guard_calls.append(1)
cache = worker.EpisodeCache(cfg, guard, lambda *args: preparations.append(args))
keys = ["synthetic_leaf_"+str(i) for i in range(952)]
episodes = cache.get("synthetic_a", bank, keys)
check("source_bound_cache_build", cache.builds == 1 and len(preparations) == 1
      and cache.bytes == episodes.tensor_bytes() and cache.bytes <= cfg["episode_cache_cap_bytes"])
check("cache_hit_identical_object", cache.get("synthetic_a", bank, keys) is episodes and cache.hits == 1)
rejects("changed_sample_keys_rejected", lambda: cache.get("synthetic_a", bank, keys[::-1]))
episodes.member_hold[0] += 1
rejects("mutated_cached_episode_rejected", lambda: cache.get("synthetic_a", bank, keys))
episodes.member_hold[0] -= 1
cache.cfg = dict(cfg, episode_cache_cap_bytes=episodes.tensor_bytes())
cache.get("synthetic_b", bank, keys)
check("cache_evicts_to_bounded_single_payload", tuple(cache.values) == ("synthetic_b",)
      and cache.bytes == episodes.tensor_bytes())
base = AnonymousCladeFlagControlGNNV1()
with torch.no_grad():
    base.clade_coupling.O.weight.normal_(0, .1)
payload = dict(
    schema="anonymous_clade_flag_checkpoint_v1", hypothesis_id="H042", phase="train",
    arm="exact_clade_flags", seed=101, pass_index=30, additional_optimizer_steps=960,
    parameter_count=30788, teacher_used=False, model_kwargs=dict(MODEL_KWARGS),
    model_state={k: v.clone() for k, v in base.state_dict().items()},
)
family, optimizers, proofs, hashes = worker.build_family_v1(payload, torch.device("cpu"), guard)
check("worker_matched_initialization_hashes", len(set(hashes.values())) == 1)
check("worker_preserves_learned_H042_flag_adapter", all(
    torch.equal(model.clade_coupling.O.weight, base.clade_coupling.O.weight) for model in family.values()))
mass = torch.zeros(3, 686, 4, dtype=torch.float32)
mass[:, :, 0] = bank.spans[:, None]
steps = []
rows = worker.train_parent_pair_v1(family, optimizers, bank, mass, episodes, guard,
    after_optimizer_step=lambda arm: steps.append(arm))
check("paired_synthetic_update_exact_once_per_arm", tuple(rows) == worker.ARMS and tuple(steps) == worker.ARMS)
check("paired_synthetic_update_finite_FP32", all(row["finite_FP32"] and np.isfinite(row["loss"]) for row in rows.values()))
check("paired_gradient_lifetimes_end_before_next_arm", all(p.grad is None for model in family.values() for p in model.parameters()))
check("independent_optimizer_states_after_pair_update", all(len(opt.state)>0 for opt in optimizers.values())
      and optimizers[worker.ARMS[0]].state is not optimizers[worker.ARMS[1]].state)

class MemoryIO:
    root = ROOT
    def __init__(self):
        self.raw = {}
    def save(self, name, value):
        data = json.dumps(value, sort_keys=True).encode()
        ref = dict(path="synthetic/"+name, sha256=hashlib.sha256(data).hexdigest())
        self.raw[ref["path"]] = data
        return ref
    def read(self, ref, **kwargs):
        data = self.raw[ref["path"]]
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
        return data
    def decode_json(self, ref):
        return json.loads(self.read(ref))

memory = MemoryIO()
completion = dict(
    schema="anonymous_clade_flag_train_completion_v1",
    status="completed_fixed_30_pass_flag_TRAIN", hypothesis_id="H042", arm="exact_clade_flags",
    steps={"exact_clade_flags":960}, history=[{} for _ in range(30)],
    VAL_truth_read=False, outer_truth_read=False,
    checkpoint_groups={"30":{"exact_clade_flags":worker.WARM_CHECKPOINT}},
)
buffer = io.BytesIO(); torch.save(payload, buffer)
class SyntheticWarmReader:
    def decode_json(self, ref):
        assert ref == worker.WARM_COMPLETION
        return completion
    def read(self, ref):
        assert ref == worker.WARM_CHECKPOINT
        return buffer.getvalue()
warm = worker.load_h042_warmstart_v1(SyntheticWarmReader())
check("warmstart_binds_synthetic_completion_and_pass30", torch.equal(
    warm["model_state"]["clade_coupling.O.weight"], payload["model_state"]["clade_coupling.O.weight"]))
completion["checkpoint_groups"]["30"] = {}
rejects("warmstart_rejects_unbound_pass30", lambda: worker.load_h042_warmstart_v1(SyntheticWarmReader()))

parents = {pid: dict(
    parent_id=pid, positions=np.array([50.,150.,250.]), edges=np.array([0.,100.,200.,300.]),
    retained_indexes=np.arange(3), query_slots=np.arange(686),
    raw_query_nodes=np.arange(686), cohorts=np.full(686,"present_day"),
) for pid in worker.VAL}
ram = worker.EpisodeRAMMatrix(memory, memory, parents, {pid:bank for pid in worker.VAL}, cfg)
p = np.full((3,686,4), .25, dtype=np.float32)
checkpoint = dict(path="synthetic/checkpoint.pt",sha256="0"*64)
at, arm, pid = worker.PASSES[0], worker.ARMS[0], worker.VAL[0]
ram.seal(at, arm, pid, p, checkpoint)
rejects("incomplete95_or_fewer_prediction_barrier_rejected", ram.verify)
rejects("grid_denied_before_complete96_barrier", lambda: ram.grid(ram.records[at,arm,pid], parents[pid]))
for at2 in worker.PASSES:
    for arm2 in worker.ARMS:
        for pid2 in worker.VAL:
            if (at2,arm2,pid2) not in ram.values:
                ram.seal(at2,arm2,pid2,p,checkpoint)
ram.verify()
check("complete96_prediction_barrier", ram.verified and len(ram.values) == 96)
check("native_hold_grid_readout_after_barrier", ram.grid(ram.records[at,arm,pid],parents[pid]).shape == (3,686,4))
rejects("immutable_seal_replacement_denied", lambda: ram.seal(at,arm,pid,p,checkpoint))
truth = dict(path="synthetic/truth/full_truth_intervals.npz",sha256=hashlib.sha256(b"synthetic").hexdigest())
memory.raw[truth["path"]] = b"synthetic"
phase = worker.PhaseReader(memory,"evaluate",{pid:dict(truth_npz=truth)})
phase.ram = ram
rejects("VAL_truth_before_gate_denied", lambda: phase.read(truth))
phase.truth_gate = ram.verified
check("VAL_truth_after_complete96_gate", phase.read(truth) == b"synthetic")
full = {arm:dict(cells=[dict(parent_id=pid,groups={"primary_present_day_182":dict(
    accuracy=.6+(.002 if arm=="foreign_episode" else 0),
    boundaries=dict(f1=.5-(.005 if arm=="foreign_episode" else 0)),
)}) for pid in worker.VAL]) for arm in worker.ARMS}
criteria = worker.practical_episode_criteria_v1(full,.001,.01)
check("joint_accuracy_and_boundary_practical_criteria", criteria["meets_both_registered_practical_criteria"])
for row in full["foreign_episode"]["cells"]:
    row["groups"]["primary_present_day_182"]["boundaries"]["f1"] = .48
check("boundary_loss_over_1pp_rejects_practical_criterion",
      not worker.practical_episode_criteria_v1(full,.001,.01)["meets_both_registered_practical_criteria"])
full["foreign_episode"]["cells"][0]["groups"]["primary_present_day_182"]["boundaries"]["f1"] = None
check("undefined_boundary_does_not_pass_practical_criterion",
      not worker.practical_episode_criteria_v1(full,.001,.01)["meets_both_registered_practical_criteria"])
source = ROOT/worker.SOURCE
receipt = dict(
    schema="H045_paired_worker_CPU_synthetic_checks_v1",status="passed",
    check_count=len(checks),checks=checks,device="cpu",dtype="float32",
    worker_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    synthetic_only=True,actual_GPU_operations=False,production_data_or_checkpoint_reads=False,
    accounting_or_admission_changed=False,registered_training_executed=False,
    production_checkpoint_creation=False,synthetic_updates_per_arm=1,
)
print(json.dumps(receipt,indent=2,sort_keys=True))
if "--write-receipt" in sys.argv:
    Path(__file__).with_name("worker_cpu_synthetic_receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n")
