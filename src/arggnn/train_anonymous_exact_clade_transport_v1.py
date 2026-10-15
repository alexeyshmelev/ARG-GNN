"""H043 scientific paired training and frozen native-hold evaluation API.

Root supplies the admitted reader/writer, current resource guard and leased CUDA
device. This module has no admission, launcher, resource ledger or CLI. Importing
it performs no science, data access or execution. H040/H042 sources stay frozen.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
import math
import random
import time

from . import H040_latent_clade_execution_v1 as shared

SOURCE = "src/arggnn/train_anonymous_exact_clade_transport_v1.py"
ARMS = ("true_partner", "own_state")
TRAIN, VAL, PASSES = shared.TRAIN, shared.VAL, shared.PASSES
MODEL, SELECTION = dict(shared.MODEL), shared.SELECTION
WARM_COMPLETION, WARM_CHECKPOINT = shared.WARM_COMPLETION, shared.WARM_CHECKPOINT
OPERATOR = "anonymous_exact_changed_clade_pairs_v1"
SCIENTIFIC_REFS = (
    dict(path="src/arggnn/anonymous_exact_clade_transport_v1.py",
         sha256="441751a252f111fdf0d8af4e70e93e5a923b7bad7829f3aed7d5060d65727304"),
    dict(path="src/arggnn/anonymous_clade_flag_control_v1.py",
         sha256="d88a4639a53e63e2bff91e20d98e2af1f5cf8abc71e5fde2e02ac35e831eb8ce"),
    dict(path=shared.SOURCE,
         sha256="28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d"),
    dict(path="src/arggnn/mutation_routed_full_panel_primary_v1.py",
         sha256="a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374"),
    dict(path="src/arggnn/train_mutation_routed_full_panel_pyg_v1.py",
         sha256="c01c4d9506faae7bf537282e4d722b115b3c69cc5de1d4570758136de292dae3"),
    dict(path="src/arggnn/biological_gst_control_v1.py",
         sha256="66788b41cabcb852c739505200a641bdc93fea3c9fbb642c291d52cefc77350f"),
    dict(path="src/arggnn/mutation_routed_full_panel_fast_reader_v1.py",
         sha256="d6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03"),
    *shared.MECHANISM_REFS,
)
METRICS = dict(
    aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble",
    boundary_tolerance_bp=1000., calibration_bins=15, normalization_length_bp=2000000,
    posthoc_renormalization=False, primary="present_day182", probability_floor=False,
    purity_filter=False, secondary=["all686", "ancient504", "Bronze_Age162", "BAA14", "Neo302", "Yam26"],
    short_tract_cutoff_bp=10000.,
    support="same_ARGMix_original100bp_center_bins_clipped_to_coverage_no_missing_bins_no_tail_extrapolation",
)
RESOURCE_CAPS = (
    "CPU_bank_cap_bytes", "CPU_target_cap_bytes", "transport_cache_cap_bytes",
    "transport_payload_cap_bytes", "GPU_residency_cap_bytes",
    "GPU_transport_cap_bytes", "RAM_prediction_cap_bytes",
)
require, PhaseReader = shared.require, shared.PhaseReader


def science():
    """Initialize existing serializers lazily, after root capability validation."""
    global np, torch, h038, fit
    shared.science()
    np, torch, h038, fit = shared.np, shared.torch, shared.h038, shared.fit


def validate_protocol(cfg):
    require(
        cfg.get("schema") == "anonymous_exact_clade_transport_execution_config_v1"
        and cfg.get("hypothesis_id") == "H043"
        and cfg.get("linked_hypothesis_ids") == ["H038", "H042"]
        and cfg.get("operator") == OPERATOR and cfg.get("arms") == list(ARMS)
        and cfg.get("execution_mode") == "paired_sequential_one_parent"
        and cfg.get("seed") == 101 and cfg.get("dtype") == "float32"
        and cfg.get("passes") == 30 and cfg.get("updates_per_pass") == 32
        and cfg.get("updates_per_arm") == 960 and cfg.get("checkpoint_passes") == list(PASSES)
        and cfg.get("TRAIN_parent_ids") == list(TRAIN) and cfg.get("VAL_parent_ids") == list(VAL)
        and cfg.get("parent_order") == "shuffle_seed101_plus_pass"
        and cfg.get("model") == MODEL and cfg.get("parameters_per_arm") == 30788
        and cfg.get("base_parameters") == 29764 and cfg.get("added_parameters") == 1024
        and cfg.get("event_block_size") == 32768 and cfg.get("checkpoint_event_blocks") is True
        and cfg.get("learning_rate") == .001 and cfg.get("weight_decay") == .0001
        and cfg.get("AdamW_betas") == [.9, .999] and cfg.get("AdamW_epsilon") == 1e-8
        and cfg.get("gradient_clip_norm") == 1. and cfg.get("early_stopping") is False
        and cfg.get("first_step_qualification") == "within_continuous_training_no_extra_updates"
        and cfg.get("VAL_influences_optimization") is False
        and cfg.get("conditioned_on_H040_VAL") is False
        and cfg.get("all_H042_flags_retained") is True
        and cfg.get("selection") == SELECTION and cfg.get("metrics") == METRICS
        and cfg.get("dense_bitset_sidecars") is False
        and cfg.get("transport_arrays_saved_to_disk") is False
        and cfg.get("dense_prediction_arrays_saved") is False
        and cfg.get("warmstart_completion_ref") == WARM_COMPLETION
        and cfg.get("warmstart_checkpoint_ref") == WARM_CHECKPOINT
        and cfg.get("scientific_refs") == list(SCIENTIFIC_REFS),
        "Exact prospective H043 paired 30-pass protocol required",
    )
    threshold = cfg.get("practical_accuracy_gain_threshold")
    require(type(threshold) in (float, int) and math.isfinite(threshold) and threshold == .001,
            "Prespecified absolute accuracy threshold0.001 required")
    for name in RESOURCE_CAPS:
        require(type(cfg.get(name)) is int and cfg[name] > 0, "Positive root-reviewed cap: " + name)
    require(cfg["transport_payload_cap_bytes"] <= 1006632960,
            "Preserve the source-qualified complete payload cap")


def transport_upper_bounds_v1(bank):
    """Raw compact tensors only; root must qualify Python scratch/RSS separately."""
    n, t, nodes = int(bank.sample_index.shape[1]), int(bank.sample_index.shape[0]), int(bank.parent.numel())
    pairs = n * (t - 1)
    return dict(native_nodes=nodes, native_holds=t, haplotypes=n, pair_upper_bound=pairs,
                compact_payload_upper_bound_bytes=6 * nodes + 24 * pairs,
                event_chunks_plus_concatenation_upper_bound_bytes=48 * pairs,
                adjacent_exact_masks_only=True, Python_scratch_and_RSS_require_root_review=True)


def _payload_arrays(value):
    return dict(node_flags=value.flags.node_flags.numpy(),
                **{name: getattr(value, name).numpy() for name in
                   ("node_size", "boundary_index", "old_node", "new_node",
                    "leaf_support", "intersection_count", "weight")})


def _payload_identity(value):
    return {k: shared.array_identity(v) for k, v in _payload_arrays(value).items()}


def _bank_identity(bank):
    return {name: shared.array_identity(getattr(bank, name).numpy())
            for name in ("parent", "sample_index", "query_column")}


class TransportCache:
    """RAM-only bounded complete payloads with immutable observation-bound recipes."""
    def __init__(self, cfg, guard, before_build):
        self.cfg, self.guard, self.before_build = cfg, guard, before_build
        self.values, self.proofs = OrderedDict(), {}
        self.bytes = self.builds = self.hits = self.peak_runtime_bytes = 0
        self.build_seconds = 0.

    def get(self, pid, bank, keys):
        self.guard()
        binding = dict(bank_arrays=_bank_identity(bank), stable_sample_keys_sha256=hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode()).hexdigest())
        if pid in self.proofs:
            require(self.proofs[pid]["binding"] == binding,
                    "Transport recipe must remain bound to original topology/sample/query columns")
        if pid in self.values:
            value = self.values[pid]
            require(_payload_identity(value) == self.proofs[pid]["runtime_array_identities"],
                    "Cached compact transport changed")
            self.hits += 1
            self.values.move_to_end(pid)
            return value
        from .anonymous_exact_clade_transport_v1 import build_clade_transport_v1
        limits = transport_upper_bounds_v1(bank)
        self.before_build(pid, bank, limits)
        self.guard()
        started = time.monotonic()
        value, meta = build_clade_transport_v1(
            bank, max_payload_bytes=self.cfg["transport_payload_cap_bytes"])
        require(meta["flag_meta"]["topology_sha256"] == binding["bank_arrays"]["parent"]["sha256"]
                and meta["flag_meta"]["sample_index_sha256"] == binding["bank_arrays"]["sample_index"]["sha256"]
                and meta["query_column_sha256"] == binding["bank_arrays"]["query_column"]["sha256"],
                "Builder metadata must bind exact original bank arrays")
        proof = dict(binding=binding, metadata=meta,
                     runtime_array_identities=_payload_identity(value), raw_bounds=limits)
        if pid in self.proofs:
            require(self.proofs[pid] == proof, "Rebuilt observation-only recipe/content changed")
        else:
            self.proofs[pid] = proof
        size = value.tensor_bytes()
        require(size <= self.cfg["transport_cache_cap_bytes"],
                "One complete payload exceeds root RAM cache cap")
        while self.values and self.bytes + size > self.cfg["transport_cache_cap_bytes"]:
            _, old = self.values.popitem(last=False)
            self.bytes -= old.tensor_bytes()
            del old
        self.values[pid] = value
        self.bytes += size
        self.builds += 1
        self.build_seconds += time.monotonic() - started
        self.peak_runtime_bytes = max(self.peak_runtime_bytes, self.bytes)
        self.guard()
        return value


def load_banks(ctx, reader, ids, cache):
    from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1, read_full_cache_parent_v1
    index = read_full_cache_index_v1(ctx["observation_cache_ref"], reader, registry=ctx["registry"],
        entry_refs=ctx["entry_refs"], source_refs=ctx["observation_source_refs"])
    banks, parents, keys, total = {}, {}, {}, 0
    for pid in ids:
        ctx["guard"]()
        item = read_full_cache_parent_v1(pid, index, reader)
        bank, parent = item["model_inputs"], item["scoring"]
        require(parent["split"] == ("train" if pid in TRAIN else "val"), "Original observed split required")
        total += h038.tensor_bytes_v1(bank)
        require(total <= ctx["cfg"]["CPU_bank_cap_bytes"], "Complete CPU bank cap exceeded")
        banks[pid], parents[pid] = bank, parent
        keys[pid] = shared.stable_keys(bank, ctx["entry_refs"][pid], reader)
        cache.get(pid, bank, keys[pid])
        print(json.dumps(dict(stage="H043_observed_transport_recipe", parent_id=pid,
            native_holds=int(bank.sample_index.shape[0]), native_nodes=int(bank.parent.numel()),
            events=cache.proofs[pid]["metadata"]["event_count"], CPU_bank_bytes=total)), flush=True)
        del item
    seal = ctx["output"].save("all_%d_observed_RAM_transport_recipes.json" % len(ids), dict(
        schema="H043_observation_barrier_v1", hypothesis_id="H043", parent_ids=list(ids),
        observation_cache_ref=ctx["observation_cache_ref"],
        entry_refs={p: ctx["entry_refs"][p] for p in ids}, recipes=cache.proofs,
        all_observations_precede_phase_truth=True, CPU_bank_tensor_bytes=total,
        exact_bitsets_saved=False, transport_arrays_saved_to_disk=False,
        scientific_refs=list(SCIENTIFIC_REFS)))
    return banks, parents, keys, seal


def build_family_v1(warm, device, guard):
    """Independent learned tensors and AdamW states, matched original CPU seed."""
    from .anonymous_exact_clade_transport_v1 import (
        AnonymousExactCladeTransportGNNV1, load_h038_transport_warmstart_v1)
    family, optimizers, proofs, hashes = {}, {}, {}, {}
    for arm in ARMS:
        guard()
        torch.manual_seed(101)
        model = AnonymousExactCladeTransportGNNV1(**MODEL, arm=arm,
            event_block_size=32768, checkpoint_event_blocks=True)
        proofs[arm] = load_h038_transport_warmstart_v1(model, warm)
        hashes[arm] = shared.state_identity(model.state_dict())
        family[arm] = model.to(device=device, dtype=torch.float32)
        optimizers[arm] = torch.optim.AdamW(family[arm].parameters(), lr=.001,
            weight_decay=.0001, betas=(.9, .999), eps=1e-8)
    require(len(set(hashes.values())) == 1, "Matched learned initial states required")
    require(not ({p.data_ptr() for p in family[ARMS[0]].parameters()}
                 & {p.data_ptr() for p in family[ARMS[1]].parameters()}),
            "Arms must not share learned tensor storage")
    require(all(not o.state for o in optimizers.values()), "Fresh independent optimizer states required")
    return family, optimizers, proofs, hashes


def _memory_start(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _memory_end(device):
    if device.type != "cuda":
        return dict(device_type=device.type, real_GPU_qualification=False)
    torch.cuda.synchronize(device)
    return dict(device_type="cuda", real_GPU_qualification=True,
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))


def train_one(model, optimizer, bank, mass, transport, guard, *, qualify=False, after_optimizer_step=None):
    """One actual update. Only Python scalars survive; gradient storage is cleared."""
    from .biological_gst_control_v1 import exact_span_loss
    guard()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    require(mass.dtype == torch.float32 and all(p.dtype == torch.float32 for p in model.parameters()),
            "FP32 parameters and full-parent target mass required")
    if qualify:
        _memory_start(bank.parent.device)
    started = time.monotonic()
    try:
        with torch.autocast(device_type=bank.parent.device.type, enabled=False):
            logits = model.forward_train_holds(bank, transport=transport, control="aligned")
            require(logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()),
                    "Finite full-parent FP32 logits required")
            loss = exact_span_loss(logits, mass)
        require(loss.dtype == torch.float32 and bool(torch.isfinite(loss)), "Finite FP32 loss required")
        loss.backward()
        require(all(p.grad is None or (p.grad.dtype == torch.float32
                    and bool(torch.isfinite(p.grad).all())) for p in model.parameters()),
                "Finite FP32 gradients required")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if after_optimizer_step is not None:
            after_optimizer_step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                "Finite updated FP32 parameters required")
        require(all(not isinstance(v, torch.Tensor) or (v.dtype == torch.float32
                    and bool(torch.isfinite(v).all()))
                    for state in optimizer.state.values() for v in state.values()),
                "Finite independent FP32 AdamW states required")
        row = dict(loss=float(loss.detach()), gradient_norm=float(norm),
                   seconds=time.monotonic() - started, finite_FP32=True)
        if qualify:
            row["memory"] = _memory_end(bank.parent.device)
        del logits, loss, norm
    finally:
        optimizer.zero_grad(set_to_none=True)
    require(all(p.grad is None for p in model.parameters()), "Gradient lifetime must end before next arm")
    guard()
    return row


def train_parent_pair_v1(family, optimizers, bank, mass, transport, guard, *, qualify=False, after_optimizer_step=None):
    """Same bank/payload/target, two sequential independent autograd lifetimes."""
    require(tuple(family) == ARMS and tuple(optimizers) == ARMS, "Fixed paired arm order required")
    rows = {}
    for arm in ARMS:
        require(all(p.grad is None for model in family.values() for p in model.parameters()),
                "No gradients may survive across independent arm steps")
        callback = None if after_optimizer_step is None else lambda a=arm: after_optimizer_step(a)
        rows[arm] = train_one(family[arm], optimizers[arm], bank, mass, transport, guard,
                              qualify=qualify, after_optimizer_step=callback)
    return rows


def _release_parent(residency):
    # Original residency implementation is reused, with one current parent only.
    residency.resident.clear()
    residency.bytes = 0


def run_train(ctx, input_ref):
    from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    cfg, output, guard, device = ctx["cfg"], ctx["output"], ctx["guard"], ctx["device"]
    reader = PhaseReader(ctx["reader"], "train", ctx["bindings"])
    cache = TransportCache(cfg, guard, ctx["before_transport_build"])
    checkpoints, history, steps, qualifications = {}, [], {a: 0 for a in ARMS}, {}
    def record_step(arm):
        steps[arm] += 1
    try:
        banks, parents, keys, observation_seal = load_banks(ctx, reader, TRAIN, cache)
        require(set(cache.proofs) == set(TRAIN), "All32 observed recipes must precede TRAIN truth")
        reader.truth_gate = True
        targets, target_bytes, helpers = {}, 0, fit.metric_helpers_v1()
        for pid in TRAIN:
            guard()
            target = FullPanelExternalTargetV1(parents[pid], ctx["bindings"][pid], "TRAIN_loss", reader)
            mass = h038.native_hold_train_mass_v1(parents[pid], banks[pid], target, helpers)
            target_bytes += int(mass.nbytes)
            require(target_bytes <= cfg["CPU_target_cap_bytes"], "External TRAIN mass cap exceeded")
            targets[pid] = torch.from_numpy(mass)
            del target, mass
        warm = shared.load_warmstart(reader)
        family, optimizers, initialization, initial_hashes = build_family_v1(warm, device, guard)
        del warm
        residency = h038.FullParentResidencyV1(banks, targets, device, cfg["GPU_residency_cap_bytes"])
        output.save("initialization.json", dict(hypothesis_id="H043", initialization=initialization,
            initial_state_sha256=initial_hashes, observation_seal=observation_seal,
            external_TRAIN_mass_bytes=target_bytes, independent_model_and_optimizer_states=True,
            paired_sequential_shared_parent=True, optimizer_moments_and_step_reset=True,
            VAL_influences_optimization=False, conditioned_on_H040_VAL=False))
        for at in range(1, 31):
            order = list(TRAIN)
            random.Random(101 + at).shuffle(order)
            sums, started = {a: 0. for a in ARMS}, time.monotonic()
            for number, pid in enumerate(order, 1):
                guard()
                transport_cpu = cache.get(pid, banks[pid], keys[pid])
                require(transport_cpu.tensor_bytes() <= cfg["GPU_transport_cap_bytes"],
                        "Complete GPU transport exceeds root cap")
                bank, mass = residency.get(pid)
                transport = transport_cpu.to(device)
                first = at == 1 and number == 1
                rows = train_parent_pair_v1(family, optimizers, bank, mass, transport, guard,
                    qualify=first, after_optimizer_step=record_step)
                for arm in ARMS:
                    sums[arm] += rows[arm]["loss"] / 32
                    if first:
                        qualifications[arm] = dict(rows[arm], parent_id=pid, pass_index=1,
                            optimizer_step=1, part_of_continuous_960_update_training=True,
                            no_extra_qualification_updates=True,
                            native_holds=int(bank.sample_index.shape[0]),
                            native_nodes=int(bank.parent.numel()),
                            transport_tensor_bytes=transport_cpu.tensor_bytes())
                if first:
                    output.save("first_real_parent_step_qualification.json", dict(
                        hypothesis_id="H043", arms=qualifications, continuous_training=True,
                        no_extra_updates=True, root_guard_checked_after_each_step=True))
                del transport, transport_cpu, bank, mass, rows
                _release_parent(residency)
                if number % 8 == 0:
                    print(json.dumps(dict(stage="H043_paired_TRAIN_progress", pass_index=at,
                        complete_parents=number, steps=steps.copy(),
                        elapsed_seconds=time.monotonic() - started)), flush=True)
            require(steps == {a: 32 * at for a in ARMS}, "Exactly32 parent updates per arm/pass required")
            row = dict(pass_index=at, steps=steps.copy(), mean_TRAIN_NLL=sums,
                       parent_order=order, full_parent_updates_per_arm=32,
                       seconds=time.monotonic() - started)
            history.append(row)
            output.save("TRAIN_pass_%02d.json" % at, row)
            if at in PASSES:
                groups = {}
                for arm in ARMS:
                    guard()
                    groups[arm] = output.save("%s.pass_%02d.pt" % (arm, at), dict(
                        schema="H043_transport_checkpoint_v1", hypothesis_id="H043", phase="train", arm=arm,
                        model_state={k: v.detach().cpu() for k, v in family[arm].state_dict().items()},
                        optimizer_state=optimizers[arm].state_dict(), pass_index=at,
                        additional_optimizer_steps=steps[arm], seed=101, model_kwargs=MODEL,
                        parameter_count=30788, input_ref=input_ref, config_ref=ctx["config_ref"],
                        observation_seal=observation_seal, warmstart_checkpoint_ref=WARM_CHECKPOINT,
                        initial_state_sha256=initial_hashes[arm], teacher_used=False,
                        optimizer_state_reset=True, scientific_refs=list(SCIENTIFIC_REFS),
                        conditioned_on_H040_VAL=False), "torch")
                checkpoints[str(at)] = groups
                output.save("pass_%02d_checkpoints.json" % at, groups)
        require(steps == {a: 960 for a in ARMS} and set(checkpoints) == {str(p) for p in PASSES},
                "Complete paired30-pass/960-update/six-checkpoint schedule required")
        result = dict(schema="H043_transport_train_completion_v1",
            status="completed_fixed_30_pass_paired_TRAIN", hypothesis_id="H043",
            input_ref=input_ref, config_ref=ctx["config_ref"], arms=list(ARMS),
            checkpoint_groups=checkpoints, steps=steps, history=history,
            initialization=initialization, initial_state_sha256=initial_hashes,
            warmstart_checkpoint_ref=WARM_CHECKPOINT, observations_seal=observation_seal,
            first_step_qualification=qualifications, VAL_truth_read=False, outer_truth_read=False,
            selection_deferred=True, conditioned_on_H040_VAL=False,
            scientific_refs=list(SCIENTIFIC_REFS), transport_arrays_saved_to_disk=False,
            exact_bitsets_saved=False, transport_builds=cache.builds, transport_cache_hits=cache.hits,
            transport_build_seconds=cache.build_seconds, peak_cached_transport_tensor_bytes=cache.peak_runtime_bytes)
        output.save("completed.json", result)
        return result
    except Exception as error:
        output.save("failed.json", dict(schema="H043_transport_failure_v1", hypothesis_id="H043",
            phase="train", status="failed_partial_preserved", error_type=type(error).__name__,
            error=str(error), logical_steps=steps.copy(), completed_checkpoint_groups=checkpoints,
            first_step_qualification=qualifications, automatic_replay=False,
            VAL_truth_read=False, outer_truth_read=False))
        raise


class TransportRAMMatrix(shared.RAMMatrix):
    """H043 seal metadata and arm set; unchanged original current-grid semantics."""
    def seal(self, at, arm, pid, value, checkpoint):
        key = (at, arm, pid)
        require(at in PASSES and arm in ARMS and pid in VAL and key not in self.values,
                "Unique registered frozen prediction cell required")
        value = np.array(value, dtype=np.float32, copy=True, order="C")
        bank, parent = self.banks[pid], self.parents[pid]
        require(value.shape == (len(bank.spans), 686, 4) and np.isfinite(value).all()
                and np.all(value >= 0) and np.allclose(value.sum(-1), 1, rtol=1e-5, atol=1e-6),
                "Complete native-hold FP32 probabilities required")
        require(self.bytes + value.nbytes <= self.cfg["RAM_prediction_cap_bytes"], "Frozen RAM cap exceeded")
        value.setflags(write=False)
        geometry = {k: shared.array_identity(parent[k]) for k in
            ("positions", "edges", "retained_indexes", "query_slots", "raw_query_nodes", "cohorts")}
        geometry["readout_hold"] = shared.array_identity(bank.readout_hold.numpy())
        packet = dict(schema="H043_native_hold_RAM_seal_v1", hypothesis_id="H043",
            pass_index=at, arm=arm, parent_id=pid, checkpoint_ref=checkpoint,
            probability=shared.array_identity(value), geometry=geometry,
            scientific_refs=list(SCIENTIFIC_REFS), prediction_array_durable=False,
            targets_opened_before_seal=False)
        ref = self.output.save("%s.pass_%02d.%s.RAM_seal.json" % (arm, at, pid), packet)
        self.bytes += int(value.nbytes)
        self.values[key], self.records[key], self.packets[key] = value, ref, packet
        self.lookup[ref["path"]] = key

    def verify(self):
        require(set(self.values) == {(at, a, p) for at in PASSES for a in ARMS for p in VAL},
                "All6 passes x2 arms x8 VAL predictions must precede truth")
        for key, value in self.values.items():
            require(not value.flags.writeable and shared.array_identity(value) == self.packets[key]["probability"]
                    and self.reader.decode_json(self.records[key]) == self.packets[key],
                    "Frozen native probabilities or JSON seal changed")
        self.output.save("all96_VAL_RAM_sealed_before_truth.json", dict(hypothesis_id="H043",
            checkpoint_passes=list(PASSES), arms=list(ARMS), VAL=list(VAL),
            complete_native_RAM_prediction_count=96, native_probability_bytes=self.bytes,
            dense_prediction_arrays_saved=False, all96_sealed_before_any_VAL_truth=True,
            seals=[self.records[k] for k in sorted(self.records)]))
        self.verified = True


def load_checkpoints_v1(reader, train, cfg):
    require(train.get("schema") == "H043_transport_train_completion_v1"
        and train.get("status") == "completed_fixed_30_pass_paired_TRAIN"
        and train.get("hypothesis_id") == "H043" and train.get("arms") == list(ARMS)
        and train.get("steps") == {a: 960 for a in ARMS}
        and train.get("outer_truth_read") is False and train.get("VAL_truth_read") is False
        and train.get("conditioned_on_H040_VAL") is False
        and train.get("scientific_refs") == list(SCIENTIFIC_REFS)
        and len(set(train["initial_state_sha256"].values())) == 1,
        "Actual complete matched paired H043 training required")
    train_cfg = reader.decode_json(train["config_ref"])
    validate_protocol(train_cfg)
    validate_protocol(cfg)
    require({k: v for k, v in train_cfg.items() if k not in RESOURCE_CAPS}
            == {k: v for k, v in cfg.items() if k not in RESOURCE_CAPS},
            "Frozen science/threshold must persist; only root resource caps may change")
    require(set(train["checkpoint_groups"]) == {str(p) for p in PASSES},
            "Exactly six prescribed checkpoint groups required")
    states, checkpoints = {}, {}
    for at in PASSES:
        require(set(train["checkpoint_groups"][str(at)]) == set(ARMS), "Exactly two arms/checkpoint required")
        for arm in ARMS:
            ref = train["checkpoint_groups"][str(at)][arm]
            cp = torch.load(io.BytesIO(reader.read(ref)), map_location="cpu", weights_only=False)
            require(cp.get("schema") == "H043_transport_checkpoint_v1"
                and cp.get("hypothesis_id") == "H043" and cp.get("phase") == "train" and cp.get("arm") == arm
                and cp.get("pass_index") == at and cp.get("additional_optimizer_steps") == 32 * at
                and cp.get("seed") == 101 and cp.get("parameter_count") == 30788 and cp.get("model_kwargs") == MODEL
                and cp.get("teacher_used") is False and cp.get("optimizer_state_reset") is True
                and cp.get("scientific_refs") == list(SCIENTIFIC_REFS)
                and cp.get("warmstart_checkpoint_ref") == WARM_CHECKPOINT
                and cp.get("initial_state_sha256") == train["initial_state_sha256"][arm]
                and cp.get("input_ref") == train["input_ref"] and cp.get("config_ref") == train["config_ref"]
                and cp.get("observation_seal") == train["observations_seal"]
                and cp.get("conditioned_on_H040_VAL") is False,
                "Exact actual checkpoint provenance required")
            require(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                        and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),
                    "Finite raw FP32 learned checkpoint states required")
            states[at, arm], checkpoints[at, arm] = cp["model_state"], ref
            del cp
    return states, checkpoints


def paired_parent_differences_v1(selected, threshold, *, group="primary_present_day_182"):
    cells = {a: {c["parent_id"]: c["groups"][group] for c in selected[a]["cells"]} for a in ARMS}
    require(all(set(cells[a]) == set(VAL) for a in ARMS), "Eight independent paired VAL parents required")
    rows = []
    for pid in VAL:
        true, own = cells["true_partner"][pid], cells["own_state"][pid]
        finite_nll = not true.get("NLL_is_infinite", bool(true.get("infinite_nll_query_bp", 0)))
        finite_nll = finite_nll and not own.get("NLL_is_infinite", bool(own.get("infinite_nll_query_bp", 0)))
        row = dict(parent_id=pid, accuracy_delta_true_minus_own=true["accuracy"] - own["accuracy"],
            NLL_improvement_own_minus_true=(own["nll"] - true["nll"]) if finite_nll else None,
            either_NLL_infinite=not finite_nll)
        for metric in ("brier", "calibration_ece"):
            if metric in true and metric in own:
                row[metric + "_improvement_own_minus_true"] = own[metric] - true[metric]
        rows.append(row)
    effects = [r["accuracy_delta_true_minus_own"] for r in rows]
    mean = sum(effects) / len(effects)
    result = dict(population=group, statistical_unit="independent_ARG_parent",
        paired_parent_count=8, cells=rows, mean_paired_accuracy_delta_true_minus_own=mean,
        sample_SD_paired_accuracy_delta=math.sqrt(sum((v - mean)**2 for v in effects) / 7),
        accuracy_effect_sign_counts=dict(positive=sum(v > 0 for v in effects),
            zero=sum(v == 0 for v in effects), negative=sum(v < 0 for v in effects)),
        checkpoint_selection_per_arm=SELECTION, leaf_or_window_pseudoreplication=False,
        threshold_is_significance_or_novelty_claim=False)
    result["metric_effect_summaries"] = {}
    effect_names = [name for name in rows[0] if name.endswith(("_minus_own", "_minus_true"))]
    for name in effect_names:
        defined = [r[name] for r in rows if r[name] is not None]
        count = len(defined)
        average = sum(defined) / count if count else None
        result["metric_effect_summaries"][name] = dict(defined_parents=count, undefined_parents=8-count,
            mean=average, sample_SD=(math.sqrt(sum((v-average)**2 for v in defined) / (count-1))
                if count > 1 else None), positive=sum(v > 0 for v in defined),
            zero=sum(v == 0 for v in defined), negative=sum(v < 0 for v in defined))
    if group == "primary_present_day_182":
        result.update(practical_accuracy_gain_threshold=threshold,
                      reaches_practical_accuracy_threshold=mean >= threshold)
    return result


def run_evaluate(ctx, input_ref):
    from .anonymous_exact_clade_transport_v1 import AnonymousExactCladeTransportGNNV1
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg, output, guard, device = ctx["cfg"], ctx["output"], ctx["guard"], ctx["device"]
    reader = PhaseReader(ctx["reader"], "evaluate", ctx["bindings"])
    train = reader.decode_json(ctx["train_completion_ref"])
    states, checkpoints = load_checkpoints_v1(reader, train, cfg)
    cache = TransportCache(cfg, guard, ctx["before_transport_build"])
    banks, parents, keys, observation_seal = load_banks(ctx, reader, VAL, cache)
    ram = TransportRAMMatrix(output, reader, parents, banks, cfg)
    reader.ram = ram
    residency = h038.FullParentResidencyV1(banks, {}, device, cfg["GPU_residency_cap_bytes"])
    models = {a: AnonymousExactCladeTransportGNNV1(**MODEL, arm=a, event_block_size=32768,
        checkpoint_event_blocks=True).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        for a in ARMS}
    for pid in VAL:
        guard()
        transport_cpu = cache.get(pid, banks[pid], keys[pid])
        require(transport_cpu.tensor_bytes() <= cfg["GPU_transport_cap_bytes"], "Complete GPU payload cap exceeded")
        bank, _ = residency.get(pid)
        transport = transport_cpu.to(device)
        for at in PASSES:
            for arm in ARMS:
                guard()
                models[arm].load_state_dict(states[at, arm], strict=True)
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    probability = models[arm].forward_train_holds(bank, transport=transport, control="aligned").softmax(-1)
                ram.seal(at, arm, pid, probability.cpu().numpy(), checkpoints[at, arm])
                del probability
        del bank, transport, transport_cpu
        _release_parent(residency)
    del models, residency, states
    ram.verify()
    reader.truth_gate = True
    selected, history, full = {}, {}, {}
    helpers = fit.metric_helpers_v1()
    for arm in ARMS:
        validation = FullPanelPrimaryValidationV1(parents, ctx["bindings"], reader, helpers, cfg)
        rows = []
        for at in PASSES:
            guard()
            cells, means = validation.score([(p, ram.records[at, arm, p]) for p in VAL])
            row = dict(pass_index=at, cells=cells, equal_parent_metrics=means,
                       checkpoint=checkpoints[at, arm], selection_rule=SELECTION)
            rows.append(row)
            output.save("%s.pass_%02d.primary_metrics.json" % (arm, at), row)
        best = min(rows, key=lambda row: h038.selection_key_v1(row["equal_parent_metrics"], row["pass_index"]))
        selected[arm], history[arm] = best, rows
        cells, means = validation.full_final([(p, ram.records[best["pass_index"], arm, p]) for p in VAL])
        full[arm] = dict(cells=cells, equal_parent_metrics=means, selection_revised=False)
        output.save(arm + ".selected_full_diagnostics.json", full[arm])
        del validation
    paired = paired_parent_differences_v1(selected, cfg["practical_accuracy_gain_threshold"])
    populations = set(full[ARMS[0]]["cells"][0]["groups"])
    require(all(set(c["groups"]) == populations for a in ARMS for c in full[a]["cells"]),
            "Same complete selected diagnostic populations required")
    all_population_effects = {group: paired_parent_differences_v1(
        full, cfg["practical_accuracy_gain_threshold"], group=group) for group in sorted(populations)}
    result = dict(schema="H043_transport_VAL_completion_v1", status="completed_original_H038_selection",
        hypothesis_id="H043", input_ref=input_ref, config_ref=ctx["config_ref"],
        train_completion_ref=ctx["train_completion_ref"], observations_seal=observation_seal,
        selection_rule=SELECTION, selected=selected, history=history, selected_full_diagnostics=full,
        paired_selected_parent_differences=paired,
        all_selected_population_paired_effects=all_population_effects, optimization_performed=False,
        all96_RAM_seals_before_any_VAL_truth=True, RAM_native_probability_bytes=ram.bytes,
        dense_prediction_arrays_saved=False, outer_truth_read=False, conditioned_on_H040_VAL=False)
    output.save("completed.json", result)
    return result


def run_context(ctx, input_ref):
    """Scientific entry point only. Root retains all admission/lease authority."""
    require(ctx.get("root_authorizer") == "root_agent" and ctx.get("admitted") is True
        and ctx.get("phase") in ("train", "evaluate") and ctx.get("arm") == "joint_pair"
        and callable(ctx.get("guard")) and callable(ctx.get("before_transport_build")),
        "Independently admitted root paired context required")
    validate_protocol(ctx["cfg"])
    for ref in SCIENTIFIC_REFS:
        shared.bound(ref)
    require(ctx["reader"].root.resolve() == shared.ROOT and ctx["device"].type == "cuda",
            "Original project reader and root-qualified leased CUDA device required")
    require(ctx["reader"].decode_json(ctx["config_ref"]) == ctx["cfg"],
            "Hash-bound frozen root config required")
    ids = TRAIN if ctx["phase"] == "train" else VAL
    require(set(ctx["bindings"]) == set(ids), "Exact phase targets only, no outer bindings")
    require(set(ctx["entry_refs"]) == set(TRAIN + VAL), "Original40 observation entries only")
    for pid, row in ctx["bindings"].items():
        require(row["split"] == ("train" if pid in TRAIN else "val")
            and row["truth_npz"]["path"].endswith("/" + pid + "/truth/full_truth_intervals.npz"),
            "Original same-parent phase target binding required")
    ctx["guard"]()
    science()
    return run_train(ctx, input_ref) if ctx["phase"] == "train" else run_evaluate(ctx, input_ref)
