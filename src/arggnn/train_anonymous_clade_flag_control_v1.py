"""Prospectively fixed exact-clade-flag TRAIN and RAM-sealed VAL source.

No admission or execution by default. Root supplies a distinct hash-bound
context factory and owns the current writer, reader, lease and resource guard.
The frozen H040 worker, context and guards are imported/read only, never edited.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import time

from . import H040_latent_clade_execution_v1 as shared

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/train_anonymous_clade_flag_control_v1.py"
HERE = "research/reviews/H040_clade_flag_execution_source_v1"
CONTROL_ID = "H040_exact_clade_flag_control_v1"
ARM = "exact_clade_flags"
TRAIN, VAL, PASSES = shared.TRAIN, shared.VAL, shared.PASSES
MODEL, SELECTION = dict(shared.MODEL), shared.SELECTION
WARM_COMPLETION, WARM_CHECKPOINT = shared.WARM_COMPLETION, shared.WARM_CHECKPOINT
SCIENTIFIC_REFS = (
    dict(path="src/arggnn/anonymous_clade_flag_control_v1.py",
         sha256="d88a4639a53e63e2bff91e20d98e2af1f5cf8abc71e5fde2e02ac35e831eb8ce"),
    dict(path=shared.SOURCE,
         sha256="28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d"),
    *shared.MECHANISM_REFS,
)
METRICS = dict(
    aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble",
    boundary_tolerance_bp=1000., calibration_bins=15,
    normalization_length_bp=2000000, posthoc_renormalization=False,
    primary="present_day182", probability_floor=False, purity_filter=False,
    secondary=["all686", "ancient504", "Bronze_Age162", "BAA14", "Neo302", "Yam26"],
    short_tract_cutoff_bp=10000.,
    support="same_ARGMix_original100bp_center_bins_clipped_to_coverage_no_missing_bins_no_tail_extrapolation",
)
require, bound, metadata = shared.require, shared.bound, shared.metadata
PhaseReader = shared.PhaseReader


def science():
    global np, torch, h038, fit
    shared.science()
    np, torch, h038, fit = shared.np, shared.torch, shared.h038, shared.fit


def validate_protocol(cfg):
    require(
        cfg.get("schema") == "anonymous_clade_flag_execution_config_v1"
        and cfg.get("control_id") == CONTROL_ID and cfg.get("arm") == ARM
        and cfg.get("arms") == [ARM]
        and isinstance(cfg.get("hypothesis_id"), str) and cfg["hypothesis_id"]
        and cfg["hypothesis_id"] != "ROOT_ASSIGN_BEFORE_ADMISSION"
        and cfg.get("linked_hypothesis_ids") == ["H038", "H040"]
        and cfg.get("seed") == 101 and cfg.get("dtype") == "float32"
        and cfg.get("passes") == 30 and cfg.get("updates_per_pass") == 32
        and cfg.get("updates_per_arm") == 960
        and cfg.get("checkpoint_passes") == list(PASSES)
        and cfg.get("TRAIN_parent_ids") == list(TRAIN)
        and cfg.get("VAL_parent_ids") == list(VAL)
        and cfg.get("model") == MODEL and cfg.get("parameters_per_arm") == 30788
        and cfg.get("base_parameters") == 29764 and cfg.get("added_parameters") == 1024
        and cfg.get("learning_rate") == .001 and cfg.get("weight_decay") == .0001
        and cfg.get("AdamW_betas") == [.9, .999] and cfg.get("AdamW_epsilon") == 1e-8
        and cfg.get("gradient_clip_norm") == 1. and cfg.get("early_stopping") is False
        and cfg.get("VAL_influences_optimization") is False
        and cfg.get("conditioned_on_H040_VAL") is False
        and cfg.get("selection") == SELECTION and cfg.get("metrics") == METRICS
        and cfg.get("dense_bitset_sidecars") is False
        and cfg.get("flag_arrays_saved_to_disk") is False
        and cfg.get("dense_prediction_arrays_saved") is False
        and cfg.get("warmstart_completion_ref") == WARM_COMPLETION
        and cfg.get("warmstart_checkpoint_ref") == WARM_CHECKPOINT
        and cfg.get("scientific_refs") == list(SCIENTIFIC_REFS),
        "Exact prospective flag-control 30-pass protocol required",
    )
    for name in (
        "CPU_bank_cap_bytes", "CPU_target_cap_bytes", "flag_cache_cap_bytes",
        "flag_preparation_cap_bytes", "GPU_residency_cap_bytes",
        "GPU_flag_cap_bytes", "RAM_prediction_cap_bytes",
    ):
        require(type(cfg.get(name)) is int and cfg[name] > 0,
                "Positive root-reviewed cap: " + name)


def flag_preparation_plan(native_nodes, haplotypes):
    """Planning allowance, not a Python allocator/RSS proof. Root guards RSS."""
    require(type(native_nodes) is int and native_nodes > 0
            and type(haplotypes) is int and haplotypes >= 3, "Observed dimensions required")
    return dict(
        output_uint8_bytes=2 * native_nodes,
        child_link_int32_bytes=8 * native_nodes,
        # Deliberately uses all N, though exact masks exist for two trees only.
        Python_container_planning_allowance_bytes=2048 * native_nodes + 128 * haplotypes,
        total_planning_allowance_bytes=2058 * native_nodes + 128 * haplotypes,
        masks_retained_across_parents=False, RSS_guard_is_authoritative=True,
    )


class FlagCache:
    """Bounded RAM-only LRU and sealed deterministic recipes before target access."""
    def __init__(self, cfg, guard):
        self.cfg, self.guard = cfg, guard
        self.values, self.proofs = OrderedDict(), {}
        self.bytes, self.builds, self.hits = 0, 0, 0
        self.build_seconds, self.peak_runtime_bytes = 0., 0

    def get(self, pid, bank, keys):
        self.guard()
        if pid in self.values:
            self.hits += 1
            value = self.values[pid]
            require(shared.array_identity(value.node_flags.numpy())
                    == self.proofs[pid]["runtime_array_identity"],
                    "Cached observed node flags changed")
            self.values.move_to_end(pid)
            return value
        from .anonymous_clade_flag_control_v1 import build_clade_flags_v1
        n, h = int(bank.parent.numel()), int(bank.sample_index.shape[1])
        limits = flag_preparation_plan(n, h)
        require(limits["total_planning_allowance_bytes"]
                <= self.cfg["flag_preparation_cap_bytes"],
                "Flag preparation planning allowance exceeds root cap")
        start = time.monotonic()
        flags, meta = build_clade_flags_v1(bank)
        proof = dict(
            metadata=meta, runtime_array_identity=shared.array_identity(flags.node_flags.numpy()),
            stable_sample_keys_sha256=hashlib.sha256(
                json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
            preparation_plan=limits,
        )
        if pid in self.proofs:
            require(self.proofs[pid] == proof,
                    "Rebuilt exact flag recipe/content changed after target access")
        else:
            self.proofs[pid] = proof
        size = flags.tensor_bytes()
        require(size <= self.cfg["flag_cache_cap_bytes"],
                "One complete flag payload exceeds RAM cache cap")
        while self.values and self.bytes + size > self.cfg["flag_cache_cap_bytes"]:
            _, old = self.values.popitem(last=False)
            self.bytes -= old.tensor_bytes()
        self.values[pid] = flags
        self.bytes += size
        self.builds += 1
        self.build_seconds += time.monotonic() - start
        self.peak_runtime_bytes = max(self.peak_runtime_bytes, self.bytes)
        self.guard()
        return flags


def load_banks(ctx, reader, ids, cache):
    from .mutation_routed_full_panel_fast_reader_v1 import (
        read_full_cache_index_v1, read_full_cache_parent_v1,
    )
    index = read_full_cache_index_v1(
        ctx["observation_cache_ref"], reader, registry=ctx["registry"],
        entry_refs=ctx["entry_refs"], source_refs=ctx["observation_source_refs"],
    )
    banks, parents, keys, total = {}, {}, {}, 0
    for pid in ids:
        ctx["guard"]()
        item = read_full_cache_parent_v1(pid, index, reader)
        bank, parent = item["model_inputs"], item["scoring"]
        require(parent["split"] == ("train" if pid in TRAIN else "val"),
                "Original observed split differs")
        total += h038.tensor_bytes_v1(bank)
        require(total <= ctx["cfg"]["CPU_bank_cap_bytes"],
                "Complete CPU observation banks exceed root cap")
        banks[pid], parents[pid] = bank, parent
        keys[pid] = shared.stable_keys(bank, ctx["entry_refs"][pid], reader)
        cache.get(pid, bank, keys[pid])
        print(json.dumps(dict(
            stage="exact_clade_flags_observed_recipe", parent_id=pid,
            native_holds=int(bank.sample_index.shape[0]), native_nodes=int(bank.parent.numel()),
            node_flag_bytes=cache.proofs[pid]["metadata"]["node_flag_bytes"],
            CPU_bank_bytes=total,
        )), flush=True)
        del item
    seal = ctx["output"].save(
        "all_%d_observed_RAM_flag_recipes.json" % len(ids),
        dict(
            schema="anonymous_clade_flag_observation_barrier_v1",
            control_id=CONTROL_ID, hypothesis_id=ctx["cfg"]["hypothesis_id"], arm=ARM,
            parent_ids=list(ids), observation_cache_ref=ctx["observation_cache_ref"],
            entry_refs={p: ctx["entry_refs"][p] for p in ids}, recipes=cache.proofs,
            all_observations_precede_phase_truth=True, exact_bitsets_saved=False,
            flag_arrays_saved_to_disk=False, CPU_bank_tensor_bytes=total,
            scientific_refs=list(SCIENTIFIC_REFS),
        ),
    )
    return banks, parents, keys, seal


def train_one(model, optimizer, bank, mass, flags, guard):
    from .biological_gst_control_v1 import exact_span_loss
    guard()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    with torch.autocast(device_type=bank.parent.device.type, enabled=False):
        logits = model.forward_train_holds(bank, flags=flags)
        loss = exact_span_loss(logits, mass)
    require(bool(torch.isfinite(loss)), "Finite full-parent FP32 TRAIN loss required")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "Finite updated parameters required")
    row = dict(loss=float(loss.detach()), gradient_norm=float(norm),
               seconds=time.monotonic() - start)
    del logits, loss, norm
    optimizer.zero_grad(set_to_none=True)
    guard()
    return row


def run_train(ctx, input_ref):
    from .anonymous_clade_flag_control_v1 import (
        AnonymousCladeFlagControlGNNV1, load_h038_flag_warmstart_v1,
    )
    from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    cfg, output, guard, device = ctx["cfg"], ctx["output"], ctx["guard"], ctx["device"]
    reader = PhaseReader(ctx["reader"], "train", ctx["bindings"])
    cache = FlagCache(cfg, guard)
    checkpoints, history, steps = {}, [], 0
    try:
        banks, parents, keys, observation_seal = load_banks(ctx, reader, TRAIN, cache)
        require(set(cache.proofs) == set(TRAIN),
                "All32 observation/flag recipes must precede every TRAIN target")
        reader.truth_gate = True
        targets, target_bytes = {}, 0
        helpers = fit.metric_helpers_v1()
        for pid in TRAIN:
            guard()
            target = FullPanelExternalTargetV1(
                parents[pid], ctx["bindings"][pid], "TRAIN_loss", reader,
            )
            mass = h038.native_hold_train_mass_v1(
                parents[pid], banks[pid], target, helpers,
            )
            target_bytes += int(mass.nbytes)
            require(target_bytes <= cfg["CPU_target_cap_bytes"],
                    "External native-hold TRAIN mass exceeds root cap")
            targets[pid] = torch.from_numpy(mass)
            del target, mass

        warm = shared.load_warmstart(reader)
        guard()
        torch.manual_seed(101)
        model = AnonymousCladeFlagControlGNNV1(**MODEL)
        initialization = load_h038_flag_warmstart_v1(model, warm)
        initial_hash = shared.state_identity(model.state_dict())
        del warm
        require(sum(p.numel() for p in model.parameters()) == 30788,
                "Exact matched parameter count required")
        model = model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=.001, weight_decay=.0001,
            betas=(.9, .999), eps=1e-8,
        )
        residency = h038.FullParentResidencyV1(
            banks, targets, device, cfg["GPU_residency_cap_bytes"],
        )
        output.save("initialization.json", dict(
            schema="anonymous_clade_flag_initialization_v1",
            control_id=CONTROL_ID, hypothesis_id=cfg["hypothesis_id"], arm=ARM,
            initialization=initialization, initial_state_sha256=initial_hash,
            observation_seal=observation_seal, external_TRAIN_mass_bytes=target_bytes,
            checkpoint_passes=list(PASSES), flag_arrays_saved_to_disk=False,
            same_H040_CPU_seed_projection_order_and_state_layout=True,
            VAL_influences_optimization=False, conditioned_on_H040_VAL=False,
            optimizer_moments_and_step_reset=True,
        ))
        for at in range(1, 31):
            order = list(TRAIN)
            random.Random(101 + at).shuffle(order)
            mean_loss, started = 0., time.monotonic()
            for number, pid in enumerate(order, 1):
                guard()
                flags_cpu = cache.get(pid, banks[pid], keys[pid])
                require(flags_cpu.tensor_bytes() <= cfg["GPU_flag_cap_bytes"],
                        "Complete GPU flag payload exceeds root cap")
                bank, mass = residency.get(pid)
                flags = flags_cpu.to(device)
                row = train_one(model, optimizer, bank, mass, flags, guard)
                steps += 1
                mean_loss += row["loss"] / 32
                del flags, bank, mass, flags_cpu
                if number % 8 == 0:
                    print(json.dumps(dict(
                        stage="exact_clade_flags_TRAIN_progress", pass_index=at,
                        complete_parents=number, steps={ARM: steps},
                        elapsed_seconds=time.monotonic() - started,
                    )), flush=True)
            require(steps == 32 * at, "One update per complete TRAIN parent required")
            row = dict(
                pass_index=at, steps={ARM: steps}, mean_TRAIN_NLL={ARM: mean_loss},
                full_parent_updates=32, seconds=time.monotonic() - started,
            )
            history.append(row)
            output.save("TRAIN_pass_%02d.json" % at, row)
            if at in PASSES:
                guard()
                ref = output.save("%s.pass_%02d.pt" % (ARM, at), dict(
                    schema="anonymous_clade_flag_checkpoint_v1",
                    control_id=CONTROL_ID, hypothesis_id=cfg["hypothesis_id"],
                    phase="train", arm=ARM, model_state={
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    }, optimizer_state=optimizer.state_dict(),
                    pass_index=at, additional_optimizer_steps=steps, seed=101,
                    model_kwargs=MODEL, parameter_count=30788,
                    input_ref=input_ref, config_ref=ctx["config_ref"],
                    observation_seal=observation_seal,
                    warmstart_checkpoint_ref=WARM_CHECKPOINT,
                    initial_state_sha256=initial_hash, teacher_used=False,
                    optimizer_state_reset=True, scientific_refs=list(SCIENTIFIC_REFS),
                    conditioned_on_H040_VAL=False,
                ), "torch")
                checkpoints[str(at)] = {ARM: ref}
                output.save("pass_%02d_checkpoints.json" % at, {ARM: ref})
        require(steps == 960 and set(checkpoints) == {str(p) for p in PASSES},
                "Complete30-pass/960-update/six-checkpoint schedule required")
        result = dict(
            schema="anonymous_clade_flag_train_completion_v1",
            status="completed_fixed_30_pass_flag_TRAIN",
            control_id=CONTROL_ID, hypothesis_id=cfg["hypothesis_id"], arm=ARM,
            input_ref=input_ref, config_ref=ctx["config_ref"],
            checkpoint_groups=checkpoints, steps={ARM: steps}, history=history,
            initialization=initialization, initial_state_sha256=initial_hash,
            warmstart_checkpoint_ref=WARM_CHECKPOINT, observations_seal=observation_seal,
            VAL_truth_read=False, outer_truth_read=False, selection_deferred=True,
            conditioned_on_H040_VAL=False, scientific_refs=list(SCIENTIFIC_REFS),
            flag_arrays_saved_to_disk=False, exact_bitsets_saved=False,
            flag_preparation_builds=cache.builds, flag_cache_hits=cache.hits,
            flag_preparation_seconds=cache.build_seconds,
            peak_cached_flag_tensor_bytes=cache.peak_runtime_bytes,
        )
        output.save("completed.json", result)
        return result
    except Exception as error:
        output.save("failed.json", dict(
            schema="anonymous_clade_flag_failure_v1", control_id=CONTROL_ID,
            hypothesis_id=cfg["hypothesis_id"], phase="train", arm=ARM,
            status="failed_partial_preserved", error_type=type(error).__name__,
            error=str(error), logical_steps={ARM: steps},
            completed_checkpoint_groups=checkpoints, automatic_replay=False,
            VAL_truth_read=False, outer_truth_read=False,
        ))
        raise


class FlagRAMMatrix:
    """All48 single-control native-hold predictions sealed before any VAL truth."""
    def __init__(self, output, reader, parents, banks, cfg):
        self.output, self.reader, self.parents, self.banks, self.cfg = (
            output, reader, parents, banks, cfg,
        )
        self.values, self.records, self.lookup, self.packets = {}, {}, {}, {}
        self.bytes, self.verified = 0, False

    def seal(self, at, pid, value, checkpoint):
        key = (at, pid)
        require(key not in self.values, "Frozen probability replacement forbidden")
        value = np.array(value, dtype=np.float32, copy=True, order="C")
        bank, parent = self.banks[pid], self.parents[pid]
        require(value.shape == (len(bank.spans), 686, 4)
                and np.isfinite(value).all() and np.all(value >= 0)
                and np.allclose(value.sum(-1), 1, rtol=1e-5, atol=1e-6),
                "Complete native-hold FP32 probabilities required")
        self.bytes += int(value.nbytes)
        require(self.bytes <= self.cfg["RAM_prediction_cap_bytes"],
                "Complete frozen RAM prediction matrix exceeds root cap")
        value.setflags(write=False)
        geometry = {k: shared.array_identity(parent[k]) for k in (
            "positions", "edges", "retained_indexes", "query_slots", "raw_query_nodes", "cohorts",
        )}
        geometry["readout_hold"] = shared.array_identity(bank.readout_hold.numpy())
        packet = dict(
            schema="anonymous_clade_flag_native_hold_RAM_seal_v1",
            control_id=CONTROL_ID, hypothesis_id=self.cfg["hypothesis_id"], arm=ARM,
            pass_index=at, parent_id=pid, checkpoint_ref=checkpoint,
            probability=shared.array_identity(value), geometry=geometry,
            scientific_refs=list(SCIENTIFIC_REFS),
            prediction_array_durable=False, targets_opened_before_seal=False,
        )
        ref = self.output.save("%s.pass_%02d.%s.RAM_seal.json" % (ARM, at, pid), packet)
        self.values[key], self.records[key], self.packets[key] = value, ref, packet
        self.lookup[ref["path"]] = key

    def verify(self):
        require(set(self.values) == {(at, p) for at in PASSES for p in VAL},
                "All6 passes x8 VAL parents required before truth")
        for key, value in self.values.items():
            require(not value.flags.writeable
                    and shared.array_identity(value) == self.packets[key]["probability"]
                    and self.reader.decode_json(self.records[key]) == self.packets[key],
                    "Frozen probability content/JSON seal differs")
        self.output.save("all48_flag_VAL_RAM_sealed_before_truth.json", dict(
            schema="anonymous_clade_flag_complete_VAL_barrier_v1",
            control_id=CONTROL_ID, hypothesis_id=self.cfg["hypothesis_id"], arm=ARM,
            checkpoint_passes=list(PASSES), VAL=list(VAL),
            complete_native_RAM_prediction_count=48, native_probability_bytes=self.bytes,
            dense_prediction_arrays_saved=False, all48_sealed_before_any_VAL_truth=True,
            seals=[self.records[k] for k in sorted(self.records)],
        ))
        self.verified = True

    def grid(self, record, parent):
        require(self.verified and record["path"] in self.lookup,
                "Verified complete prediction matrix required")
        key = self.lookup[record["path"]]
        require(record == self.records[key] and key[1] == parent["parent_id"],
                "Current same-parent RAM seal required")
        value = self.values[key]
        require(not value.flags.writeable
                and shared.array_identity(value) == self.packets[key]["probability"],
                "Immutable native probabilities changed")
        return value[self.banks[key[1]].readout_hold.numpy()]


def run_evaluate(ctx, input_ref):
    from .anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg, output, guard, device = ctx["cfg"], ctx["output"], ctx["guard"], ctx["device"]
    reader = PhaseReader(ctx["reader"], "evaluate", ctx["bindings"])
    train = reader.decode_json(ctx["train_completion_ref"])
    require(
        train.get("schema") == "anonymous_clade_flag_train_completion_v1"
        and train.get("status") == "completed_fixed_30_pass_flag_TRAIN"
        and train.get("control_id") == CONTROL_ID and train.get("arm") == ARM
        and train.get("hypothesis_id") == cfg["hypothesis_id"]
        and train.get("steps") == {ARM: 960}
        and train.get("outer_truth_read") is False
        and train.get("VAL_truth_read") is False
        and train.get("conditioned_on_H040_VAL") is False
        and train.get("scientific_refs") == list(SCIENTIFIC_REFS),
        "Actual complete prospectively specified flag TRAIN required",
    )
    states, checkpoints = {}, {}
    for at in PASSES:
        guard()
        ref = train["checkpoint_groups"][str(at)][ARM]
        cp = torch.load(io.BytesIO(reader.read(ref)), map_location="cpu", weights_only=False)
        require(
            cp.get("schema") == "anonymous_clade_flag_checkpoint_v1"
            and cp.get("control_id") == CONTROL_ID and cp.get("arm") == ARM
            and cp.get("hypothesis_id") == cfg["hypothesis_id"]
            and cp.get("pass_index") == at and cp.get("additional_optimizer_steps") == 32 * at
            and cp.get("parameter_count") == 30788 and cp.get("model_kwargs") == MODEL
            and cp.get("teacher_used") is False and cp.get("optimizer_state_reset") is True
            and cp.get("scientific_refs") == list(SCIENTIFIC_REFS)
            and cp.get("warmstart_checkpoint_ref") == WARM_CHECKPOINT
            and cp.get("initial_state_sha256") == train["initial_state_sha256"]
            and cp.get("input_ref") == train["input_ref"]
            and cp.get("config_ref") == train["config_ref"]
            and cp.get("conditioned_on_H040_VAL") is False,
            "Exact actual six-checkpoint flag provenance required",
        )
        require(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                    for v in cp["model_state"].values()), "Raw FP32 learned states required")
        states[at], checkpoints[at] = cp["model_state"], ref
        del cp
    cache = FlagCache(cfg, guard)
    banks, parents, keys, observation_seal = load_banks(ctx, reader, VAL, cache)
    ram = FlagRAMMatrix(output, reader, parents, banks, cfg)
    reader.ram = ram
    residency = h038.FullParentResidencyV1(
        banks, {}, device, cfg["GPU_residency_cap_bytes"],
    )
    model = AnonymousCladeFlagControlGNNV1(**MODEL).to(
        device=device, dtype=torch.float32,
    ).eval().requires_grad_(False)
    for pid in VAL:
        guard()
        flags_cpu = cache.get(pid, banks[pid], keys[pid])
        require(flags_cpu.tensor_bytes() <= cfg["GPU_flag_cap_bytes"],
                "Complete GPU flag payload exceeds root cap")
        bank, _ = residency.get(pid)
        flags = flags_cpu.to(device)
        for at in PASSES:
            guard()
            model.load_state_dict(states[at], strict=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                probabilities = model.forward_train_holds(bank, flags=flags).softmax(-1)
            ram.seal(at, pid, probabilities.cpu().numpy(), checkpoints[at])
            del probabilities
        del bank, flags, flags_cpu
    del model, residency, states
    ram.verify()
    reader.truth_gate = True
    validation = FullPanelPrimaryValidationV1(
        parents, ctx["bindings"], reader, fit.metric_helpers_v1(), cfg,
    )
    rows = []
    for at in PASSES:
        guard()
        seals = [(p, ram.records[at, p]) for p in VAL]
        cells, means = validation.score(seals)
        row = dict(pass_index=at, cells=cells, equal_parent_metrics=means,
                   checkpoint=checkpoints[at], selection_rule=SELECTION)
        rows.append(row)
        output.save("%s.pass_%02d.primary_metrics.json" % (ARM, at), row)
    best = min(rows, key=lambda row: h038.selection_key_v1(
        row["equal_parent_metrics"], row["pass_index"],
    ))
    chosen = [(p, ram.records[best["pass_index"], p]) for p in VAL]
    cells, means = validation.full_final(chosen)
    full = dict(cells=cells, equal_parent_metrics=means, selection_revised=False)
    output.save(ARM + ".selected_full_diagnostics.json", full)
    result = dict(
        schema="anonymous_clade_flag_VAL_completion_v1",
        status="completed_original_H038_selection_for_flag_control",
        control_id=CONTROL_ID, hypothesis_id=cfg["hypothesis_id"], arm=ARM,
        input_ref=input_ref, train_completion_ref=ctx["train_completion_ref"],
        observations_seal=observation_seal, selection_rule=SELECTION,
        selected={ARM: best}, history={ARM: rows}, selected_full_diagnostics={ARM: full},
        optimization_performed=False, all48_RAM_seals_before_any_VAL_truth=True,
        RAM_native_probability_bytes=ram.bytes, dense_prediction_arrays_saved=False,
        outer_truth_read=False, conditioned_on_H040_VAL=False,
    )
    output.save("completed.json", result)
    return result


def run_context(ctx, input_ref):
    require(ctx.get("root_authorizer") == "root_agent" and ctx.get("admitted") is True
            and ctx.get("phase") in ("train", "evaluate") and callable(ctx.get("guard")),
            "Distinct independently admitted root context required")
    validate_protocol(ctx["cfg"])
    for ref in SCIENTIFIC_REFS:
        bound(ref)
    require(ctx.get("arm") == ARM
            and ctx["reader"].root.resolve() == ROOT and ctx["device"].type == "cuda",
            "Distinct flag arm and existing root-qualified reader/leased CUDA device required")
    require(ctx["reader"].decode_json(ctx["config_ref"]) == ctx["cfg"],
            "Hash-bound root config must equal supplied protocol")
    ids = TRAIN if ctx["phase"] == "train" else VAL
    require(set(ctx["bindings"]) == set(ids), "Exact phase target keys, no outer bindings")
    require(set(ctx["entry_refs"]) == set(TRAIN + VAL),
            "Original32TRAIN/eightVAL observed entries required")
    for pid, row in ctx["bindings"].items():
        require(row["split"] == ("train" if pid in TRAIN else "val")
                and row["truth_npz"]["path"].endswith("/" + pid + "/truth/full_truth_intervals.npz"),
                "Original same-parent phase truth binding required")
    ctx["guard"]()
    science()
    return run_train(ctx, input_ref) if ctx["phase"] == "train" else run_evaluate(ctx, input_ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(
            status="source_only_not_executed", source=SOURCE, control_id=CONTROL_ID,
            arm=ARM, root_assigns_hypothesis_id=True, passes=30, updates_per_arm=960,
            checkpoint_passes=list(PASSES), parameters_per_arm=30788,
            train_parents=32, val_parents=8, future_VAL_RAM_prediction_count=48,
            exact_bitset_disk_bytes=0, flag_array_disk_bytes=0,
            root_context_required=True, conditioned_on_H040_VAL=False,
        ), indent=2))
        return 0
    require(args.input and args.expected_input_sha256,
            "Pinned active distinct root input capability required")
    record = dict(path=args.input, sha256=args.expected_input_sha256)
    spec = metadata(record)
    require(
        spec.get("schema") == "anonymous_clade_flag_root_execution_input_v1"
        and spec.get("control_id") == CONTROL_ID and spec.get("arm") == ARM
        and spec.get("root_authorizer") == "root_agent" and spec.get("status") == "active"
        and spec.get("phase") in ("train", "evaluate"),
        "Active distinct root flag-control capability required",
    )
    ref = spec["root_context_module_ref"]
    bound(ref)
    require(ref["path"].startswith("research/reviews/") and ref["path"].endswith(".py"),
            "Separately pinned root-owned execution factory required")
    module_spec = importlib.util.spec_from_file_location(
        "anonymous_clade_flag_root_context", ROOT / ref["path"],
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    ctx = module.build_context_v1(record, spec)
    require(ctx["phase"] == spec["phase"] and ctx["arm"] == spec["arm"]
            and ctx["cfg"]["hypothesis_id"] == spec["hypothesis_id"],
            "Root context phase/arm/hypothesis differs from active capability")
    run_context(ctx, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
