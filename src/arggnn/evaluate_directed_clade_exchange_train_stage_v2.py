"""Frozen H039 checkpoints: native-hold RAM seals before exact original VAL scoring."""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, os, resource, time
from pathlib import Path
from dataclasses import fields, replace

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/evaluate_directed_clade_exchange_train_stage_v2.py"
HERE = "research/reviews/H039_frozen_checkpoint_evaluation_v1"
ARMS = ("plain_continuation", "generic_unary", "directed")
VAL = tuple("ap_expanded_val_%03d" % i for i in range(8))
COUNTS = dict(plain_continuation=29764, generic_unary=29784, directed=29784)
MODEL = dict(hidden_dim=32, expected_haplotypes=952, site_chunk_size=256, checkpoint_sites=True)
SELECTION = "earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass"
CAP = 15000000
REPAIR = "research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"
MARKER = "src/arggnn/project_budget_v9_consumer.py"
LAUNCHER = "scripts/launch_reserved_gpu_authorized_external_share_v3.py"
ROLES = {"config", "runtime", "storage", "source_manifest", "phase_registration", "generation_config",
    "registration", "observation_cache", "target_bindings", "witness_sidecar_manifest",
    "train_input", "train_config", "train_phase_registration", "checkpoint_manifest"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_ref(record, suffix=None):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Exact path/SHA reference required")
    p, h = record["path"], record["sha256"]
    require(isinstance(p, str) and Path(p).as_posix() == p and not Path(p).is_absolute()
            and ".." not in Path(p).parts and isinstance(h, str) and len(h) == 64
            and all(c in "0123456789abcdef" for c in h), "Canonical project reference required")
    if suffix:
        require(p.endswith(suffix), "Required reference suffix: " + suffix)
    path = ROOT / p
    require(path.resolve() == path and path.is_file() and path.stat().st_nlink == 1,
            "Canonical singly linked project file required: " + p)
    return path


def bound(record):
    path = canonical_ref(record)
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == record["sha256"], "Bound bytes changed: " + record["path"])
    return payload


def metadata(record):
    canonical_ref(record, ".json")
    return json.loads(bound(record))


def validate_protocol(cfg):
    require(cfg.get("schema") == "H039_frozen_checkpoint_evaluation_config_v1"
        and cfg.get("status") == "active" and cfg.get("execution_guard_ready") is True
        and cfg.get("phase") == "frozen_checkpoint_evaluation" and cfg.get("hypothesis_id") == "H039"
        and cfg.get("arms") == list(ARMS) and cfg.get("seed") == 101 and cfg.get("dtype") == "float32"
        and cfg.get("evaluation_passes") in ([5], [5, 10]) and cfg.get("original_evaluation_passes") == [5, 10]
        and cfg.get("model") == MODEL and cfg.get("parameter_counts") == COUNTS
        and cfg.get("selection") == SELECTION and cfg.get("output_cap_bytes") == CAP
        and cfg.get("qualification_allowance_bytes") == 2000000
        and cfg.get("VAL_parent_ids") == list(VAL) and cfg.get("TRAIN_truth_authorized") is False
        and cfg.get("outer_truth_authorized") is False and cfg.get("optimization_authorized") is False
        and cfg.get("all24_RAM_seals_before_any_VAL_truth_per_pass") is True
        and cfg.get("dense_prediction_arrays_saved") is False
        and cfg.get("probability_representation") == "immutable_CPU_native_hold_FP32_with_exact_readout_hold"
        and cfg.get("budget_guard") == "journal_terminal_retired_scopes_credit_v1",
        "Exact active root frozen-checkpoint RAM-sealed evaluation protocol required")


def verify_source_manifest(record):
    manifest = metadata(record)
    require(manifest.get("schema") == "H039_frozen_checkpoint_evaluation_source_manifest_v1"
        and SOURCE in manifest["sources"] and len(manifest["required_consumer_source_paths"]) <= 40,
        "Reviewed actual inference/scoring source manifest required")
    require(set(manifest["required_consumer_source_paths"]).issubset(manifest["sources"]),
        "Consumer sources must belong to reviewed source map")
    for p, h in manifest["sources"].items():
        require(p.endswith(".py"), "Only Python source in execution source manifest")
        bound(dict(path=p, sha256=h))
    return manifest


def load_repair(cfg):
    helper_ref = cfg["helper_refs"]["storage_adapter"]
    require(helper_ref["path"] == REPAIR, "Only targeted marker-repair storage adapter")
    bound(helper_ref); metadata(cfg["marker_repair_authority_ref"])
    spec = importlib.util.spec_from_file_location("H039_frozen_eval_storage", ROOT / REPAIR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_repair_v1(cfg["marker_repair_authority_ref"])
    return module


class EvaluationReader:
    """Actual truth reads require the complete immutable RAM prediction matrix."""
    def __init__(self, base, truth_refs, checkpoint_refs):
        self.base = base
        self.root = base.root
        self.truth = {(r["path"], r["sha256"]) for r in truth_refs}
        self.checkpoints = {(r["path"], r["sha256"]) for r in checkpoint_refs}
        self.observed = set()
        self.ram = None
        self.allowed_passes = ()

    def register_observed(self, record, parent_id):
        require(parent_id in VAL and parent_id in record["path"] and "/truth/" not in record["path"],
            "Only registered eight VAL observation arrays")
        canonical_ref(record, ".npz")
        self.observed.add((record["path"], record["sha256"]))

    def read(self, record, **kwargs):
        key = (record["path"], record["sha256"])
        if "/truth/" in record["path"]:
            require(key in self.truth and self.ram is not None and self.allowed_passes,
                "VAL truth denied before every requested pass's all24 RAM seals")
            require(all(p in self.ram.verified_passes for p in self.allowed_passes),
                "Full matrix barrier required before first and every later VAL truth read")
        elif record["path"].endswith(".npz"):
            require(key in self.observed, "No TRAIN/outer/arbitrary array reads in frozen evaluation")
        elif record["path"].endswith(".pt"):
            require(key in self.checkpoints, "Only exact registered actual frozen checkpoints")
        return self.base.read(record, **kwargs)

    def decode_json(self, record):
        canonical_ref(record, ".json")
        return json.loads(self.read(record))


def inputs(record):
    spec = metadata(record)
    require(spec.get("schema") == "H039_frozen_checkpoint_evaluation_input_v1"
        and spec.get("status") == "active" and spec.get("root_authorizer") == "root_agent"
        and spec.get("hypothesis_id") == "H039" and spec.get("phase") == "frozen_checkpoint_evaluation"
        and spec.get("evaluation_authorized") is True and spec.get("VAL_truth_score_authorized") is True
        and spec.get("optimization_authorized") is False and spec.get("fit_authorized") is False
        and spec.get("TRAIN_truth_authorized") is False and spec.get("outer_truth_authorized") is False
        and spec.get("pyg_authorized") is True and spec.get("device") == "cuda" and spec.get("seed") == 101
        and spec.get("arms") == list(ARMS) and set(spec["refs"]) == ROLES,
        "Active root VAL-only frozen checkpoint evaluation capability required")
    refs = spec["refs"]
    cfg = metadata(refs["config"]); validate_protocol(cfg)
    require(cfg["helper_refs"]["storage_adapter"]["path"] == REPAIR, "Targeted actual storage adapter")
    require(spec["evaluation_passes"] == cfg["evaluation_passes"], "Requested fixed passes differ")
    manifest = verify_source_manifest(refs["source_manifest"])
    require(cfg["source_manifest_ref"] == refs["source_manifest"], "Reviewed source map differs")
    require(spec["evaluation_source_ref"] == dict(path=SOURCE,sha256=manifest["sources"][SOURCE]),
        "Seal must bind the actual evaluator source")
    from arggnn import census_tsinfer_fit_v1 as fit
    from arggnn.native_argformer_exec import validate_lease, apply_and_validate_title
    from arggnn.mutation_routed_full_panel_observations_v1 import validate_registry_v1
    lease = validate_lease(); apply_and_validate_title()
    fit._check_active_v1(spec); fit.require_logical_job_v1(lease, spec)
    require(fit.canonical_job_v1(spec["job_id"]), "Canonical evaluation job required")
    for r in refs.values():
        canonical_ref(r, ".json")
    generation = metadata(refs["generation_config"])
    registration = metadata(refs["registration"])
    roles = validate_registry_v1(generation["parent_registry"])
    require(tuple(roles["val"]) == VAL and registration["parent_registry"] == generation["parent_registry"]
        and registration["config_ref"] == refs["generation_config"], "Original eight VAL registry required")
    train = metadata(refs["train_input"]); train_cfg = metadata(refs["train_config"])
    train_phase = metadata(refs["train_phase_registration"])
    require(train["phase"] == "train_only" and train["hypothesis_id"] == "H039"
        and train["refs"]["config"] == refs["train_config"]
        and train["refs"]["phase_registration"] == refs["train_phase_registration"]
        and train["refs"]["generation_config"] == refs["generation_config"]
        and train["refs"]["registration"] == refs["registration"]
        and train["refs"]["observation_cache"] == refs["observation_cache"]
        and train["refs"]["witness_sidecar_manifest"] == refs["witness_sidecar_manifest"]
        and train_cfg["additional_updates_per_arm"] == 320 and train_cfg["checkpoint_passes"] == [5, 10]
        and train_cfg["early_stopping"] is False and train_cfg["VAL_influences_optimization"] is False
        and train_cfg["model"] == MODEL and train_cfg["parameter_counts"] == COUNTS
        and train_cfg["selection"] == SELECTION and train_phase["later_all24_seals_before_any_VAL_truth"] is True,
        "Actual fixed training input/config/phase lineage required")
    for p in manifest["scientific_lineage_source_paths"]:
        require(manifest["sources"][p] == train["source_sha256"][p], "Frozen trained inference/scorer source differs: " + p)
    plan = metadata(refs["checkpoint_manifest"])
    passes = cfg["evaluation_passes"]
    require(plan.get("schema") == "H039_frozen_checkpoint_set_v1"
        and plan["train_input_ref"] == refs["train_input"]
        and plan["train_config_ref"] == refs["train_config"]
        and plan["train_phase_registration_ref"] == refs["train_phase_registration"]
        and plan["evaluation_passes"] == passes and set(plan["checkpoint_groups"]) == {str(p) for p in passes},
        "Prospectively registered actual checkpoint set required")
    checkpoint_refs = []
    for p in passes:
        group = metadata(plan["group_receipt_refs"][str(p)])
        require(group["pass_index"] == p and group["steps"] == {a: 32*p for a in ARMS}
            and group["all3_actual_states_retained"] is True and group["VAL_predictions_executed"] is False
            and group["VAL_truth_read"] is False and group["checkpoints"] == plan["checkpoint_groups"][str(p)]
            and set(group["checkpoints"]) == set(ARMS), "Actual all3 frozen pass checkpoint receipt required")
        checkpoint_refs.extend(group["checkpoints"].values())
    bindings = metadata(refs["target_bindings"])
    require(set(bindings["parents"]) == set(VAL) and bindings.get("targets_opened") is False,
        "Exactly eight VAL target metadata bindings required")
    for p, b in bindings["parents"].items():
        require(b["split"] == "val" and b["truth_npz"]["path"] ==
            generation["output_root"] + "/" + p + "/truth/full_truth_intervals.npz",
            "Original same-parent VAL truth binding")
        provenance = metadata(b["generation_receipt"])
        require(provenance["parent_id"] == p and
            provenance["files"]["truth/full_truth_intervals.npz"]["sha256"] == b["truth_npz"]["sha256"],
            "Original same-parent VAL generation provenance required")
    phase = metadata(refs["phase_registration"])
    require(phase.get("schema") == "H039_frozen_checkpoint_evaluation_registration_v1"
        and phase["status"] == "registered" and phase["root_authorizer"] == "root_agent"
        and phase["evaluation_passes"] == passes and phase["original_evaluation_passes"] == [5,10]
        and phase["selection"] == SELECTION and phase["all24_RAM_seals_before_any_VAL_truth_per_pass"] is True
        and phase["checkpoint_manifest_ref"] == refs["checkpoint_manifest"]
        and phase["train_phase_registration_ref"] == refs["train_phase_registration"]
        and phase["optimization_authorized"] is False, "Root registered exact RAM-sealing phase required")
    for name in ("max_cpu_rss_bytes", "max_cuda_reserved_bytes"):
        require(type(spec.get(name)) is int and spec[name] > 0, "Explicit admitted resource cap")
    require(spec["max_cuda_reserved_bytes"] <= 50000000000, "Authorized external-sharing CUDA cap")
    reader = EvaluationReader(fit.ConsumerReaderV1(),
        [b["truth_npz"] for b in bindings["parents"].values()], checkpoint_refs)
    storage = reader.decode_json(refs["storage"]); adapter = load_repair(cfg)
    needed = [dict(path=p, sha256=manifest["sources"][p]) for p in manifest["required_consumer_source_paths"]]
    needed += list(refs.values())  # storage is self-referential and is checked separately below.
    needed = [r for r in needed if r != refs["storage"]]
    needed += [cfg["base_fit_config"], cfg["helper_refs"]["storage_adapter"],
        cfg["helper_refs"]["actual_launcher"], cfg["helper_refs"]["v9_consumer_marker"],
        cfg["marker_repair_authority_ref"], cfg["external_GPU_sharing_authorization_ref"]]
    require(all(r in storage["reservation"]["consumer_bindings"] for r in needed),
        "Actual execution sources/metadata/repair/marker/launcher missing from reservation")
    require(cfg["helper_refs"]["actual_launcher"]["path"] == LAUNCHER
        and cfg["helper_refs"]["v9_consumer_marker"]["path"] == MARKER, "Existing v2 sharing launcher and v9 marker")
    adapter.validate_storage_v9(spec, cfg, storage,
        source_ref=dict(path=SOURCE, sha256=manifest["sources"][SOURCE]),
        config_ref=refs["config"], runtime_ref=refs["runtime"])
    return spec, cfg, manifest, plan, train, generation, bindings["parents"], reader, storage, adapter, lease


def array_identity(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return dict(dtype=a.dtype.str, shape=list(a.shape), nbytes=int(a.nbytes),
        sha256=hashlib.sha256(memoryview(a).cast("B")).hexdigest())


class RAMSeals:
    """Immutable native-hold values; published JSON binds bytes, mapping, CP and sources."""
    def __init__(self, output, reader, spec, parents, geometry, observations):
        self.output, self.reader, self.spec = output, reader, spec
        self.parents, self.geometry, self.observations = parents, geometry, observations
        self.values, self.records, self.barriers, self.verified_passes = {}, {}, {}, set()

    def seal(self, p, arm, at, probability, checkpoint_ref):
        import numpy as np
        require((at,arm,p) not in self.values, "No prediction replacement")
        value = np.array(probability, dtype=np.float32, copy=True, order="C")
        t = self.geometry[p]["native_holds"]
        require(value.shape == (t,686,4) and np.isfinite(value).all() and np.all(value >= 0)
            and np.allclose(value.sum(-1), 1, rtol=1e-5, atol=1e-6), "Complete raw finite native-hold FP32 probabilities")
        value.setflags(write=False)
        packet = dict(schema="H039_native_hold_RAM_prediction_seal_v1", parent_id=p, arm=arm, pass_index=at,
            seed=101, checkpoint_ref=checkpoint_ref, checkpoint_manifest_ref=self.spec["refs"]["checkpoint_manifest"],
            source_manifest_ref=self.spec["refs"]["source_manifest"], evaluation_source_ref=self.spec["evaluation_source_ref"],
            train_input_ref=self.spec["refs"]["train_input"], train_config_ref=self.spec["refs"]["train_config"],
            phase_registration_ref=self.spec["refs"]["phase_registration"],
            probability=array_identity(value), geometry=self.geometry[p]["identities"], observations=self.observations[p],
            representation="CPU_native_hold_FP32", exact_original_grid_mapping="native_probabilities[readout_hold]",
            prediction_array_durable=False, prediction_immutable_in_worker_RAM=True, dense_prediction_arrays_saved=False,
            targets_opened_before_this_seal=False, reproduction="actual_checkpoint_plus_original_observations_and_source_manifest")
        ref = self.output.save("%s.pass_%02d.%s.RAM_seal.json" % (arm,at,p), packet)
        self.values[at,arm,p] = value
        self.records[at,arm,p] = ref
        return ref

    def verify_pass(self, at):
        require({(a,p) for t,a,p in self.values if t == at} == {(a,p) for a in ARMS for p in VAL},
            "All3 x all8 actual native-hold probability arrays required")
        for arm in ARMS:
            for p in VAL:
                record = self.records[at,arm,p]
                packet = self.reader.decode_json(record)
                require(packet["probability"] == array_identity(self.values[at,arm,p])
                    and not self.values[at,arm,p].flags.writeable
                    and packet["geometry"] == self.geometry[p]["identities"]
                    and packet["observations"] == self.observations[p]
                    and packet["source_manifest_ref"] == self.spec["refs"]["source_manifest"],
                    "Immutable RAM content/geometry/source seal changed")
        return [(p,self.records[at,a,p]) for a in ARMS for p in VAL]

    def publish_barrier(self, at):
        records = self.verify_pass(at)
        barrier = self.output.save("pass_%02d_all24_RAM_sealed.json" % at, dict(
            schema="H039_all24_RAM_prediction_barrier_v1", pass_index=at,
            arms=list(ARMS), VAL_parent_ids=list(VAL), seals=[r for _,r in records],
            complete_prediction_count=24, all24_actual_probabilities_immutable_in_CPU_RAM=True,
            before_any_VAL_truth_read=True, checkpoint_manifest_ref=self.spec["refs"]["checkpoint_manifest"],
            source_manifest_ref=self.spec["refs"]["source_manifest"],
            no_prediction_array_written_to_disk=True))
        self.reader.decode_json(barrier)
        self.barriers[at] = barrier
        self.verified_passes.add(at)

    def grid(self, at, arm, p):
        require(at in self.verified_passes, "Verified full24 pass barrier")
        packet = self.reader.decode_json(self.records[at,arm,p])
        require(packet["probability"] == array_identity(self.values[at,arm,p])
            and not self.values[at,arm,p].flags.writeable, "Current immutable native probability seal")
        # One transient parent/arm expansion. Exact original order, columns and FP32 values.
        return self.values[at,arm,p][self.geometry[p]["readout_hold"]]


def load_models(plan, at, spec, reader, device, guard):
    import torch
    from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
    from arggnn.directed_clade_exchange_crf_v2 import DirectedCladeExchangeCRFV2
    family = {}
    for arm in ARMS:
        guard()
        cp = plan["checkpoint_groups"][str(at)][arm]
        payload = torch.load(io.BytesIO(reader.read(cp)), map_location="cpu", weights_only=False)
        require(payload.get("arm") == arm and payload.get("seed") == 101 and payload.get("pass_index") == at
            and payload.get("phase") == "train_only" and payload.get("additional_optimizer_steps") == 32*at
            and payload.get("model_kwargs") == MODEL and payload.get("parameter_count") == COUNTS[arm]
            and payload.get("teacher_used") is False and payload.get("optimizer_state_reset") is True
            and payload["input_ref"] == spec["refs"]["train_input"]
            and payload["train_config_ref"] == spec["refs"]["train_config"]
            and payload["phase_registration_ref"] == spec["refs"]["train_phase_registration"],
            "Actual same fixed-training checkpoint content/provenance required")
        state = payload["model_state"]
        require(all(isinstance(v,torch.Tensor) and v.dtype == torch.float32 for v in state.values()),
            "Raw FP32 frozen model state")
        torch.manual_seed(101)
        model = MutationRoutedFullPanelPyGGNNV1(**MODEL) if arm == ARMS[0] else DirectedCladeExchangeCRFV2(**MODEL)
        model.load_state_dict(state, strict=True)
        require(sum(p.numel() for p in model.parameters()) == COUNTS[arm], "Original model capacity")
        model.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
        family[arm] = model
        del payload, state
    return family


def load_VAL_observations(spec, train, generation, reader, guard):
    import numpy as np
    from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1, read_full_cache_parent_v1
    from arggnn.directed_clade_exchange_witness_v1 import read_parent_sidecar_v1, PARENTS, SOURCE as WITNESS_SOURCE
    from arggnn.directed_clade_exchange_crf_v1 import CladeExchangeInputsV1
    refs = spec["refs"]
    manifest = reader.decode_json(refs["witness_sidecar_manifest"])
    witness_ref = dict(path=WITNESS_SOURCE, sha256=hashlib.sha256((ROOT/WITNESS_SOURCE).read_bytes()).hexdigest())
    require(manifest["schema"] == "directed_clade_exchange_full_observation_sidecars_v1"
        and manifest["builder_ref"] == witness_ref and manifest["observation_cache_ref"] == refs["observation_cache"]
        and manifest["entry_refs"] == train["observation_entry_refs"]
        and manifest["source_refs"] == train["observation_cache_source_refs"]
        and manifest["parent_ids"] == list(PARENTS) and set(manifest["parents"]) == set(PARENTS)
        and manifest["all40_observed_banks_complete"] is True and manifest["targets_opened"] is False
        and manifest["outer_included"] is False, "Original complete sealed observation witness lineage")
    index = read_full_cache_index_v1(refs["observation_cache"], reader, registry=generation["parent_registry"],
        entry_refs=train["observation_entry_refs"], source_refs=train["observation_cache_source_refs"])
    banks, parents, geometry, observations = {}, {}, {}, {}
    for p in VAL:
        guard()
        cache_ref = index["parents"][p]["array_ref"]
        sidecar_ref = manifest["parents"][p]
        sidecar_meta = reader.decode_json(sidecar_ref)
        reader.register_observed(cache_ref,p); reader.register_observed(sidecar_meta["array_ref"],p)
        item = read_full_cache_parent_v1(p,index,reader)
        arrays, sidecar = read_parent_sidecar_v1(sidecar_ref,reader,parent_id=p,
            observation_entry_ref=train["observation_entry_refs"][p],cache_parent_ref=cache_ref)
        bank = CladeExchangeInputsV1.attach(item["model_inputs"],arrays,record=sidecar)
        parents[p] = item["scoring"]
        require(parents[p]["split"] == "val", "Only original VAL banks")
        for k in ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts"):
            parents[p][k].setflags(write=False)
        lookup = np.array(bank.readout_hold.numpy(),copy=True,order="C")
        require(lookup.dtype == np.int64 and lookup.shape == (len(parents[p]["positions"]),)
            and np.all(lookup >= 0) and np.all(lookup < bank.sample_index.shape[0]), "Exact original readout_hold mapping")
        lookup.setflags(write=False)
        identities = {k:array_identity(parents[p][k]) for k in
            ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")}
        identities["readout_hold"] = array_identity(lookup)
        identities["native_spans"] = array_identity(bank.spans.numpy())
        geometry[p] = dict(native_holds=int(bank.sample_index.shape[0]), readout_hold=lookup, identities=identities)
        observations[p] = dict(observation_entry_ref=train["observation_entry_refs"][p],
            cache_parent_ref=cache_ref, witness_sidecar_ref=sidecar_ref, witness_array_ref=sidecar["array_ref"])
        banks[p] = bank
        del item, arrays, sidecar
    return banks, parents, geometry, observations


def run(record):
    spec,cfg,manifest,plan,train,generation,bindings,reader,storage,adapter,lease = inputs(record)
    import numpy as np
    import torch
    import torch_geometric
    from arggnn import census_tsinfer_fit_v1 as fit
    from arggnn.biological_candidate import configure_fit, check_fit_lease
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1, means_v1, PRIMARY
    from arggnn.mutation_routed_fast_primary_v1 import mass_cache_v1, score_primary_v1
    require(str(torch.__version__) == cfg["runtime"]["torch"] and torch.version.cuda == cfg["runtime"]["torch_cuda"]
        and torch_geometric.__version__ == cfg["runtime"]["torch_geometric"], "Original pinned inference runtime")
    # Initialize the existing lazy serializer runtime before any evidence write.
    fit.np = np
    fit.torch = torch
    output = adapter.PilotOutputV9(spec["output_directory"],storage,reader)
    adapter.budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
    device,runtime,actual_lease = configure_fit("cuda",reader.decode_json(cfg["base_fit_config"]),output.path)
    require(actual_lease == lease, "Qualified original lease")
    torch.cuda.set_per_process_memory_fraction(min(1.,spec["max_cuda_reserved_bytes"]/
        torch.cuda.get_device_properties(device).total_memory),device)
    output.used = sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
    require(output.used <= cfg["qualification_allowance_bytes"], "Qualification exceeds admitted 2MB")
    def guard():
        fit._check_active_v1(spec);check_fit_lease(lease);fit.require_logical_job_v1(lease,spec)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= spec["max_cpu_rss_bytes"], "Admitted CPU RSS")
        require(torch.cuda.max_memory_reserved() <= spec["max_cuda_reserved_bytes"], "Admitted CUDA reserve")
    started = time.monotonic()
    receipt = dict(schema="H039_frozen_checkpoint_evaluation_evidence_v1",status="running",hypothesis_id="H039",
        input_ref=record,source_manifest_ref=spec["refs"]["source_manifest"],checkpoint_manifest_ref=spec["refs"]["checkpoint_manifest"],
        evaluation_passes=cfg["evaluation_passes"],original_evaluation_passes=[5,10],runtime=runtime,
        optimization_performed=False,TRAIN_truth_read=False,outer_truth_read=False,dense_prediction_arrays_saved=False)
    output.save("started.json",receipt)
    ram = None
    try:
        # Checkpoint bytes/provenance are verified before observation arrays and before any VAL truth.
        families = {at:load_models(plan,at,spec,reader,device,guard) for at in cfg["evaluation_passes"]}
        banks,parents,geometry,observations = load_VAL_observations(spec,train,generation,reader,guard)
        ram = RAMSeals(output,reader,spec,parents,geometry,observations)
        for at in cfg["evaluation_passes"]:
            for p in VAL:
                guard(); cpu_bank = banks[p]
                needed = sum(t.numel()*t.element_size() for f in fields(cpu_bank)
                    for t in (getattr(cpu_bank,f.name) if isinstance(getattr(cpu_bank,f.name),tuple) else (getattr(cpu_bank,f.name),)))
                free,_ = torch.cuda.mem_get_info(device)
                headroom = min(int(free),spec["max_cuda_reserved_bytes"]-torch.cuda.memory_reserved())-cfg["workspace_reserve_bytes"]
                require(needed <= headroom, "Admitted complete-parent inference residency")
                values = {f.name:tuple(t.to(device) for t in getattr(cpu_bank,f.name)) if isinstance(getattr(cpu_bank,f.name),tuple)
                    else getattr(cpu_bank,f.name).to(device) for f in fields(cpu_bank)}
                bank = replace(cpu_bank,**values)
                for arm in ARMS:
                    guard(); model = families[at][arm]
                    with torch.inference_mode():
                        logits = model.forward_train_holds(bank,control="aligned" if arm == ARMS[0] else arm)
                        probability = logits.softmax(-1)
                        require(probability.dtype == torch.float32, "Unchanged raw FP32 inference")
                        ram.seal(p,arm,at,probability.detach().cpu().numpy(),plan["checkpoint_groups"][str(at)][arm])
                    del logits,probability
                del bank,values
            ram.publish_barrier(at)
            print(json.dumps(dict(stage="H039_frozen_all24_native_RAM_sealed",pass_index=at,
                barrier_ref=ram.barriers[at],VAL_truth_read=False)),flush=True)
        # Stronger than per-pass gating: seal every requested pass before the first truth read.
        for at in cfg["evaluation_passes"]:
            ram.verify_pass(at)
        reader.ram = ram;reader.allowed_passes = tuple(cfg["evaluation_passes"])
        helpers = fit.metric_helpers_v1()
        caches = {}
        all_seals = [r for at in cfg["evaluation_passes"] for _,r in ram.verify_pass(at)]
        for p in VAL:
            guard()
            target = FullPanelExternalTargetV1(parents[p],bindings[p],"VAL_after_sealed_predictions",reader,sealed_records=all_seals)
            caches[p] = mass_cache_v1(parents[p],target.target_mass(parents[p],helpers),
                target_provenance=target.provenance,expected_present_queries=182)
            del target
        rows = {a:[] for a in ARMS}
        for at in cfg["evaluation_passes"]:
            for arm in ARMS:
                cells = []
                for p in VAL:
                    guard(); probability = ram.grid(at,arm,p)
                    group = score_primary_v1(probability,caches[p],backend="numpy")
                    cells.append(dict(parent_id=p,split="val",prediction_seal=ram.records[at,arm,p],
                        checkpoint_ref=plan["checkpoint_groups"][str(at)][arm],groups={PRIMARY:group}))
                    del probability
                means = means_v1(cells)
                row = dict(arm=arm,pass_index=at,optimizer_steps=32*at,
                    checkpoint=plan["checkpoint_groups"][str(at)][arm],cells=cells,equal_parent_metrics=means,
                    all24_barrier_ref=ram.barriers[at])
                row_ref = output.save("%s.pass_%02d.primary_scores.json" % (arm,at),row)
                rows[arm].append(row)
                print(json.dumps(dict(stage="H039_frozen_original_VAL_primary_score",arm=arm,pass_index=at,
                    primary=means[PRIMARY],score_ref=row_ref)),flush=True)
        final = cfg["evaluation_passes"] == [5,10]
        selected = {}
        if final:
            for arm in ARMS:
                def key(row):
                    primary = row["equal_parent_metrics"][PRIMARY]
                    return (-primary["accuracy"],float("inf") if primary["NLL_is_infinite"] else primary["nll"],row["pass_index"])
                selected[arm] = min(rows[arm],key=key)
        receipt.update(status="completed_H039_frozen_checkpoint_evaluation_v1",history=rows,
            selected=selected if final else None,selection_final=final,selection_deferred=not final,
            selection_rule=SELECTION,all_requested_pass_seals_before_first_VAL_truth=True,
            all24_barrier_refs=ram.barriers,RAM_native_probability_bytes=sum(v.nbytes for v in ram.values.values()),
            original_readout_hold_mapping_exact=True,scorer="unchanged_H038_FP32_Present182_numpy",
            optimization_performed=False,optimizer_instantiated=False,training_scope_written=False,
            elapsed_seconds=time.monotonic()-started,owned_output_bytes=output.used)
        guard();output.save("completed.json",receipt)
        return receipt
    except Exception as error:
        receipt.update(status="failed_partial_evaluation_preserved",error_type=type(error).__name__,error=str(error),
            published_all24_barriers=ram.barriers if ram else {},automatic_replay=False,
            elapsed_seconds=time.monotonic()-started)
        output.save("failed.json",receipt);raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input");parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_frozen_evaluation_not_executed",source=SOURCE,
            output_cap_bytes=CAP,representation="native_hold_RAM_seals",original_evaluation_passes=[5,10],
            all24_seals_before_truth=True,optimization_performed=False)))
        return 0
    require(args.input and args.expected_input_sha256,"Explicit root SHA-bound evaluation input")
    run(dict(path=args.input,sha256=args.expected_input_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
