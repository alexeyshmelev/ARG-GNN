"""One-tile FP32 H033 repacking qualification. Execution needs a fresh root input."""
from __future__ import annotations
import argparse, io, json, tempfile, time
from types import SimpleNamespace
from . import train_interleaved_tree_tv_gnn_v1 as h033
from .interleaved_tree_tv_gnn_v1 import InterleavedTreeTVGNNV1, prepare_interleaved_tree_tv_bank_v1
from .interleaved_tree_tv_repack_v1 import InterleavedTreeTVRepackGNNV1, repack_interleaved_tree_tv_bank_v1
fit, ROOT = h033.fit, h033.ROOT
SOURCE = "src/arggnn/qualify_interleaved_tree_tv_repack_v1.py"
IDENTITY, PARENT, TILE = "bio_paths_pilot_train_000:1", "bio_paths_pilot_train_000", 1; CAP, QUALIFICATION_ALLOWANCE = 50000000, 32000000
def require(value, message):
    if not value:
        raise ValueError(message)
def build_one_tile_v1(original, cfg, reader, output, storage, device):
    """Frozen H033/H035 native extraction and joins, restricted to TRAIN tile 1."""
    import tskit
    from .persistence_trace_kernel_v1 import MetricTreeHoldV1
    from .project_storage_full_reservation_v1 import budget_check_v9
    np, torch = fit.np, fit.torch
    refs = original["observed_parents"][PARENT]
    prepared, entry = (reader.decode_json(refs[k]) for k in ("prepared_manifest", "observed_parent_entry"))
    source_ref = entry["original_observed_sample_manifest"]
    source, inferred = reader.decode_json(source_ref), reader.decode_json(refs["observed_roster"])
    emitter = reader.decode_json(original["refs"]["emitter_bindings"])["parents"][PARENT]
    emitted = reader.decode_json(emitter["observed_roster"])
    maps = list(map(h033.roster_v1, (source, inferred, emitted)))
    require(prepared["parent_id"] == PARENT and prepared["parent_split"] == "train"
            and prepared["files_sha256"]["observed_roster.json"] == refs["observed_roster"]["sha256"]
            and "tsinfer" in entry["inferred_tree"]["condition"].lower(), "Frozen TRAIN producer differs")
    require(set(maps[0]) == set(maps[1]) == set(maps[2]) and all(
        row[field] == other[key][field] for key, row in maps[0].items() for other in maps[1:]
        for field in ("role", "age_generations", "observed_cohort", "observed_reference_class")),
        "Observed identity/role/date/reference join differs")
    query, pure = ([row for row in source if row["role"] == role]
                   for role in ("query", "pure_source_reference"))
    require(len(query) == 120 and len(pure) == 96 and all(
        row["observed_reference_class"] is None for row in source if row["role"] != "pure_source_reference"),
        "Only genuine observed R96 labels and actual Q120 may enter")
    binding = reader.decode_json(original["refs"]["target_bindings"])["parents"][PARENT]
    coordinate = prepared["coordinate_contract"]
    offset, coverage = coordinate["local_to_original_offset_bp"], coordinate["coverage_original_bp0"]
    roles = [SimpleNamespace(actual_id=row["sample_index"], role=row["role"]) for row in source]
    loaded = (None, roles, 50. + 100. * np.arange(20000), offset, coverage, {"source": source_ref}, None)
    parent = fit.scoring_parent_v1({"parent_id": PARENT, "split": "train"}, None, loaded, reader, binding)
    left, right = max(TILE * 250000, parent["edges"][0]), min((TILE + 1) * 250000, parent["edges"][-1])
    indexes = np.flatnonzero((parent["positions"] >= left) & (parent["positions"] < right))
    require(len(indexes) == 2500 and cfg["tile_bp"] == 250000, "Exact original full 250kb tile required")
    with np.load(io.BytesIO(reader.read(emitter["grid_prediction"])), allow_pickle=False) as grid:
        columns = {int(node): i for i, node in enumerate(grid["query_nodes"])}
        permutation = [columns[int(maps[2][(row["sample_id"], row["haplotype"])]["inferred_tree_node_id"])] for row in query]
        require(len(columns) == 120 and np.array_equal(grid["positions_bp"], parent["positions"])
                and np.array_equal(grid["grid_indexes"], coordinate["retained_fixed100bp_indexes"])
                and np.array_equal(grid["bin_edges_bp"], parent["edges"])
                and np.array_equal(grid["positions_bp"] - grid["local_positions_bp"], np.full(len(parent["positions"]), offset)),
                "Original emitter support and coordinate join differs")
        probabilities = grid["probabilities"][indexes][:, permutation].copy()
        require(grid["supported"][indexes][:, permutation].all() and probabilities.dtype == np.float32
                and probabilities.shape == (2500, 120, 4) and np.isfinite(probabilities).all()
                and (probabilities > 0).all() and np.allclose(probabilities.sum(-1), 1., atol=1e-5, rtol=0),
                "Complete positive FP32 original emitter tile required")
    tree_ref = {k: entry["inferred_tree"][k] for k in ("path", "sha256")}
    payload = reader.read(tree_ref)
    require(output.used + len(payload) + 8192 <= output.limit, "Native temporary bytes exceed cap")
    budget_check_v9(storage, boundary=True, required_write_bytes=len(payload) + 8192)
    try:
        with tempfile.NamedTemporaryFile(dir=output.path, suffix=".observed.trees") as file:
            file.write(payload); file.flush()
            ts = tskit.load(file.name)
    finally:
        budget_check_v9(storage, boundary=True)
    del payload
    native = [int(maps[1][(row["sample_id"], row["haplotype"])]["inferred_tree_node_id"]) for row in source]
    require(len(native) == len(set(native)) == 280 and set(native) == set(map(int, ts.samples()))
            and ts.sequence_length == coverage[1] - coverage[0]
            and all(float(ts.node(node).time) == float(row["age_generations"]) for node, row in zip(native, source)),
            "Complete native sample/date join differs")
    trees, intervals = [], []
    for tree in ts.trees():
        lo, hi = max(left, tree.interval.left + offset), min(right, tree.interval.right + offset)
        if hi <= lo:
            continue
        union = set()
        for sample in native:
            node = sample
            while node != -1 and node not in union:
                union.add(node); node = tree.parent(node)
        packed = [int(node) for node in tree.nodes(order="postorder") if node in union]
        local = {node: i for i, node in enumerate(packed)}
        trees.append(MetricTreeHoldV1(torch.tensor([-1 if tree.parent(n) == -1 else local[tree.parent(n)] for n in packed]),
            torch.tensor([tree.time(n) for n in packed], dtype=torch.float64), torch.tensor(packed),
            PARENT + ":" + tree_ref["sha256"], torch.tensor([local[n] for n in native]),
            torch.tensor([row["sample_index"] for row in source])))
        intervals.append([lo, hi])
    bank = prepare_interleaved_tree_tv_bank_v1(trees, torch.tensor(intervals, dtype=torch.float64),
        torch.from_numpy(parent["positions"][indexes]), torch.tensor([r["sample_index"] for r in query]),
        torch.tensor([r["sample_index"] for r in pure]), torch.tensor([r["observed_reference_class"] for r in pure]),
        census=cfg["census"], sample_identity_namespace=PARENT + ":" + source_ref["sha256"], **cfg["geometry"])
    require(bank.metadata["younger_reference_count"] == 84 and bank.metadata["older_reference_count"] == 12,
            "Original H030 genuine source policy differs")
    teacher = h033.h025.models_v1(reader.decode_json(original["refs"]["frozen_baseline_config"]),
                                 torch.device("cpu"))["genomic_smoothing"]["model"]
    saved = torch.load(io.BytesIO(reader.read(original["refs"]["frozen_baseline_checkpoint"])), map_location="cpu", weights_only=True)
    require(saved["arm"] == "genomic_smoothing" and saved["epoch"] == 10, "Frozen H025 identity differs")
    teacher.load_state_dict(saved["model_state"], strict=True); teacher.to(device).eval()
    with torch.no_grad():
        baseline = teacher(torch.from_numpy(np.log(probabilities)).to(device)).detach().cpu()
    tile_parent = dict(parent, positions=parent["positions"][indexes], edges=parent["edges"][indexes[0]:indexes[-1] + 2])
    return bank, baseline, tile_parent, binding
def run_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    from .biological_candidate import configure_fit, fingerprint_sha
    from .biological_gst_control_v1 import exact_span_loss
    from .project_storage_full_reservation_v1 import PilotOutputV9, budget_check_v9, validate_storage_v9
    lease = validate_lease(); apply_and_validate_title()
    reader = fit.ConsumerReaderV1(); spec = reader.decode_json(record)
    require(spec["schema"] == "interleaved_tree_tv_repack_qualification_input_v1"
        and spec["root_authorizer"] == "root_agent" and spec["status"] == "active"
        and spec["qualification_authorized"] is True and spec["TRAIN_loss_authorized"] is True
        and spec["device"] == "cuda" and lease["physical_gpu"] == 2 and spec["output_cap_bytes"] == CAP,
        "Fresh root one-tile GPU2 qualification authority required")
    fit._check_active_v1(spec); fit.require_logical_job_v1(lease, spec)
    require(spec["source_sha256"].get(SOURCE) == fit._hash(SOURCE), "Exact driver pin required")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    refs = spec["refs"]; original = reader.decode_json(refs["original_input"])
    cfg = reader.decode_json(original["refs"]["config"]); completed = reader.decode_json(refs["original_completed"])
    first = reader.decode_json(refs["first_training_tile"])
    require(completed["input_ref"] == refs["original_input"] and completed["status"].startswith("completed_")
        and all(v["steps"] == 160 for v in completed["models"].values()) and first["identity"] == IDENTITY
        and completed["models"]["interleaved"]["selected"]["checkpoint"] == refs["selected_checkpoint"],
        "Immutable completed H033 interleaved checkpoint and first-tile identity required")
    require(refs["original_completed"]["path"] == "experiments/results/interleaved_tree_tv_gnn_v1_seed101_GPU2/completed.json"
            and spec["source_sha256"].get("src/arggnn/interleaved_tree_tv_repack_v1.py") ==
            "60e331e9a5001adcc0649d4f194c39c36f95c8e3dcf64456e692271e80c09e99", "Frozen source/run bindings required")
    for path, digest in original["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for reference in cfg["helper_refs"].values():
        reader.read(reference)
    reader.read(refs["runtime"]); storage = reader.decode_json(refs["storage"]); fit._science_v1(); torch = fit.torch
    require(cfg["helper_refs"]["actual_full_storage_launcher"] in storage["reservation"]["consumer_bindings"],
            "Actual full-storage launcher must be reserved")
    validate_storage_v9(spec, spec, storage, source_ref={"path": SOURCE, "sha256": fit._hash(SOURCE)},
                        config_ref=record, runtime_ref=refs["runtime"])
    output = PilotOutputV9(spec["output_directory"], storage, reader)
    measure = lambda: sum(max(p.stat().st_size, p.stat().st_blocks * 512) + 8192 for p in output.path.rglob("*") if p.is_file())
    before = measure(); require(before + QUALIFICATION_ALLOWANCE <= output.limit, "32MB CUDA allowance exceeds cap")
    budget_check_v9(storage, boundary=True, required_write_bytes=QUALIFICATION_ALLOWANCE)
    try:
        device, runtime, lease = configure_fit("cuda", reader.decode_json(cfg["base_fit_config"]), output.path)
    finally:
        budget_check_v9(storage, boundary=True)
    require(measure() - before <= QUALIFICATION_ALLOWANCE, "CUDA qualification exceeds 32MB allowance")
    output.used = measure(); h033.resources_v1(spec, storage, lease)
    bank32, baseline, parent, binding = build_one_tile_v1(original, cfg, reader, output, storage, device)
    bank256 = repack_interleaved_tree_tv_bank_v1(bank32)
    require(all(bank32.metadata[k] == first[k] for k in ("native_vertices", "replicated_vertices", "packed_block_count")),
            "Same recorded actual tile geometry required")
    seal = output.save("observation_bank_seal.json", dict(identity=IDENTITY, targets_opened=False,
        bank32_sha256=fingerprint_sha(bank32), bank256_sha256=fingerprint_sha(bank256),
        baseline_sha256=fingerprint_sha(baseline), support_sha256=fingerprint_sha(parent), metadata=bank32.metadata))
    reader.read(seal); h033.resources_v1(spec, storage, lease)
    mass = torch.as_tensor(fit.ExternalTargetV1(parent, binding, "TRAIN_loss", reader).target_mass(parent, fit.metric_helpers_v1()),
                           dtype=torch.float32, device=device)
    saved = torch.load(io.BytesIO(reader.read(refs["selected_checkpoint"])), map_location="cpu", weights_only=True)
    require(saved["arm"] == "interleaved" and saved["input_ref"] == refs["original_input"], "Selected state provenance differs")
    rows, retained = [], []
    for packing, factory, cpu_bank in ((32, InterleavedTreeTVGNNV1, bank32), (256, InterleavedTreeTVRepackGNNV1, bank256)):
        h033.resources_v1(spec, storage, lease)
        torch.cuda.empty_cache(); model = factory().to(device).train(); model.load_state_dict(saved["model_state"], strict=True)
        bank, frozen = cpu_bank.to(device), baseline.to(device)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); start = time.monotonic()
        logits = model(bank, frozen, control="interleaved")
        torch.cuda.synchronize(device); forward = time.monotonic() - start
        loss = exact_span_loss(logits, mass); require(bool(torch.isfinite(loss)), "Nonfinite TRAIN loss")
        torch.cuda.synchronize(device); start = time.monotonic(); loss.backward()
        torch.cuda.synchronize(device); backward = time.monotonic() - start
        gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters()}
        require(all(bool(torch.isfinite(g).all()) for g in gradients.values()), "Nonfinite parameter gradient")
        rows.append(dict(packing=packing, forward_seconds=forward, backward_seconds=backward,
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)), peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            loss=float(loss.detach()), block_count=len(bank.blocks), readouts=bank.readout_count, native_vertices=bank.metadata["native_vertices"]))
        output.save(f"packing_{packing}.json", rows[-1])
        retained.append((logits.detach().cpu(), gradients)); del model, bank, frozen, logits, loss
    a, b = retained; torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
    errors = {k: float((a[1][k] - b[1][k]).abs().max()) for k in a[1]}
    for key in errors: torch.testing.assert_close(a[1][key], b[1][key], rtol=1e-5, atol=1e-6)
    h033.resources_v1(spec, storage, lease); reader.read(record)
    return output.save("qualification.json", dict(status="completed_descriptive_single_tile_qualification", input_ref=record,
        identity=IDENTITY, observation_seal=seal, runtime=runtime, order=[32, 256], runs=rows, optimizer_steps=0,
        logits_max_abs_error=float((a[0] - b[0]).abs().max()), gradient_max_abs_errors=errors,
        loss_abs_difference=abs(rows[0]["loss"] - rows[1]["loss"]), rtol=1e-5, atol=1e-6, accuracy_claim=False, novelty_claim=False, measured_speed_claim=False))
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--input"); parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed", source=SOURCE, sha256=fit._hash(SOURCE)))); return 0
    require(args.input and args.expected_input_sha256, "Pinned root input required")
    print(json.dumps(run_v1({"path": args.input, "sha256": args.expected_input_sha256}))); return 0
if __name__ == "__main__": raise SystemExit(main())
