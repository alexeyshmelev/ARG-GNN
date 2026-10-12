# Native AP root scoring assistance

Source-only draft, 2026-10-12. The helper was syntax-parsed, not imported or run. No prediction, observation array or truth file was opened by this reviewer. No score, numerical-readiness claim, admission, GPU action or Git change occurred. The immutable novelty review is untouched.

The existing completion metadata records five epochs, 5,190,874 realized rows from 5,200,000 requested, a 9,126-row shortfall, and all eight VAL prediction seals. AP uses its fixed final epoch-5 model, with no VAL selection.

## Reusable exact APIs

- mutation_routed_full_panel_primary_v1.FullPanelExternalTargetV1: full686 external C4 interval target, instantiated only after all AP seals are verified.
- target.target_mass(parent, helpers): exact FP64 bin/truth intersections.
- mutation_routed_fast_primary_v1.mass_cache_v1(..., expected_present_queries=182), then score_primary_v1(..., backend="numpy"): exactly H038's FP32 primary physical-span accuracy/NLL, retaining mixed spans and infinite NLL.
- mutation_routed_full_panel_primary_v1.means_v1(cells): same unweighted eight-parent FP32 reduction.
- census_tsinfer_fit_v1.metric_helpers_v1(): existing intersections helper.
- expanded_native_ancestralpaths_v1.native_focal_metrics_v1(): separate six-path unweighted focal accuracy/confusion and precision.

AP's common_C4_ref already contains raw source_probabilities collapsed as p1+p5,p2+p6,p3,p4 BEFORE argmax. Feed that unchanged to the H038 metric. Ana is the same slot-0 source as H038 ANA. Do not rerun collapse with the contracts function, which uses FP64 arithmetic, or call the old stable-logit scorer.

FullPanelPrimaryValidationV1 directly expects the H038 five-field NPZ schema, whereas AP has a different sealed schema. The draft uses its constituent exact APIs with strict AP-to-H038 coordinate/sample validation. It verifies all eight common C4 and native6 products before opening any target, then scores one parent at a time. FullPanelExternalTargetV1 receives the eight verified AP manifest refs as its held-out seal gate. The complete NPZ verification gate is enforced first by this helper.

AP6 is sealed only at native SNP focal positions. The old TruthReader hardcodes120 queries and cannot be reused for full686. The draft's small root-only adapter validates the same externally bound six-path interval archive after the seal gate, joins by original SNP positions and query columns, and calls the existing native_focal_metrics_v1. It retains repeated source focal pairs. It produces no AP6 full-grid physical-span metric.

## Concrete root entry

After root reviews/pins this helper, use the existing project Python/NumPy runtime. No TensorFlow runtime, checkpoint load, new inference, training worker or GPU lease is required for scoring.

    import importlib.util
    from arggnn import census_tsinfer_fit_v1 as fit

    path = ROOT / "research/reviews/expanded_native_AP_score_root_assistance_v1/root_score_helper.py"
    spec = importlib.util.spec_from_file_location("root_native_AP_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # index is the root-verified H038 cache index.
    # root_bindings is the root-verified target-binding metadata.
    parents = {p: module.read_h038_scoring_only_v1(index, p, reader)
               for p in module.VAL_IDS}
    bindings = {p: root_bindings["parents"][p] for p in module.VAL_IDS}
    result = module.score_root_v1(
        completed_ref=reviewed_completed_ref,
        parents=parents, bindings=bindings, reader=reader,
        helpers=fit.metric_helpers_v1(),
        on_parent=root_guarded_parent_json_writer,
    )
    root_guarded_summary_json_writer(result)

Alternatively reuse already validated H038 scoring-parent objects to avoid another observed-cache read. Reader must independently SHA-verify exact repository-local records. Use the bound completion JSON record and original generation/target refs. Do not use the H038 training inputs_v1/run_v1 entry, which requires a GPU fit capability.

Strict equality checks cover grid centers, clipped edges, retained indexes, original query slots/raw nodes, coverage and sample/haplotype keys. A mismatch must stop the matched comparison. The helper does not crop to an intersection, interpolate, impute tails or reorder unknown node namespaces.

## Realistic output scope

First produce eight compact per-parent JSON reports and one summary: Present182 C4 accuracy/NLL with H038 arithmetic, all-eight parent mean, native AP6 focal overall/by-cohort accuracy and 6x6 confusion/precision, denominator/coverage and immutable provenance. A 10 MB compact-JSON output cap is ample for this scope. The helper returns reports and delegates writes to root's existing guarded writer.

It does not produce predictions, copied NPZs, checkpoints, plots, per-query tract JSON, full calibration/boundary/dosage diagnostics or outer scores. Those broader diagnostics can later call helpers.full_span_score separately if root selects that scope. They use FP64 reporting and must not replace the H038 primary FP32 table.

At the maximal20,000-bin support, one C4 array is about220 MB. The full686 FP64 class-span target mass is about439 MB, and the Present182 FP32 cache about58 MB. Native AP6 focals are small, at nominal50,000 examples per parent. The draft avoids holding all eight dense C4 arrays, but reads each compressed C4 product twice, once for the complete seal gate and once while scoring. Peak process memory can exceed1 GB with temporary arrays and compressed bytes, plus any caller-owned observation banks. NFS/decompression and mass integration may dominate runtime. No throughput or peak-memory figure has been measured.

The draft still needs root execution review and numerical verification. Syntax parsing alone is not validation of actual AP/H038 support joins or scores.
