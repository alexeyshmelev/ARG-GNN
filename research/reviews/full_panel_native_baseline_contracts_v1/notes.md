# Native full-panel contracts v1

The module is a pure metadata/array contract library. It opens no files, creates no outputs, imports no native models, exposes no CLI, and supplies no GPU/storage/scheduling or experiment admission. The separate config is deliberately **planned_not_execution_guard_ready**, with every execution stage marked **planned_unimplemented**. Root must integrate actual observed readers, byte checks, native execution and real truth-file capability enforcement next.

Owned files:
- src/arggnn/full_panel_native_baseline_contracts_v1.py
- configs/full_panel_native_baselines_v2.json
- research/reviews/full_panel_native_baseline_contracts_v1/receipt.json
- this notes.md

The fixed split is bound by exact bytes/SHA256, not just nominal parent names: 000/001 TRAIN, nominal TRAIN002 INTERNAL_VAL, both prespecified discovery parents OUTER_DISCOVERY. Original generation tags, streams/config/source stay separate from downstream roles.

## API for root integration

- validate_split_bytes(original_bytes), validate_policy_bytes(policy_bytes, split_bytes): validate the root freeze and planned policy. They verify supplied JSON bytes, not referenced files.
- collapse_ap6_to_c4(probabilities), ap6_c4_argmax(probabilities): sum native probabilities as (p1+p5,p2+p6,p3,p4), then C4 argmax. Derived arithmetic is float64 and never renormalizes, masks, smooths or clips. Native six-class argmax remains a separate response.
- validate_observed_roster(rows, require_argmix_native_ids=False): accepts the actual observed JSON roster. It checks all952 slots, original tsk_0..475/haplotype/cohort order, null query classes, source-deme donor classes, unequal counts and all34 older prototypes. It returns immutable ObservedRoster identities/classes. Native ARGMix can require its existing node=2*sample_row+hap constraint, rejecting noncontiguous IDs. AP/ARGformer may use arbitrary actual unique inferred IDs.
- parent_namespace(parent_id, tree_sha256), validate_path_namespace(...), full_path_identity(namespace, full_nodes): bind each ARG vocabulary/checkpoint/path to a parent and inferred graph. Full path identity includes its leaf before token skipping/truncation, and numeric IDs from other ARGs remain distinct.
- validate_physical_support(inference["post_five_iterations"], original_centers_bp=None): checks the exact observed-only schema and all centers inside retained coverage. It returns original/local centers, clipped bin edges, exact query-bp denominator and both full-parent excluded and within-observed unscored mass. Bins do not extrapolate to an uncovered tail or an observed interval without a retained center.
- validate_site_coordinate_join(site_join, coverage_original_bp0=..., declared_site_count=...): requires complete native site IDs, strictly ordered original site IDs (attrition gaps retained), integer SNP coordinates, local+offset=original, VCF POS=original+1 and first-SNP/last-SNP+1 coverage.
- authorize_truth_use(parent_id, purpose, method=..., policy=..., ...): checks TRAIN_SUPERVISED_ROW_SELECTION, TRAIN_GRADIENTS, NATIVE_INTERNAL_VAL_SELECTION or SEALED_OUTER_SCORE against frozen downstream roles. Only current-release ARGMix has query-label internal VAL selection. ARGformer query ancestry cannot enter either training-purpose gate.
- native_ap_stride(S, cohort, traversal="TRAIN"|"HOLDOUT"): uses exact rational /3 overhead even with two downstream TRAIN parents. A zero stride fails rather than changing the observation grid or parent.
- validate_argformer_transduction(...): checks same-parent vocabulary/encoder namespace, all952 MLM leaves, all266 pure-reference contrastive leaves, no query ancestry and preserved4096 logical negative set. Outer observation-only per-ARG fitting is explicit transduction, not frozen-checkpoint transfer.
- raw_probability_nll(probabilities, fractional_truth_mass, weights): computes an unfloored proper score on supplied handmade or subsequently admitted arrays. Any weighted positive truth mass with zero native probability yields null numeric NLL plus nll_is_infinite and its affected query-bp. Zero-weight rows do not introduce an artificial infinity.
- run_handmade_checks(validated_policy, split_bytes): eight decisive synthetic checks without actual parent/tree/NPZ/model/target inputs.

Outer truth intents require declarations for all five prespecified prediction products on both outer parents, plus frozen selection. **A declaration is not proof of a prediction seal.** This library cannot ensure that a caller's path/hash names a real immutable artifact, or enforce filesystem access. Future guarded scorers must verify all referenced prediction and selection bytes before opening truth, and reject changed/stale policy/source bindings. No raw file-opening helper is supplied here.

The config pins existing qualified helper algorithms for reuse and records native source references. It does not import or rewrite any classifier, tokenizer, Rust extractor, pooler or retrieval implementation. It explicitly retains AP bounded-sampler/source-stride adaptations, current ARGMix source extraction/head/VAL objective/overlap behavior, and ARGformer released example objective/transduction/retrieval constraints. Paper/source and execution-adaptation differences remain labeled. Native full-panel memory and equivalence remain unqualified.

## Actual verification performed

Eight handmade checks passed: exact split bytes, source probability collapse before argmax, unequal/older donor eligibility and ARGMix ID rejection, ARG-specific path namespaces/observed-only transduction, TRAIN/VAL/OUTER label gates and complete seal declarations, exact AP /3 thresholds, clipped physical/SNP joins, and zero-vote infinite NLL. Synthetic seal declarations used fabricated fixture names/hashes solely to test gate shape; no actual seal was claimed.

The two already completed parents' observed inference/roster JSON metadata also passed. Their actual tree hashes were carried as metadata bindings; tree bytes, genotypes, scoring NPZs, target arrays and checkpoints were not opened. Original roster hashes matched inference receipts, strict ARGMix ID correspondence passed, both complete SNP coordinate maps/retained grids passed, and the34 older donors were retained. These are schema/source checks, not preparation or baseline results.

All thirteen pinned helper/author source files were rehashed and matched config. An AST check confirmed no Torch/TensorFlow import. Python ran only through source scripts/project_env.sh and .runtime/env/bin/python -B on anogena. No installs, fits/inference, GPU, scheduling/storage audit or Git operation occurred. The user1TB policy override remains root-owned and this contract adds no storage limit/reservation.

To repeat the pure handmade verification from the repository on anogena:

~~~python
from pathlib import Path
from arggnn import full_panel_native_baseline_contracts_v1 as c
split = Path("research/reviews/full_panel_native_comparison_split_v1/split.json").read_bytes()
policy = c.validate_policy_bytes(Path("configs/full_panel_native_baselines_v2.json").read_bytes(), split)
print(c.run_handmade_checks(policy, split))
~~~

Run that snippet with the project environment sourced, CUDA_VISIBLE_DEVICES empty and .runtime/env/bin/python -B. The receipt binds module/config/split hashes, public function signatures/API SHA256, all individual check outcomes and completed observed-metadata summaries.
