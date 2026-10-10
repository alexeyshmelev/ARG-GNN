# Source-native AncestralPaths biological CPU pilot v1

This is an H020 runner prepared for the four new biological development parents. It has not performed biological fitting. It uses the already-qualified isolated TensorFlow CPU prefix and the unchanged source-native Dense bridge. It is not a reconstruction of the paper's stated CNN, an exact quantitative reproduction, or a final TEST analysis.

## Frozen fitting method and explicit adaptations

The actual pinned `training.py` is SHA256 `13b4f30ab0df75fc461b764ae2cdbd1fdc071ff45200e6401e84faa565508e4f`. Lines 168-177 reshape to 5x9 and fit Flatten, Dense512/ReLU, Dropout0.2, Dense6/softmax with Adam and categorical cross-entropy for five epochs. Batch32 and shuffle=True are Keras `fit` defaults, made explicit here. There are 26,630 parameters. Model seeds are 101, 202 and 303. The final epoch5 model is retained; VAL does not choose a checkpoint, stopping rule or calibration transform.

The TRAIN sampler follows source lines 31-35 and 50-156: observed site IDs in increasing order, a cohort-dependent initial stride, then sample order, path-specific caps, and repeated passes with `step = step + step/2`. The source can reuse the same site/query example on later passes. This runner preserves replacement and reports unique examples, duplicate counts, each pass's stride and realized path counts. It adapts hardcoded 952-haplotype cohort ranges to the frozen observed 280-haplotype manifest and its 120 query roles, with two independent TRAIN parents instead of author parents 3,5,6.

Per TRAIN parent, the native targets are present_day50,000; Bronze_Age50,000; BAA12,500; Neo25,000; Yam25,000, totaling162,500. The path cap is12,500 per non-BAA path and6,250 per BAA path. Original per-parent candidate-scan budgets and the source's 10,000/3 overhead are retained when calculating the initial stride with actual cohort size. Expected total fit rows are325,000 if every quota fills. TRAIN labels may legitimately stratify supervised examples. They never enter features. A zero stride, quota stall or the disclosed128-pass bound is recorded as failure; caps are not silently relaxed and missing classes are not deleted.

The source's inconsistent default `-nn=4` conflicts with its hardcoded 5x9 reshape. The runner retains the already-reviewed five-row native bridge. It avoids source preallocation/unfilled-row behavior by fitting only actually selected rows. No original author file is edited or executed.

## Observed-only inputs and efficient exact native features

TRAIN and VAL require completed Relate-derived observations, with explicit tree hashes and sample-index-to-inferred-node joins. `cleaned_oracle.trees` and truth/census paths are rejected as primary inputs. Features use only inferred parent/child relationships and actual leaf membership, eight declared observed cohorts, and inferred ancestor ages. Columns are Bronze_Age, BAA, Yam, Neo, WHG, EHG, Ana, CHG, age/1500. Present-day leaves are excluded from reference shells, ancient focal self-inclusion is retained and disclosed, empty shells are skipped, and root rows are padded with-15. The training integer-age convention is used for both fitting and prediction.

The cache computes eight-cohort descendant counts once per observed tree, extracts each unique first-ancestor path once within that tree, and deduplicates the resulting informative shell/age signatures across trees. It preserves the native float64 repeated-add normalization before conversion to Keras float32. Grid and actual inferred observed-site coordinates map to compact cached feature indices. It therefore avoids walking each full subtree for all20,000 coordinates x120queries x5rows. Prediction evaluates each cached feature vector once and restores exact physical/query order. A single actual first observed tree is compared with the unchanged bridge, requiring identical float32 features for all120queries before cache completion. Cache provenance records this equivalence check, tree/manifest hashes, counts and no truth access.

The external truth reader is a separate capability. TRAIN opens it only for supervised label sampling/loss. VAL opens it after raw observed predictions have been written and hashed. It reads full truth interval labels, never supplies features, and never changes VAL/query eligibility or site/bin inclusion.

## Finalized post-Relate manifest contract

Default path: `data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json`. This is the native output of the frozen [`infer_biological_paths_relate_v1.py`](../scripts/infer_biological_paths_relate_v1.py), whose code/config/doc hashes are bound in the fitting config. No provisional wrapper schema is used.

Require top-level `kind=biological_paths_relate_v1`, `status=four_development_parents_inferred_pre_and_post_five_iterations`, `primary_condition=post_five_size_iterations_primary`, and exactly the four `replicates`. Each parent must have `status=inferred_both_conditions`, its correct split and `truth_or_oracle_read=false`.

Read the hashed `inferred_tree` for `post_five_size_iterations_primary`; the separately hashed `observed_sample_manifest`; its separately hashed `source_manifest`; all280 `sample_index_to_inferred_tree_node_id` mapping objects; and hashed `completion_provenance` / `commands_provenance`. Original roster SHA is checked against the frozen dataset source roster. Derived roster SHA is checked against its own manifest, and every original observed attribute is preserved while the actual node-ID fields are verified against the explicit mapping. The two roster hashes are not required to match.

Use `coordinate_contract` and the matching `post_five_iterations` receipt. The cleaned tree is LOCAL `[0,coverage_right-coverage_left)`, with `physical_original = local_tree_position + coverage_left`. Grid centers and scoring bins remain original physical coordinates; tree intervals are shifted by the exact offset during cached feature selection. Actual local inferred SNP positions have the offset added before TRAIN truth-label joins and are checked against the producer's physical site join.

Retain exactly `retained_fixed100bp_indexes` from the hashed original20,000-center grid. Clip the corresponding original100bp bin edges to `coverage_original_bp0`. Do not fill a partial covered edge bin if its center is outside coverage. Retained bins therefore define the exact prediction support, which can be narrower than inferred coverage. Keep original grid indexes, the local offset, declared coverage, represented bin support, and2,000,000bp normalization in caches and raw prediction files.

TRAIN requires sites on the actual post-size inferred tree sequence because the author sampler uses `tree_rel.sites()`; it does not substitute oracle sites or invent an alternative sampler.

## VAL endpoints

The two VAL parents retain exactly the declared original-grid centers inside post-Relate coverage and all120 explicit queries. The source grid has20,000 fixed100bp bins; uncovered tails and partial edge bins without retained centers are excluded using observed coverage alone. Primary results use the64 present-day query haplotypes (32diploids). The56 ancient query haplotypes (28diploids) are a named secondary analysis, including observed ancient self-inclusion. All-query results and native per-cohort results are also saved. There are zero TEST parents.

Raw six-path probabilities are retained in each sealed prediction NPZ. Native argmax accuracy and six-axis confusion use the author orientation: predicted rows, truth columns, unweighted retained fixed midpoints. The C4 endpoint sums probabilities first: `(p1+p5,p2+p6,p3,p4)`, then takes argmax. It does not collapse the native argmax labels.

C4 predictions are piecewise constant over the retained physical100bp bins, clipped at the declared coverage edges. Metrics integrate exact intersections with complete external truth intervals on precisely this retained-bin support, including truth switches within a prediction bin. The denominator is exact scored query-bp. Report original2Mb normalization, physical coverage and represented-support fractions separately; no uncovered tail is extrapolated or treated as a prediction. Per parent and seed, save query-bp accuracy, NLL, Brier, supported-class macroF1/balanced accuracy, truth-row/prediction-column query-bp confusion, class spanIoU, exact span-weighted raw-probability calibration/ECE15, and paired-diploid dosage MAE/MSE. No probability floor is used. A zero true-class probability records infinite-NLL query-bp and a null NLL rather than fabricating a finite value.

Resolution-limited predicted tracts extend only to represented-bin support endpoints. Clip merged full truth tracts to that same support, retain their original physical endpoints/lengths, and disclose support censoring. Save per-query visible length summaries, one-to-one matching of internal boundaries at1,000bp, and exact correct-class overlap for tracts whose original full truth length is<=10,000bp. Coverage/support endpoints are never counted as ancestry switches. Parent/seed cells remain separate descriptive development results; queries/bins are not independent experimental units. No pooled scientific promotion, oracle-primary result or holdout claim is generated.

## Execution and evidence

Only root runs fitting. All outputs and caches remain under a fresh repository-local output directory. No installations, GPU access or author CLI execution are needed. From the repository root:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python \
  scripts/train_ancestralpaths_biological_v1.py --check-only

# After the completed inferred-observation manifest is supplied:
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python \
  scripts/train_ancestralpaths_biological_v1.py \
  --observed-manifest data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json \
  --output results/ancestralpaths_biological_pilot_v1
```

`--check-only` is the small actual source/schema check: it opens no truth arrays, imports no TensorFlow, performs no cache extraction or fit, and creates no output. If a completed observed manifest is present it also checks that explicit interface. Scientific execution retains cache receipts, sampler selections/counts, model checkpoints/optimizer settings, all raw probabilities, per-parent metrics, failures.jsonl and receipt.json. Stage failures remain visible and are not repaired through fallback inputs or relaxed quotas.
