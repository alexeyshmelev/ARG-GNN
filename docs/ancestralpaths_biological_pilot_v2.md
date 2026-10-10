# AncestralPaths biological pilot v2: realized source-quota rows

Status: source prepared, no fit executed by this agent. H020 v2 is an explicitly approved new condition after the strict v1 TRAIN sampler stalled before any optimizer fit. The source-native Dense model, five epochs, Adam, batch32, seeds101/202/303, observed features and all scoring functions remain unchanged. This is not the paper's stated CNN or an exact quantitative reproduction.

## Preserved v1 outcome and narrow change

V1 execution receipt SHA256 is `ef24a030ccb84c88569a5507961f8dfc5014ad32d1aee343fdf73fdcc4348baf`. It has `seeds={}` and `biological_fit_ran=false`. The failing TRAIN parent is `bio_paths_pilot_train_001`, cohort Neo: 12,500 Ana rows and11,231 WHG rows, totaling23,731 of25,000; shortfall1,269. There are19,992 distinct site/query examples and3,739 source-style replacements.

The native initial stride is3 and later strides are3×(3/2)^p. For integer site IDs, pass p admits multiples of3^(p+1). At pass8, the numerator19,683 exceeds the10,147 observed sites, leaving only site0. That pass adds zero because no uncapped WHG row is available there. Future native passes cannot fill this quota. This diagnoses the declared stride schedule on this parent; it does not claim that every possible all-site sampler lacks WHG.

V2 changes only the handling of an unfilled quota: retain every genuine row accepted under the unchanged source schedule/caps/replacement, disclose the shortfall, continue subsequent cohorts including Yam, then fit only realized rows. It does not raise caps, reset class counts, switch site grids, pad phantom zeros, delete native classes, or select VAL inputs. Zero initial strides and other actual errors remain failures. The existing128-pass/no-progress stopping safeguards remain disclosed.

Requested per-parent budgets remain present_day50,000, Bronze_Age50,000, BAA12,500, Neo25,000 andYam25,000, totaling162,500. The caps remain12,500 per non-BAA path and6,250 per BAA path. Each cohort records candidate budget, initial and later strides, cumulative six-path counts, actual/unique rows, duplicate replacements, quota shortfall, stopping reason and whether the target filled. V1 stopped before train_001/Yam, so its v2 realization is unknown until root executes. A total323,731 rows would result if that Yam target fills and other observed v1 outcomes repeat; the actual v2 total is always recorded.

## Frozen helpers and observed cache reuse

`scripts/train_ancestralpaths_biological_v1.py` remains unchanged at SHA256 `826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056`. V2 imports this hash-bound definition module for source inspection, the native model, the separate truth reader, prediction sealing and all metric functions. V1 code/config/docs/results are read-only inputs.

Reuse all four feature NPZs from `checkpoints/ancestralpaths_biological_pilot_v1/feature_cache/`. Bind each cache SHA256, its metadata SHA256 and the preserved execution receipt. Require successful first-tree native float32 equivalence, no feature truth access, exact observed tree/manifest identity, explicit actual inferred node order, original query order, physical SNP join, retained original grid indexes, clipped bin edges, coverage and local offset. No unchanged feature extraction or equivalence test is rerun.

The completed Relate manifest is pinned to SHA256 `6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28`. All280 observed haplotypes and120 explicit queries retain their frozen roles/cohorts. Ancient observed self-inclusion stays disclosed. TRAIN supervision may determine source class-stratified examples; it never enters the observed feature bank.

## Unchanged evaluation and receipts

Use exactly the v1 prediction/scoring function. Raw six probabilities and C4 probability sums are written and hashed before the VAL truth reader is opened. The C4 collapse sums1+5 and2+6 before argmax. Native confusion remains predicted rows/truth columns. Exact C4 intersections use the unchanged retained-center clipped physical100bp bins and exact scored query-bp denominator, with original2Mb normalization and coverage/support reporting. Uncovered tails and edge fragments without retained centers are excluded by observations alone.

Present64 is primary and ancient56 secondary, with all-query and native per-cohort results preserved. Dosage, calibration, tract lengths/censoring and boundary metrics are mechanically unchanged through the frozen import. No VAL checkpoint/early-stopping/calibration fitting or final TEST analysis is added.

New output defaults to `checkpoints/ancestralpaths_biological_pilot_v2/`. Each parent receives a complete realized selection NPZ and sampler receipt; source targets, actual counts and shortfalls are recorded before fitting. Each seed retains the final epoch5 checkpoint, optimizer settings, native weight change, all raw probabilities and per-parent metrics. The execution receipt links the v1 failure, cache/source identities and new repeat reason. Failures are preserved; quota shortfall is a named method condition rather than a silently corrected success.

## Root execution

The repeated-condition rationale is H014 native CPU API qualification plus H020 v1 source-schedule failure before fitting. V2 completes a new biological fit condition while reusing answered extraction work. Root owns the ledger entry, audit, execution and Git.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python \
  scripts/train_ancestralpaths_biological_v2.py --check-only

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python \
  scripts/train_ancestralpaths_biological_v2.py \
  --output checkpoints/ancestralpaths_biological_pilot_v2
```

The small check-only command reads frozen source/evidence and observed cache arrays, creates no outputs, performs no extraction, imports no TensorFlow and opens no external truth. Root's actual fit is separate. The preparing agent executed no fit, installation, GPU job or Git operation.
