# Full-panel AncestralPaths discovery data preparation v1

This prepares one common ancestry dataset for our whole-tree candidate and the genuine AncestralPaths, released ARGMix and ARGformer competitors. Native AncestralPaths six-path scoring remains an author-workflow endpoint. **Explicitly derived C4 is the common comparison endpoint**, including BAA, on the same post-population-size Relate observations and matched physical support. This is not an AP-only comparison or an exact replication of the published CNN.

Owned files are [config](../configs/ancestralpaths_full_panel_discovery_v1.json) and [producer](../src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py). This implements data preparation from the [benchmark plan](../research/reviews/original_paper_benchmark_next_v1/plan.md), with common C4 competition preserved. No model, native inference or stochastic simulation ran while preparing these files.

## Frozen scientific definition

The source is AncestralPaths commit 77db3cfae100641c35218b87e1451e14f3074666, byte-identical to the paper-cited accepted Zenodo 8319452 archive. Exact raw seeds/bytes and the paper CNN are unavailable. The archived executable classifier is Dense, as established in the [source handoff](ancestralpaths_CNN_source_handoff_v1.md). Code has no named licence in the inspected archive. Existing Relate 1.2.4 academic-use authorization is retained.

The roster retains every author diploid, original within-cohort age-file order, and the author's simulation order:

| Cohort | Diploids | Haplotype slots | Role |
| --- | ---: | --- | --- |
| present_day |91 |0–181 |query |
| Bronze_Age |81 |182–343 |query |
| BAA |7 |344–357 |query |
| Neo |151 |358–659 |query |
| Yam |13 |660–685 |query |
| WHG |51 |686–787 |pure reference |
| EHG |43 |788–873 |pure reference |
| Ana |25 |874–923 |pure reference |
| CHG |14 |924–951 |pure reference |

Totals are 476 diploids / 952 haplotypes, 686 queries and 266 pure references. Present 91 is hardcoded at age 0; the provided GBR age rows are ignored, as in the source. Ancient focal self-inclusion remains available to native features. References older than census 350 are retained. Native names are tsk_0 through tsk_475; independent sample identity includes parent_id, sample_id and haplotype.

The original 11-population demography is preserved, including all Ne values, Bronze 166 / Yam 177 / Neo 259 / BAA 169 pulses, source mixture fractions, census 350 and splits 600/800/1500. Each parent is 2 Mb, r1e-8 and mu1.25e-8. Explicit current msprime defaults are Hudson, JC69, discrete coordinates and diploid SampleSets. Preserve the source's single 91-diploid present SampleSet followed by individual ancient diploid SampleSets.

Root 101 defines three TRAIN and two discovery-holdout parents. All ancestry, mutation, initial-Relate and size-helper streams are stored in the config before generation. The size helper uses base+1..5 and reuses base+5 for its final branch re-estimation. No stream is shared across parents or purposes; the intentional final-size reuse is declared. The native scripts had no supplied seeds. The earlier overlapping TRAIN 3,5,6 / TEST 5–9 split is corrected. There is no VAL, original-paper TEST claim, favorable-parent replacement or automatic retry.

## Data separation and common endpoints

Generation stores census ancestry and six-path/C4 targets only under truth/. Mutation occurs on a census-stripped genealogy using the existing guarded helper. This explicit observation adapter removes census-only topology/annotations before mutation. It creates a new versioned realization, not the author's original bytes. No cleaned oracle tree is exported.

Native paths 1–4 are non-BAA Ana/CHG/WHG/EHG; paths 5–6 are BAA Ana/CHG. The C4 class order is Ana, CHG, WHG, EHG, with explicit project spelling alias ANA→Ana. Native probability collapse is (p1+p5,p2+p6,p3,p4), before argmax. C4 is a derived ontology, not the paper's separately trained BAA-excluded model.

Observed files contain only phased genotypes, cohort/date/role/reference metadata, an exact SNP/VCF coordinate join, a constant map and observation-defined 100 bp scoring coordinates. Queries have null observed reference classes. Truth breakpoints never define model windows or evaluation-coordinate selection. No feature bank, sampling, fitting or scoring is performed by this producer.

All competitors use the same full observed panel; their released representations/reference selection/objectives/readouts must remain method-specific. ARGMix's current 15-reference/log-distance model is a released-workflow condition rather than April replication. ARGformer on these European Relate trees remains distinct from its original American tsinfer/tsdate task. Preserve observed-only MLM/reference-label supervision, genuine native retrieval and hard-vote zero probabilities without invented NLL floors. Root must freeze each full-roster method adapter before fitting.

Common metrics use all 686 queries, equal-parent matched query-bp accuracy/NLL, each cohort and all-query summaries, with present_day 182 reported separately. Use fixed 100 bp centers, clipped bin intersections and the exact common denominator. Report excluded tails, attrition, empty inferred support and failures. Native six-path SNP/focal sample accuracy and predicted-row/true-column precision remain separate. Holdout targets open only after all predictions are sealed. Discovery results can inform the next design; confirmation requires fresh parents.

## Native sampler realizability at 2 Mb

The released TRAIN formula is step=floor(S/((candidate_scan_budget+10000/3)/H)), where S is the final post-population-size inferred site count and H is that cohort's query haplotype count. The candidate_scan_budget is 100k/100k/25k/50k/25k for present/Bronze/BAA/Neo/Yam. It is a per-parent scan budget, not a denominator of 50k or 100k sites. HOLDOUT uses step=floor(S/(20000/H)).

| Cohort | H | Minimum TRAIN S for step≥1 | Minimum HOLDOUT S for step≥1 |
| --- | ---: | ---: | ---: |
| present_day |182 |568 |110 |
| Bronze_Age |162 |638 |124 |
| BAA |14 |2024 |1429 |
| Neo |302 |177 |67 |
| Yam |26 |1090 |770 |

The maximum TRAIN requirement is 2024 inferred SNPs; HOLDOUT requires 1429. Thus 2 Mb is not intrinsically stride-impossible. At an illustrative S=20000, TRAIN strides are 35/31/9/113/18, all positive. Actual variant counts have not been generated or measured.

The config freezes these exact rational denominators and thresholds. The post-size inference receipt checks actual S using observations only, before any sampler/heldout target access. A zero stride fails the cell, preserves the receipt and does not change the formula, length, grid or parent streams. If the fixed cell fails, a longer-length experiment needs a prospectively frozen new version for all parents, never favorable replacement of the failing parent.

Positive strides do not establish that class quotas can fill. Keep native caps and repeated-pass schedule; the already declared v2 realized-row shortfall policy is a derived sampler condition. No native sampler is executed or silently adapted by this producer. Source evidence is training.py lines 29–53 and testing_bygroup.py lines 78–84/108–111 at the pinned source.

## Native Relate path

Relate reads a strict nine-file observed whitelist. It never opens truth/, generation.json or any oracle. For a separate relate stage, root admission must bind the nine observed file hashes. For an all stage, the just-generated observed hashes bind the inference input in memory. Original observed bytes are checked again after inference.

The chain preserves ConvertFromVcf → GenerateSNPAnnotations → RelateParallel (N 30k, mu1.25e-8, ordered actual ages/map, eight threads) → five EstimatePopulationSize iterations (eight threads) → ConvertToTreeSequence. The primary input is the final post-size tree. Existing native executables, helpers, installation and licence provenance are pinned. The author-native threshold 0.5 is retained, unlike the reduced pilot's disclosed threshold 0. --noplot omits plotting; years-per-generation 28 is explicit; mutation-rate normalization stays off and final branch re-estimation runs. The native chunk memory argument is 2 GB; it is not an assertion about total resident memory.

Existing mechanical adapters retain phased biallelic segregating nucleotide sites with MAC ≥ 1, count all attrition, normalize converter ID2=NA without reordering, and change only the map header. Synthetic contig 22 asserts no empirical chromosome/build/map. VCF coordinate is original SNP bp0 + 1. Conversion subtracts one and clips to first retained SNP through last retained SNP+1. The receipt retains coverage, local-to-original offset, every site join and actual flagged inferred sample IDs. Nodes/metadata are cleaned without changing inferred node order. Uncovered tails are not extrapolated. Native .anc/.mut/.trees and command logs are preserved.

## Root execution and output admission

The default CLI validates source/config/runtime/tools without writing outputs. The minimum read-only check is:

~~~bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python \
  -m arggnn.prepare_ancestralpaths_full_panel_discovery_v1 --check-handmade
~~~

This passed: full roster/split/stream validation, native six-path→C4 mapping, exact stride boundary checks at 2023/2024 TRAIN sites and 1428/1429 HOLDOUT sites, a tiny hidden-truth counterfactual with identical cleaned observations, and noncontiguous inferred sample IDs. It made no files and called neither sim_ancestry/sim_mutations nor native tools. Runtime is Python 3.11 / msprime 1.3.4 / tskit 1.0.3 / NumPy 2.4.6. Full-parent memory, timing and numerical outputs remain unqualified.

Each --execute invocation handles exactly one fixed parent. Use --stage simulate first; relate consumes existing observed output; all chains both stages. Root registers the new full-panel condition, supplies CPU runtime/resource admission, freezes hashes and starts execution:

~~~bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python \
  -m arggnn.prepare_ancestralpaths_full_panel_discovery_v1 \
  --parent ap_full_train_000 --stage simulate --execute \
  --expected-config-sha256 <root-reviewed-config-sha256> \
  --expected-source-sha256 <root-reviewed-producer-sha256> \
  --admission <root-owned-admission.json>
~~~

Admission must contain job_id, parent_id, allowed_stages, output_directory, exact source_ref/config_ref/runtime_ref objects, and the active root storage record. The output_directory must be data/ancestralpaths_full_panel_discovery_v1/<parent_id>. The reservation binds those refs plus the pinned storage components. A relate-only admission additionally contains observed_input_refs, a filename→{path,sha256} dictionary for all nine whitelist files, each bound in reservation.consumer_bindings. Runtime_ref names root's matching CPU qualification record; no installation is requested here.

The producer uses **project_storage_full_reservation_v1.budget_check_v9** at nested-file/phase boundaries. It does not use retry-v3 or inspect global processes. Root should reserve the conservative 4,000,000,000-byte per-parent ceiling. Planned native phase growth has 1,000,000,000-byte headroom; recursive own-scope checks fail on overflow. Native TMPDIR is inside that parent's relate/tmp scope. Phase checks do not claim continuous monitoring or a measured memory bound. Native steps time out after 7200 seconds and stop their newly created command group on timeout/interruption. Root owns scheduling.

A fresh simulate/all call writes freeze.json containing all five streams before truth generation, then observed/, truth/, generation.json and stage completion. A relate-only call checks the same freeze, writes relate/inputs, native files/logs, cleaned post-size trees, observed sample mapping and inference.json. Failures preserve owned partial output and attempt a guarded failure receipt; no other scope is removed or altered. The producer performs no GPU operation, model fitting, package installation or Git operation.
