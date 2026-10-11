# Next original-paper ancestry benchmark: full-panel AncestralPaths discovery

**Recommendation:** run one fresh, single-seed, full-panel 2 Mb AncestralPaths comparison. Use the archived author's demographic/observation recipe and actual six-path Dense classifier, with a disjoint parent split and the disclosed realized-source-schedule sampler. Compare it with a prospectively named six-path whole-tree arm. This is a **released-author-workflow benchmark with declared derived sampling/split/model conditions**, not an exact replication of the paper's CNN, original bytes or reported accuracy.

The public source is unusually actionable: the paper-cited [Zenodo 8319452](https://doi.org/10.5281/zenodo.8319452) and [commit 77db3cfae100641c35218b87e1451e14f3074666](https://github.com/AliPearson/AncestralPaths/tree/77db3cfae100641c35218b87e1451e14f3074666) contain 17 byte-identical files. Their runnable model is Flatten → Dense512/ReLU → Dropout0.2 → Dense6/softmax, not a specified CNN. Publication/source discrepancies and provenance are established in the [native protocol](../../../docs/ancestralpaths_native_protocol_v1.md) and [CNN source handoff](../../../docs/ancestralpaths_CNN_source_handoff_v1.md). Apply the [evaluation contract's fidelity labels](../../../docs/paper_dataset_evaluation_contract_v1.md).

## Available assets and preparation still needed

| Available now | Missing or prospective |
| --- | --- |
| Cached source, supplied `example/mesoneo_ages_pop.txt`, constant map, simulator, Relate chain, native training/testing/painting scripts, article and Supplementary Note 3. | Original simulation bytes, author seeds/runtime lock, original CNN layer definition/config/checkpoint. |
| [Native bridge](../../../src/arggnn/ancestralpaths_native_bridge.py), guarded truth joins, six-path probabilities/metrics, qualified TensorFlow2.15.1/Keras2.15 CPU runtime at `.runtime/conda/envs/ancestralpaths_cpu_v1`. | Fresh full-panel simulated VCF/truth and post-population-size Relate observations for the new split. No verified full476-parent dataset is established by the reduced pilot. |
| [Reduced execution config](../../../configs/biological_paths_execution_pilot_v1.json), source-equivalent demographic events and prior Relate/feature preparation. [V2 sampler policy](../../../docs/ancestralpaths_biological_pilot_v2.md). | A new full-roster manifest/config and runner adapter. Existing config has140 diploids/280 haplotypes and fixed120 queries, so it cannot simply be reused as the native panel. |

Keep the source snapshot read-only. Its archive supplies no named code licence; preserve that provenance uncertainty rather than assigning the article's licence to the code. Existing reduced-panel fits are answered development conditions, not new full-panel measurements.

## One discovery cell

Freeze these fields before generating or inspecting ancestry truth:

- **Seed:** root101, with separately recorded ancestry/mutation child streams for each of five parents and model initialization/shuffling. The author scripts supply no seeds. These newly chosen seeds and recorded current dependency versions do not recover the original data.
- **Parents:** three fresh TRAIN parents and two fresh DISCOVERY-HOLDOUT parents, each2 Mb. Assign parent identities/streams before windowing. No VAL or early stopping: retain final epoch5. The release's TRAIN3,5,6 versus TEST5–9 overlap is deliberately corrected. Discovery holdout can inform the next protocol, so later confirmation must use new parents.
- **Panel:** full476 diploids/952 haplotypes in author simulation order: present91, Bronze81, BAA7, Neo151, Yam13, WHG51, EHG43, Ana25, CHG14. Score all686 admixed haplotypes; retain266 pure-source haplotypes and every observed sample in inference. Preserve within-cohort age-file order, hardcoded present91 at age0, and actual cohort-derived pulses (Bronze166/Yam177/Neo259/BAA169), all remaining source events, census350, r1e-8 and mu1.25e-8.
- **TRAIN rows:** retain source requested budgets per parent (GBR50k, Bronze50k, BAA12.5k, Neo25k, Yam25k), caps12.5k per non-BAA path/6.25k per BAA path, native site-ID stride schedule and repeated rows. Maximum requested total487,500. Prospectively adopt v2: retain only genuine accepted rows after its bounded no-progress/128-pass guard, disclose every shortfall, then continue other cohorts. Zero initial stride or invalid joins fail the cell. No phantom rows, new grids, rebalanced caps or favorable-parent replacement. Both methods receive the identical realized TRAIN row multiset.
- **HOLDOUT:** native observation-defined site/focal traversal,10k examples per five cohorts per parent (100k requested total), no class balancing. Derive the selected pairs without reading holdout truth; the source's test caps exceed the target and cannot affect selection. Record any failure/shortfall instead of filling zeros or drawing replacement parents.

This is a small data/seed cell, not a prediction that full-roster candidate memory or wall time is already qualified. Root first qualifies one TRAIN parent and one complete candidate tile, retains that work, then completes the predetermined cell. No scientific job was run to prepare this plan.

## Native observations, readout and scoring

Follow [author `relate.sh`](../../../.runtime/author_sources/ancestralpaths_77db3cfae100/relate.sh): ConvertFromVcf → GenerateSNPAnnotations → RelateParallel (N30k, mu1.25e-8, actual ordered ages/map, eight threads) → **five EstimatePopulationSize iterations** → ConvertToTreeSequence. Both methods consume the same final post-population-size trees, retained SNPs, coordinate map, cohorts and dates. Pin the actual Relate/runtime versions and full input/output hashes. Annotations contain observed cohort identities only.

Dense input stays5×9: Bronze, BAA, Yam, Neo, WHG, EHG, Ana, CHG, int(age)/1500. Preserve incremental non-recounted shells, skipped empty shells, present exclusion, ancient focal self-inclusion and -15 root padding. Fit categorical CE/Adam, batch32, shuffleTrue, five epochs. Mechanical rank/API fixes remain those already documented.

Targets come only from separately held simulated census350 ancestry at the exact inferred physical SNP coordinate: non-BAA Ana/CHG/WHG/EHG paths1–4; BAA Ana/CHG paths5–6. Truth/census nodes and truth-defined breakpoints never enter observations or model coordinates. Seal all six probabilities and sample/site/parent identities before opening holdout truth.

Primary comparison is paired six-path categorical sample accuracy, retaining the parent×cohort table and predicted-row/true-column confusion/precision. Use argmax of six probabilities, with no added smoother or cohort-dependent masking. C4 probability sums (p1+p5,p2+p6,p3,p4), physical-span scores and calibration are explicitly secondary derived endpoints. Two discovery parents do not establish robust uncertainty or biological superiority.

## Can our whole-tree model use these observations?

**Yes in principle; the current H032 checkpoint/driver cannot be plugged in unchanged.** [H032](../../../docs/reference_dependence_gnn_v1.md) binds Q120/R96, a four-output head and frozen H025 four-logit skip. Full952/Q686/R266 and native six-path supervision require a new adapter.

For this benchmark, name the candidate **AP6-DenseSkip whole-tree**, a disclosed hybrid: retain the H032 static whole-tree propagation and four observed pure-source donor classes, extend its readout to six paths, and replace H025's four-logit skip with pre-softmax six logits from the newly fitted, frozen native Dense model. Preserve all observed leaves/cohort/date information and current observed-age family/centering rules; report empty support and singleton-age centering effects. Do not turn six ancestry targets into donor metadata. The generic [IntervalLineageGNN](../../../src/arggnn/models.py) already has a class-parametrized head, but that does not qualify this H032 adaptation.

Prepare/freeze this adapter before the cell. Use complete observation-defined250 kb tiles, sparse gathering of the identical selected TRAIN rows, and duplicate multiplicities in categorical loss. Five complete tile epochs give120 tile updates across three TRAIN parents; retain final epoch5 without holdout selection. Record candidate batching, parameters and both the frozen Dense and residual fitting costs. This is matched supervision, not equal compute. Full-roster memory/timing requires actual qualification, with no dense FP64 donor matrix.

## Scale-up and deferred alternatives

After discovery, freeze architecture/sampler/metrics and expand to three200 Mb TRAIN parents and five additional independent200 Mb TEST parents, then independent root seed streams (e.g.202/303) under the same frozen protocol. Preserve complete planned coverage and parent-level results. This reaches the published sequence-length/split scale **while retaining the released Dense model and declared sampler**. It still differs from the paper's50k training-example statement and missing CNN. A separate50k sampling condition needs its own frozen selection rule; increasing length alone does not qualify exact-paper reproduction.

Do not spend this cell on April ARGMix or exact ARGformer: [ARGMix revision evidence](../../../docs/argmix_paper_revision_search_v1.md) establishes that the released15-reference/log-distance model is not the April25-reference/linear-distance model, and its original configuration/data remain missing. [ARGformer fidelity evidence](../../../docs/paper_dataset_evaluation_contract_v1.md) leaves original AmericanAdmixture30 Mb sample units, simulator/seeds, paths/reference corpus and LAI readout unresolved. Both remain useful separately named workflows, but AncestralPaths offers the most complete realizable original-paper-related next ancestry task.

Root owns experiment registration, full-panel configuration/adapter qualification, scheduling and execution. This file is planning evidence only.
