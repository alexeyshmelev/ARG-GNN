# H058: one useful alignment test, with a narrow claim

**Retain the already registered matched H058 comparison after its existing prerequisites. It tests whether explicit same-SNP reference alignment improves tract discrimination. The inspected source does not introduce a distinct operator for ordered graph collections, and no priority or predictive gain is established.** No new arm, tuning sweep, budget, experiment or execution is recommended.

## Exact trained-baseline limitation

This is the European C4 H045 own-state condition:32 full2Mb TRAIN parents,8 reused development VAL parents,952 observed haplotypes,686 queries and266 references. H058 starts from completed H045 own-state pass30. Its continuation control is the unmodified H045 arm already trained within H050,not H050's added cut-memory arm.

The baseline computes19 channels at each haplotype/SNP:8 donor-group log frequencies for that haplotype's allele,8 call masks,panel log frequency,entropy and call indicator. An MLP then mean/max pooling reduces them separately for each haplotype/native hold BEFORE tree interaction (mutation_routed_whole_tree_gnn_v1.py:281–341). H045 retains the first complete up/outside sweep,clade/episode injections,second complete sweep and full genomic BiGRU (ordered_clade_episode_gnn_v1.py:374–403;backbone:277–279).

The bottleneck is **no explicit query/reference SAME-SNP carrier comparison before pooling in that feature route**. The five branch-support fields are symmetric split-concordance summaries. H050 memory acts on already formed states. This does not prove the complete baseline lacks related information:topology,branch support,reference classes/ages and pooled SNP features encode correlated evidence indirectly. H058 adds a computation/inductive bias from existing observations,not an independent data modality or an expressivity impossibility theorem.

Earlier H045 foreign-episode results motivate the question:Present182 parent-mean accuracy96.8270±0.5870%,boundaryF1 at1kb18.1558±1.3790%,and aggregate905 matched,3103 missed,5063 extra boundaries across8 VAL parents (next_useful_biological_mechanism.../REPORT.md:7–11). These are earlier development results for the foreign-episode arm,not fresh measurements or the matched continuation's scores. Weak boundary recovery does not identify missing alignment as its cause. H058 retains native-hold-constant output;it does not test finer within-hold resolution or explain H057's representability gap.

## Actual mathematics and novelty limit

For tree/hold t,define

    W_t(q,r) = product over path(q,r) edges e of gamma_t(e), 0<gamma_t(e)<1.
    N_qsg(a) = sum over called references r in group g of W_t(q,r) 1[allele_rs=a].
    D_qsg = sum_a N_qsg(a).
    z_qsg = N_qsg(allele_qs)/D_qsg - f_sg(allele_qs).

The learned gate uses9 geometry/support fields and first-sweep states of two endpoints,once per native hold. The code propagates8 source×age groups×4 allele slots using ordinary additive recipient-excluding tree sweeps,then gathers the query allele BEFORE pooling (same_marker_packed_csr_v2.py:342–389,462–474). It appends8 relative log masses and8 availability channels,mean/max pools24 features to48,and applies48→8→32 with zero output initialization (411–459). The714 extra parameters are74 gate+384 input+256 output;declared total32526.

This is learned positive unique-path weighting,normalized allele matching,centering,pooling and a residual. Each new statistic is computed within a tree and injected before H045's second sweep (integration_hooks_v3.py:61–78;the v2 numerical body is retained). Across-tree processing remains the inherited episode machinery/BiGRU. H058 adds no tree-to-tree carrier,shared-event posterior,donor-continuity state or graph-collection transport. Its exact composition may be a useful domain specialization;that does not establish a new primitive or historical priority.

Direct mathematical limits:

- Equal within-group path weights,monomorphic groups or one called donor give zero allele-frequency contrast. Mass/availability may remain useful.
- A factor shared by every donor in a group cancels in N/D. The query pendant-edge factor changes mass but cannot change contrast. Gate parameters need not be uniquely identifiable from ancestry performance.
- Initial gamma=.95 gives hop-count attenuation. Unary subdivision changes the product. This is not an evolutionary transition probability,a calibrated ancestry posterior or a representation-invariant branch metric.
- Matching is identity-by-state support,not identified descent or certified population membership at350 generations. References remain proxies.
- Pooling discards within-hold SNP order and donor identity. The original genomic BiGRU supplies order at the retained hold scale.

| Existing primary material inspected | Established overlap and limit |
|---|---|
| Ralph,Thornton & Kelleher,2020,[Efficiently Summarizing Relationships in Large Samples](https://doi.org/10.1534/genetics.120.303253),cached full text:vector sample weights/Definition3 | Vector sample/allele accumulation and tree-sequence statistics are established. Its ordinary site statistic is tree-independent,not literally H058's learned query-specific path kernel. |
| Browning etal.,2023,[FLARE](https://doi.org/10.1016/j.ajhg.2022.12.010),cached full Methods:HMM emissions | Query/reference allele matching and ordered reference-copying ancestry inference are established. FLARE retains ancestry×donor history;H058 marginalizes donors per SNP. |
| Whitehouse etal.,2024,[msae223](https://doi.org/10.1093/molbev/msae223),cached full text:GCN architecture/Fig.1 | Whole marginal-tree GNN plus node/tree-sequence GRUs is established. Relate's missing internal-node correspondence is explicit. Tasks/readout differ from H058. |
| [Matching Networks](https://arxiv.org/html/1606.04080),Eqs.1–2;[TPN](https://arxiv.org/html/1805.10002v5),Eqs.3–5 | Existing repository primary-excerpt receipt records learned positive support weighting/shared graph label propagation. This task read that receipt,not newly retrieved full papers. Attention/all-walk operators are not exactly H058's centered unique-path allele statistic. |

The same-marker review,H050/H051 priority audits,and interval-tree audit consistently distinguish biological utility from established propagation/sequence ingredients. H058 supplies no new mathematical distinction overturning them. Exhaustive priority for the exact composition remains unresolved. Do not relabel conditional memory,tree diffusion,GST feedback or this residual as a newly invented general operator.

## What the ONE registered experiment can establish

Keep aligned,alignment-null,and exact H045 continuation. The two new arms share the H045 initializer,714 extra parameters,new weights,and fresh independent AdamW;each receives960 scheduled complete-parent updates,30 passes,seed101,FP32,unchanged exact-span loss and6 checkpoints. All inherited weights remain trainable. No early stopping or disposable qualification updates are added.

The null permutes complete reference diploid sequences within source×younger/older groups once per parent,constant across SNPs. Original backbone/gate inputs stay aligned. It preserves group allele/call counts and each moved sequence's LD,but alters branch attachment,exact age within family,and spatial callability. Aligned-versus-null tests that combined ADDITIONAL alignment,not pure topology or exact dates alone. Older Ana has one diploid and cannot move. Moved identity does not prove informative allele evidence changed.

Continuation controls additional training of the original body;aligned-versus-null controls extra capacity/budget. A sham may also be harmful,so both contrasts are necessary. A win remains one-seed discovery evidence on8 repeatedly reused development parents,not independent confirmation,priority,calibration or broad biological generalization.

Select by frozen equal-parent Present182 accuracy,then NLL,then earlier pass—not boundaryF1. Registered utility needs mean paired boundaryF1 gain≥1.00percentage point over BOTH controls,span-accuracy loss≤0.05point against either,and no mean NLL regression. Current source additionally requires available/nonworsening uncensored true-short-tract recall for positive utility (producer_v1.py:388–424).

The source requires96 NEW products (2arms×6checkpoints×8parents) sealed before new truth. Approved old CPU-control reuse preserves its historical6-checkpoint/8-parent pretruth selection separately. Explicitly authorized same-backend frozen-control replay gives a joint144-product barrier. The earlier four-arm marker-feedback design's192 count is not H058's count.

## Actionable feedback within the registered experiment

1. **Resolve continuation and diagnostic completeness before the pilot.** CONTINUATION_REUSE_AUDIT.md supports historical TRAIN identity and identifies old CPU-selected pass20. Actual ROOT source/data/target/runtime/inference/selection attestation remains necessary. Old CPU metrics may lack H058 short-tract diagnostics:paired_criteria_v1 correctly leaves utility UNKNOWN if absent,even when accuracy/F1/NLL gates pass. Use only the existing separately authorized diagnostic/control-replay route if required. No silent retraining,reselection or automatic replay follows here.
2. **Report the frozen null's strength.** Retain its single seed and per-group moved/called support. Do not reseed immovable/weak strata. Identity movement can overstate effective allele perturbation;where informative change is unresolved,qualify a negative mechanism conclusion.
3. **Keep the biological endpoint interpretable.** Report parent-paired deltas/sampleSD in percentage points,all7 cohort rows,precision/recall,event counts,typed transitions,and existing≤1/2/5/10kb tract strata/edge censoring. Separate genuine short-tract recovery from false-tract deletion. Pooled counts and equal-parent meanF1 are different summaries.
4. **Separate arithmetic/fit from utility.** The latest registry amendment reports corrected one-hold FP32 PASS with0 optimizer updates. That does not establish full-parent fit or efficacy. Retain exact full952/686/all-sites/all-vertices binding and the first scheduled update's measurement. Do not crop data or detach history.
5. **Stop instead of searching.** Losing either control,capacity-only benefit or damaged real short-tract recall ends this condition under its frozen interpretation. A convincing pass can motivate later independent confirmation under separate authority;it does not authorize gamma/rank/seed/threshold sweeps or a novelty label.

Source,registration/review metadata and cached primary text only were read. No scientific import,observation/model/checkpoint/truth/prediction payload access,process/device/resource/guard probe,job,installation,accounting,Git action or edit outside this fresh review directory occurred.
