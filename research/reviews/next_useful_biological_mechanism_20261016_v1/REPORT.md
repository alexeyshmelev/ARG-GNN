# One conditional mechanism: marker-specific allele support on observed tree paths

Date: 2026-10-16. **Source-only recommendation: retain one narrow utility refinement of Hypothesis 2 in the existing next-architecture report. This is not a newly qualified operator or a new registered condition.** Do not implement or fit it while H050 evaluation is pending. H050 scored outputs were not opened.

The question is whether **which labelled reference carries the query-matching allele on which observed branch, at the same SNP**, adds useful evidence beyond source-wide allele frequencies, pooled mutation concordance and the current whole-tree predictor. Keep native-hold output resolution and the original full-parent BiGRU. Adding a marker-grid decoder at the same time would confound this question with the separately owned marker-feedback proposal.

## Evidence and biological claim

H045 foreign_episode reaches Present182 accuracy 96.8270 ± 0.5870% but boundary F1 18.1558 ± 1.3790% and recall 22.5890%. Across eight reused VAL parents there are 905 matched, 3,103 missed and 5,063 extra boundaries at 1 kb. Its accuracy increment over matched own_state is only 0.009568 percentage points. These are development observations from the saved residual-error report, not new measurements. They establish weak transition discrimination. They do not establish that a missing allele/topology interaction caused the errors.

The explicit SNP route first encodes source-frequency/callability tuples, then mean/max pools by hold before tree processing. The separate branch route pools discrepancies of allele splits without retaining source-specific, marker-specific allele carriers. The whole model can still encode correlated SNP and reference information indirectly. No claim that the complete baseline is invariant to reference-allele reassignment, or lacks LD, follows.

C4 is source population at 350 generations BP. A topology boundary does not identify a C4 switch. The 266 pure-source references are proxies with known observed metadata, not certified ancestors. Their older/younger split at 350 is an existing feature convention. Older references are not projected onto a census lineage. The 686 query labels remain unknown inputs. Cohort ages, pulse times and tract complexity are coupled in these simulations. This model must not impose a cohort-wide tract length or interpret its learned path attenuation as an evolutionary posterior.

## Exact proposed route

Start all arms from the same completed H045 own_state pass30 checkpoint and retain its SNP inputs, mutation support, episode adapter, every original tree vertex and sampled haplotype, two complete PyG sweeps, full genomic BiGRU, classifier and native-hold-to-100 bp mapping. All inherited weights remain trainable.

For each native hold, use a small shared Torch gate to compute one attenuation gamma_e in (0,1) per observed edge from its existing nine geometry/support channels and the first-sweep states of its two endpoints. Compute it once per hold, not per marker. Let g index the same eight observed source × age-family groups, and b index actual allele categories. At SNP s, seed a reference sampled node r by

    x_rs[g,b] = 1[group(r)=g, called_rs, allele_rs=b].

All query and internal-node seeds are zero. Two recipient-excluding additive tree sweeps over every original vertex compute

    m_(u→v)[g,b] =
        gamma_(u,v) * (x_us[g,b] + sum_(w≠v) m_(w→u)[g,b]).

Thus, at a query q,

    N_qs[g,b] =
        sum_(r in g, called_rs) W_qr * 1[allele_rs=b],
    W_qr = product_(e on path(q,r)) gamma_e,
    D_qs[g] = sum_b N_qs[g,b].

For a called query allele a_qs, form eight contrasts

    z_qs[g] = N_qs[g,a_qs]/D_qs[g] - f_sg(a_qs),

where f is the same marker's ordinary, unweighted, called-reference frequency within g. Include eight relative log masses log(D_qs[g]/n_called_sg) and eight availability masks. Missing query calls or genuinely absent called-reference groups produce zero features with masks, never an imputed ancestry. Queries are the only recipients of this new feature route, so no reference self-copy subtraction is needed.

Use FP32 log-domain rerooting for N and D. Distinguish an empty source group from numerical underflow. This requires the ordinary finite-gradient conventions for empty log-sums, already relevant to the project's H030 source. The quantities are predictive weighted allele summaries, not independent likelihood factors or calibrated ancestry probabilities.

Mean/max pool these 24-channel contrasts/masses/masks within the original hold, project through a small rank-8 adapter, and inject a zero-output-initialized 8-to-32 residual at the actual query sampled nodes before the second complete tree sweep. The original full-parent BiGRU and direct classifier perform the final prediction. Zero output weights preserve the common initial H045 function exactly. No new chain, attention, ODE, donor-indexed posterior, topology-to-switch rule, finer grid or tuned smoothing is added.

The same-marker query-allele lookup occurs before pooling. Without it, linear source/allele propagation and linear site pooling could commute and simply recreate another pooled reference statistic. Even with this lookup, the resulting summary is tree-weighted identity-by-state evidence. It does not retain which donor explains successive SNPs, and cannot answer the separately owned donor-continuity/remix hypothesis.

## One minimal falsification

**Hypothesis:** conditioned on the unchanged strong predictor, correctly aligned source–allele–branch support improves real C4 boundary discrimination beyond the same additional capacity with misaligned support.

Use three arms, with identical data, initialization, parent order, loss, precision and optimizer budget:

| Arm | Distinction |
| --- | --- |
| Aligned route | The reference allele/call arrays occupy their actual observed reference nodes. |
| Alignment sham | Only inside the new route, apply one fixed parent-wide permutation of complete reference diploid blocks within source × younger/older age-family strata. Keep it constant across SNPs. Gates and original backbone observations remain untouched. |
| Exact H045 continuation | The original model, without unused additional weights, receives the same further training. |

The sham preserves each group's allele and called-reference counts at every SNP and preserves the observed reference haplotype sequences/LD. It disrupts their extra route's association with individual observed branches and exact sample dates within an age family. It is an alignment ablation, not a biologically valid replacement dataset. It does not preserve exact reference ages of assigned haplotype sequences. Single-diploid strata, notably older ANA, cannot move. Report observation-only moved/called support by group with one frozen seed, without reseeding. Weak movement makes that part of the mechanism contrast inconclusive.

No fourth new mechanism is needed. Beating continuation alone could be capacity or additional-training benefit. Beating sham alone could coexist with damage to the strong predictor. Both contrasts are necessary.

If root later elects to register this refinement, retain 32 TRAIN/eight VAL parents, all 952 observed haplotypes/686 queries, every SNP and native hold, exact span loss, FP32, seed101, fresh independent AdamW and 30 further passes/960 whole-parent updates per arm. That is 2,880 updates, 18 scheduled checkpoint files and 144 arm/checkpoint/VAL-parent probability products sealed before truth. Preserve the original accuracy/NLL/earlier-pass selection rule. No boundary-based reselection.

Proposed practical gate, to freeze before outputs: at least +1.00 percentage point equal-parent Present182 boundary F1 over EACH control, no more than 0.05 point span-accuracy loss against EITHER, and no mean NLL regression. This supports boundary utility only. Report paired parent differences/sample SD, accuracy, NLL, Brier/ECE, precision/recall/F1, matched/missed/extra counts and all seven scoring rows. Freeze common tract strata and chromosome-edge censoring before scoring. An apparent gain obtained by deleting genuine short tracts cannot support improved tract discrimination. H038 fragmentation numbers must not be reused as H045/H050 residual measurements.

H050's pending results cannot select these definitions. If H050 meets a registered utility gate, its independent confirmation remains the priority and it becomes a necessary practical comparator in later interpretation. Do not silently replace this common initializer with a validation-selected H050 checkpoint. If H050 fails, do not rename its memory as this route. Reused VAL and one optimizer seed provide discovery evidence only.

## Closest verified primary precedents and novelty limit

| Primary material directly read | Overlap and limit |
| --- | --- |
| Ralph, Thornton & Kelleher (2020), [Efficiently Summarizing Relationships in Large Samples](https://doi.org/10.1534/genetics.120.303253), Framework/Definition 3/Algorithm B. Cached full text in the shared-switch review. | Additive sample-weight accumulation on trees, allele-weight summaries, and efficient tree-sequence statistics are established. The paper's site statistics can be tree-independent, so it is not literally this learned path-weighted residual. It defeats novelty based merely on propagating source/allele channels or avoiding sample-pair matrices. |
| [tskit author C source](https://github.com/tskit-dev/tskit/blob/0e38fc3d3c98ba35f4c84ed0be359d539bf75724/c/tskit/trees.c#L891), function lines 891–1055, freshly retrieved. | Reference-set indicators, ancestor-count propagation, focal exclusion and span-normalized genealogical nearest-neighbour proportions are implemented. It uses nearest ancestor topology rather than this learned positive all-path weighting and marker matching. Reference-conditioned tree aggregation is not new. |
| Gasteiger et al., [Predict then Propagate](https://arxiv.org/html/1810.05997v6), Section 3, freshly read. | Neural prediction combined with reference/node propagation and adjustable influence range is established. Its all-walk PageRank kernel is not the exact unique-path tree operator here. Neither a new label-propagation principle nor a generic GNN contribution follows. |
| Browning et al. (2023), [FLARE](https://doi.org/10.1016/j.ajhg.2022.12.010), Hidden Markov Model/emissions. Cached full primary Methods and pinned author updater code. | Matching query alleles to reference alleles and using marker order for ancestry are established. FLARE retains ancestry × donor history and separates switching processes. This route sums donors at each marker and is not a reproduction or a replacement for the native baseline. |
| [Whitehouse et al. (2024)](https://doi.org/10.1093/molbev/msae223), Fig.1/GCN architecture, and ARGMix/ARGformer cached primary XML. | Whole-tree GNN plus genomic GRU, reference-informed tree ancestry, TMRCA-conditioned inference and genealogy embeddings/retrieval are established. The proposed interaction is narrower than these categories. |

The gate/path-product sum is ordinary tree message passing and positive-kernel weighting. The rank adapter and residual placement are established. **No genuinely novel operator is proved here.** The only unproved, potentially useful contribution is the exact controlled finding that same-marker source–allele–branch alignment adds tract discrimination after a strong ordered whole-tree predictor. An exact-combination priority search remains incomplete. If a new general architecture is a hard requirement, this candidate does not currently satisfy it and must not be marketed as one.

## Compute and decision blockers

For S SNPs and V_t native vertices in the containing tree, additional arithmetic is O(sum_s V_t(s) * 8*A_s), with two scalar-channel sweeps. Gates cost one small shared evaluation per native edge/hold. Biallelic sites have 16 channels, maximum four-allele sites have 32. Marker blocks and exact checkpoint/recomputation can limit the working messages to O(B*V_max*8*A_max), plus pooled/native-hold outputs. For B=16 and biallelic sites, one FP32 message buffer is 1.024 MB per 1,000 native vertices. This is not peak autograd memory or measured runtime.

There is no query-by-reference matrix, full 32-channel nonlinear GNN rerun per SNP, or new 20,000-bin sequence tensor. Existing full-tree, SNP and BiGRU costs remain. Learned log-domain rerooting over every SNP may still be expensive and cannot inherit the old preprocessing runtime. No O(edge edits) speed or actual fit is established. The first scheduled complete-parent update, if separately authorized, must decide feasibility with all gradients retained. Only physical GPU0 and the existing project storage limits could apply to any later execution.

Critical blockers are (1) H050 utility is pending, (2) no saved diagnostic attributes current errors to this missing alignment, (3) the proposed sham has immovable strata, (4) per-SNP learned tree work lacks measured full-parent feasibility, and (5) architectural priority and independent biological generalization remain unproved. These block a novelty or superiority claim. They do not justify manufacturing a new condition.

Only this report was written. No scientific payloads, truth, probabilities, checkpoints or H050 scored outcomes were opened. No model, fixture, qualification, fit, scorer, simulation, GPU/resource probe, admission, registration, ledger mutation, installation or Git operation occurred.

## Source provenance

Required inputs read: AGENTS.md, residual_biological_errors_20261015_v1/REPORT.md, H050 HANDOFF/scientific_config.inactive.json, next_biological_architecture_after_H050_20261016_v1/REPORT.md, and interval_tree_operator_novelty_20261016_v1/REPORT.md. Additional targeted sources were the H045 encoder and current manuscript, mutation-observation API, C4/protocol audits, donor/shared-switch/attachment reports, H030 reference-kernel source and the cached primary passages cited above.

Fresh retrieval hashes: tskit trees.c SHA256 076d12b47d02fe848d1519ac42db7a07c4bd0c40f7a05277755ea393e679219a at commit 0e38fc3d3c98ba35f4c84ed0be359d539bf75724. APPNP v6 HTML SHA256 2a257ed9c637ed678d4b95b65889f1c9d46290fb25ad1a5402cadc9460085253. Tskit statistics manual SHA256 e8423e3e1c049a41b56588991700b5b70ad64b7df5cfd808f02c88298bce4436. A fresh PMC request for Ralph et al. returned a browser challenge, so the already cached full primary text was used instead.

Local source hashes at this read:

- src/arggnn/mutation_routed_whole_tree_gnn_v1.py: 254f5155e31f81ae675e2bd08d38c97c9bf130d47e98efa2deea4c7344d9e073
- src/arggnn/reference_logcavity_gnn_v1.py: 8970aa2e82a43b45d1ec10b2eaae6e2203dab12356a4b968bea3c65b2e21fc0b
- research/reviews/post_H050_shared_switch_primary_novelty_20261016_v1/tree_sequence_statistics_full.txt: eb417ddb0e6753c4f76bb0747ab320eab4ee175bfc0a231cbbbe76bcf3bbeb2e
- research/reviews/post_H046_donor_haplotype_condition_20261015_v1/flare_primary_html.txt: f35fc23cff57d02fcfb513bd6bce570a7cfcf4cea8ba2cf977154c5022a5b4a2
- research/reviews/H043_priority_fresh_v2/interval_persistent_dag_v1/whitehouse2024_full_primary.txt: 21c1fe4d8346f9bf7a618c1809ed128141df71fe820dbd1d230ac73276f373f7
- research/reviews/ordered_operator_novelty_independent_20261015_v1/ARGMix.xml: 383bf243185583820dbb35cb8bbf8b680be7f06197d1e26d01e29db1488c63fe
- research/reviews/ordered_operator_novelty_independent_20261015_v1/ARGformer.xml: 53ab26710804f0b9ca6f355095893b365413caf56cf7bb722b39907f6db0619f
