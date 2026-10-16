# After H050: learn tract evidence before pooling it away

**Retain two conditional hypotheses. Register or implement neither before H050 utility is known. Neither currently establishes a new general GNN operator or an A* novelty claim.** The biological target is distinguishing real C4 changes from false short tracts, beyond another memory or smoothing wrapper.

## Evidence and no-repeat boundary

H050 genuinely completed30passes/960FP32updates per arm/24checkpoints,3840updates total. Its CLI exit1 is preserved. Full validation has no ancestry outcome in the inspected snapshot. Triple-support conditional-mean transport overlaps established higher-order memory. Its matched controls determine utility, not priority.

H045 foreign_episode gives96.8270±0.5870% Present182 span accuracy but18.1558% boundaryF1 and22.5890% recall at1kb, with905matched/3103missed/5063extra calls across eight reused VAL parents. The matched accuracy gain is only0.009568pp. H048 failed both utility criteria. H057's TRAIN grid-minus-hold oracle gap is only0.104443pp. This narrowly supports limited finer-readout headroom, not an achievable gain or an explanation of H045's different VAL errors. H038 fragmentation numbers are not H045/H050 measurements.

The current SNP route encodes allele-frequency/callability tuples, then mean/max pools within native holds before tree propagation. That route loses marker order before graph processing. The complete model can still encode LD indirectly. Prior audits already reject changed-clade shared-switch factors and unqualified attachment mixtures. Do not revive those or duplicate the owned donor-copying/remix HMM.

## Hypothesis1: ordered marker evidence before the second tree sweep

Retain the existing unregistered marker/tree-feedback design, not a duplicate condition. Start every arm from H045 own_state pass30. Learn rank8 ordered100bp-bin states from existing SNP embeddings, first-sweep sampled-node context, call counts and geometry. Keep the four specified depthwise-separable Conv1d blocks, kernel3 and dilations1/8/64/512. Span-average their states into each native hold, inject a zero-initialized8-to32 residual at actual sampled nodes before the second complete PyG sweep, and add the common zero-initialized marker readout at original scoring centers. Keep the full genomic BiGRU.

**Question:** does ordered local evidence help when it changes genealogical computation, rather than only final scores?

Primary control: identical ordered computation/parameters, injected at sampled-node output after the second sweep. Both retain the same second tree sweep and marker head. The existing hold-mean input ablation removes localization, and unchanged H045 continuation exposes extra-training gains. The earlier1240-parameter addition is analytic, not an implemented count.

Early versus late routing also changes the focal node's nonlinear processing. It does not isolate foreign-haplotype exchange alone. A tie rejects the early-routing claim. Fewer false calls with worse true-short-tract recall rejects the biological improvement.

**Novelty boundary:** GST interleaves graph/set feedback, Graph Matching Networks exchange context inside node updates, Whitehouse et al. combine tree GNNs and genomic recurrence, and FLARE/RFMix exploit marker order. The potential finding is useful placement of ordered evidence in a whole-tree predictor, not a new cross-graph primitive. This differs from H044's fixed SNP boxes and H033's frozen-emitter tree/TV alternation.

## Hypothesis2: marker-resolved source evidence through the complete tree

A distinct question is whether *which reference alleles occupy which observed branches at a marker* matters beyond global source frequencies. Keep source channels rather than a donor-copying HMM or persistent donor posterior.

For each complete local tree, compute an edge attenuation gamma_e in(0,1) using a small shared Torch gate on existing first-sweep context, branch geometry, dates and observed support. Compute gates once per native hold. At SNP s, seed genuinely labelled reference sampled nodes with called-allele indicators x_u[c,b], indexed by source c and observed allele b. Query nodes seed zero. On every original vertex, use two directed tree sweeps

    m_(u→v)[c,b] = gamma_(u,v) *
                   (x_u[c,b] + sum_(w≠v) m_(w→u)[c,b]).

The query receives exactly

    A_qs[c,b] = sum_(references r of class c)
               product_(e on path(q,r)) gamma_e *
               1[called_rs and allele_rs=b].

Form local source-normalized allele agreement and called-reference mass, contrasted with the SAME marker's ordinary source-frequency feature. Missing query calls or zero source mass get availability masks. A mixed NF/Yam reference without a known source label cannot be seeded as a known ancestry. No true tree, ancestry switch, persistent internal-ancestor ID or attachment posterior enters this operator.

Pool these marker-resolved contrasts into the SAME100bp bins and use the SAME ordered rank8 module/readout in all arms. A zero residual preserves the common initial H045 function. Retain every original SNP, sampled haplotype and native vertex.

**Paired control:** keep the original backbone observations untouched, but permute allele/callability assignments to reference sampled-node positions ONLY inside the new route, within fixed observed source×age strata at each SNP. This preserves marginal allele counts and active parameters while disrupting allele-to-topology alignment. Freeze an observation-only permutation seed. It is an alignment ablation, not a native competitor or a biologically valid replacement dataset. Include unchanged H045 continuation. The ordinary ordered-marker model is an additional descriptive comparison, with capacity differences disclosed.

The route must beat BOTH sham alignment and continuation. Beating continuation alone could reflect extra capacity or marker features. No donor-identity-continuity conclusion follows, since identities are summed at every marker and no donor-indexed state is carried.

**Novelty boundary:** positive tree-path kernels, additive tree message passing, label propagation and gated GNNs are established. AncestralPaths/ARGMix/ARGformer already use reference-informed genealogy. The narrower mechanism to investigate is marker-specific allele/source/branch alignment before temporal aggregation, rather than another global frequency statistic. An exact-mechanism literature search is required before claiming architectural priority. The current combination alone does not meet the requested novelty bar.

Torch/PyG can batch these scalar-channel sweeps by native hold and marker block, reusing depth schedules/scatter. It does not rerun the32-dimensional nonlinear backbone per SNP or construct query-by-reference matrices. Additional work is O(sum_s N_t(s)*4*A_s), with block storage O(B*N_t*4*A_s). This is still per-marker tree work, not O(edge edits) or measured speed. Full-gradient block recomputation may be needed. Feasibility belongs to the first scheduled complete-parent update, not another discarded fixture campaign.

## Same-data falsification after H050

Do not change the common H045 initializer after observing H050. If H050 passes, preserve its independent-confirmation plan and compare any eventual boundary model against that stronger predictor, with training/initialization differences disclosed. If it fails, do not rename and rerun its memory.

Retain original32TRAIN/eightVAL parents,952observed haplotypes/686queries, source metadata/dates, every SNP/tree, exact support, FP32, seed101, fresh AdamW and30full passes/960whole-parent updates per arm. Preserve six checkpoints, original accuracy/NLL selection and complete arm/checkpoint/parent seals before VAL truth. No boundary-based reselection or tuned smoothing.

Proposed practical gate: at least+1.00pp equal-parent Present182 boundaryF1 versus EACH primary control, at most0.05pp span-accuracy loss versus either, and no NLL regression. Freeze this before outputs. It supports boundary utility, not an ancestry-accuracy win. Report parent-paired changes/sampleSD, accuracy/NLL/Brier/ECE, precision/recall/F1 and matched/missed/extra counts, all seven cohort rows, true-short-tract recall and false short tracts/false-switched bp per Mb. Freeze tract strata, chromosome-edge censoring and1kb matching before scoring.

C4 is source population at350BP, not sample cohort. Topology changes do not certify source switches. Reused VAL is discovery data. Neither hypothesis establishes top1 or a complete paper without independent simulations, faithful native ARGMix/ARGformer/FLARE comparisons and appropriate real-data interpretation.

## Sources and scope

Source: mutation_routed_whole_tree_gnn_v1.py:281–341,343–388 and the H050 encoder splice at direct_readout_cut_memory_v1.py:123–147. Results: README.md, residual_biological_errors_20261015_v1/REPORT.md, H057_completed_TRAIN_resolution_audit_root_20261016_v1/REPORT.md and genuine H050 actual_outcome.root_v1.json. Reviewed the existing H049/H050 novelty, scaling, shared-switch rejection, attachment rejection, marker-feedback/reassessment and donor-identity reports, plus H026–H035 reference/kernel/fusion conclusions.

Primary passages re-read: cached [GST Section3.1](https://arxiv.org/html/2606.05116v1), [Graph Matching Networks Section3.2](https://proceedings.mlr.press/v97/li19d.html), and [FLARE HMM methods](https://doi.org/10.1016/j.ajhg.2022.12.010). Earlier verified audits support the Whitehouse, higher-order-memory and neural-BP boundaries. No exhaustive priority search is claimed.

Only this report was written. No arrays, genotype/tree/truth/checkpoint/probability payloads, model/scorer/simulation/job, guard/admission, registration, implementation or Git action occurred.
