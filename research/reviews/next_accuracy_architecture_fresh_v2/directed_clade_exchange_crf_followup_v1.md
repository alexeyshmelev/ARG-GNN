# One next operator: directed clade-exchange C4 readout

**Status, 2026-10-12:** source-only proposal for root decision after actual H038 learning curves. No implementation, experiment ID, admission, feature extraction, array/truth read or run. Live H038 and native sources remain unchanged.

**Scientific hypothesis.** Given the same full inferred trees, observed alleles/mutation support and H038 sample-aligned GRU, a directed exchange of a query's rooted-triplet affinity from an observed source-a donor to a source-b donor carries ancestry-transition information beyond generic tree turnover and recurrent smoothing. It should improve the frozen H038 primary present-day182 parent-mean accuracy, with NLL and boundary metrics explaining the change, over a control that retains turnover but removes destination-source information. A genealogy edit is not an ancestry switch. Donor source labels describe observed proxies, not pure query clades or verified ancestral identity.

## Operator and exact retained inputs

Keep H038's complete PyG upward/cavity sweeps, mutation-concordance gates, observed SNP encoder, width32 bidirectional GRU over all952 stable sample columns, and C4 emission head. Add one four-state chain CRF over the native ordered holds for each of686 query samples. Its emissions are the current GRU/head logits. Forward-backward returns soft C4 marginals, never Viterbi hard labels.

For a fixed query q and stable donor samples r from source a and s from source b, count a witness at boundary i when the old observed tree resolves (q,r)|s and the new observed tree resolves (q,s)|r. Within each individual tree, resolution means the first pair's LCA is a strict descendant of the equal other-pair LCAs. Use local topology only. Never compare internal node IDs between trees. Mark tied, unresolved or sampled-ancestor triplets explicitly as unresolved. Ancient samples stay at their real observed ages and nodes, with no census350 substitution or fabricated descendants.

Let W_i(q,a,b) be the available-pair-normalized witness count for a!=b. Let D be W normalized across destinations within each source row, with an all-zero row when no witness exists. Cache g_i(q), the fraction of resolved triplets whose topology changes, and u_i(q), their unresolved fraction. Existing mutation support remains in the H038 encoder; the new witness feature initially uses topology alone to keep the hypothesis narrow.

Define exit probability and source transition:

    p_i(q,a) = sigmoid(c_a + s*log1p(delta_i/1000bp) + u*g_i(q) + v*u_i(q))
    P_i(q,a,a) = 1 - p_i(q,a)
    P_i(q,a,b) = p_i(q,a)*softmax_{b!=a}(beta_ab + gamma*D_i(q,a,b))

delta_i is the distance between adjacent native-hold centers. Use log P as the CRF boundary potential. There are exactly20 extra learned scalars: four c_a, three shared exit-feature coefficients, twelve off-diagonal beta_ab and one gamma. Both comparison arms have29784 parameters. Gamma is unconstrained, initialized to zero. Set c_a=log(3), all other new coefficients=0 initially, making every transition row uniform and the initial marginals recover the current emission softmax mathematically. A negative learned gamma would contradict the proposed positive affinity-exchange interpretation.

No query truth, true ARG, ARGMix predictions, frozen teacher logits or old baseline prediction arrays enter feature construction or model inputs. TRAIN truth remains only the supervised loss, with the established external target boundary.

## Concrete feature/runtime plan

Use K=8 persistent donor sentinels per parent: two distinct observed individuals per source, selected deterministically from the youngest and oldest available observed age strata, with stable sample/haplotype keys breaking ties. Keep these identities fixed along the chromosome. If a class has fewer usable donors, retain actual availability and masks. All266 donors still participate in the full H038 encoder; only edit-witness extraction is sparse.

Build an LCA/topology index for each complete observed tree. Query-to-sentinel and sentinel-to-sentinel LCA lookups suffice. At K=8 there are48 ordered cross-source donor pairs per query. Extract boundary witnesses in small hold blocks, aggregate directly to the twelve W fields plus g/u, then discard triplet temporaries. Use a new observation-only sidecar after a future root freeze; do not rewrite the live cache. No all-sample N-by-N matrix, branch transport solver or new genealogy inference is needed.

For T=1000,Q=686, feature extraction entails about33million simple ordered-pair witness evaluations per parent. The fourteen FP32 fields occupy about38.4MB per parent, about1.54GB over40 parents before headers/masks. CRF forward-backward visits about22million four-by-four transition entries per parent. Batch all queries within each scan and use the existing PyTorch runtime; do not add a solver or environment. These are operation/storage estimates, not measured throughput. Sequential scan/kernel-launch cost and autograd memory must be measured once if root selects the design, not assumed negligible.

Emit one C4 marginal per native hold and reuse the exact existing hold-to-original100bp-grid map. This preserves H038's native-hold TRAIN class-span mass aggregation and full-grid VAL output, including mixed spans, coverage clipping and every query. It cannot recover switches inside an unchanged hold, the same resolution limitation as current H038. Do not add purity filtering, missing-bin removal, tail extrapolation or probability floors.

## Discriminating same-capacity comparison

Only two newly registered arms: directed W versus destination-removed W. In the control, replace each off-diagonal row by

    W_control(a,b) = sum_{c!=a}W(a,c) * n_b / sum_{c!=a}n_c

where n_b is observed proxy availability, never a truth-derived ancestry prior. If the denominator is absent, the original row is zero and remains zero. This preserves each boundary's source-row witness total, event presence, g/u, span, donor availability, local emissions and recurrent inputs. It removes the witnessed destination. The self/total-exit gate receives identical generic inputs in both arms; all20 parameters, initialization, optimizer, budget and checkpoint rule are identical. Separately fitted gates may diverge, so do not claim identical fitted rates.

For efficiency, root may warm-start both arms from the same selected H038 checkpoint and use a single shared, predeclared short continuation budget, tentatively10 complete-parent passes (320 TRAIN-parent updates per arm), with end-to-end gradients. This is a proposal, not a resource authorization. Keep the current32TRAIN/8VAL split, one seed and exact existing selection/scoring protocol. Record reused-VAL selection and do not infer fresh-parent confirmation. Do not launch another matrix or seed sweep.

A convincing lead requires directed destination information to beat its control on the frozen primary metric, with a useful NLL/boundary tradeoff and recorded added time/memory. A tie, gamma near zero, or improvement only over unmodified H038 supports generic smoothing rather than this biological mechanism. If H038 curves still show unresolved underfitting, root should first decide whether further fitting is the more useful action.

## Closest prior art and scope

The nearest published operator is Maples et al., **RFMix: A Discriminative Modeling Approach for Rapid and Robust Local-Ancestry Inference**, American Journal of Human Genetics (2013), which couples local random-forest predictions through a conditional random field: https://pmc.ncbi.nlm.nih.gov/articles/PMC3738819/. CRF/HMM ancestry smoothing and source-transition potentials are established. ARGMix's current transformer plus genomic GRU is already close prior for the retained genealogy/reference recurrence, see Shanks et al. (2026), https://doi.org/10.64898/2026.04.05.714076. This proposal does not claim the combined stack is novel.

The possible contribution is specifically an observation-only directed three-sample clade-exchange potential, using stable sampled identities while leaving inferred ancestors anonymous. Literature priority remains unresolved. H029/H030 carrier histories did not establish coherent-history accuracy gains, H032 static Gram was negative, H033 tree/TV interleaving tied, and H035 signed tree/alignment differences lost. This operator uses none of their carrier matching, Gram gates, TV solves or signed/path-sum compositions. Its claim must rest on the directed-versus-destination-removed comparison.

Source basis read-only: research/experiment_ledger.md H029-H035/H038; research/reviews/next_accuracy_architecture_fresh_v2/proposal.md; src/arggnn/mutation_routed_full_panel_pyg_model_v1.py; src/arggnn/mutation_routed_whole_tree_gnn_v1.py; src/arggnn/train_mutation_routed_full_panel_pyg_v1.py. The ledger status can lag the parent's live v5 dispatch; no H038 accuracy result is asserted here.
