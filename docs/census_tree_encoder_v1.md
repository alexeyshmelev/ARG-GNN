# Census whole-tree encoder v1

H021, condition `census_cut_exchange_proto_v1_source_only`, 2026-10-10. [src/arggnn/census_tree_encoder_v1.py](../src/arggnn/census_tree_encoder_v1.py) adds an executable ordinary whole-tree GNN to the separately sealed [census-cut exchange core](census_cut_exchange_v1.md). The backbone, tree sweeps, mean aggregation, projections, decoder and gates are inherited. The unvalidated biological constraint remains the observed census-cut relation plus support-loss retention. No first/novel backbone, predictive benefit or measured performance is claimed.

The new source/test/doc and [separate author bundle](../research/reviews/census_tree_encoder_v1_author/README.md) were created after the 36-test core was pinned. The core module, tests, document and its sealed receipts remain byte-for-byte unchanged. The root owns independent review, array/native adapters, scaler/feature provenance, fitting admission, scheduling, integration and Git.

## Actual sample-ancestor scope

A marginal tree can expose rows for isolated non-sample nodes used by other marginal trees. Treating every stored node or every apparent root as part of the current whole tree can invent extra graph components and incorrect identity counts. The root reports 279 connected internal nodes and 280 sampled leaves per actual Relate tree, with other stored internal nodes isolated in that marginal tree. Those numbers are root-attributed actual-audit metadata, not recomputed here.

`connected_observed_tree_v1` takes supplied CPU parent/time/F47 tables and actual sample-node joins, traces the ancestor union of every declared sample, and excludes all other non-sample rows. It exports a compact connected observed tree, unchanged selected F47, a global native-index provenance map and an administrative exclusion count. Those indexes and counts never become learned features. All samples must belong to a single complete dated component. A sample-bearing second root, an invalid ancestor path or a cycle fails. The adapter does not discard a sample to make a tree connected.

`prepare_connected_whole_tree_inputs_v1` then calls the frozen checked core preparation. It suppresses anonymous nonroot unary vertices and preserves root, every branching interior and every sampled leaf. The resulting provenance maps back to the supplied marginal-node table, including sparse global indexes. The original supplied table is not mutated. Native parent-child endpoint time differences provide additive retained edge durations.

This first prototype deliberately retains the core's all-and-only-sampled-leaves contract. A sample attached to a branching/unary interior is rejected rather than silently omitted or reinterpreted. The current Relate panel uses sampled leaves. Admitting internal ancestral samples requires a separately designed representation and qualification.

The adapters operate on explicitly supplied tensors. They do not load a real TS, project bank, scaler, checkpoint, truth array or hidden population field. Actual file/array/scaler and feature joins remain unqualified.

## Observation-only node and edge interface

`retained_observed_features_v1` rechecks complete connected parents, native CPU float64 times, stored CPU F47, native-index provenance, sample-leaf joins, all-node sweep membership, parent-child sweep ordering and exact fixed endpoint durations. The F47 encoded-age column6 must match `log(1+native_age)/log(10001)` under the frozen core tolerance. All retained feature rows are finite.

The neural node input has 51 channels: the stored observed F47 plus encoded native age, root flag, leaf flag and branching flag. The edge input is `log(1+retained_duration)/log(10001)`. Node type derives only from current observed topology. Age and endpoint duration derive only from observed native times. No sample ID, native/local vertex ID, group ID, Q/R role, population metadata or hidden census label is embedded. Actual IDs perform sample gathers only. The caller must certify that the original F47 itself contains only permitted observations. A generic numeric interface cannot recover or verify external feature/scaler provenance.

## Batched forest sweeps

`prepare_retained_observed_forest_v1` creates checked CPU height/depth batches for one or more independent whole trees. Height is zero at leaves and one plus maximum child height at an interior. Depth is zero at the root and one plus parent depth at a child. The forest has graph offsets, concatenated observed features, parent indexes, child counts, upward level members/edge children/destination ranks and downward level members. Every retained node appears once in each sweep and every retained edge appears once as an upward child and once as a downward parent relation. Python topology loops belong to administrative preparation, outside neural vertex actions. This metadata can be cached for already qualified inputs.

`CensusWholeTreeEncoderV1` applies tensor batches across the entire forest. Let b_v be tanh of the node input projection. An upward level computes

\[
a_v=\operatorname{mean}_{u\in\mathrm{children}(v)}
 (M_\uparrow h_u+T_\uparrow e_u),
\quad h_v=\tanh U_\uparrow[b_v,a_v].
\]

Leaves have zero aggregate. Messages are linear batches followed by `index_add` into level-local parent ranks. Each level's updated states enter the forest state with functional `index_copy`. The downward sweep computes

\[
c_v=M_\downarrow d_{p(v)}+T_\downarrow e_v,
\quad d_v=\tanh U_\downarrow[h_v,c_v],
\]

with zero root context. A final shared projection produces `tanh O[h_v,d_v]`. No neural operation runs in a sample-by-node Python loop. The neural loop runs once per checked height/depth level, batching all trees at that level. The encoder accepts a single checked tree, a sequence of checked trees or a prepared forest. `encode_whole_tree_contexts_v1` invokes one forest encoder call and gathers every tree's Q/R contexts by actual sample IDs.

The source-derived dense neural work is O(Nd²+Ed²), where N/E refer to the selected retained forest. Deep or unbalanced trees still require many sequential levels. Functional `index_copy` allocates an N-by-d state at each level, which incurs repeated state-copy traffic. Autograd retains the level/message/index computations and downstream exchange/readout activations. A conservative memory budget can grow with level count times N-by-d, and metadata caching does not prove a peak-memory bound. Device launch count, achieved utilization, actual memory, transfer costs and fitting throughput remain unmeasured. Source batching is not evidence that the required 10x fitting gate is met.

## Shared executable GNN

`CensusTreeGNNV1` owns exactly one forest encoder, one shared list of `CensusCutExchangeLayerV1` layers and one scored-query decoder. Its default depth is two. All five controls use the same parameter objects for the encoder, source projections, gates, residual LayerNorms and decoder. First-layer whole-tree contexts are identical across controls. Source/gate changes at later depths follow each declared dynamics contrast.

The wrapper checks graph count, persistent native sample dates, Q/census age agreement and observed cut topology against the checked bank. Proposed, eta0 and raw-diagonal require observed memberships. Permuted groups use the fixed global membership control from the core. The pointwise local contrast is independent of group assignments and requires target-local R contexts. Anchor and target contexts retain all dynamics Q. The final common decoder selects only the declared scored Q and returns their actual identities along with logits and contexts. No ancestry label is passed to the forward call.

The donor bank is an explicitly prequalified observed input. The forward call does not repeat the Python Q-by-R MRCA search. Source-only bank preparation belongs to the frozen core and should occur once for matched observed trees. This wrapper does not establish native donor-file/scaler provenance or certify that a separately supplied bank arose from the same native arrays. Such joins require root qualification before fitting. Numerical donor/source checks in the core remain active.

| Control | Shared components | Inactive parameters in the changing-support toy |
| --- | --- | --- |
| Proposed | Forest encoder, source projections, gates, decoder, depth | None |
| Eta0 | Same exact objects | Eta in both layers |
| Fixed global permuted groups | Same exact objects and observed donor bank | None |
| Raw-diagonal propagation | Same exact objects and finite source/mass measure | None |
| Pointwise local | Same exact objects, coordinate-local donors and depth | Damping and eta in both layers |

## Handmade CPU qualification

[tests/test_census_tree_encoder_v1.py](../tests/test_census_tree_encoder_v1.py) passed **30 cases in 3.36 s**, wrapper duration4.15 s, with empty CUDA visibility, one CPU thread, disabled pytest cache and a project-local basetemp. Earlier successful source snapshots are preserved:24 cases before the connected-union adapter,26 after that adapter,28 after full-GNN derivative/checks and29 after forest batching. The final thirtieth case constructs a real `tskit.TableCollection` entirely by hand with two marginal trees, six sampled leaves,15 stored nodes and five irrelevant non-sample rows in the left marginal tree. It performs no stochastic genealogy simulation and reads no actual project arrays. Excluding those rows leaves encoder and census groups exactly unchanged.

The suite covers explicit age/type/duration features, absence of learned identity features, every retained vertex's nonzero F47 gradient, every active neural parameter's finite/nonzero gradient, meaningful root/branching-interior and older-reference influence, actual Q/R/native-node permutation, exact anonymous unary insertion invariance, unary-root retention, sampled-interior rejection, sparse native provenance, cycle/disconnected-sample failures, forest-vs-separate output/parameter-gradient parity, shared source/encoder/gate/decoder objects, every contrast at two depths, local genomic independence, finite genomic dependence, scored roster/date/cut-bank mismatches and unchanged sealed-core hashes.

Output/permutation/forest equivalence tolerances are rtol1e-10/atol1e-12 in float64 and rtol2e-4/atol2e-5 in float32. Unary and irrelevant-node invariance use exact equality where the retained representation is identical. A float64 five-point check perturbs both F47 inputs (encoded-age column fixed) and every active parameter through the complete forest encoder plus two exchange depths and decoder. Step2e-4, rtol2e-8/atol1e-9 yields derivative−97.3971876587 and absolute difference1.78e-9 from the finite difference.

Tiny-fixture proposed/permuted/raw-diagonal forwards have40 active parameter tensors, eta0 has38, and local has36. All retained feature rows receive nonzero finite gradients in every tested float64/float32 control. The smallest retained-node gradient L1 over these fixtures is8.996e-5. Perturbing a branching interior, root and age600 pure reference changes query contexts by4.30e-4,6.61e-4 and1.98e-5 respectively. Altering a reference feature in the other physical hold changes a graph0 finite GNN output by0.00218, while the pointwise local graph0 output remains exactly unchanged. These are implementation dependence checks on handmade inputs, not ancestry performance findings. The [numerical receipt](../research/reviews/census_tree_encoder_v1_author/tiny_fixture_numbers.json) records each dtype/control.

The metric collector initially attempted to reuse one prepared differentiable feature graph across multiple backward calls. Its diagnostic and original script are retained. The collector was corrected to create fresh handmade feature graphs for each control. No encoder behavior or tolerance was changed for that diagnostic.

## Limits and later admission

There was no actual data/holdout or prepared-bank array access, native inference, fitting, CUDA/GPU job, profile, package installation, Git action or storage scan. All project writes remain in the separately owned remote source/test/doc/author namespace. The sealed exchange core and ten frozen source hashes remain unchanged. No native feature/scaler/donor join, native model parity, actual tree-shape/support distribution, predictive result, efficiency gate or independent reviewer verdict is established.

Root review and separately admitted native integration must verify the sample-ancestor union, pure-reference roster, observed F47/scaler provenance, branch durations, donor bank, chronology, same-weight contrasts and scored decoder alignment. Horizon/tile/grid sensitivity and independent parent-level NLL/calibration/accuracy confirmation remain separate. Reconstructed dates, panel dependence, strict query/context date policy and singleton loss of ancestry evidence remain limitations of H021.

Prior-art credit and exact claim boundaries follow the [biological-kernel equation audit](../research/reviews/biological_kernel_design_v2/README.md), including [GraphSSM](https://arxiv.org/html/2406.00943), [MP-SSM](https://arxiv.org/html/2505.18728), normalized hypergraph/partition diffusion, graph-SPDE covariance, dynamic graph kernels, TMRCA pooling and Relate descendant-set comparison. Ordinary whole-tree message passing is an inherited backbone and adds no new general kernel claim.
