# Biological kernel design v2

Source-only audit, 2026-10-10. No implementation, experiment or novelty finding. Root owns integration and execution. Existing three-seed candidates and seven attribution controls continue unchanged.

The root-supplied pilot supports no superiority claim: present accuracy is 84.8906% age, 84.8410% plain, 84.1769% matched GRU, 85.5537% native current-release ARGMix and 82.0603% released AP three-seed mean. Candidate NLL 0.42725 versus ARGMix 0.43532 crosses the accuracy ranking. Two TRAIN/two VAL 2 Mb parents, unmatched seeds and different native selection objectives cannot establish independent benefit. Age/plain are effectively tied. C4 is ANA/CHG/WHG/EHG at census 350. Q184 includes 120 scored queries and 64 unlabeled context haplotypes. Hours of candidate fitting versus native ARGMix's reported 136.5 s / 20 epochs make efficiency a prerequisite.

No defensible general kernel-novelty claim survives. Kron reduction, normalized bipartite/hypergraph diffusion, heat/OU covariance, dynamic graph kernels, adaptive temporal GNNs and set attention are inherited. H019 already rejects unconstrained learned low-rank incidence as novel. Freshly retrieved primary texts verify GraphSSM Eq. (5), graph-dependent input filtering with fixed channel generator and Appendix B.2 persistent-node alignment, MP-SSM Eqs. (2–4), graph-shift state exchange plus nonlinear decoding, and graph-SPDE Sections 4.3–4.4 covariance/heat kernels. DiffPool Eq. (6) learns soft incidence. Relate's `BranchAssociation` already compares descendant sets. The proposed distinction below is a specific ancestry constraint, with priority unresolved.

Source-inferred bottlenecks are hypotheses, not profiled causes. `anchor_kernel.py:_layout` inserts all 100 bp readouts into events. Two scans/layer then perform two dense basis transforms/event. At Q184, S3, H8, L2, P≈2500 and G≈116/tile, numerator-plus-mass transforms require approximately 34.4 billion forward MACs/tile. Indexing eigenvectors by every segment materializes approximately 1.08 GB float32 repeated bases. `bool` checks on GPU rates, sources, mass, loss and individual parameter gradients can synchronize. `load_bank` hashes/loads full bundles and transfers every geometry; root reports approximately 38 GB prepared banks. Native ARGMix instead batches 16 chunks, flattens transformer rows and invokes a batched GRU. Its own scalar checks remain, and information scope differs.

First pursue faithful engineering separately: retain spectral states within each unchanged hold, scan only hold boundaries/anchors, batch exact partial-hold readouts, and reconstruct each readout once/direction. Validate failures through consolidated device flags before optimizer steps. A losslessly derived execution view can retain only required geometry/fields, with original sealed banks preserved. Decode 120 targets while retaining 184 anchor states. These changes require output/gradient parity, fail-closed numerical checks and fresh reviewed namespaces. They supply no mathematical novelty and must not alter running jobs.

One new design merits specification: **census-cut exchange with soft clade-turnover retention**. From every complete observed tree, partition Q184 by
\[
 i\sim_gj\iff t_g(\mathrm{MRCA}(i,j))\le c,\qquad c=350.
\]
All Q184 samples are younger than 350. Infer groups from topology/observed ages, never census flags or hidden populations. Let C_g(q) be q's group of persistent query IDs, including q. Internal IDs play no role. On a correct tree, a common ancestor before 350 implies a common census lineage. Different groups need not have different ancestry. On reconstructed trees this is an uncertain sufficient-evidence constraint, not certified ancestry identity.

Let P_g be one-hot group incidence and
\[
 S_g=P_g\operatorname{diag}(P_g^T1)^{-1}P_g^T,
 \quad K_g=\delta I+\lambda(I-S_g),
\]
\[
 E_g(\Delta)=e^{-\delta\Delta}S_g+
 e^{-(\delta+\lambda)\Delta}(I-S_g).
\]
S is an orthogonal block-averaging projection. This exponential is standard partition diffusion, computable through segmented sums without eigensolves. At a genuine tree boundary b, use
\[
 \rho_{b,q}=\frac{|C_{g-1}(q)\cap C_g(q)|}
 {|C_{g-1}(q)\cup C_g(q)|},\quad
 J_b=\operatorname{diag}(\rho_{b,q}^{\eta}),\quad h^+=J_bh^- .
\]
Rates are global per layer/branch/channel, lambda,delta>0 and eta≥0. Lambda/delta use inverse normalized-coordinate units, eta is dimensionless. Because q persists, rho>0. Clade overlap is a declared soft retention relation, never true ancestor correspondence or a recombination posterior. Subdivision of an unchanged hold gives J=I. Serial insertion preserves groups. The proposed biological relation combines horizon-limited exchange with query-specific loss of contextual support across genomic boundaries.

Use an ordinary whole-tree upward/downward encoder after suppressing anonymous unary vertices, with observed F47 and additive endpoint branch durations. This retains all branching interiors and their information. Donor pooling uses inherited observed-TMRCA kernels over all 96 pure references: W_qr=exp(-g(t_MRCA(q,r))/sigma_s)/sum_r exp(-g(t_MRCA(q,r))/sigma_s), with g(t)=log(1+t)/log(10001) and sigma=(0.05,0.1,0.2). Use stored encoded ages and log-softmax evaluation. References older than 350 remain proxy evidence and cannot be inserted into census-cut groups. At fixed 32 anchors use bounded q/r signals, sources s=q+lambda*Wr and matching mass m=1+lambda*W1. Compose E and J in physical bp / 2 Mb order, complete the reverse kernel by transpose, normalize identical numerator/mass scans, and retain the existing gated readout/depth pattern. Readouts never create impulses.

E and J are nonnegative contractions. The unnormalized completed kernel has the inherited covariance certificate: within-hold noise covariance 2K maintains covariance I, while boundary innovation I-JJ^T maintains it across jumps. Positive source masses preserve bounded normalized messages. These properties do not bound the whole network or establish cross-input PSD. Tree-age uncertainty and panel dependence are limitations. Singleton cuts can erase useful between-lineage ancestry evidence. If all groups are singleton, exchange vanishes and only query-wise temporal filtering remains. Record group-size/support distributions before fitting.

With n_e event subholds, G trees, V nodes, A=32 anchors, B=3 branches, H=8 latent channels, L=2 layers and d=32 hidden width, propagation falls from O(L*n_e*B*Q²*H) to O(L*n_e*B*Q*H), with O(GVd²) whole-tree encoding and O(ABQRH) donor pooling. Q184 group indexes occupy approximately 2.7 MB across 3718 original trees. The core avoids Q² basis banks; readout gates O(LP120d(d+H)) and backward activations remain substantial. No measured speedup follows.

Minimum matched contrasts share encoder, F47, donor map, anchors, decoding and fitting budget: proposed versus eta=0, a fixed global permutation of Q group memberships preserving group sizes/turnover statistics, raw-diagonal propagation with identical finite impulses, and shared-encoder local prediction. Compare existing dense candidate/GRU and unchanged native ARGMix separately. Qualify subdivision, sample permutation, boundary conventions, mass and gradients before fitting. Prespecify dating/horizon and tile/grid sensitivity plus positive/negative ordering fixtures. Reject turnover attribution if eta=0 ties, ancestry constraints if permuted groups tie, memory claims if local/diagonal controls tie, and efficiency if matched 20-epoch fitting fails a 10× prototype speed gate. Freeze independent parent-level NLL/calibration/accuracy confirmation before opening new outcomes. No new fit is authorized here.

Primary-source URLs, equation checks, source line anchors and administrative hashes are recorded in [the audit bundle](../research/reviews/biological_kernel_design_v2/README.md).
