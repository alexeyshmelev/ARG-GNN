# One provisional mechanism: correlated reference-support messages

The hypothesis is that local ancestry corrections sometimes require distinguishing repeated donor support from several genealogically separated supports, or identifying conflicting reference labels attached to related inferred lineages. This is predictor-side observed information. It is not query ancestry supervision, true ancestral continuity, a population migration model or calibrated posterior uncertainty.

## Exact field

Use every H030 native node/edge, analytic350 point and existing reference placement. Younger reference sources remain at valid350 points. Older sources remain at their actual sampling nodes and ages. Keep the two families separate, including absent older ANA. For family g, retain H030's additive edge cost c_g(e) and define

κ_g(u,v)=exp(-Σ[e on path(u,v)] c_g(e)).

Thus κ(r,r)=1, colocated sources have κ=1, and zero-cost edges need no contraction or invented labels. This exponential tree covariance is established tree OU machinery. Its role here is an explicit *dependence proxy*, not a claim that donor ancestry labels follow an OU process.

Reference r of observed class c_r receives exactly ω_r=1/n[g,c_r]. At node v let

M_v=Σ_r ω_r κ(v,r) e[c_r],  Z_v=1ᵀM_v,
p_v=M_v/Z_v,  π_vr=ω_r κ(v,r)/Z_v.

This preserves H030's class-balanced p exactly for the same positive age coefficients and donor subset. The new support Gram is

C_v=Σ[r,s] π_vr π_vs κ(r,s) e[c_r] e[c_s]ᵀ.

C is symmetric, nonnegative and PSD. It is BᵀKB for B[r,c]=π_vr 1[c_r=c]. It is **not posterior covariance** of the census class. Diagonal entries and 1ᵀC1 describe weighted correlated support. Under an independent-donor K=I condition, 1/(1ᵀC1) is the ordinary weighted effective count. Off-diagonal entries measure related support carrying different observed labels, rather than certified topology errors.

Calculate corresponding full, subtree and outside fields using exactly their respective donor subsets and masks. Empty-family fields return zero with false masks. No probability floor or fabricated absent-class evidence is introduced.

## Explicit reroot computation

For each family and each node v, define Y_v=Σ[r placed at v] ω_r e[c_r]. Maintain three messages for a component viewed from its boundary node:

M_v=Σ_r ω_r κ(v,r)e[c_r],
H_v=Σ_r ω_r κ(v,r)²e[c_r],
G_v=Σ[r,s] ω_rω_s κ(v,r)κ(v,s)κ(r,s)e[c_r]e[c_s]ᵀ.

A local colocated-source component has (M,H,G)=(Y_v,Y_v,Y_vY_vᵀ). Across an edge with a=exp(-c), its incoming component is

(M_i,H_i,G_i)=(a M_child, a² H_child, a² G_child).

Combine independent directional components at v by

M=Σ_i M_i,  H=Σ_i H_i,
G=Σ_i G_i + Σ[i<j](H_iH_jᵀ + H_jH_iᵀ).

The cross term follows because a source pair in different components has a path through v. Full C is G/(1ᵀM)². Upward messages use local and child components. Downward messages merge local, parent-side and all sibling components excluding the recipient child. Prefix/suffix component merges give every exclusion without subtracting nearly equal totals. Incremental merging adds G_new=G_old+G_i+H_oldH_iᵀ+H_iH_oldᵀ, then H_new=H_old+H_i.

All terms are positive sums, permitting masked log-domain evaluation to avoid tiny-mass divisions. Numerical conditioning and autograd memory still require qualification. The source-derived arithmetic is O(N×4²) per family and forest sweep; no R²-per-query bank or linear-scan speed claim is needed. Additive edge potentials preserve these kernel fields under source-free serial subdivision. The finite nonlinear GNN retains H030's explicit lack of whole-model subdivision invariance.

## Minimal learned use and falsification

Keep H030's centered p contrasts as the only value source. Add the ten symmetric C entries per family to the same whole-tree encoder/decoder gates, so they modulate reference propagation instead of creating a free calibration path. Keep width, frozen H025 ordered skip, output support, source policy, centering, initialization and fitting budget matched. Add no new carrier recurrence. This is one changed evidence mechanism.

The primary cheap comparison is genuine C versus K=I, with p exactly unchanged. For K=I, colocated distinct donors contribute Σ_r ω_r² e[c_r]e[c_r]ᵀ to the local G, **not** Y_vY_vᵀ. Its edge transforms remain the same, but cross-component Gram terms vanish and local individual self terms replace the aggregate outer product. Off-diagonal inputs are then zero, reducing input rank and potentially active gate-weight count. Equal template counts do not remove this difference. Report active counts and retain the ordinary p-only/static control.

A stronger capacity-preserving comparison uses K'=PKPᵀ while retaining actual π, p, donor labels and source locations. Use one preregistered parent-wide bounded reference permutation, restricted within family/class/coarse observed-date diploid blocks, with both haplotypes and genomic permutation coherence retained. This preserves K's eigenvalues, unit diagonal and PSD but breaks donor-dependence alignment with the actual query affinity weights. All arms train under their own fixed condition. Report moved fraction and singleton bins; if the restricted roster gives negligible movement, the sham is uninformative. Its calculation may require a bounded dense R96 Gram implementation, so feasibility is unmeasured and it must not be silently substituted for the linear tree recurrence.

A tie with independence weakens a useful dependence claim. A tie with a well-displacing alignment sham weakens attribution to actual donor genealogy. A gain over p-only but not these informed controls supports extra readout/capacity rather than the proposed biology. Examine parent-level NLL/calibration, short tracts and boundaries alongside accuracy. Reused VAL selection remains discovery evidence. No performance promise follows from the formula.

## Closest work and claim boundary

- Garba, Nye, Lueg and Huckemann, [Information geometry for phylogenetic trees](https://arxiv.org/abs/2003.13004v2), §4.1 Eqs8–9: stationary tree OU covariance exp(-path distance). The dependence kernel is established.
- Wang and Ji, [Second-Order Pooling for Graph Neural Networks](https://arxiv.org/abs/2007.10467v1), §3.2 Eq3: HᵀH and subsequent bilinear/attentional pooling. Second-order graph statistics and neural use are established.
- Felsenstein, [Phylogenies and the Comparative Method](https://doi.org/10.1086/284325), and Sibbald/Argos, [Weighting aligned protein or nucleic acid sequences to correct for unequal representation](https://doi.org/10.1016/S0022-2836(99)80003-5), identify the longstanding correlated-observation/unequal-representation setting. Only bibliographic metadata was newly checked for these two here.
- Standard Gaussian GLS/GP conditioning already uses K and aᵀKa. This proposal does not invert K, infer a Gaussian ancestry posterior or invent a covariance propagation primitive. Project Kron/GP operators, H030 rerooting and existing GNN gates are additional direct predecessors.

The specific candidate is an exact query-conditioned, class-balanced donor-support Gram coupled to whole-tree reference gates. Its priority and usefulness are unresolved. On other ancestry tasks the class basis/source-date policy must be requalified. Independent biological data require corroboration without treating proxy cohort labels as per-site truth. This static field alone cannot support a claim of a novel ordered graph-set architecture.
