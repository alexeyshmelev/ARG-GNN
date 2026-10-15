# H043 exact changed-clade pairing: source-only design audit

**Verdict:** reject the nearest-only operator as a replacement for H042 or as a claim of complete changed-clade transport. It does have an exact, at-most-n leaf-event construction. A small, matched experiment is defensible only as an additional foreign-state channel on top of all H042 local flags. No accuracy or novelty result is established. Root has not been asked to launch a new pipeline.

## What the object means

Use the same full observed panel, n=952 stable (sample_id,haplotype) keys, in two adjacent inferred marginal trees. Compare exact proper descendant-leaf sets, excluding singleton and full-panel clades, with H042's deepest representative for an equal-clade unary chain. No internal node ID is matched across trees.

For every leaf x, take A(x), its lowest removed old-clade ancestor, or null; and B(x), its lowest added new-clade ancestor, or null. If both are null, emit nothing. Otherwise aggregate the pair (A,B), retaining integer support k_AB = number of leaves assigned that pair. One-sided pairs are explicit. These are leaf-conditioned changes in observed genealogical context. They do not identify a recombination recipient, an ancestral individual, physical movement of ancestry mass or homologous internal nodes. Different ARG histories can generate the same two marginal topologies, and inferred-tree noise can generate the same features.

The raw contingency has marginals equal to nearest-assignment leaf counts, not full clade sizes. k_AB/n describes observed support. It is not an optimal-transport solution. Signed learned updates are not probabilities or a biological conservation law.

## Exactness and the coverage counterexample

There is one raw event per affected leaf, so raw events and deduplicated pairs are each <=n per boundary. Real endpoint contributions are <=2n. This covers every leaf with any changed proper ancestor. It does **not** cover every changed clade.

Consider old (((a,b),(c,d)),(e,f)) and new (((a,c),(b,e)),(d,f)). The removed old clade {a,b,c,d} is hidden behind removed cherries {a,b} and {c,d}; no leaf chooses it. The added clade {a,b,c,e} is likewise hidden behind added cherries {a,c} and {b,e}. Both changed ancestor states are absent from the pair graph. All H042 gained/lost flags must therefore remain present, including zero-support ancestor clades. This retains O(n) changed-node descriptors without pretending to enumerate every clade-overlap pair.

Other decisive cases:

- ((a,b),c) -> (a,(b,c)) gives (ab,null) for a, (ab,bc) for b and (null,bc) for c. One-sided events occur even for complete, fully binary panels. They do not mean a leaf or lineage was unobserved.
- Resolve a polytomy, or contract a cherry: all new or old clade events may be one-sided. Rejecting them would selectively discard legitimate observed topology changes.
- A subtree wholly outside the four H040 landmarks can rearrange internally while every eligible one-query/two-landmark triplet is unchanged. The proposed construction avoids that anchor blind spot, but retains the nested-ancestor blind spot above.
- Identical proper clades with changed node times/branch lengths give no event. Recombination can also leave sampled topology unchanged. This is a topology descriptor, not complete ARG-event detection.
- Naively climbing each leaf's ancestors is quadratic on a comb. A root-to-leaf traversal carrying the most recent changed representative is required on each tree.
- Equal endpoint hidden states force the existing H040 antisymmetric tensor and event-MLP messages to zero. A generic unary/geometry term removes that algebraic constraint, but learned weights, output initialization and aggregation can still yield zero.
- A clade born in a middle hold and lost at its next boundary can receive cancelling positive/negative messages. Separate role features and the retained H042 flags help, but no universal noncancellation guarantee is valid.

## Minimal candidate, with a genuine 1,024-weight control

Keep the H042 body, local flag term, two shared whole-tree sweeps, all sampled leaves and original genomic GRU/head. Let U,V,W map32->8 and O map8->32 without biases: exactly1,024 weights, as H042. Reuse these projections for both local and cross-tree branches. Preserve the original H042 rank term:

    L_v = gained_v * (U h_v) * (V h_v) - lost_v * (U h_v) * (W h_v)

For each deduplicated pair, use a fixed eight-dimensional geometry vector:

    g(A,B,k) = [1, log1p(|A|)/log1p(n), log1p(|B|)/log1p(n),
                |A intersection B|/n, |A symmetric_difference B|/n,
                k/n, is_null(A), is_null(B)]

Null is the empty set with zero hidden state. It is never a negative array index passed to a gather. Count-support marginals r_A=sum_B k_AB and c_B=sum_A k_AB include the explicit null row/column. Use w_AB=k_AB/sqrt(r_A*c_B), a normalized weighted bipartite adjacency, not a transport probability.

    z_old = GELU(U h_A + V h_B + g(A,B,k))
    z_new = GELU(U h_B + W h_A + g(B,A,k))
    R_A -= w_AB*z_old, if A is real
    R_B += w_AB*z_new, if B is real
    injection_v = O(L_v + R_v)

Use fixed branch coefficient1; freeze the geometry/scaling before outcomes. The constant coordinate permits a nonzero rank feature when equal endpoint states are zero. O remains zero-initialized to preserve the matched H038 warm start, so initial output injection is intentionally zero. There is no claim that every learned event must remain nonzero. The two directional messages need not conserve their sum.

This is a design specification, not implemented code. Sharing the1,024 weights is a constraint for a first falsification experiment, not a claim that this is the optimal architecture.

## Decisive matched controls and rejection rules

Freeze these comparisons before H040/H042 VAL outcomes influence the design:

1. **H042 local flags:** unchanged baseline, reused under its existing frozen protocol rather than rerun unchanged without a repeat reason.
2. **Local geometry / partner blind:** same local term, true endpoints, support weights, geometry, parameters, projection/gather/scatter workload and nonlinear branch. Replace each present partner hidden state with the recipient's own hidden state; keep a genuinely absent partner zero.
3. **True foreign state:** the candidate above, differing from control2 only in the foreign hidden-state gather.
4. **Misassigned partner state, optional trained control:** keep actual destinations, local states, weights and geometry, but prospectively permute only foreign projected states within boundary/role and fixed support-count strata. Record how many events actually change. This is an explicitly corrupted information channel, not an exact clade correspondence. An inference-only shuffle is a distribution-shift diagnostic, not a substitute for the trained control.

All arms keep the same original H038 warm start, parameter/body initialization, sample/parent splits,32TRAIN/eightVAL,30passes/960updates, six checkpoints, FP32, optimizer, targets and primary selection/scorer. Controls2/3 match extra computation; baseline1 shares parameters/body but lacks that extra nonlinear work. Freeze training plans for all new arms before opening any VAL. Each frozen evaluation must seal all48 checkpoint-parent cells per arm before that arm's VAL truth; a two-arm paired evaluator retains96 seals. Do not pool windows/leaves as independent experimental replicates.

The primary scientific unit is an independent ARG parent. Report the eight paired parent differences in original Present182 exact-span accuracy, mean effect and uncertainty, with unchanged NLL/diagnostics as secondary. Freeze a practical minimum effect before evaluation; this audit invents no success threshold and promises no power from eight parents/one seed.

- Candidate equals control2: reject the foreign-state-transfer explanation, even if both beat H042. Geometry/local turnover may explain the gain.
- Misassigned partners work as well: reject a claim that correct cross-tree alignment is responsible.
- Improvement exists only under a changed support/selection rule or extra tuning: reject the matched accuracy claim.
- All-parent cost/cache bounds fail: reject feasibility under current caps; do not silently subsample leaves, discard boundaries or filter clades.
- All claims of complete clade coverage, ancestral homology or optimal transport remain rejected regardless of classification accuracy.

Before training, a separately authorized tiny structural prototype should test the symbolic counterexamples, binary/polytomy/unary cases, stable-leaf and internal-node relabelling, duplicate collapse, null endpoints, exact dedup support, coordinate reversal roles and zero/equal-state behavior. That prototype has not been run here.

## Cost and practical gate

For T holds and N serialized vertices, P<=n(T-1). Exact Python-integer bitsets take O(N*ceil(n/w)+P*ceil(n/w)) worst-case word work, not asymptotic O(n) preparation. Expected hash aggregation adds O(nT); deterministic sorting adds O(nT log n). Keep only adjacent-tree bitsets. Reusing H042 traversal contracts avoids extra ancestral-ID assumptions.

A conservative compact record is24 bytes: three int32 indices, support/intersection int32 counts and a float32 normalized weight. Add4N node-size bytes and existing2N H042 flags. Thus the complete static object is <=24n(T-1)+6N bytes. For an illustrative10,000 strict-binary holds, N=19,030,000 and the object is342,637,152 bytes. This is a bound example, not measured data. The worst-case event-record bytes are6.48 times smaller than H040's all5,692-triples-change bound; it is not an observed sparsity or speedup.

The three rank8 node projections alone are96N bytes, about1.83GB in that example. Budget an additional1–3 rank8 residual buffers and checkpointed event blocks; never retain all P autograd messages. Whole-tree sweeps/site encoding/GRU remain. Unary vertices can make N exceed the strict-binary illustration.

A40-parent cache at that illustrative size would exceed the existing4GiB witness cache. Rebuilding exact masks every30-pass round robin may dominate runtime; a disk cache would need a fresh retained-scope reservation and verified count bounds. No compression saving, wall time, GPU fit or current storage headroom is assumed. resource_forecast.json gives formulas and thresholds. Any future qualification/admission is root-owned on physical GPUs4/5 only.

## Prior work and defensible novelty

Exact cluster comparison and matching are established, as are dynamic GNNs and transport over tree-like structures. See literature_verified.md and the retrieval receipts. Particularly close are Bogdanowicz/Giaro's rooted Matching Cluster distance and Smith's generalized RF split pairing. Neither makes this nearest-leaf rule a new tree-distance or matching principle. TGN/EvolveGCN preclude broad novelty claims for cross-snapshot messaging or changing node sets; Tree Mover's Distance concerns GNN computation trees and hierarchical OT, not inferred genealogical-node homology.

A defensible future contribution would be the specific sparse, leaf-conditioned coupling of anonymous changed clade contexts, with full local-flag coverage, demonstrated alignment-specific improvement and measured cost on the frozen ancestry task. This limited source/abstract audit does not establish first publication or an absence of closer prior art. There is no new biological discovery yet.

Only this review folder was written. No model/pipeline, registration, observation array, private target, H040 VAL result, GPU job, admission, package or Git state was touched.
