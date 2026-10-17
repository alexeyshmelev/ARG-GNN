# FP32 ordered marker kernel: bounded source handoff

17 October2026. **GO to source integration/registration of a representative matched experiment using positive log-domain fields. Do not use the original signed-event formula as its numerical engine.** The four explicit toy cases pass features, masks, finite gate gradients and independent path enumeration. This establishes bounded API/math feasibility, not ancestry utility, production scalability, historical priority or execution authority.

## Actual H058 semantics preserved

The exact inspected source SHA is `081acadf8f55adec0a5d16c6430c30fe5771ee4a307f5387d32138e1659a5e61`. The earlier report seal remains `af7d46068175d00f7b733e5abe29855b356fe66bacb3c7d449b207184a3a3dc6`.

- The same child edge gamma is applied in either path direction. No child-degree division occurs in H058's added path evidence; degree controls the sibling scan. The inherited neural encoder's degree scaling is separate.
- Eligible seeds are known references only. Group is class+4×1[age>=350]. The fixed within-group sequence routing applies to references. Query lookup uses the original query genotype; its ancestry class stays unknown.
- H058 gathers **outside-only** at a query node. References beneath an internally sampled query are excluded. Source-age frequencies and the log-mass normalizer still use *all* called references in the group.
- Missing reference calls seed no allele slot; called alleles0..3 remain distinct. Missing queries mask all24 output features. The last eight pooling-mask channels follow query-called status.
- For group called count>0 but zero outside mass, H058 retains availability1, matching0, and safe_logD=0. The prototype preserves this unusual internal-query edge behavior rather than silently correcting it.

The dense oracle is an independent small-tree implementation of the positive log-message algebra. It was checked against direct enumeration of every eligible reference path. The actual H058 module was not imported or executed. This is **not an H058 qualification or replay**.

## Why signed events fail and the stable correction

The real-arithmetic identity in the earlier report remains correct. Directly exponentiating group masses, subtracting ancestor outside terms and reconstructing alleles from signed branch changes is not stable enough in FP32. Three intentionally hard cases produce maximum24-feature errors **32.000198**, **0.05771446** and **144.0** against positive messages. They isolate tiny sole-called support hidden by ancestor subtraction, subtraction of a dominant missing reference, and exp(-144) probability underflow. No probability floor or negative-mass clamp was used to hide those failures.

The correction builds a **positive cover** of the called references at each site: maximal subtrees whose reference calls share an allele, plus singleton internally sampled references when their own call differs from descendant calls. Missing regions contribute nothing. Reference sets in this cover are disjoint, so every called donor appears exactly once. Query genotypes do not determine the cover. It is a lossless algebraic representation, not a mutation/recombination posterior.

For C_g(q,v), the mass below v but outside the query subtree, disjoint clades use K(q,v)U_vg; descendants of q contribute zero. When v is a strict ancestor of q, sum the nonnegative local-reference and sibling contributions along q→v instead of subtracting K(q,v)O_vg from O_qg. The prototype evaluates all these contributions and final allele totals in the log domain. This avoids both signed cancellation and probability underflow while preserving differentiable edge gates.

An explicit structural fallback chooses dense positive messages when the cover's query/path operation proxy exceeds a full-tree pass. It uses topology and cover size, not labels/truth, a numerical agreement oracle or detached gates. It is a conservative *toy proxy*, not measured runtime optimization. It fell back on three of four normal markers and the long chain; the other two hard cases used the stable sparse path.

## Bounded results

Four trees/cases cover seven synthetic marker rows in total. The largest has32 nodes,17 references,4 queries, all8 source-age groups, a multifurcation, internally sampled query/reference nodes, recurrent assignments,4 called alleles, missing calls, a within-group route permutation and a root query with empty outside support. Hard cases have8,4 and10 nodes.

All stable feature/mask comparisons pass at atol4e-5,rtol2e-5; gate-gradient comparisons pass at atol8e-5,rtol4e-4. The maximum observed stable feature and gate-gradient differences are each **1.1920929e-7**; hard-case differences are0. These are CPU FP32 results, not FP64 or exact-roundoff claims. The nonzero gate direction is0.0031818002 analytically versus0.0031944362 by FP32 central difference at epsilon0.002, absolute difference1.2636e-5. The differentiated variables are the FP32 edge log-gates, including their clamp path; a learned gate projection is not instantiated. Its parameter gradient would follow by the chain rule after integration. No optimizer update occurred.

The returned marker-axis API permutes correctly with a marker permutation. Mean/max pooling remains invariant, while a position-weighted moment changes. Thus retaining this API supplies the earlier missing observed arrangement; this check does not establish a fitted sequence model or useful tree feedback.

An initial post-comparison fixture assertion expected a negative contrast in one channel that correctly had zero matching frequency. Its captured failure and initial source are retained. The assertion now checks the relevant eligible groups. Attempt3 passed the corrected bounded numerical checks. Attempt4 adds the ordered-marker API check and returns0. The first outer1 delivered no traceback and remains explicitly uncertain; its same-source repetition identified the later fixture error, without inventing missing original output.

## Cost and remaining implementation boundary

Let P_s be the positive cover size and D query path depth. With prepared adjacency/group fields, the intended straightforward positive evaluation is O(NG+QG sum_s P_s D), plus exact cover preparation, calls/frequencies, sequence states, outputs and both neural sweeps. Naive cover construction from raw calls costs O(SN); raw call access is at least Omega(SH). P_s can be O(number of references), even when a signed mutation encoding is small. A shared positive ancestor-segment data structure could change the D factor; none is implemented or claimed here.

The literal correctness prototype uses explicit Python path walks, repeated path/child queries and convenience recomputation. Its path nesting can cost D^2 and it is not the implementation of a claimed efficient bound. The hybrid wrapper also repeats preparation for its tiny per-site calls. Production must prepare adjacency/U/O once per hold, vectorize/chunk readouts and retain exact autograd before any cost claim. Dense fallback restores the established worst-case full-per-marker tree computation, not a sublinear guarantee. Its encoding overhead remains additional. Globally changing learned gates require rebuilding group fields; tree edge differences alone do not preserve a cache.

The API ends at `[markers,queries,24]` features/masks. It implements no ancestry model, second-sweep injection, readout, data loader, training framework or regenerator. ROOT can now design the specific matched comparison: identical ordered evidence/marker recurrence with an equally informed one-way readout versus zero-initialized feedback before the second complete-tree pass. Keep complete tree/sample processing, information access, extra capacity, training and marker outputs matched. Boundary/short-tract gains must be tested without erasing true short tracts. This prototype supplies no efficacy or resource-fit result.

**Decision:** sufficient bounded numerical evidence to proceed with that representative experiment's source integration and registration. Actual training still needs its complete reviewed implementation and ROOT's ordinary execution process. No speedup or ancestry conclusion is supported. The signed-event fast path is a no-go without a different stable representation; the positive-cover/dense-fallback route is the tested candidate.

Only the assigned new directory was authored. No scientific observations/truth/checkpoints/probabilities, GPU calls, scientific model imports, optimizer/training, installations, resource/guard operations, registration or Git actions occurred. The existing CPU runtime was used with CUDA hidden, one Torch/OMP/MKL thread, and bytecode writes disabled.
