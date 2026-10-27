# Compact observed-interval architecture assessment

A local function representation is a credible American architecture experiment. It can avoid the repeated hidden states of the same observed node across marginal trees. It is a new, lossy latent model, not an exact replacement for TreeLoom. The European Relate converter creates a different storage regime: internal nodes are local to each tree, so the same large compression factor is unavailable there. No architecture was implemented or trained in this assessment, and no runtime improvement or predictive equivalence is established.

The immediate exact optimization remains the separately prepared first/forcing forward fusion. Keep that candidate separate from this architecture investigation and from execution's block-packing work.

## Evidence and scope

All byte units below are decimal. The compact American observed tree was fully SHA-256 authenticated against the original TRAIN000 manifest. The observed tskit file includes its site and mutation tables. The site-by-sample genotype payload, private truth, checkpoints, validation and TEST arrays were not opened by the compact assessment. The genotype header was read only. No expanded observation bank, observation disk cache or GPU context was created. The project cap remains 1,000,000,000,000 bytes.

| Quantity | American TRAIN000 |
|---|---:|
| Sequence length | 30,000,000 bp |
| Samples | 1,600 |
| Marginal trees | 77,472 |
| Stored nodes | 106,762 |
| Stored edge intervals | 599,269 |
| Expanded node occurrences | 205,369,272 |
| Expanded sample-tree cells | 123,955,200 |
| Sites | 227,474 |
| Site-sample categorical observations | 363,958,400 |
| Cut-group occurrences at the inherited horizon | 37,545,876 |
| Observed tree file | 34,665,236 B |
| tskit table memory reported by tskit | 41,140,234 B |
| One expanded FP32 width-32 node array | 26,287,266,816 B |
| Expanded upward and outward arrays | 52,574,533,632 B |

The expanded node count was reconstructed from compact edge-difference incidence changes and matched the known expanded count. Cut groups were counted from crossing edges plus eligible roots, with direct ancestor-walk checks on four marginal trees. The bounded measurement completed in 3.35 seconds with 145,092,608 B peak RSS. These are measurement costs, not model timings.

A wording correction to the frozen count receipt is necessary: `site_payload_or_truth_read:false` means that the genotype matrix payload and private truth were not read. Loading the observed tree necessarily read its compact site and mutation tables. The original receipt is preserved; the fusion byte addendum records this clarification.

## Node identity is an input convention, not demonstrated biological continuity

American tsinfer node IDs are identities in the supplied inferred graph. Reuse of one row over several intervals supports computation reuse within that observed representation. It does not establish that an inferred internal row is a uniquely identified biological ancestor, nor that two different rows denote different biological individuals. Learned embeddings must be equivariant to arbitrary internal-node renumbering; numerical node IDs should not be learned features.

The exact pinned European Relate converter source has SHA-256 `452298c9a5ad60159d9f86dd402f71798e8a88108f2bb217fcb941d28778bfc4`. It adds samples once, then adds N-1 fresh internal node rows in each marginal-tree iteration. Lines 383-384 set `node_const = tree_count * (data.N - 1)`; lines 423 and 432-433 add the new internal nodes and offset internal edge endpoints. The original cleaner preserves IDs during clipping, so it does not restore internal sharing.

A bounded Kastore parser read only the first 5,184 bytes of each observed tree file: header, descriptors and keys, with no array payload. The parser was validated against the authenticated American counts. European TRAIN000 has 90,940,327 node rows and 181,878,750 edge rows, exactly matching `952 + 951 * 95,625` and `1,902 * 95,625`. The European file is 8,791,314,172 B. Its full contents were not rehashed in this check: the receipt binds the declared tree hash, current stat, prefix hash and converter source. The 952 samples and 95,625 trees come from prior observation evidence, not the header parser.

For Europe, one FP32 width-32 vector per stored node already costs 11,640,361,856 B. A rank-32, width-32 coefficient array would cost 372,491,579,392 B before gradients or other state. Equal ages or repeated clades do not authorize merging these IDs as persistent ancestors. Lossless computational interning would require proving equality of the complete computation inputs, adjacency, interval conditions and attributes; it would not establish biological identity. No such merging is proposed or performed here.

## Exact tree-event reuse has a worst-case expansion problem

Let E be stored interval edges, N stored nodes, T marginal trees, H samples, and d hidden width. Compact topology storage is O(N+E). Linear subtree sufficient statistics can be updated on edge insertion/removal, often touching only affected ancestor paths. The existing tree-sequence algorithms exploit this structure.

A nonlinear tree network is harder. A leaf or edge change can alter every ancestor's nonlinear state; an outward pass can then alter descendants. With repeated layers or full-tree sweeps, changed dependencies can cover most of each marginal tree. Genotype-driven inputs also change independently of edge topology. Caching exact piecewise node functions can therefore accumulate breakpoints propagated from descendants and other inputs. Worst-case work and retained states approach the expanded occurrence count, and backward must account for the same dependencies. O(E) topology events are not proof of O(E) nonlinear training cost.

Exact incremental execution remains a possible engineering optimization after measuring the actual affected sets. It is not a guaranteed asymptotic solution and must not silently prune updates or reuse stale states.

## A concrete candidate: local projected graph functions

Represent the hidden function of observed node v as one vector `c[v,b]` on each active position bin b. A node's active domain A_v is the union of its incident edge intervals; samples have full-sequence support. Retain every original edge interval as a half-open gate `I_e(x) = 1[left_e <= x < right_e]`. Zero is used outside A_v, with an explicit active mask so that absence is distinct from a learned zero feature.

For a simple upward stage, use position-independent weights within a bin:

```
z_v(x) = a_v + U c_old[v,b(x)]
         + sum_{e=(u -> v)} I_e(x) W c[u,b(x)]
c_new[v,b] = integral_{A_v intersect b} sigma(z_v(x)) dx
             / length(A_v intersect b)
```

Here arrows describe message direction, not a reinterpretation of the tree edge table. Static age/cohort features can enter a_v. Sample cohorts and ages are public observed inputs, including every ancient query's self and ancestor occurrences in the European setting. Upward and downward stages use separate weights and masks. A verified topological order of the observed union DAG can implement a recursive upward sweep; a fixed number of synchronous graph layers is an alternative with a different receptive field. Equal age ties must not be resolved by an unjustified numeric-ID order.

This is the projection of a nonlinear update, not the nonlinear update of a projected aggregate: in general `P sigma(z) != sigma(P z)`. For box functions and the affine, additive message above, z is constant between incident edge endpoints and grid endpoints. Sorting those local events, incrementing the active transformed-message sum, applying sigma on each constant segment, and accumulating segment length computes the defined integral exactly up to floating-point arithmetic. No full sample-by-tree hidden array is needed. Projection after each node/stage is the architectural approximation. It removes high-frequency position information and allows observed information from different positions in a bin to mix.

Attention, nonlinear edge messages depending on position, nonconstant kernels, or smoother basis functions require a separately specified integration rule. Averaging degrees before normalization is generally wrong; if degree normalization is used, update the degree at the same interval events. Unnormalized additive messages are the simplest first candidate. Biases, empty supports and roots need explicit semantics.

Let M be active node-bin pairs and Z edge-bin intersections for one direction. Pretransforming each node-bin state once costs O(M d^2), and an event sweep costs O((Z+M)d), after reusable topology sorting. A recursive upward and downward pair visits both directions. Position-independent shared W is essential to this bound; arbitrary dense edge-specific d-by-d transforms can instead cost O(Z d^2). Backward can accumulate gradients through the same segment integrals. Retaining every segment activation costs O((Z+M)d); checkpointing/recomputation trades time for memory. GPU sorting, ragged event scheduling, topological dependencies, reduction order and backward workspace remain unimplemented engineering risks.

## Measured local-basis storage, without constructing the expanded arrays

For each node the measurement unions integer bin ranges touched by its compact incident intervals. Samples receive every bin. Edge-bin intersections are counted arithmetically. These are exact counts for these grids, not an estimated sampling rate.

| Nominal bin width | Bins | M: active node-bin vectors | Z: edge-bin intersections | One d=32 FP32 array | Two arrays |
|---|---:|---:|---:|---:|---:|
| 128,000 bp | 235 | 737,786 | 1,219,202 | 94,436,608 B | 188,873,216 B |
| 32,000 bp | 938 | 2,607,525 | 3,081,824 | 333,763,200 B | 667,526,400 B |
| 8,000 bp | 3,750 | 10,065,073 | 10,532,230 | 1,288,329,344 B | 2,576,658,688 B |

The 32 kb pair is about 78.8 times smaller than the existing pair of expanded node arrays. This is a logical hidden-state ratio, not a whole-program memory or speedup claim. Gradients, segment saves, topology indices, raw observations, weights and allocator/workspace are excluded. At 32 kb an unnecessarily materialized width-32 edge-bin message array would add 394,473,472 B; streamed transformed-node gathers avoid requiring that complete array. For one message direction, `2Z + M` bounds segments from message endpoints alone. A safe bound including active-domain endpoints from both incident directions is `4Z + M`, or 14,934,821 at 32 kb, before merging coincident events. The local-support receipt labels the smaller quantity as an upper bound; it excludes these additional active-domain mask events and must not be used as a complete nonlinear-work bound. Dense channel transformations alone remain roughly 2.67 billion scalar multiply-accumulate terms per full M-by-32-by-32 transform. Arithmetic and irregular scheduling cannot be inferred from state bytes alone.

The support measurement used 192,778,240 B peak RSS and 3.64 seconds in the existing project Python environment, with one CPU thread setting and a 4 GiB address-space limit. The initial system-Python attempt stopped at a missing NumPy import and produced no data; no packages were installed. This measurement loaded only the compact observed tree, whose full SHA was verified again.

A useful first research comparison is 32 kb versus 8 kb, with the 128 kb grid as a deliberately coarse control. These are candidate resolutions, not approved replacements. A multiresolution extension can allocate finer local coefficients around observed edge/site changes, using observation-only criteria fixed without target truth. It must report its measured M and Z and a hard memory bound. Adding every propagated breakpoint can return to the original expansion problem.

## Controlled low-rank kernels are another approximation

For a global rank R representation, `h_v(x) = sum_r c[v,r] phi_r(x)`. One FP32 coefficient array with d=32 costs:

| Rank R | American coefficient bytes | Dense per-edge R-by-R overlap matrices if stored |
|---|---:|---:|
| 8 | 109,324,288 B | 153,412,864 B |
| 16 | 218,648,576 B | 613,651,456 B |
| 32 | 437,297,152 B | 2,454,605,824 B |
| 64 | 874,594,304 B | 9,818,423,296 B |

The interval mask generally mixes basis coefficients through `A_e[k,r] = integral_{left_e}^{right_e} phi_k(x) phi_r(x) dx`. Arbitrary dense mixing costs O(E R^2 d), not O(E R d). At R=32 that is about 19.64 billion coefficient-channel terms for one mixing pass. Computing these integrals analytically avoids storing all matrices but does not by itself remove the dense arithmetic. Local overlap sparsity, a proven structured fast transform, or an explicitly approximate diagonal/low-rank mask treatment is needed. Each imposes conditions that must be stated.

Global Fourier or smooth low-rank functions can ring or blur at ancestry boundaries; periodic boundary assumptions and endpoint handling are additional choices. Thirty-two global coefficients over 30 Mb should not be advertised as resolving arbitrary short ancestry tracts. A local compact basis is preferable for the first assessment because its spatial approximation and memory tradeoff are easier to audit. No claim is made that it is statistically optimal.

## Genotypes, target support and interpolation

The original 363,958,400 genotype categories remain an input cost. Every sample and every site must enter a streaming encoder, with missingness and site positions preserved as inputs. A concrete initialization can aggregate learned site embeddings into local coefficients while retaining site counts and within-bin position features. Such aggregation is noninvertible and differs from the inherited nonlinear per-tree site pooling. Reading every observation preserves input coverage, not all information after projection.

A learned site MLP over all cells can dominate compute and backward memory. Tiles plus recomputation can bound memory, but weights change each update, so stale learned encodings cannot be treated as exact caches. No disk observation cache is allowed. The architecture must separately budget genotype encoding and graph propagation; neither disappears merely because node functions are compact.

Decode at every original registered target sample and physical coordinate, retaining original registry identity, sampling multiplicities and loss weights. Box functions use the containing half-open bin; piecewise linear functions interpolate their specified coefficients, and multiresolution functions evaluate their local basis. Evaluate full dense predictions in streaming blocks if requested rather than storing every sample-tree state. Registered target positions remain unchanged even when the latent resolution changes.

Short ancestry tracts and true boundaries can lie inside a bin. In the American observed graph, 75.4%, 49.8% and 26.4% of edge spans are shorter than 128 kb, 32 kb and 8 kb respectively. Edge spans are not measured ancestry tract lengths, so these percentages are a representation warning, not an ancestry error estimate. The shortest edge interval is 1 bp; median edge span is 32,288 bp. No fixed coarse basis can guarantee those changes are retained. Averaging may suppress rare classes, displace boundaries, couple nearby targets and erase narrow tracts. A fine local residual using exact observed genotype/graph context at queries is a possible extension, with its own measured cost; it is not a free recovery of all discarded information.

## European disposition

Do not deploy the American per-node rank representation unchanged on Relate output. For the current European converter there is no shared internal state across trees to exploit. A streaming tree encoder could release each tree's internal activations and accumulate sample functions or local summaries, retaining full graph and genotype coverage. It would save retained hidden states but still process the 181.9 million edge rows, and nonlinear backward would need checkpointing or regeneration. Only sample IDs provide established cross-tree identity in this output.

A European fast path using linear sufficient statistics plus learned local readout would be a further architectural restriction and needs an explicit comparison. Neither age equality nor clade similarity is a valid substitute for the missing observed identity. Changing or simplifying the inference converter is outside this architecture-only assessment.

## Relationship to established methods

Tree-sequence storage and edge-difference traversal are established. Kelleher et al., *Efficient Coalescent Simulation and Genealogical Analysis for Large Sample Sizes* (2016), describes the succinct genealogical representation: https://doi.org/10.1371/journal.pcbi.1004842. Ralph, Thornton and Kelleher, *Efficiently Summarizing Relationships in Large Samples: A General Duality Between Statistics of Genealogies and Genomes* (2020), establishes efficient genealogical/genomic statistics: https://doi.org/10.1534/genetics.120.303253. Those results do not automatically give the same complexity to arbitrary learned nonlinear states.

Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (2020), uses event streams, node memory and graph operators: https://arxiv.org/abs/2006.10637. Xu et al., *Inductive Representation Learning on Temporal Graphs* (TGAT, 2020), uses functional time encoding and temporal attention: https://arxiv.org/abs/2002.07962. Applying such memory to genomic coordinate requires explicit node identity and direction semantics. Genome position is not ancestor age, and a causal scan along position is not interchangeable with a bidirectional genomic model.

Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations* (2020), parameterizes function-space operators with Fourier kernels: https://arxiv.org/abs/2010.08895. Low-rank functions, local bases, multiresolution representations and projected nonlinear operators are established numerical/neural techniques. Their combination with interval genealogy is not by itself a new general method. A possible contribution would be a carefully specified observed-graph model, efficient differentiable interval scheduling, and demonstrated ancestry accuracy/resource tradeoffs. The retrieved source/abstract evidence is a focused comparison, not an exhaustive novelty search or proof of priority.

## Gates for a future prototype

Before any training: implement a bounded CPU example and compare the projected operator with a brute-force piecewise integration of the same defined operator. Check interval splitting invariance, node renumbering equivariance, disconnected roots, missing observations, zero-support bins, endpoint conventions, gradient finite differences and checkpointed backward. Do not compare it with exact TreeLoom and label a projection difference a bug.

Before a full-parent launch: measure actual topology/event indices, genotype encoding cost, retained activations and allocator peaks; obtain ROOT's shared-host/process-family admission; preserve the 1 TB cap and no-cache policy. No launch is authorized by this report.

Before a predictive claim: keep the original observed graphs, sites, samples and target registries. Fix architecture/resolution using prospective TRAIN/validation rules. Compare matched training budgets and seeds, all ancestry classes, cohort groups, calibration, short tracts and boundary displacement. Use the same prospective independent European validation registry for original TreeLoom, cohort8 TreeLoom and CNN if making a comparative claim. One random-initialization TRAIN diagnostic does not measure learned accuracy. Preserve TEST for the existing final evaluation contract.

The evidence supports a bounded American local-function prototype as a separate research architecture. It does not support immediate live replacement, a general European compression claim, or an invention claim.

## Reading inventory and remaining gaps

`READ_INVENTORY_v1.json` lists every literature/source item used for this assessment, its hash, and the actual reading extent. The TGN, TGAT and FNO evidence is abstract-page reading; the tree-sequence papers are Crossref metadata/available abstracts. No full paper PDF or author implementation was read. The pinned Relate converter was read in full, Kastore format snippets were inspected, and the inherited execution/model call paths listed in the inventory were read. This supports the scoped comparisons above, not an exhaustive literature review.
