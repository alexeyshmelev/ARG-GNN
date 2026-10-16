# Same-marker proposal: source specification before implementation

This handoff implements the existing unregistered Hypothesis2 refinement in `research/reviews/next_useful_biological_mechanism_20261016_v1/REPORT.md`. The read-only registry snapshot contains56 decisions throughH057, no same-marker-support ID, and no authority to register or execute it. No new experiment name or ID is introduced.

The original prose leaves source blockers: no exact single-hold indexing API, no finite-gradient rule for all-empty log sums, no activation-memory treatment across marker blocks, and no concrete mapping into H045. These are addressed here before writing the isolated source. The scientific utility question and native-hold output resolution are retained.

## Complementary-split issue

`src/arggnn/mutation_routed_tree_observations_v1.py:262–309` explicitly enumerates BOTH complementary biallelic allele sets and uses `min(difference,ncalled-difference)`. Its scores are intentionally unpolarized. Multiplying equal complementary scores by source allele frequencies and summing can give `score * (f_g(0)+f_g(1)) = score`. That is no longer source-discriminating. This source must NOT derive the new feature from those five pooled support channels.

The pending positive all-path construction avoids that specific collapse. It seeds labelled references by actual allele at EACH SNP, propagates their32 source×age×allele channels, selects the focal query allele, and ONLY THEN pools sites. It never sums source frequencies over matched complementary allele sets. For two references in one group with opposite alleles, the query-matching weighted frequency is `w0/(w0+w1)` or `w1/(w0+w1)`, not their sum1. Subtracting unweighted1/2 leaves a nonzero contrast if weights differ. If all path weights are equal, the contrast is correctly zero. This is desirable loss of extra topology information, not an orientation bug.

Directional cavity messages retain which side of each recipient edge supplied evidence. The final all-path sum is symmetric in path orientation, and needs no arbitrary derived/ancestral allele polarity. Simultaneously permuting the allele categories at one SNP leaves the query-matched result invariant. VCF allele0 is never asserted ancestral.

## Concrete source fixes

- One original hold at a time, marker blocks of at most16. Live messages are `[V_hold, B*8*4]`, never whole-parent node×site×source arrays.
- CPU observation-only layout preparation certifies complete disjoint trees, all sampled leaves, source/age grouping and query-label exclusion. It retains every unary node. Group order is `4*(observed_age>=350)+observed_source_class`, matching the existing eight donor channels.
- Categorical input is bounded to -1 or0..3. Missing calls have no seed. Optional original per-site allele cardinalities1..4 are checked during preparation. Four fixed allele slots include unused zero-mass categories. No best mutation placement, winning allele, minimum-discrepancy tie or unary representative is selected.
- Empty log sums explicitly return negative infinity without differentiating an all-negative-infinity `logsumexp`. Prefix/suffix log sums give sibling exclusion without subtracting nearly equal masses. A finite sentinel is used only for all-empty cumulative prefixes. Effective edge log attenuation is constrained to[-16,-0.0001], a disclosed numerical/model restriction that keeps every nonempty path above that sentinel. No biological calibration is claimed.
- Nonreentrant checkpointing of each trainable marker block is mandatory. The stored block outputs are `[Q,24]` sums/max/counts. This bounds retained message activations by rematerializing blocks, while keeping the original genotype views, per-hold gate vector, reduction outputs and their gradients. It does not prove full-model peak memory or runtime.
- Fixed seed/count/path aggregation is evidence engineering. `BoundedEdgeAttenuationV1` and its small residual adapter are a separate UNPROVEN trainable extension using known operations. Supplying fixed external log attenuations to the evidence module does not make it a learned operator. The code does not instantiate or execute either path during authoring.

The source is a review artifact only. Root must freeze these explicit restrictions before any actual experiment. The gate parameterization is more specific than the original unconstrained positive-gate prose. No live source, registry, accounting or launcher is modified.
