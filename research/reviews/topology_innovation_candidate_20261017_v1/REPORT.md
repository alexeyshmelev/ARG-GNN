# Topology-innovation candidate assessment — 17 October 2026

**Decision: no-go. No new candidate or experiment is proposed.** A leaf-resolved message between topology-edit endpoints would avoid graph-level pooling, but the concrete version supported by the available observations is already H043. Adding event memory, a joint switch factor, soft matching or a distance-aware decoder would return to examined directions or established inference. This is a bounded source/literature assessment, not a proof that no useful ordered-tree architecture exists.

The current European result establishes the unresolved prediction problem, not its cause: TreeLoom has 97.0778±0.5674% span accuracy and 18.3807±1.3141% boundary F1, versus matched NoMem 97.0000±0.4978% and 19.0488±1.4275%. SD is across eight reused development simulations, with one optimizer seed. There is no independent-test or overall-ranking conclusion.

## What the inference observations actually supply

Available information is the phased observed variants and callability, ordered point-inferred marginal trees and node dates, genomic coordinates/spans, stable sampled-haplotype identities/ages, and permitted reference labels. Exact sampled-descendant sets can be derived. The inspected Relate export does not authenticate persistent internal-ancestor identity. Neither true recombination/prune/recoalescence histories, population paths, query ancestry at census350, nor an inferred-tree/event posterior is supplied.

For a query younger than the census age tau, the biological target is

    Y_tq = population(A_tq(tau)).

The ancestry boundary indicator is

    B_tq = 1[Y_tq != Y_(t+1),q].

Observed topology turnover D_t = clades(T_t) symmetric_difference clades(T_(t+1)) does not determine B_tq. Even a genuine recombination can retain the same source ancestry. A single subtree-prune-and-regraft can change clades containing samples whose lineage was never pruned: ((a,b),(c,d)) -> (b,((a,c),d)) by moving only a. Newly formed {a,c} and {a,c,d} do not identify a shared switch among their descendants. This counterexample was already recorded in the shared-switch rejection and is not a new result of this review.

The useful missing object would be evidence that distinguishes a source-changing lineage transition from within-source recombination and inference error. Renaming a deterministic clade contingency as an event posterior supplies no such evidence. Uniform averaging over compatible SPR explanations supplies an arbitrary mixture, not a sequence-conditioned posterior.

## The topology-edit operator already tested

Direct source inspection confirms H043 builds nearest removed/added proper-clade endpoints f-(q), f+(q) for each stable leaf, with explicit null endpoints. Its count and normalization are

    K_ab = sum_q 1[f-(q)=a, f+(q)=b],
    S_ab = K_ab / sqrt(K_a. K_.b).

It then computes edge-conditioned old/new messages from both endpoint hidden states and observed geometry, scatters them to the full trees, and performs the second shared tree sweep. This imports information before query readout and is materially richer than independently smoothing pooled graph embeddings. It is also an ordinary MPNN on a deterministic sparse bipartite correspondence graph. See anonymous_exact_clade_transport_v1.py:172–205,381–453.

The closest representative matched-budget falsifier has therefore already run: H043 true_partner versus own_state, identical geometry/flags/support, 30,788 parameters, H038 initializer, fresh AdamW, seed101, FP32, 32 TRAIN parents, eight reused VAL parents, and 960 complete-parent updates per arm. Only the foreign-state gather changes. True partner gives 96.5182±0.6201% versus 96.5029±0.6006%, a 0.0153-point difference below the registered 0.10-point threshold. **Do not repeat this test unchanged.** It does not disprove every possible topology mechanism.

Other answered comparisons close the obvious substitutions: H035 signed compositions lost to path sums; H045/H046 foreign episode/flank contexts failed practical accuracy gates; H048 graph-conditioned persistence failed both utility rules; H050 higher-order cut history failed both rules. H039 and H040 had positive development contrasts, which must be retained, but H042's simpler local flags exceeded H040's tensor/event controls at matched added budget. H044 localized SNP evidence worsened boundary F1, and H047 cohort-risk reweighting worsened its matched accuracy/NLL. These outcomes do not justify a cosmetic successor. H058 is a separate, already registered within-tree same-marker alignment test, not an untested cross-tree mechanism discovered here.

## Closest verified primary work

- Li et al., *Graph Matching Networks for Learning the Similarity of Graph Structured Objects* (ICML2019), §3.2 Eqs4–11: cross-graph node-state matching messages enter node updates before final graph readout. H043 substitutes known leaf-anchored sparse correspondence and genomic adjacency for dense attention. [Primary](https://proceedings.mlr.press/v97/li19d.html).
- Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (2020), §3.1: event messages update node memory, which enters subsequent graph embeddings. A new name for carrier/event memory would not distinguish this principle. [Primary](https://arxiv.org/html/2006.10637v3).
- Escrig Molina et al., *Graph Set Transformer* (2026 preprint), §3.1 Steps2–4: graph tokens exchange collection information and return it to nodes. Leaf-resolved ordered exchange differs from that bottleneck, but early collection feedback itself is established. [Primary](https://arxiv.org/html/2606.05116v1).
- *ARGformer*, §2.1, explicitly learns repeated ancestral-node co-occurrence across samples and intervals. Its identifier assumption must not be silently transferred to this export. [DOI](https://doi.org/10.1093/bioinformatics/btag438).
- *Graph transformer for ancient ancestry inference* (ARGMix preprint), §§2.1/4.1, uses reference-informed TMRCA attention. [DOI](https://doi.org/10.64898/2026.04.05.714076).
- Irving-Pease et al., *The selection landscape and genetic legacy of ancient Eurasians*, describes simulated ancestry-path supervision and inferred genealogies for chromosome painting. [DOI](https://doi.org/10.1038/s41586-023-06705-1).

Cached primary method text, not titles alone, was inspected for GMN, TGN, GST, ARGformer and ARGMix. For the Nature paper, the cached main-text ancestry-stratified methods description was inspected, not its supplement anew. These sources do not claim exact prior implementation of every domain composition. A useful controlled combination could still be publishable, but no new general operator or priority follows from the present evidence.

## Cost, leakage and stopping rule

For H043's existing sparse route, with N total native vertices, rank r, hidden width d and E_delta correspondence cells, projected messages cost O(N*d*r + E_delta*r) arithmetic and O(N*r + E_delta*r) working activation storage before the retained complete-tree sweeps and genomic BiGRU. E_delta <= n(T-1) because each stable leaf votes once per boundary. Exact descendant-bitset bookkeeping additionally depends on panel bit width; this is not constant-cost ancestor matching. These are analytic bounds, not timing or memory measurements. A single tree edit can alter outside messages throughout a tree, so no exact O(number of edits) whole-network speed claim follows.

Using simulator recombination identities, true population paths, true ancestor joins or boundary-derived features would change the inference information and leak targets. Observation-only clade matching is legitimate but does not certify biological homology. Older proxy labels do not pin census350 states. Reusing development outcomes to select an edit support, gate or threshold leaves a development experiment even if predictions are sealed before scoring.

No additional matched experiment is recommended because no surviving candidate is materially distinct. Reopening needs a specific new observable or qualified observation model, an explicit non-redundant computation, and a falsifier that isolates that computation beyond H043/local flags and ordinary sequence context. This prerequisite is not an invitation to add another heuristic posterior or to repeat the old pilot.

Scope: instructions, manuscript, ledger/decisions, source text, prior architecture/novelty reports and cached primary literature only. No model/array/truth/checkpoint payload read, model import/execution, experiment, installation, resource/guard/process query, registration or Git action occurred. Only this assigned review directory was written.
