# Novelty assessment during H038
Updated 2026-10-12.

The current whole-tree PyG model, cavity sweeps, observed SNP gates and genomic bidirectional GRU do not yet establish a distinct machine-learning contribution. H038 tests a feature ablation on new independent parents. Its outcome cannot by itself establish the priority of a new operator.

Three possible next directions remain hypotheses:

- Partial relational transport between adjacent inferred trees. Fused and unbalanced Gromov–Wasserstein matching are established methods. A useful contribution would need a bounded, ambiguity-aware coupling of changed tree components, with a gain over a transport sham preserving marginal and unmatched mass. Correspondences cannot be called verified ancestral continuity.
- Directed ancestry transitions from changes in donor/query triplets. A CRF or ancestry HMM is established. The question is whether directed topology-change evidence adds information beyond emissions, generic boundary rate and a GRU receiving the same features. A gain only from smoothing is insufficient.
- Exact genomic-span recurrence using a learned sample projector. Graph diffusion, matrix exponentiation, low-rank state matrices, selective state-space models and resolution invariance already exist. A potentially useful distinction is changing whole-tree subspaces over genuinely persistent sample rows. It requires gains over a span-aware diagonal state model and a correspondence sham.

The projector formula is an L2 contraction with two decay scales. Arbitrary orthogonal projectors do not necessarily preserve nonnegative ancestry mass. Exact interval subdivision invariance requires unchanged generator and forcing with exactly integrated forcing. Constant topology alone is insufficient when SNP features change. Ridge regularization breaks the idempotence assumed by the closed form. Building the basis also costs O(Nr² + r³), beyond applying it in O(NrD).

Closest references inspected by the research agent:

- Fused Gromov–Wasserstein: https://arxiv.org/abs/1811.02834
- Unbalanced Gromov–Wasserstein: https://arxiv.org/abs/2009.04266
- RFMix: https://pmc.ncbi.nlm.nih.gov/articles/PMC3738819/
- GRAND: https://arxiv.org/abs/2106.10934
- Graph diffusion convolution: https://arxiv.org/abs/1911.05485
- Graph Neural ODEs: https://arxiv.org/abs/1911.07532
- S4: https://arxiv.org/abs/2111.00396
- Mamba: https://arxiv.org/abs/2312.00752
- S4ND: https://arxiv.org/abs/2210.06583

No new operator was implemented or trained in this assessment. No priority or superiority claim is supported. Closest prior art and decisive controls must accompany the next registered experiment.
