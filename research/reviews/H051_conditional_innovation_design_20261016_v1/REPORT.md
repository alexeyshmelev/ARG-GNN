# H051 conditional innovation: source-only design, 2026-10-16

The reusable Torch/PyG prototype and 61 focused synthetic CPU cases are complete. H051 is an inactive prospective hypothesis, with no biological result or permission to register or fit. The identity arm is blocked on source feasibility. This worker changed only this new review directory, not registered/admitted implementations, experiment ledgers, guards, datasets, truth, checkpoints or Git state.

## One hypothesis and its present blocker

Within one authenticated inferred-tree condition, does equality of actual adjacent cut-boundary edge endpoints improve conditional innovation for false short ancestry tracts while retaining true short tracts, beyond cut-only fixed/count shrinkage and the same full-tree feature body?

This is an identity-utility question. Shrinkage, conditional expectation, graph memory and shared genealogical IDs already have prior art. An inferred edge can persist and still be wrong. Persistence is not calibrated ancestry confidence.

The current loader creates compact parents from original tskit node keys, then drops the internal-key mapping. The observation and whole-tree input records have no global internal-node identity field. A future sidecar can preserve original keys that genuinely exist, but cannot recover identity absent from the producer.

The inspected default official Relate exporter allocates fresh internal nodes inside each marginal-tree loop, using a tree-dependent offset. Cross-boundary endpoint equality is therefore unavailable/uninformative for that condition. Equal clades, times, packed indices and approximate branch matching must not be relabeled as supplied shared identity. If the gate degenerates, do not register this identity hypothesis on that source. A producer with genuine shared inferred IDs, such as a separately reviewed tsinfer condition, must be used by every arm. Its comparison against Relate pipelines is not an architecture-only gain.

## Operator and scope

For aligned persistent witnesses and hard cuts A,B,C, let Pi_X=L_X D_X^-1 L_X^T be the occupied-partition leaf conditional-mean projector. The implementation is

    J = (Pi_BC - Pi_B) Pi_AB
    carry_alpha = Pi_B + alpha_B (Pi_BC - Pi_B) Pi_AB.

Pi_B=Pi_B Pi_AB. Since B is coarser than AB and BC, Pi_BC-Pi_B is an orthogonal projector. Thus ||J||_2<=1, Pi_B J=J Pi_B=0, and J is the zero operator exactly when empirical A and C are conditionally independent given B. This is an occupied-table statement, not biological conditional independence.

Observation-only alpha_B in [0,1], constant inside B, gives a leaf-L2 contraction preserving B means. For cell states this means the count-weighted norm, not unweighted cell-row norm. It does not establish stability/calibration/identifiability of the learned model. Stable-leaf memory retains richer history.

Old state lives on occupied AB cells, output on BC cells. Triple is alpha1, marginal is alpha0, and fixed alpha.5 tests global damping. Count alpha is n_B/(n_B+16). The proposed identity alpha is s_B/(n_B+16), where s_B counts leaves whose valid (parent global key,child global key) cut-boundary edge pair is identical immediately before/after B-to-C. Missing/root=(-1,-1) is excluded. Endpoint equality, not edge row ID, avoids row-fragmentation artifacts. IDs never enter numerical embeddings.

## Implementation and complexity

`conditional_innovation_v1.py` supplies a parameter-free nn.Module, Torch index-add or native PyG SumAggregation, arbitrary trailing feature dimensions, leaf/AB input bases, and disjoint ragged batches. Fixed/count gates need no identity sidecar or namespace. No model, file reader, optimizer, random initialization, implicit scan, reset, detach or truncation is supplied.

The eleven int64 plan vectors hold 3H+2K_AB+K_B+2K_BC+3Q<=11H elements, at most88H tensor bytes/event excluding Python metadata. Every occupied count and triple count Q<=H. CPU construction uses sort-based unique, expected O(H log H). Feature transport uses O((H+Q+K_AB+K_B+K_BC)F) work/storage up to constants, without H-by-H or H-cubed production tensors. This does not bound a complete sequence's learned activations.

Within-carrier exponential relaxation is separate from event transport: rho=exp(-distance/scale), state'=rho*state+(1-rho)*forcing. Units must explicitly match bp or morgan. Scalar/per-cell distances and ragged event clocks require no padding. Constant forcing on a fixed carrier has the subdivision semigroup, not whole-model/tree-record refinement invariance. An optional monotone cumulative-map helper integrates Morgan distance, including plateaus. H051 keeps bp for every arm to avoid confounding the gate.

The source provenance record is an assertion, not a leakage detector or signature. Root must authenticate input/builder lineage. The API cannot identify targets disguised as observed cuts/endpoint keys.

## Prior art and novelty judgment

Whitehouse et al., *Tree Sequences as a General-Purpose Tool for Population Genetic Inference* (2024, DOI10.1093/molbev/msae223), already combine graph attention, node GRU and tree-sequence GRU. Their methods explicitly say the node GRU may exploit consistency in Relate's node ordering. Leaves have consistent sample order, internal nodes time order. This is not a wholly identity-blind baseline. Explicit global-key equality cross-tree edges are not specified, but times/topology/order may proxy identity. Their128-tree caps/subsampling, with task-specific sampling exceptions, provide scaling context, not novelty.

ARGformer (2026, DOI10.1093/bioinformatics/btag438), Section2.1, explicitly tokenizes unique inferred-node identifiers and uses repeated ancestral nodes across paths/intervals. Shared identity itself is prior art. Memory networks, higher-order/de Bruijn graph models and temporal GNNs also predate lifted overlap states/event memory.

Endpoint witnesses can vary with cuts fixed, establishing a possible extra channel relative to cut memberships. That does not establish information gain over full observed trees. If times/topology/order determine matching, equality gating supplies explicit computation bias. Utility must survive full-feature controls. This constrained mechanism is testable, but currently insufficient by itself for an AAAI novelty claim.

## Verification and prospective decision

`fixtures_actual_receipt_v1.json` records61 actual CPU passing cases. Features/autograd are FP32 with stated tolerances. Immutable cumulative-map and count/clock metadata use float64 where specified. Tiny dense projectors are test oracles only. Coverage includes algebra, conditional-independence zero, contraction/B means, gates, provenance/namespace rejection, leaf/key equivariance, ragged shape/clocks, state/scale gradients, native PyG equivalence, units and linear storage. These are engineering checks, not ancestry accuracy or short-tract evidence.

`prospective_protocol_v1.json` fixes matched controls, metrics and falsification. Root must freeze source/split/warmstart/evaluator hashes, genuine shared-ID feasibility and training/admission decisions before registration. No such data audit or run occurred here. Failure beyond fixed/count shrinkage, unacceptable true-short loss, full-feature redundancy or unavailable shared IDs rejects the proposed contribution. API and pinned source evidence are in `API_INTEGRATION.md` and `source_evidence_v1.json`.
