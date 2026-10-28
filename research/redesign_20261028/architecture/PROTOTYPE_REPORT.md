# Bounded TreeLoom prototype and CPU evidence

The proposed bounded architecture is implemented in `window_model_frozen_v1.py`, SHA256 `e0528c8e0d01b69512fe588dc74ec966a3855489abf4160a8ef4b9111e5a4e12`. The development `window_model.py` has the same bytes at this handoff. The frozen file is a new finite-context model and is not compatible with historical checkpoints. Original models, author baselines, datasets, jobs and Git were not changed.

## Implemented predictor

Two complete native-forest PyG upward/downward cavity sweeps share learned weights. Between them, three observed-time cut partitions (100, 350 and 1000 generations) exchange projected states through persistent sampled leaves, alongside a separate same-leaf route. The second sweep distributes those messages within each complete current tree. All selected nodes, edges, leaves and SNPs enter the computation. Multiple roots and polytomies are supported; sampled interior nodes are explicitly rejected.

Ancient samples older than a cut remain singleton computational carriers with a false validity feature. They are not claimed to biologically coexist at that younger cut time. A second feature marks root saturation. Internal-node numbers never enter learned features.

Observed metadata has 17 channels: nine cohort categories, five reference-role categories including unknown, observed age and query/reference role. SNP evidence uses known reference classes and all observed cohort categories, including query cohorts, to calculate smoothed observed allele frequencies. These are genotype-derived quantities, not query ancestry. SNP encoding is evaluated once per categorical allele and gathered to samples, then mean/max pooled per selected tree. The classifier also receives the focal SNP encoding and focal observed metadata directly.

This intentionally differs from historical TreeLoom in more than finite context: it omits the old learned GRU and old exact-clade flag/episode adapters, adds explicit observed-cohort frequency evidence and a direct focal readout, and gates tree edges with observed ages/duration without the old explicit branch-concordance summary. The proposal retains the biological observations in selected full trees, not all old engineered feature tensors or old equations. A formal A-versus-B study must align its local encoder and focal/metadata inputs before attributing gains to communication.

The `structure`, `leaf_only` and `no_cross` modes have the same registered parameter tensors. `no_cross` skips communication and reuses the first sweep's leaf result, because the former second sweep had the same base, weights and gates. `leaf_only` skips the cut branches. Neither control is intentionally slowed by unused transport. Unused parameters have `grad=None` in these efficient modes and must not trigger an all-parameters-connected training assertion. The toy qualification compares missing gradients as mathematical zero. At width 16, no_cross has 30 connected tensors/4,854 elements, leaf_only 33/5,118, and structure 42 tensors. The width-32 model has 23,822 total parameters.

## Context representations

Consecutive intervals use exact double integrals of an exponential physical-distance kernel. Matrices are constructed separately per window with storage O(sum_w T_w^2), then applied by cut-to-leaf gather, within-window temporal multiplication and leaf-to-cut mean. There is no clade-pair matrix or sample-pair matrix.

Uniform-coordinate packets have one forced focal and random context draws. The adapter duplicates the complete observed tree and its observed SNP payload for each draw, including repeated draws selecting the same native tree. This bounded reference does not deduplicate learned computation. Target coordinate tokens therefore remain distinct before nonlinear updates. Source quadrature weights are 1/(K-1) for the uniform random draws; the forced focal has zero random-integral weight and its own local residual route. No native interval span is multiplied into this path. The normalized kernel mean is a ratio estimator, not an unbiased nonlinear full-context predictor.

No end-to-end subdivision invariance is claimed. Fixed-coordinate routing is invariant to serialization changes only when the observed field at each draw is unchanged; current per-native-interval SNP mean/max/logcount pooling can change that field after interval subdivision. Canonical coalescing or a fixed-coordinate SNP neighborhood is not implemented in this prototype. The exact tested split identity concerns the linear interval kernel alone.

## Evidence

`CPU_TESTS.json` records finite forward/backward, leaf and node permutations, multiple windows sharing one panel, collated panels, no cross-example leakage, constant and leaf-mass preservation of cut transfer, singleton/empty-SNP/missing-focal handling, forests, polytomies, ancient-singleton masks, zero-span rejection, label-whitelist rejection, core selection, coordinate multiplicity and weighting, coordinate-draw permutation, and efficient-control parity.

Leaf permutation error is zero. Node-renumbering and independent-window comparison errors are at most 2.98e-8. The source/refined integral-kernel error is 4.44e-16. Efficient no_cross and leaf_only logits and all compared gradients match the preserved pre-skip implementation exactly on the toy qualification. The no_cross core output is unchanged after perturbing a neighbor SNP, while structure changes by 3.80e-6. Direct focal SNP and cohort perturbations both change the output.

`REAL_COORDINATE_CPU_PROBE.json` uses the actual TRAIN000 uniform-coordinate packet `../data/train000_uniform_coordinate_window_v2.npz`: five full trees, 9,515 nodes, 952 persistent samples, 100 SNPs and one present-day focal row. With two CPU threads and width 32, observation preparation took 0.0651 s, structure forward 0.2281 s, structure backward 0.1625 s, and efficient no_cross forward 0.0631 s. Peak process RSS was 1,018,765,312 bytes. All 42 learned tensors received finite, nonzero gradients from a numerical squared-logit probe.

The earlier consecutive five-tree packet has 39 SNPs and four BAA focal rows. Its recorded forward/backward times were 0.1930/0.1530 s and peak RSS 935,759,872 bytes. `REAL_CPU_PROBE_v1.json` predates efficient controls and the coordinate adapter, so its old no_cross timing is not a current speed comparison. Single calls are not benchmark confidence intervals or GPU timing estimates.

The first neighbor-connectivity test failed because the synthetic perturbation merely exchanged complementary SNP rows, preserving the model's mean/max pooling. The corrected test changes one observed call. Failure logs are preserved. Independent review identified and fixed shared-panel expansion for multiple within-parent windows; corresponding regression tests now pass. Source snapshots v1/v2/v3 preserve earlier implementations.

No training labels were read by these model probes. No optimizer update, ancestry accuracy calculation, CUDA initialization, GPU execution or learning experiment was performed. These results qualify a bounded implementation and basic mathematical contracts, not predictive value, convergence, throughput at training scale or generalization.

## Public API for root-owned experiments

Import all classes and preparation functions from the SAME frozen module, since Batch type identity is checked:

```python
from research.redesign_20261028.architecture import window_model_frozen_v1 as wm
obs = wm.load_observations(npz_path)  # never accesses the NPZ labels member
batch = wm.prepare_observations(obs) # CPU immutable observation routes
model = wm.WindowTreeLoom(width=32, classes=6, scales=3)
logits = model(batch, mode="structure")
core_logits = model.core_logits(batch, mode="structure")
```

For independent packets, call `prepare_observations(collate_observations([obs1, obs2, ...]))`. For an in-memory raw data packet, explicitly select `{k: raw[k] for k in wm.OBS_KEYS}`. If it contains sampled-coordinate context, call `wm.coordinate_observations(selected, {k: raw[k] for k in wm.SOURCE_CONTEXT_KEYS})` before collation. Never pass the raw labels dictionary to preparation. Training targets stay in the caller and align with focal-row order; `batch.core_focal_mask` selects core rows.

`batch.to(device)` moves all observation tensors and prepared routes. Both models and batches must be bound to the admitted device by the root-owned runner. Cache only these immutable observations/routes between updates; every learned activation, site encoding, gate and transport value is recomputed. Draws can be prospectively fixed or resampled according to a registered policy. The framework does not authorize a GPU or a training run.

Independent review passed nine additional tests on the frozen SHA above (`../review/`). Two limits remain explicit. First, no_cross still encodes the entire matched context packet even though only the focal tree affects its loss; its timing is not the fastest possible single-tree implementation. Second, the sampled kernel presently normalizes ordinary exponential weights. Extremely small learned length scales can underflow every positive context-source weight for a distant forced focal, producing a zero denominator. Defaults were finite in the probes. Root-owned diagnostics must reject nonfinite losses/used gradients, and wider length-scale tuning should first use stable log-weight normalization in a separately versioned successor. The frozen v1 source is unchanged.
