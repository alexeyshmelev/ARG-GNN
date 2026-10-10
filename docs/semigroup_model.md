# Experimental graph-conditioned affine semigroup candidate

## Scientific status

This is a candidate design and CPU implementation, with no novelty or performance claim. Matrix-exponential flow, zero-order-hold discretization, dissipative symmetric/skew parameterization, semigroup composition and ordered products are established mathematics and prior model ingredients. Input-conditioned dynamics overlap selective/liquid state-space models. The full-matrix comparison tests cross-state mixing, not the existence of order sensitivity.

Diagonal affine systems already preserve order through input-dependent forcing. Even when diagonal homogeneous matrices commute, affine augmented generators have commutator forcing term `A_i*b_j-A_j*b_i`, generally nonzero. The diagonal control is therefore indispensable. Existing GraphSSM also needs its exact-ZOH mode compared under clearly disclosed encoder differences, rather than only its paper discretization.

Prior-art investigation is separate. [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951) explicitly uses input-dependent state transitions, and [Lipschitz Recurrent Neural Networks](https://arxiv.org/abs/2006.12070) constructs stable continuous-time recurrent units. Their detailed bilinear and symmetric/skew formulations must be credited before asserting architectural distinction. The source titles and abstracts were independently verified through the arXiv API. This note makes no first-method claim.

## Graph representation and generator

Use an independent `SharedGraphEncoder` at every marginal tree, optionally on the same all-node observed-reference augmentation as every other study family. Query-leaf states `h[g,q,H]` are matched by persistent identities, with no learned ID embeddings. They generate a latent `d`-state affine dynamics matrix and forcing for each graph/query.

```
A_g = rate * (-diag(softplus(d_g)+epsilon) - U_g U_g^T
              + V_g W_g^T - W_g V_g^T)
b_g = rate * tanh(B(h_g))
```

Factors are bounded `tanh` projections and scaled by `1/sqrt(rank)`. The rate is a learned positive scalar. The default latent dimension is 8, low-rank factors have rank 2 and initial rate is 20 per normalized genomic-coordinate unit. These defaults are pilot choices, not validated hyperparameters. A diagonal control uses the same local bank, damping, rate, forcing and readout, omitting cross-state factors.

The local encoder and generator never depend on record counts, interval midpoints, previous latent states or tree index. That restriction makes redundant splitting leave source generators unchanged. The current implementation transports each persistent query's latent state independently. It does not exchange dynamic states across queries or implement clade-matching transport.

## Exact ordered and set transports

Form the augmented affine generator `M_g=[[A_g,b_g],[0,0]]`. The constant-generator flow over distance `delta` is `E_g(delta)=matrix_exp(delta*M_g)`. Applying the upper block updates `z` without accumulating floating-point drift in a repeated homogeneous constant coordinate.

For ordered mode, full-interval exponentials are batched once. Forward boundary states use increasing genomic order. Backward boundary states use the reversed intervals with positive durations, never the unstable inverse `exp(-delta*A_g)`. At fixed query coordinate `x` in interval `[a_g,b_g)`, forward and backward states include exact partial flows of lengths `x-a_g` and `b_g-x`.

Boundary states start at zero at the supplied tile endpoints. Tile endpoints remain fixed and independent of tree segmentation. Moving the tile boundary changes the available context, and is outside the subdivision guarantee. The source coordinate's scale must be recorded. Normalized physical distance is not interchangeable with genetic distance across chromosome lengths or recombination maps.

The set control instead evaluates `matrix_exp(sum_g interval_width_g*M_g)` from zero and broadcasts that context to all fixed query positions. This removes physical arrangement while retaining span weights and local target-tree features. It is a weighted-set control. Rearrangement can still change the local feature at a coordinate, so compare global contexts or predictions where the local target graph is preserved.

A stronger order-free control, `homogeneous_set`, uses the same span-weighted generator average for the left/right partial distances at each query coordinate. It therefore retains position and boundary-length effects while discarding arrangement. For a constant generator field it matches ordered transport exactly. This avoids attributing simple partial-versus-full-span readout effects to ordering.

With bidirectional symmetric readout, the classifier concatenates local `h`, forward+backward state and their squared difference. Chromosome reversal swaps the scans and preserves predictions. A concatenating directional readout is available explicitly, with no reversal guarantee. Forward-only mode supports a directional representation, but offline ancestry inference defaults to two-sided context.

## Established guarantees and limits

For one unchanged generator, `E(delta1+delta2)=E(delta2)*E(delta1)`. Replacing an interval by arbitrary disjoint contiguous subintervals with identical graphs/features therefore leaves forward/backward states and logits unchanged at common query coordinates, up to floating-point error. Different generators need not commute. The set sum is additive under the same refinement operation.

`Sym(A)=-rate*(diag(softplus(d)+epsilon)+U*U^T)` is negative definite. With `alpha=rate*epsilon`, the homogeneous flow satisfies `||exp(delta*A)||_2 <= exp(-alpha*delta)`. Products over changing intervals are contractive as well. This applies to differences of states evolving under the same forcing. A forced trajectory can grow in norm, and the augmented affine generator itself need not be dissipative. Do not claim every augmented transition matrix is a contraction.

Node relabeling, common arbitrary persistent-ID renaming and graph serialization permutations preserve outputs. The model is sensitive to genuine physical arrangement. The symmetric bidirectional readout makes complete reversal an equivalence, which is appropriate for the chromosome interpretation but not every ordered-graph application.

These properties do not prove better accuracy, expressivity relative to general SSMs, robustness to inferred-tree error or biological identifiability. The full matrix offers channel coupling and conditional rotations. Whether those help after strong reference features is an empirical question.

## API and controls

`SemigroupGraphModel(input_dim,num_classes,hidden_dim=64,num_layers=2,latent_dim=8,generator_rank=2,matrix_kind='full',aggregation='ordered',bidirectional=True,symmetric_readout=True,initial_rate=20,minimum_damping=1e-3,validate=True)` consumes the existing GraphSequence contract and returns logits `[P,Q,K]`. Setting `num_classes=1` gives raw scalar regression outputs `[P,Q,1]` with no softmax.

- `matrix_kind='diagonal'`: exact affine-ZOH selective control, retaining ordering through forcing
- `aggregation='span_set'` or `SpanSetSemigroupModel`: full-span weighted-generator-sum set control
- `aggregation='homogeneous_set'`: arrangement-free, position-aware homogeneous-average control
- `symmetric_readout=False`: explicit directional concatenation
- `bidirectional=False`: forward-only control
- `.generator_bank(seq)`: local bank plus `A`, `b` and augmented matrices in original graph order
- `.encode_context(seq)`: local, forward and optional backward features at fixed coordinates
- `.configuration_manifest()`: serializable options, parameter counts, schema and implementation/local-encoder SHA256 values. No files are written

The diagonal parameter schema omits U/V/W projection heads entirely. Its allocated parameter count therefore reflects its active generator. `model.inactive_parameter_count()` still reports identically zero difference-feature head weights in the full-span symmetric set control. The full-span set readout also has an identically zero forward/backward difference feature, unlike the ordered and homogeneous-set modes. Report allocated and structurally inactive counts, and consider matched-active-budget controls if a gain survives initial testing. No registry, frozen protocol or existing model implementation is modified by this module.

## Cost and discriminating experiments

For `G` graphs, `Q` queries and latent dimension `d`, full-step exponentials cost roughly `O(G*Q*(d+1)^3)`. Fixed-coordinate partial exponentials cost `O(P*Q*(d+1)^3)` per direction. Boundary scans cost `O(G*Q*d^2)`. The first implementation batches exponentials but uses Python sequential state scans, with one small matrix-vector operation per interval. GPU launch overhead can dominate at large `G`. A parallel affine scan or one-sequence compiled kernel would be a subsequent efficiency implementation, requiring separate validation.

At `d=8`, augmented matrices are 9x9 and tensor storage is small relative to dense cross-node attention. This is eight states per tracked query. In the current GraphSSM adaptation, `state_dim=16` means sixteen states per hidden channel, giving 512 states per node at width 32. Matching the numeric `state_dim` option does not match dynamic state storage or active capacity. The CPU probe below does not establish GPU throughput or peak memory. Matrix-exponential autograd stores additional intermediates. Extreme rates may create numerical stiffness despite mathematical dissipativity. Test precision and throughput at actual tree counts before scheduling a broad run.

The lead controls are full versus diagonal exact affine dynamics, ordered versus span-set, and exact-ZOH GraphSSM under matched observed inputs and disclosed local encoders. Match coordinate windows, readout direction and training protocol. Synthetic tests should separate structural tree changes without ancestry changes from actual ancestry switches. Evaluate calibration, switch localization and short-tract recall, rather than relying on a single accuracy improvement.

CPU tests cover exact affine subdivision, diagonal forcing order sensitivity, homogeneous dissipativity/contraction, variable graph shapes and finite gradients, node/ID/serialization symmetries, common-coordinate refinement, chromosome reversal and arrangement sensitivity versus the set control. These tests verify implementation properties only.

## Recorded CPU engineering probe

The final parameter schema passed all **10 CPU tests** on 2026-10-10 (`pytest tests/test_semigroup_model.py -q`, 2.41 seconds). This includes scalar regression and provenance output in addition to the flow, symmetry, control and gradient checks above.

A bounded label-free probe ran at 2026-10-10T02:19:55.534778+00:00 on `data/ancient_common_input_pilot_v2/train/rep000/prepared.pt`. The observed input has **61,236 nodes, 972 graphs, 8 tracked queries, 128 scoring positions and 48 input columns**. Each model used hidden width 32, two graph layers, latent dimension 8, rank 2, four outputs, initial rate 20, two-sided symmetric readout and `validate=False`. The environment was Python 3.11.17, PyTorch 2.6.0+cu124, CPU, one PyTorch thread and `CUDA_VISIBLE_DEVICES=''`.

For each configuration, initialization used seed 42. Evaluation mode retained gradients. One untimed forward/backward pass warmed the model, followed by three measured passes with `output.square().mean()` as the loss. There was no optimizer update or ancestry-label access. Input loading, contract validation, finite-value checks and loss construction were outside the timed forward/backward regions. Every output and every present parameter gradient was finite.

| Dynamics / aggregation | Allocated parameters | Structurally inactive parameters | Forward seconds, median (range) | Backward seconds, median (range) |
| --- | ---: | ---: | ---: | ---: |
| full / ordered | 8,165 | 0 | 0.345 (0.329–0.358) | 0.605 (0.605–0.624) |
| diagonal / ordered | 6,581 | 0 | 0.320 (0.319–0.321) | 0.585 (0.584–0.589) |
| full / homogeneous_set | 8,165 | 0 | 0.219 (0.218–0.219) | 0.083 (0.080–0.088) |
| full / span_set | 8,165 | 32 | 0.216 (0.210–0.218) | 0.058 (0.057–0.058) |

These timings supersede the earlier diagonal probe that retained unused U/V/W heads. They establish a modest CPU feasibility observation for this replicate, without trained predictive results, a hardware-normalized speed comparison, peak-memory measurement or a GPU throughput estimate. Ordered transport still uses sequential Python interval scans. The diagonal control uses the same augmented matrix-exponential routine, so this probe does not benchmark an optimized diagonal SSM kernel.

Recorded SHA256 values:

- Prepared observed tensor: `92d1f4e438de6fe4792096a0af096064cd8ae632b7cb9d8ea77925ff6e9daa6a`
- Semigroup implementation: `4a7a909b214d02f929c569907e97d72aa53e08a42c1b7843a75997448463ca3c`
- Local encoder dependency `models.py`: `8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5`

The configuration manifests include the schema and active/inactive parameter counts. The configuration hash covers the serialized configuration, while source hashes separately identify the implementation and local encoder. These hashes are provenance fields, not a hash of trained checkpoint weights. The four configuration hashes and raw timed passes are retained here for audit:

```json
[
  {
    "matrix_kind": "full",
    "aggregation": "ordered",
    "configuration_sha256": "55b2991229a1ca809d16211de97360aec2ec71b5893f54233421c01ba8138354",
    "measurements": [
      {
        "forward_s": 0.32875117380172014,
        "backward_s": 0.6236906014382839
      },
      {
        "forward_s": 0.34519067127257586,
        "backward_s": 0.6051493752747774
      },
      {
        "forward_s": 0.35846538469195366,
        "backward_s": 0.6051365649327636
      }
    ]
  },
  {
    "matrix_kind": "diagonal",
    "aggregation": "ordered",
    "configuration_sha256": "f4201a376a13c2911e99e50a853b56cd36a04e5db3f8a6f39d0fe7326b220e80",
    "measurements": [
      {
        "forward_s": 0.31922553572803736,
        "backward_s": 0.5843487419188023
      },
      {
        "forward_s": 0.32092047203332186,
        "backward_s": 0.5886416379362345
      },
      {
        "forward_s": 0.3200934687629342,
        "backward_s": 0.5850836345925927
      }
    ]
  },
  {
    "matrix_kind": "full",
    "aggregation": "homogeneous_set",
    "configuration_sha256": "dd2096cd5bf401d9856ca7a84b6ba1856481bbf68851d1dc6c2ed7af3f8f9f08",
    "measurements": [
      {
        "forward_s": 0.21898410934954882,
        "backward_s": 0.08306779712438583
      },
      {
        "forward_s": 0.21916094981133938,
        "backward_s": 0.08829025831073523
      },
      {
        "forward_s": 0.21782780345529318,
        "backward_s": 0.07986792456358671
      }
    ]
  },
  {
    "matrix_kind": "full",
    "aggregation": "span_set",
    "configuration_sha256": "7c39cb0ef083b6daddc9d693c953eabf9374b9f23534050d1a3fe1c6da6d66cc",
    "measurements": [
      {
        "forward_s": 0.21645610872656107,
        "backward_s": 0.058091175742447376
      },
      {
        "forward_s": 0.2178247245028615,
        "backward_s": 0.05770109407603741
      },
      {
        "forward_s": 0.20987196266651154,
        "backward_s": 0.05746135301887989
      }
    ]
  }
]
```
