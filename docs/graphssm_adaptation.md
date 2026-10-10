# Equation-based GraphSSM adaptation

Status: implemented and checked on tiny CPU fixtures on 2026-10-10. No empirical
accuracy result or exact published reproduction is claimed. This control exists
to make the comparison to temporal graph state-space models concrete.

## Authoritative references and provenance

Jintang Li et al., *State Space Models on Temporal Graphs: A First-Principles
Study*, NeurIPS 2024, [arXiv:2406.00943v2](https://arxiv.org/abs/2406.00943v2),
[full equations and appendices](https://arxiv.org/html/2406.00943v2).

The [official repository](https://github.com/EdisonLeeeee/GraphSSM) was inspected
at pinned commit **8431e061b7e740b930b36e1ec3e07ed2ea1f57ea**:

- [Cell and mixer definitions](https://github.com/EdisonLeeeee/GraphSSM/blob/8431e061b7e740b930b36e1ec3e07ed2ea1f57ea/graphssm/layers/ssm.py).
- [GraphSSM model assembly](https://github.com/EdisonLeeeee/GraphSSM/blob/8431e061b7e740b930b36e1ec3e07ed2ea1f57ea/graphssm/models/ssm.py).
- [S6 variant](https://github.com/EdisonLeeeee/GraphSSM/blob/8431e061b7e740b930b36e1ec3e07ed2ea1f57ea/graphssm/models/s6.py).
- [MIT license](https://github.com/EdisonLeeeee/GraphSSM/blob/8431e061b7e740b930b36e1ec3e07ed2ea1f57ea/LICENSE), copyright 2024 Jintang Li.

The project implementation is independent and paper-based. It does not vendor
source files or require the official repository's PyTorch Geometric, einops or
NumPy runtime. PyTorch and the project's existing tensor contract suffice.

## Preserved components

The model implements the GraphSSM-S4-style SISO family described by Equation (9),
Equation (10) and Algorithm 1. It is substantially more specific than appending
a generic SSM to independently pooled graph vectors.

1. Each block first applies a learnable graph diffusion approximation. The native
   implementation uses a GraphSAGE-style affine root-plus-incoming-neighbor mean
   map, matching the form of default unnormalized mean `SAGEConv`. Every node
   participates, including anonymous internal vertices. The supplied native
   spatial edges remain within each graph.
2. Only the first block mixes consecutive diffused representations, as advised
   by Remark 1 and Appendix B.3. The interpolation mixer implements Equations
   (43), (44):

   \[
   \rho=\operatorname{softplus}(W_\rho[z_{\rm previous}\Vert z_{\rm current}]+b_\rho),
   \qquad
   \xi=\operatorname{sigmoid}(W_\xi[z_{\rm previous}\Vert z_{\rm current}]+b_\xi),
   \]
   \[
   \widehat z=\rho\odot(\xi\odot z_{\rm previous}+(1-\xi)\odot z_{\rm current}).
   \]

   Current nodes lacking a predecessor bypass the mixer, per Appendix B.2.
   The argument order is previous then current, matching the inspected official
   interpolation implementation. Later blocks use ordinary current diffusion.
3. Each channel has a real diagonal state-space memory with
   \(U\in\mathbb R^{V\times D\times S}\),
   \(A=-\exp(\texttt{log\_nA})\in\mathbb R^{D\times S}\),
   and learned \(B,C\in\mathbb R^{D\times S}\). Initialization follows the
   paper and official HiPPO-real indexing convention \(A_n=-(n+1)\) for
   \(n=1,\ldots,S\). B starts at one, and C uses Xavier initialization.
4. The default recurrence is the paper's Equation (9) input approximation:

   \[
   U_l=\exp(\Delta_l A)\odot U_{l-1}
       +\Delta_l\,\widehat z_l\odot B,
   \qquad
   Y_l=\sum_s U_{l,:,s}\odot C_{:,s}.
   \]

   Raw U is carried into the next interval, separately from projected Y.
5. Blocks apply ReLU to Y and add a learned affine projection of the previous
   layer's features, following Equation (10). Optional dropout defaults to zero
   and is an explicit regularization adaptation when enabled.
6. At each graph boundary, the memory follows Equation (21) and Appendix B.2.
   Persistent leaves retain their state by stable identity. Newly appearing
   vertices receive zero memory. Removed vertices are dropped. A leaf that
   disappears and later reappears does not resurrect memory from a graph before
   its absence.

## Explicit task adaptations

- **Known genomic spans:** the scan sorts graphs by observed interval starts.
  Delta is the supplied width of the current normalized interval. The current
  graph is held over its own interval and updated at its right endpoint. This
  applies the snapshot recurrence to fully observed interval-bearing graphs.
  The paper supports supplied gaps and introduces learned softplus gaps when
  timestamps are unavailable. We do not replace known widths with learned gaps.
- **Partial node correspondence:** only nonnegative `node_ids` denote stable
  sample identities. Every -1 is an anonymous graph-local vertex, so internal
  vertices always initialize fresh across graph boundaries. This applies the
  paper's alignment rule using only the correspondence available in the input.
  No graph-local array position is treated as a persistent identity.
- **Per-position predictions:** the original code classifies nodes at the last
  snapshot. This task needs predictions for queried leaves at a fixed genomic
  grid. We select each query leaf's encoded representation in its containing
  tree and apply the classifier. Predictions are piecewise constant within a
  containing tree. They use snapshot-end representations and are not exact
  fractional-position continuous-time states.
- **Optional bidirectional offline inference:** `bidirectional=True` runs the
  same learned blocks over reverse genomic order and concatenates forward and
  reverse node representations before classification. Reverse gaps remain
  positive interval widths. This exposes both sides of the observed sequence
  for the noncausal ancestry task. It is not the published forward architecture.
  Report the option and its additional compute explicitly.
- **Optional exact ZOH cell:** `discretization="zoh"` replaces the paper's
  \(\Delta B\) input approximation by the exact single-hold factor from
  Equation (8):

  \[
  U_l=\exp(\Delta_l A)\odot U_{l-1}
      +\frac{\exp(\Delta_l A)-1}{A}\odot B\odot\widehat z_l.
  \]

  The implementation uses `expm1` and a near-zero exprel series to avoid
  cancellation and unstable derivatives for very small rates or widths.
  This is a separately labeled numerical variant, not the default GraphSSM
  update. A low-level constant-driving cell has the usual ZOH semigroup
  property. That does not establish interval-subdivision invariance of the
  complete stacked model with representation mixing and anonymous-node resets.
- **Finite arithmetic:** log rates are guarded only at floating-point exponent
  limits so the diagonal poles stay finite and negative under extreme optimizer
  values. Delta-times-rate is formed in log space to avoid overflow before its
  decay rounds to zero, while positive subnormal widths remain unrounded.
  Ordinary parameter values are unchanged. This guard and stable ZOH
  evaluation are numerical implementation choices, not scientific novelty.

The tensor input is the same observed `GraphSequence.x` and native graph
structure used elsewhere in the project. Stable integer IDs select alignment
and query ordering. They are never embedded as learned numerical features.
Clades are validated as shared metadata but are not additional learned inputs
in this control. Truth ancestry labels are absent from the model interface.

## Paper versus pinned source distinction

The inspected official SISO cell uses C shaped [D,S,S], returns a projected
[V,D,S] tensor, and its model loop passes that tensor back into the next cell
call. Algorithm 1 in the paper instead specifies C shaped [D,S], raw memory
U [V,D,S] and output Y [V,D]. This implementation follows the paper's stated
state/output separation. It therefore does not claim source-identical behavior.
The official source was used to verify initialization, mixing and model assembly,
not silently treated as interchangeable with every paper equation.

The proposed control is properly labeled **GraphSSM adaptation** or
**GraphSSM-S4-style adaptation**. Do not label it an exact GraphSSM reproduction,
S5/S6 implementation, newly invented interval operator, or a fully
subdivision-consistent model.

## API and verification

`GraphSSMAdaptation(input_dim, num_classes, hidden_dim=64, num_layers=2,
state_dim=16, discretization="paper", bidirectional=False, dropout=0.,
validate=True)` returns logits `[P,Q,K]` from `GraphSequence`.

`num_layers` is the number of GraphSSM blocks. Report this separately from the
spatial encoder depth of other families. Interval widths are in normalized global
genomic units, so pole time scales and memory order should receive the same
predeclared tuning budget as other model hyperparameters. For the strongest
noncausal comparison, include the bidirectional option and document it.

Twenty-three tiny CPU checks passed in **2.53 seconds** on 2026-10-10. Tests cover:

- Direct numerical agreement with Equation (9), including the C projection and
  separation of raw state from output.
- Equation (8) ZOH factors and constant-hold subdivision at the cell level.
- Zero, very narrow and subnormal intervals, stiff dynamics, overflowing
  delta-times-rate products and extreme rate guards, with finite gradients.
- Equations (43), (44), including bypass of new nodes.
- Correct stable-leaf alignment, anonymous resets, vanished/reappearing leaves
  and gradient flow through aligned memory.
- Varying internal-node counts, output shape, full training gradients, node-index
  permutations and stable-identity renaming.
- Storage serialization invariance, substantive coordinate-order sensitivity,
  fixed query/grid ordering, forward causality and optional reverse context.

Reproduce with the project environment sourced and GPUs disabled:

```bash
cd /workspace-SR003.nfs2/ARG-GNN
. scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARGGNN_PYTHON" -m pytest tests/test_graphssm_baseline.py \
  --basetemp=/workspace-SR003.nfs2/ARG-GNN/.runtime/tmp/pytest-graphssm -q
```

These tests establish implementation behavior on small fixtures. They are not
accuracy, runtime-scaling, benchmark-ranking or novelty evidence. Root owns
registry integration, experimental scheduling, tuning, commits and publication
claims. No packages were installed and no GPU job was launched for this work.
