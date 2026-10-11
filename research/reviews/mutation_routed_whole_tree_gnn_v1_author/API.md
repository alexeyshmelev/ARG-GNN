# Mutation-routed whole-tree model API

Source-only prototype. Predictive accuracy, novelty and actual full-parent runtime remain unestablished. This handoff owns the neural source and this directory only. No data loader, trainer, optimizer fit, GPU invocation or Git operation is included.

## Entry points

Source: src/arggnn/mutation_routed_whole_tree_gnn_v1.py.

- MutationRoutedWholeTreeInputsV1.from_arrays(mapping, device="cpu") converts exactly the fields below from compact NumPy or Torch observations. It rejects any extra key, including a target key. Float observations become FP32, lookup arrays int64, donor masks bool, and genotype retains its integer dtype.
- validate_inputs_v1(bank, expected_haplotypes=280) performs CPU shape, finite-value, missingness, dated-tree, full-leaf-panel and strict sweep-level checks once during preparation. It validates representation, not biological source provenance.
- bank.to(device) transfers the complete compact bank without changing its dtype.
- MutationRoutedWholeTreeGNNV1(hidden_dim=32, expected_haplotypes=280, site_chunk_size=256, checkpoint_sites=True) has 29,764 parameters.
- model.forward_holds(bank, control="aligned") returns raw logits [T,H,4], decoding exactly once per original native interval/haplotype.
- model(bank, control="aligned") returns raw logits [R,Q,4] gathered from native holds, preserving requested query order.
- Allowed controls are "aligned" and "shuffled". They use the same literal parameter objects. The sole selected input difference is branch_support_true versus branch_support_sham. Independently fitted arms must start from identical state_dict bytes and use the same schedule, as frozen by the primary agent.

Default H=280 is the declared development panel. Its preparation validation requires actual Q=120 and exactly24 observed references in each C4 class. An expected_haplotypes override exists for bounded handmade fixtures and an explicitly frozen future panel, not silent sample removal. Loss, scorer and external truth access belong to the future driver. There is no target, teacher, baseline-logit or residual-skip argument.

## Exact observation fields

N is the total number of native vertices across complete trees. T is original native marginal intervals. H is persistent sampled haplotypes. S is original observed sites in covered support. A is padded categorical allele width1..4, Q requested actual queries, and R fixed readout centers.

| Field | Shape / Torch dtype | Meaning |
| --- | --- | --- |
| parent | N / int64 | Forest parents, roots=-1. No cross-tree edges. |
| node_time | N / FP32 | Displayed observed generations, nonnegative, rounded parents not younger. Raw-derived duration remains positive. |
| sample_index | T,H / int64 | Actual native vertex lookup for every persistent sampled column. |
| upward_levels | tuple of int64 vectors | Every vertex once, strict children-before-parent levels. |
| upward_children | matching tuple of int64 vectors | Original children for parents in each upward level. |
| upward_parent_rank | matching tuple of int64 vectors | For each child, rank of its parent within that upward level. |
| downward_levels | tuple of int64 vectors | Every nonroot once, strict parent-before-child levels, roots excluded. |
| edge_geometry | N,4 / FP32 | Four observed geometry channels below. Root rows zero. |
| branch_support_true | N,5 / FP32 | Genuine observed support fields below. |
| branch_support_sham | N,5 / FP32 | Fixed branch-placement permutation of complete support vectors. |
| genotype | S,H / integer | Categorical allele index or -1 missing; never an embedding index. |
| log_donor_allele_frequency | S,A,8 / FP32 | Log proxy-panel allele frequencies, not ancestry predictions. |
| donor_call_mask | S,8 / bool | Per-site available younger/older donor-class groups. |
| panel_allele_frequency | S,A / FP32 | Observed total-panel frequencies. A called allele must have positive frequency. |
| site_entropy | S / FP32 | Observed nonnegative categorical entropy. |
| site_hold | S / int64 | Genuine containing interval. Nondecreasing in original site order. |
| spans | T / FP32 | Positive physical interval widths in bp, contiguous original coverage. |
| readout_hold | R / int64 | Exact containing native interval for every scored center. |
| query_column | Q / int64 | Actual-query persistent columns, unique, carrying reference_class=-1. |
| reference_class | H / int64 | Only observed pure proxies: ANA0,CHG1,WHG2,EHG3; others=-1. |
| sample_age | H / FP32 | Observed sampling generations, unchanged across all intervals. |

Raw CPU observation dates remain float64 metadata in preprocessing. It verifies strict original parent chronology and computes log1p(parent-child duration) from those original dates before converting derived geometry to FP32. The model bank casts displayed node dates to FP32 and checks only non-reversed order. Equal rounded parent/child dates are accepted when the separate raw-derived duration channel is positive. Every native branch remains present, including unary branches. No epsilon date perturbation, pruning, or GPU FP64 is used. Preparation metadata explicitly certifies raw chronology and records rounded-clock collapse. Exact repeated sampled ages are checked in the model representation; original date/provenance verification remains a preprocessing obligation. If the separately derived positive duration feature itself becomes unrepresentable, preparation must report that issue rather than silently repairing it.

Geometry channel order:

1. log1p(child observed age)/log(10001).
2. log1p(parent observed age)/log(10001).
3. log1p(native positive duration)/log(10001).
4. Descendant sampled count/H.

Support channel order:

1. Exact allele-split support, tied best branches sharing support, normalized by callable splits.
2. Closest support weighted by1-minimum discrepancy, normalized by callable splits.
3. Mean orientation-free discrepancy over proper called allele splits.
4. Raw callable split count.
5. Empty evidence flag0/1.

Root support is [0,0,0,0,1] in both arms. Empty support rows have all first four fields zero. The model transforms count by log1p(count)/log(100001) inside both gates, leaving other support fields unchanged. Preprocessing must establish that sham is a fixed, within-tree, geometry-stratified permutation of full support vectors. The neural validator does not reconstruct the sham or certify its scientific strength.

The eight donor groups retain younger/older × C4 ordering, with absent classes and common group-callability following the original H028 contract. Inactive donor values may be zero or -inf. Called, available donor log frequencies must be finite and <=0. Padded unobserved alleles can be inactive. Original multiallelic sites remain present.

## Neural computation and retained gradients

Each site/haplotype forms nineteen channels only for its current processing chunk:8 gathered log frequencies,8 available-group flags, log panel allele frequency, site entropy, and called flag. All nineteen channels are zeroed for a missing call. Called sites pass through a shared19→32→32 MLP. Mean/max reductions pool within original hold and haplotype, alongside observed count, empty flag and physical span. Empty pools use zero mean/max, with explicit emptiness. No full S×H×19 tensor is cached.

Optional non-reentrant checkpointing recomputes site MLP and local reductions during backward. Its inputs are compact original genotype/frequency slices. It retains gradients through learned site encoding and subsequent pooling. Count division uses clamp_min(1) for an empty-pool denominator, not an output probability floor.

The upward sweep includes every native vertex/edge. Geometry/support gates condition learned child messages, with child-count normalization. The downward sweep forms parent outside state plus parent local input and sibling messages, subtracting the receiving child's recorded upward message. Receiving-child subtree content is introduced only at final leaf context, not into outside recursion. Learned gates, sweeps and classifier are ordinary established GNN machinery.

The leaf sequence retains every original T×H state. One vectorized bidirectional GRU processes the complete parent with batch dimension H, augmented by span and positive previous/next interval-center distances. These are derived from contiguous widths. There is no artificial250kb reset, detached history, inferred-ancestor recurrence or numeric-node embedding. Physical x and ancestral age t remain distinct.

The classifier acts on native [T,H] states, then exact readout/query lookup produces [R,Q,4]. Thus repeated scoring bins do not replicate complete-tree encodings or large hidden vectors per readout. The final head uses ordinary initialization, rather than a zero residual correction. Its outputs are raw logits for an external unchanged softmax/NLL scorer.

## Minimal integration example

~~~python
from arggnn.mutation_routed_whole_tree_gnn_v1 import (
    MutationRoutedWholeTreeInputsV1,
    MutationRoutedWholeTreeGNNV1,
    validate_inputs_v1,
)

# observations is prepared entirely from the frozen observed-only source.
bank = MutationRoutedWholeTreeInputsV1.from_arrays(observations.model_arrays())
validate_inputs_v1(bank)
model = MutationRoutedWholeTreeGNNV1()
logits = model(bank, control="aligned")     # [R,120,4], CPU example
sham_logits = model(bank, control="shuffled")
# External driver owns authorized TRAIN loss and sealed VAL prediction.
~~~

Compatible preprocessing source is src/arggnn/mutation_routed_tree_observations_v1.py. Its MutationRoutedTreeObservationsV1.model_arrays() supplies the exact keys. Neither source opens actual data or targets by import.

Do not feed model logits as targets to another model, project older references down to time350, invent query source labels, center over same-age query predictions, merge producers, detach history, discard native holds, or infer physical support from ancestry truth.

## Bounded checks and limitations

model_receipt.json records CPU FP32 checks on two complete four-haplotype trees, including multiallelic observations and missing calls. Both width8 and the proposed width32 cell were checked. At width32, all38 parameter blocks receive finite gradients. Native-node renumbering differs by at most2.98e-8, sampled-column reordering and allele relabeling are exactly invariant in the fixture. Empty-site backward is finite. All-missing tuples ignore arbitrary unavailable frequency/entropy values. Identical support fields produce bit-identical control outputs. Swapping actually distinct support vectors changes logits. Support gradients cross both genomic directions, confirming no detached sequence boundary in the fixture. Same-level parent/child schedules and extra target keys are rejected.

These are basic correctness checks, not a broad parity matrix, accuracy evaluation, native-competitor reproduction, full-parent resource profile or proof of biological utility. Runtime/activation memory are unmeasured. The current prototype checkpoints site encoding; complete-tree batching/checkpointing for larger actual parents belongs to a separately reviewed execution layer. Full-parent ordered sequence support must survive any future batching. The primary agent freezes the matched initial20 whole-parent updates and any explicit continuation. No fitting is authorized by this API file.

The chronology follow-up used a five-vertex tree containing a native unary branch at raw CPU ages10000 and10000.00001. Displayed FP32 ages both equal10000, while its raw-derived normalized log-duration is1.0857189636226394e-6. Integration preserves the original parent array and all five vertices, accepts the rounded tie, and has finite width32 FP32 outputs/gradients. Reversed displayed chronology and a zero duration feature are rejected. The receipt records exact values and preprocessing certification. No dates were adjusted and no branches were pruned.
