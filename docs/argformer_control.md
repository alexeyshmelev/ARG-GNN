# ARGMix and whole-tree graph-transformer common-input controls

These are **independent adaptations**, not exact reproductions of ARGMix or
ARGformer. The primary comparator is `ARGMixTreeControl(token_scope="full_tree")`.
It uses the ARGMix paper's relative query/key/value attention on every node of
each observed marginal tree, with the same 48 observed features given to the
other neural comparators. A generic whole-tree graph Transformer and an optional
published-style nearest-reference subgraph scope are also available.

## Primary sources and what they compute

Bonet et al., **ARGformer: learning on ancestral recombination graphs with
transformers**, Bioinformatics 42, btag438 (2026),
[DOI](https://doi.org/10.1093/bioinformatics/btag438),
[primary full text via Europe PMC](https://europepmc.org/articles/PMC13501305).

Section 2.1 encodes leaf-to-root paths using an ARG-specific vocabulary of
ancestral **node identifiers**, a ModernBERT-style encoder, masked-node
pretraining with masking probability 0.30, and supervised contrastive
finetuning, Eqs. (1)-(5). The official README distinguishes local path embeddings
from downstream averaging over unique marginal paths. Its aggregation helper
gives each unique path equal weight within a haplotype and averages haplotype
means for individuals. This is neither an ordered genomic sequence Transformer
nor a whole-tree node classifier. An exact ARGformer baseline would require its
ARG-specific identity vocabulary, self-supervised training data, and reference
retrieval protocol. Those are incompatible with this study's ID-renaming
contract and common supervised local-ancestry task. **No model named
ARGformerControl is supplied**, and these controls must not be described as an
ARGformer reproduction.

Shanks et al., **Graph transformer for ancient ancestry inference**, bioRxiv
v1, posted April 7, 2026,
[DOI](https://doi.org/10.64898/2026.04.05.714076),
[primary full text](https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1.full).

This paper introduces **ARGMix**. Results Section 2.1 and Methods Section 4.1
describe query-centered subgraph classification inside a single marginal
coalescent tree. Methods Section 4.3 uses 25 nearest observed references,
three immediate inferred ancestors, and the queried leaf, giving 29 biological
nodes. It uses reference population tokens and TMRCA relative attention.
No learned ordered or set aggregation across neighboring trees is described
in this version of the paper. The paper trains on five simulation replicates,
uses a separate validation replicate, and selects validation loss.

Both paper licences are CC BY 4.0. The primary full ARGMix text and rendered
equations were read in the in-app browser after shell retrieval met a
Cloudflare challenge. ARGformer full text was retrieved from Europe PMC.
`research/.runtime` was absent, and no matching copies were found in
`.runtime/cache`. No challenge was solved or bypassed.

## Paper equations and independent implementation

For each attention head, the implementation uses linear query, key, and value
projections of feature vectors, paper Eq. (1). Reconstructed observed node times
are `expm1(x[:,6] * log1p(10000))`, according to the shared adapter metadata.
The topology gives the observed MRCA for any pair of nodes.

The relative relation is the paper's asymmetric elapsed time, Eq. (2):

```text
psi(i,j) = floor(min(max(observed_time(MRCA(i,j)) - observed_time(i), 0), 1000))
```

The floor operation is our explicit discretization of continuous inferred
generations for a lookup table. The lower bound is a numerical guard for valid
monotone genealogies. There are 1,001 biological relation bins. Tables for
relative query, key, and value are learned and shared across Transformer layers,
as in the inspected official implementation.

Paper Eqs. (3)-(6) are evaluated as:

```text
s_ij = (q_i dot k_j + q_i dot Pquery[psi(i,j)]
                     + k_j dot Pkey[psi(i,j)]) / sqrt(head_width)
a_ij = softmax_j(s_ij)
z_i  = sum_j a_ij * (v_j + Pvalue[psi(i,j)])
```

The relative **key term uses k_j**, as printed in the paper. The current
official `tmrca_attn.py` inspected at commit
`c2481248eb0de5fda659d68f59c51e0e27becd20` gathers its key-relative scores
by rows, which evaluates a `k_i` term. We intentionally implement the printed
equation, not that code discrepancy. A direct dense-equation test verifies
the column-key term and the graph-encoded values on an asymmetric relation
matrix. The current repository also supports logarithmic distance transforms
and a joint genomic GRU. Neither is claimed to be part of the v1 paper
comparator implemented here.

The implementation is written independently with native PyTorch and NumPy.
No public source code was copied, vendored, executed, or installed. Official
source was inspected only to audit specification and provenance. ARGMix
declares MIT, with placeholder copyright text in its LICENSE. Its attention
file attributes earlier GRPE source. The GRPE repository declares MIT.
ARGformer declares Apache-2.0 and acknowledges its ModernBERT/MosaicBERT
lineage. None of their packages or checkpoints are dependencies.

## Three explicitly different scopes

1. **Primary: full-tree ARGMix equation adaptation.**
   `ARGMixTreeControl(48,4,token_scope="full_tree")` projects all 48 observed
   node features with linear/GELU, applies dense relative attention to every
   node in each complete tree, and reads the queried node through a four-class
   linear head after final LayerNorm. The full tree includes all reference,
   query, and internal nodes. The paper's per-query landmark selection and
   population-token embedding are removed. The per-subgraph CLS readout is
   replaced by a shared query-node classifier so one full-tree computation
   serves all queries. The relation computation uses only the common edges
   and declared observed time column, not an externally supplied pairwise
   matrix. These are material adaptations.

2. **Optional: nearest-reference ARGMix adaptation.**
   `ARGMixTreeControl(48,4,token_scope="nearest_reference")` derives nearest
   references from observed one-hot columns 0:4 and observed MRCA times.
   It takes up to 25 reference leaves, three immediate observed ancestors,
   and the query. A learned CLS token and a dedicated CLS relation bin
   provide the inspected official code's classification readout.
   The same 48 features replace population-token embeddings. Other query
   leaves are excluded from this subgraph. All references tied at the
   selection cutoff are included, rather than breaking ties with arbitrary
   IDs. Consequently the subgraph size can vary. With the present panel of
   24 reference haplotypes, the observed 64-query tree smoke test used
   **27-28 biological nodes**, plus one CLS token, because some paths contain
   only two available ancestors. This is not the published 29-node setup.
   Despite selecting a subset internally, it receives exactly the same
   observed input object and no additional truth or genotype data.

3. **Generic: whole-tree graph Transformer.**
   `WholeTreeGraphTransformerControl(48,4)` uses the same dense relative
   query/key/value attention and query head, replacing elapsed TMRCA bins
   with observed unweighted tree shortest-path bins, capped at 32 by default.
   It is an independently implemented graph-transformer family control,
   not a specific published-method reproduction.

All three produce logits `[positions, queries, 4]` in the supplied query
order. Integer IDs only join a query to its node. They are never features,
tokens, sorting priorities for tied reference selection, or embedding
indices. Descendant-landmark `clades` are not read. Class predictions within
one tree are constant across its scoring positions. No genomic smoothing,
position encoding, cross-tree sequence attention, or collection pooling is
performed. This models local ancestry directly and avoids conflating a
single-tree attention baseline with a novel cross-tree operator.

Pre-LayerNorm residual attention blocks use GELU feedforward layers of width
4D, attention/residual/feedforward dropout, and a final LayerNorm. These
engineering choices follow standard Transformer blocks and the inspected
official architecture, while learned input and output heads are adapted to
the common task. All external self-supervised pretraining and checkpoints are
absent.

## Computation, interfaces, and tuning budget

```python
from arggnn.argformer_control import ARGMixTreeControl, WholeTreeGraphTransformerControl

model = ARGMixTreeControl(
    input_dim=48, num_classes=4, hidden_dim=32, num_layers=2,
    num_heads=4, dropout=0.0, time_column=6, branch_column=7,
    time_scale=10000.0, max_rel_distance=1000,
    token_scope="full_tree",
)
logits = model(common_observed_sequence)
audit = model.configuration()
```

Geometry is preprocessed on CPU from detached observed inputs. It validates a
connected acyclic tree and monotone node times, deduplicates paired directed
edges into undirected adjacency, and disambiguates tied maximum root times
with the observed root-zero branch feature. This structural preprocessing
does not differentiate through inferred times. All 48 features also enter
the learned projection, so their learned feature path remains differentiable.
A bounded four-entry cache retains source tensors and checks their in-place
version counters. Its keys are runtime object identities for cache validity,
not biological IDs or learned model inputs. `clear_geometry_cache()` releases
the cache. Structural work and relation storage are quadratic per marginal
tree. Attention remains quadratic, intentionally retaining whole-tree access.
Only occupied relation bins are materialized for attention, avoiding the
much larger node-pair-by-feature relative-value tensor.

Parameter counts below use input width 48, four outputs, four heads, FFN
multiplier four, and the stated relation caps. Dropout does not change counts.

| Hidden width / layers | Full-tree ARGMix, cap 1000 | Nearest-reference ARGMix, cap 1000 | Full-tree hop Transformer, cap 32 |
| --- | ---: | ---: | ---: |
| 32 / 1 | 110,564 | 110,692 | 17,636 |
| 32 / 2 | 123,268 | 123,396 | 30,340 |
| 64 / 2 | 295,684 | 295,940 | 109,828 |
| 128 / 3 | 986,244 | 986,756 | 614,532 |

These are architecture accounting, **not completed tuning runs**. This
implementation task launched zero training jobs, zero GPU jobs, zero
hyperparameter trials, and evaluated no accuracy labels. Training and registry
integration belong to the primary agent. Its eventual experiment manifest must
record the number of tried configurations, seeds, parameter counts, training
examples/updates, early-stopping policy, selection metric, and wall-clock
budget per family. Give the comparators the same train/validation parents,
common windows/targets, and search budget. Larger relation tables mean equal
hidden width does not imply equal total parameters. Do not describe a
single untuned small graph Transformer as a definitive published baseline,
and do not select settings on final test outcomes. Report native subgraph
scope separately from the primary full-tree scope.

## Verification and limits

The CPU suite checks the exact relational attention expression, asymmetric
elapsed time and self relations, all-feature input gradients, parameter
gradients, node-array permutation, arbitrary nonmonotone leaf-ID renaming,
query permutation, graph serialization permutation, tensor/checkpoint
roundtrip, landmark-column independence, observed-structure sensitivity,
cutoff ties, cache invalidation, and invalid topology/time/feature width.

Run:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$ARGGNN_PYTHON" -m pytest -q tests/test_argformer_control.py tests/test_models.py
```

On 2026-10-10, the combined run passed 26 tests and 16 existing subtests.
A label-free CPU smoke check used the first eighth of an actual inferred
cohort64 training chromosome, containing 22 marginal trees, 3,850 nodes,
48 features, and 64 queries. Both whole-tree controls returned
`[16,64,4]` finite logits and finite gradients for every parameter.
One forward/backward observation at width 32/layers 2 took approximately
0.55 seconds for ARGMix and 0.32 seconds for the hop control on one CPU
thread. These are engineering smoke observations, not performance benchmarks
or ancestry accuracy results. Native scope returned `[1,64,4]` on the first
scored tree with 24 references and 27-28 biological nodes per query.

Exact reproduction is not established. The v1 paper alone does not fully
specify all architecture/optimizer/augmentation/readout details. The current
official repository has later computation and configuration choices, which
should not silently be equated with the paper. The disclosed feature,
scope, readout, time-discretization, training, and attention-indexing
differences rule out an exact reproduction claim here. Full-tree attention
may use more information and more compute than the published sampled
subgraphs. The native secondary scope provides a useful sensitivity check,
but has fewer available references and the same richer common features.

Source hashes, audited source URLs/commits/licences, architecture counts, and
verification records are in `docs/argformer_control_provenance.json`.
