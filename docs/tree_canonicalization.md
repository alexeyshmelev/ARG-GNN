# Observed-tree canonicalization comparator

Implemented CPU utility, 2026-10-10. This is a preprocessing comparator, not a
new kernel or a predictive result. Exact canonicalization can give ordinary
GNN, GRU, and Transformer models the same unary-insertion and duplicate-interval
representation contracts. A kernel model therefore needs evidence beyond
those contracts, such as attributed predictive benefits, efficiency, or an
appropriate inductive bias.

## API and downstream adapter

~~~python
from arggnn.tree_canonicalization import canonicalize_observed_trees
from arggnn.reference_encoder import ReferenceFeatureAdapter

result = canonicalize_observed_trees(
    observed_sequence,
    time_column=6, parent_branch_column=7,
    base_feature_count=8, time_scale=10000.0,
    merge_identical_adjacent=True,
    source_provenance={"split": "train", "parent_id": "rep000"},
)
prepared = ReferenceFeatureAdapter(
    8, reference_columns=(0,1,2,3), time_column=6, branch_column=7,
    time_scale=10000.0,
).precompute(result.sequence)
~~~

The default merge setting is **False**. Enable exact adjacent merging when
testing both representation contracts. The utility reads only the observed
GraphSequence fields and the declared base feature prefix. Already appended
summaries are discarded, so recompute ReferenceFeatureAdapter downstream.
For the current ancient cohort prefix, F8 becomes the same F48 adapter schema.
No source registry, model registry, configuration, dataset, or training path
was changed by this implementation.

The frozen result CanonicalizedObservedTrees contains:

| Field | Meaning |
|---|---|
| sequence | Canonical observed GraphSequence with only the declared base prefix |
| provenance | Operator note, parameters, counts, and copied source audit provenance |
| source_graph_indices | Original graph indices grouped into each chronological output hold |
| source_graph_to_canonical | Explicit original graph-to-output index mapping |
| source_node_indices | Retained source packed rows for the first representative of each merged hold |

The result supports as_dict/from_dict and weights-only tensor serialization.
Its graph sequence has the existing GraphSequence.to method. Source provenance
must be finite JSON-compatible audit data. It is copied into the provenance
field and never consulted for topology, feature construction, or merging.

## Canonicalization rules

Each graph must be a connected, acyclic, rooted observed tree. Encoded ages
must be nonnegative and decode to finite physical ages using
expm1(encoded_time * log1p(time_scale)). Every edge must have a strictly positive
decoded age difference, with one older parent for every nonroot vertex.
Persistent nonnegative node IDs must identify terminal sampled leaves.
Anonymous terminal tips are rejected. Anonymous internal vertices use the
existing -1 sentinel. Self edges, disconnected graphs, cycles, sampled internal
ancestors, duplicate sampled IDs, invalid coordinates, and invalid physical
parameters are rejected. The utility applies no epsilon floor or contraction
of equal-age vertices.

An anonymous vertex with exactly one younger child is suppressed. Each retained
child reconnects to its nearest retained older ancestor. This includes a dangling
anonymous unary root chain: its first retained descendant becomes the canonical
root, with root branch zero. Sampled leaves remain present with their original
IDs and other base fields.

The parent-branch field is recomputed from the difference of **decoded canonical
endpoint ages**, then re-encoded by log1p(duration)/log1p(time_scale). Encoded branch
lengths are never summed. Thus both the expanded and unexpanded representations
produce the same canonical endpoint feature. Branch recomputation is explicit
preprocessing and can change the original stored branch field slightly through
floating-point time decoding.

Sampled leaves are placed in leaf-ID order, and anonymous ancestors follow their
children. Matching uses leaf correspondence and recursively sorted child-subtree
signatures. Anonymous packed row numbering, input edge direction, and edge
serialization order do not affect the canonical tree. Numeric IDs, ranks,
signatures, and source-row indices are **never appended to learned input features**.
The original query order remains in query_node_ids, independently of canonical
packed leaf ordering. Retained descendant-landmark indicators are preserved.

Graphs are ordered by their physical interval. When merging is enabled, two
consecutive holds merge only if physical adjacency, canonical topology, sampled
IDs, every retained base field, and descendant-landmark indicators match exactly.
No tolerance matching, feature averaging, or merging across a genuinely different
intervening observed tree is performed. The outer physical endpoints survive,
target_graph is remapped, and positions/query IDs/window_bounds keep their
original order and precision. Labels, population truth, truth trees, and external
metadata attributes are never accessed or transformed.

## Verification and scope

The final standalone CPU suite passed **29 tests in 2.56 s**. It covers inserted
unary interiors and two-vertex unary root chains, interval duplication, anonymous
packed-row permutation, edge order/direction, stable leaf correspondence
renaming, sample/query preservation, fresh adapter recomputation, idempotence,
audit provenance copying, weights-only serialization, clipped coordinates,
genuine topology/age/feature/landmark differences, and strict admissibility.

Tests also compare ordinary IndependentGNN and GNNGRU predictions after
canonicalization and fresh F48 adapter preparation. Both models return exactly
identical fixture predictions for combined unary insertion, duplicate interval
splitting, and anonymous row permutation. Consistent stable leaf-ID renaming
also preserves predictions within numerical tolerance. A guard object raises
on any access to labels, populations, truth, tree_sequence, or metadata; the
utility succeeds without reading them.

A broader CPU run passed **156 tests in 9.48 s** across canonicalization, age,
Kron, boundary-dataset, and anchor-kernel tests. The final canonicalization-only
rerun followed an internal grouping optimization that groups packed nodes and
edges once instead of rescanning the complete sequence for every hold.

These checks establish the preprocessing contracts for observed trees and the
tested deterministic consumers. They do not establish predictive superiority,
a new mathematical mechanism, or robust inferred ancestry. A Transformer was
not separately scored here. A deterministic Transformer receiving identical
canonical tensors can also share the contracts. Stochastic training comparisons
must hold random state and preprocessing fixed.

## Read-only cohort64 admission probe

The first train parent rep000 from each existing prepared source was processed
in full on CPU with four threads and exact duplicate merging enabled. Prepared
tensor SHA256 was verified before reading the observed sequence. Reference
augmentation was recomputed after canonicalization.

| Source | Input/output graphs | Input/output nodes | Removed unary interiors/roots | Duplicate merges | Canonicalization seconds | Adapter seconds |
|---|---:|---:|---:|---:|---:|---:|
| Oracle cohort64 | 1534 / 1534 | 268450 / 268450 | 0 / 0 | 0 | 16.903 | 9.477 |
| Inferred cohort64 | 183 / 183 | 32025 / 32025 | 0 / 0 | 0 | 1.963 | 1.137 |

Both parents admitted, retained scoring positions and query order exactly,
and produced F48 augmented inputs. These original parents were already free of
anonymous unary vertices and exact duplicate adjacent holds, so their graph and
node counts stayed unchanged. This is a bounded compatibility and timing probe,
not a training or accuracy result. No labels were inspected, and no transformed
dataset files were saved.

| Provenance item | SHA256 |
|---|---|
| Oracle manifest | c9b3af5c1ebbc57a9b04d58902d7d13a12dfd240a81c879deb74f82cd2ad0385 |
| Oracle prepared rep000 | adece7df8da900c728e92bd3e8840e12b4920ea1ab14c74bdfaea6ed0d6cfacf |
| Inferred manifest | 611c7e32973c22bd35115b835279884f7b0c866136e91ae2c5f2a95daefd6fe7 |
| Inferred prepared rep000 | 88b61aceaaeba584445f97080a732dcd13787be0ca7cc6f422e643e939267599 |
| tree_canonicalization.py | 0f856ad1d188e307d9fe85604d082eae0aa41a91e9964389e6edbe6a296b894c |
| test_tree_canonicalization.py | 34d26ae5b23a1ecff2ace72003511d5ec55da94149d75beb1f6c5a49732c99fc |
| data.py | fb4e4b3ad8744731e6d2c5966cc2679f9f2d59dfd079facc280d69fbc61ef654 |
| models.py | 8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5 |
| reference_encoder.py | 039f3b25383ea92f243090c2eaf592df590768fea7abf41d15fd7c34a7715138 |
| genealogy_baselines.py | 388e84c183413a79db8e0e75c86f1ce202b4bb9999473b729be045d7459dc9f8 |

No GPU jobs, Git operations, package installations, label scoring, or shared
source edits were performed by this utility task. Root owns later integration.
