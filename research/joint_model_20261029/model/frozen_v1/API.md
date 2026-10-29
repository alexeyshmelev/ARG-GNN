# Complete sparse joint ancestry model API

The frozen core is `joint_model_v1.py`, version
`joint_complete_sparse_v1_20261029`, SHA256
`ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4`.
This is a new architecture, not an equivalent reimplementation of the 28 October
bounded cavity model. It accepts observations only. Target ancestry labels and
canonical example IDs remain external to the model.

```python
from research.joint_model_20261029.model.joint_model_v1 import (
    JointAncestryModel, load_observations, prepare_observations,
    collate_prepared, prepared_to_state, prepared_from_state,
)
from research.joint_model_20261029.model.no_cross_view import focal_only_no_cross

observations = load_observations(observation_npz)
prepared = prepare_observations(observations)
# Save this plain tensor state once; never save trainable activations in cache.
torch.save(prepared_to_state(prepared), prepared_path)
restored = prepared_from_state(torch.load(prepared_path, weights_only=True))
# Apply this view ONLY to the no_cross arm, once on CPU before LRU caching.
no_cross_prepared = focal_only_no_cross(restored)
batch = collate_prepared([restored]).to(device)
model = JointAncestryModel(width=64, layers=3, classes=6).to(device)
logits = model(batch, mode='structure')
```

`forward(batch, mode)` returns `[F, classes]` logits in concatenated canonical
input target order. Legal modes are `no_cross`, `leaf_only`, and `structure`.
`model_id(mode)` records the mode, width, layer count, and class count. All modes
have the same parameter skeleton and initialization. Unused cross parameters
have no gradients, so active capacity differs among arms.

`Batch` is an immutable dataclass. Functions and forward passes do not mutate its
input tensors. `Batch.to(device, non_blocking=False)` returns a new batch.
`Batch.pin_memory()` pins dense tensors and CSR components; a usable host pin
allocator is required. Device-specific pinning and CUDA execution need separate
integration checks. `collate_prepared` accepts CPU batches with equal sample-panel
size and creates disjoint sparse block maps; no connection crosses site windows.

Observation loading reads only `BASE_KEYS` and either `CONTEXT_KEYS` or
`MODEL_KEYS`. It never reads `labels` or `example_index` even if the archive also
contains them. Direct `prepare_observations` rejects extra keys, including labels.
Compact native packets are the registered ingestion interface. Their expansion
checks exactly five occurrences/window, one forced zero-mass focal occurrence,
four uniform draws of mass 0.25, proposal density, interval membership and window
ownership. Expanded packets are an internal construction/testing interface; they
check total source mass and routing but do not independently enforce the exact
five-draw registration. Use the compact interface for training ingestion.

A compact packet uses unique focal SNP rows `U` and target rows `F`:
`focal_tree_index[U]`, `focal_genotypes[U,H]`, `focal_allele_count[U]`,
`focal_position_local_bp[U]`, `focal_site_index[F]`, `focal_sample_index[F]`.
The focal-site index maps each target to its unique SNP row. The sample index is
used only in final output gathering, not in observation or tree encoding. Thus
all queries sharing a parent/site share one context and one encoding.

`sample_id[H]` or `[W,H]` must uniquely identify parent/sample/haplotype.
`tree_sample_id[T,H]` must agree exactly with the window roster in the declared
leaf order. No ID is a trainable feature. All terminal nodes must be in the roster;
internal unary and zero-duration nodes are retained. Forests are accepted, and
cycles, cross-tree parents, invalid ages and inconsistent routing fail explicitly.

The source coordinates are in the clipped plus/minus 50 kb domain. SNP pools
contain all observed SNPs of each selected complete native interval, which can
extend outside that radius. Report this observation extent explicitly. Exact
repeated coordinate draws remain separate quadrature occurrences.

Preparation defaults to at most 2,000,000 leaf-path entries, a 20,000,000
witness-step upper bound and 4,000,000 cross-map entries per packet. Exceeding a
bound raises an error with `no truncation`; it never crops trees, paths or targets.
The dataset caller must attach parent/site identity to such an error. Preparation
is CPU-only preprocessing; build and verify the cache before training. Cache
configuration validation checks version, radius, fixed age prior and readout
atoms. The outer dataset manifest must additionally bind source/model hashes.

`focal_only_no_cross` retains every node, leaf, SNP and native/readout map of the
focal occurrences, and removes unused context occurrences. It is a trusted
prepared-view constructor and is valid exclusively with `mode='no_cross'`.
Construct it once before CPU caching/collation. Do not feed it to cross-enabled
arms. It preserves output order and the parameter skeleton and does not change
the core hash. Its separate qualification receipt tests full outputs and every
parameter VJP against the complete-context no_cross computation.
