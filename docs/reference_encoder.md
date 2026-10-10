# Common observed-reference node adapter

## Status and role

This is a fixed domain input adapter/control. It makes observed reference relationships available at every whole-tree node before collection processing. It is not an architectural novelty or a reproduction of AncestralPaths, ARGMix or GraphSSM. The v1 protocol and result tensors remain unchanged. The model registry now adds a separately named span-weighted unordered control without changing the earlier model branches. A v2 study must record its new input schema separately.

The local diagnostic found references mostly outside the two-layer graph encoder's receptive field. This adapter removes that particular hard hop cutoff using ordinary whole-tree reference statistics. It does not add ground-truth ancestry, mutation/migration truth, internal-node populations or learned embeddings of arbitrary node identifiers.

## API and CPU cache stage

```
adapter = ReferenceFeatureAdapter(
    input_dim=F,
    reference_columns=(0, 1, 2),
    time_column=5,
    branch_column=6,
    time_scale=10000,
    kernel_scales=(0.05, 0.1, 0.2),
)
prepared = adapter.precompute(raw_cpu_sequence)
provenance = adapter.manifest_provenance(feature_names, source_tensor_sha256)
```

`prepared` is a `GraphSequence`. Its original `x` occupies the first `F` columns exactly. All edges, clades, node IDs, graph IDs, intervals, query IDs, scoring coordinates and optional window bounds are carried unchanged. New features are appended in observed-reference-category-major order. With three categories and three kernels, 30 columns are added. The current seven-column simulator input becomes 37 columns.

Run the adapter once per parent replicate on CPU before fixed-window slicing and GPU transfer. Cache prepared tensors and provenance under a new versioned artifact, with hashes of the original source tensor and prepared tensor. The adapter does not write files or mutate the original sequence. It rejects GPU precomputation and repeated augmentation. CPU traversal therefore cannot silently recur inside an epoch when using the provided wrappers.

Every comparison family may consume the exact same prepared input through its original constructor with `input_dim=adapter.output_dim`, unchanged output classes and matched chosen widths/layer counts. This includes `IndependentGNN`, set/GRU controls, `LeafTemporalTransformer`, `LeafCorrespondenceSupraMPNN`, `GraphSSMAdaptation` and `IntervalLineageGNN`. The adapter adds zero trainable parameters.

`ReferenceGNN` and `ReferenceIntervalGNN` are thin wrappers around the existing independent and interval models. They accept the original `input_dim`, output `num_classes`, width/depth, and adapter metadata. Call `model.prepare_sequence(raw_cpu_sequence)` once, then `model(prepared_sequence)` returns `[P,Q,K]`. Forward requires already augmented inputs. Interval options such as weighting/correspondence/fusion are passed through to the existing implementation.

Observed reference vocabulary size can differ from the downstream ancestry class count. Reference categories are observed sample-panel labels, which can include composite population groups. They must be one-hot in the selected feature columns and occur in each supplied complete tree. The interpretation of a category is recorded in input metadata, rather than assumed to equal a target ancestry class. Query and internal-node reference labels remain forbidden.

## All-node mathematical definition

For node `u` and reference sample `r`, define the observed relation time as the time of `MRCA(u,r)`. If the reference descends from `u`, this MRCA is `u` itself. At a labeled reference sample leaf, its self relation uses the observed sample time. At an unlabeled query leaf, no self reference is present.

The upward pass accumulates descendant reference counts per category. A downward pass accumulates outside references in sibling shells. For child `u` with parent `p`, every reference in `descendants(p) - descendants(u)` has MRCA with `u` at `p`. Together, outside and descendant terms account for all labeled references at every tree node.

For each observed reference category, append:

1. Fraction of its panel references descending from the node
2. Fraction of the whole reference panel assigned to this category
3. Minimum encoded TMRCA to a category reference
4. Mean encoded TMRCA over category references
5. Standard deviation of encoded TMRCA
6. Fraction of category references at its category-specific minimum
7. Vote share among all globally nearest-TMRCA references, sharing every minimum-time tie
8. One normalized kernel-evidence value per declared kernel scale

Encoded time is the supplied `log1p(generations)/log1p(time_scale)` field. All time statistics and kernels use this declared encoded coordinate. No positional tree midpoint or interval width enters the node augmentation.

For scale `tau`, kernel evidence is the category mean of `exp(-encoded_TMRCA/tau)`, normalized across categories. It gives each category equal normalization despite unequal panel size. Kernel sums are accumulated with `logaddexp`, so a common very old MRCA or a tiny positive scale does not make every source weight underflow to zero. Scales must be finite and at least `1e-6` in encoded-time units. They are fixed, shared declared input settings, not learned separately by comparison families.

The summaries use every observed reference, but finite moments/kernel values do not preserve every detail of the complete query-reference matrix. The original topology remains in the input. The pairwise audit primitive in `genealogy_baselines.py` can retain complete relations when needed.

## Symmetries, assumptions and cost

Node relabeling permutes the augmented rows accordingly. Arbitrary common stable-ID renaming does not change feature values. Graph serialization order is respected through metadata. Splitting an unchanged tree interval repeats identical node augmentations at the fixed scoring coordinates. The adapter therefore retains the interval model's refinement guarantee without depending on how many tree records exist.

It assumes observed connected rooted coalescent trees and a known time encoding. The observed maximum-time root is disambiguated by the optional root-zero parent-branch feature when necessary. It does not validate correctness of an inferred genealogy or repair ARG inference error. All reference samples must be terminal tree leaves under this implementation.

Topology traversal is reused from the observed-only helper in `genealogy_baselines.py`. Preparation is approximately `O(N*R*(S+1)+E)`, with `R` reference categories and `S` scales, apart from adjacency sorting. Fixed statistics use CPU NumPy. The appended tensor stores `N*R*(7+S)` values. Report preparation/cache cost alongside model compute.

Identical input information and matching layer counts do not automatically establish an identical local graph operator. In particular, the existing GraphSSM adaptation uses its own interleaved GraphSAGE diffusion blocks, while most other families use `SharedGraphEncoder`. Report that difference. A strict shared-encoder GraphSSM-head control would require a separately named implementation. Do not silently reclassify the existing adaptation.

## Provenance and validation

`manifest_provenance` returns original and added feature names, exact ordering, observed reference column roles, time normalization, kernel settings and numerical scheme, scope/self-relation definition, zero learned adapter parameters, configuration SHA256, implementation SHA256 and topology-helper dependency SHA256. An optional original source tensor SHA256 is checked and included. A dataset writer should add its prepared-tensor hash and preserve replicate splits before windowing.

CPU tests check all-node exact minima/counts, metadata and original-feature preservation, node permutations and identity renaming, interval refinement, stable very-small-scale kernels, wrapper gradients, input compatibility with all current comparison families, and feature provenance. These checks establish numerical/representation properties only.


## Recorded CPU numerical audit

An observed-only audit used the frozen `oracle_pilot_v1/train/rep000` tensor with SHA256 `91ffa8822e9b228f5a0ce2d66e0572eea2ba75c300965856f7feb3eac107c75e`. No validation/test ancestry targets were used. The audit enumerated all 1,225,296 node-reference relations across 25,527 nodes, 201 trees and 48 references per tree, including internal nodes and reference self relations. An independent ancestor-walk implementation supplied comparison values.

| Statistic | Maximum absolute difference |
| --- | ---: |
| Minimum encoded TMRCA | 0 |
| Descendant reference counts | 0 |
| Nearest-reference vote share | 2.7816e-8 |
| Mean encoded TMRCA | 2.9802e-8 |
| Standard deviation of encoded TMRCA | 1.4540e-8 |
| Normalized kernel evidence | 2.9800e-8 |

The query-position nearest-vote tensor differed from the retained independent nearest-TMRCA comparator by at most `1.1921e-7`. Those differences are at float32 rounding scale. This is a numerical information-preservation check, not a new accuracy result.

All-node statistics took 0.731 seconds on CPU for this replicate. The independent full-pair audit took 5.582 seconds. Appending 30 float32 columns requires 3,063,240 additional tensor bytes, without serialization/compression overhead. The selected feature configuration SHA256 is `90eb5f0e2c210cf793374f9c21f87d1a25ba01e03a6c5ec410039181c4551cb1` with the original recorded feature names.

One prepared window yielded `[16,16,3]` logits and finite parameter gradients for both wrappers, using a label-free squared-logit loss. No optimization step or GPU execution occurred. Eight CPU tests passed, including compatible inputs to all seven existing comparison families. Kernel parameters are fixed, so adapter parameter-gradient tests are inapplicable. Model gradients remain verified.
