# Local encoder diagnosis and whole-tree controls

## Status

This file records an exploratory local-representation diagnosis after the frozen engineering pilot. It does not change that pilot's data, protocol, source models or inference claims. The new controls are deliberately based on standard genealogy mathematics, with no novelty claim and no claim to reproduce ARGMix or AncestralPaths exactly.

## Structural concern

The independent encoder has two incoming-neighbor mean message-passing layers. Reference source one-hot features occur only at labeled sample leaves. A query's final state can therefore receive those features only from reference leaves within two graph edges. Deeper reference relationships remain represented in the input tree, but cannot directly influence that state through this encoder. Observed times and branches can still contain correlated information. Receptive-field exclusion alone does not prove that depth explains the full training failure.

The set baseline additionally broadcasts a graph-wide mean. That summary loses which references are genealogically near which query. The GRU and late transformer consume these already compressed query-leaf embeddings. Additional genomic processing cannot guarantee recovery of reference relationships that their local encoder has discarded. The interval-lineage model can access source nodes globally, but its query leaf has no descendant reference landmarks and its same-leaf matches are likewise encoded with the shallow local bank. A strong observed-tree control is needed before attributing accuracy differences to cross-tree mechanisms.

## Implemented control

`genealogy_baselines.py` constructs a rooted tree from observed connectivity and node times. An optional observed parent-branch column identifies the root if rounded time features create a tie. It never reads node population, migration records or external ancestry targets. Declared reference one-hot columns must label terminal sample leaves, and query reference labels are explicitly rejected.

An upward pass accumulates per-source descendant reference counts `c[u,k]`. For child `u` of parent `p`, the sibling reference shell is `c[p,k]-c[u,k]`. Every reference outside `u` has its most recent common ancestor either above `p` or exactly at `p`. A downward pass therefore accumulates class-conditioned TMRCA counts, minimum and multiplicity, first/second moments of encoded time, and kernel sums. At a queried leaf, these shells account for every labeled reference without a fixed hop cutoff. The summary pass costs `O(N*K*(S+1))` for `K` reference source columns and `S` kernel scales, after linear topology preparation.

The feature vector has six fields per source plus `S` normalized kernel densities: minimum encoded TMRCA, mean encoded TMRCA, standard deviation, panel-count fraction, fraction at the source-specific minimum, and globally nearest-reference vote share. Encoded time is the observed `log1p(time)/log1p(time_scale)` field. The minima can be reconstructed in generations using the recorded normalization. Finite summaries can discard details. An independent ancestor-walk primitive retains the full query-reference TMRCA matrix for audits or richer heads.

`NearestTMRCAClassifier` shares all globally minimum-time reference ties and returns log probabilities. `TMRCAFeatureMLP` can begin with this explicit deterministic prior and learn a residual correction. With `nearest_init=True`, its final layer is zero-initialized and the log nearest-reference prior is added. This starting advantage must be reported. Set `nearest_init=False` for a plain MLP on the same summary features. Neither version tests GNN superiority.

This relates to the established use of genealogy-nearest-neighbor features in [AncestralPaths](https://www.nature.com/articles/s41586-023-06705-1) and reference selection by TMRCA in [ARGMix](https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1.full). The control adapts those mathematical feature families independently, with its own summaries and panel assumptions.

## API

All classifiers output logits `[P,Q,K]` for the shared `GraphSequence` contract, supporting fixed-coordinate clipped windows. Required external metadata is `reference_columns` and `time_column`. `branch_column` is optional. Recorded time normalization is `time_scale=10000` for the current simulator.

- `ReferenceTMRCAFeatures(reference_columns, time_column, branch_column=None, time_scale=10000, kernel_scales=(0.05,0.1,0.2), validate=True)` returns flattened features `[P,Q,K*(6+S)]`. `.summarize(seq)` returns graph-level features, nearest votes, minimum TMRCA and reference counts.
- `NearestTMRCAClassifier(input_dim, num_classes, *, reference_columns, time_column, branch_column=None, time_scale=10000, validate=True)` has no trainable parameters and should be evaluated directly.
- `TMRCAFeatureMLP(input_dim, num_classes, *, reference_columns, time_column, branch_column=None, time_scale=10000, hidden_dim=64, num_layers=2, nearest_init=True, kernel_scales=(0.05,0.1,0.2), validate=True)` is trainable. `.features_and_prior(seq)` plus `.forward_from_features(features,prior_logits)` permits precomputation before neural training.
- `pairwise_reference_tmrca(seq,...)` independently reconstructs all query-reference TMRCAs and hop distances.
- `reference_hop_diagnostics(seq,...,num_layers=2)` reports observed fixed-grid reference reachability without reading ancestry targets.

Feature extraction currently uses CPU NumPy and Python tree traversals, then transfers the fixed summaries to the feature device. It is linear in tree size but is not a custom GPU implementation. Precompute features for a properly versioned control run and report extraction cost. The extractor assumes complete connected coalescent trees, labeled pure-source references and known time encoding. It does not reconstruct ARGs or correct erroneous inferred genealogies.

Node permutations, arbitrary stable-ID renaming and graph serialization order preserve outputs. Population classes remain semantic output columns. Independent per-tree controls are invariant to redundant interval splitting at common genomic query positions. This property does not use neighboring-tree evidence.

## Measured diagnostics

CPU probes used only `train/rep000` and `val/rep000` in `data/oracle_pilot_v1`. Each has 128 fixed coordinates and 16 queried haplotypes, for 2,048 query-coordinate pairs. No test ancestry labels were evaluated or used for this control.

| Quantity | Training replicate 0 | Validation replicate 0 |
| --- | ---: | ---: |
| Marginal graphs | 201 | 197 |
| Packed nodes | 25,527 | 25,019 |
| Pairs with any labeled reference within 2 hops | 27 / 2,048 (1.3184%) | 0 / 2,048 (0%) |
| Pairs with a globally nearest-TMRCA reference within 2 hops | 27 / 2,048 | 0 / 2,048 |
| Observed-feature nearest-reference accuracy | 98.3887% | 97.8516% |
| Maximum probability difference from the retained comparator | 1.1921e-7 | 1.1921e-7 |
| Initial nearest-prior MLP NLL | 0.296818 | 0.225674 |
| Probe wall time, including feature extraction and diagnostics | 1.773 s | 1.570 s |

The width-16, two-hidden-layer MLP has 771 parameters. Its residual output was zero-initialized, so these accuracy observations belong to the deterministic prior. Forward and one loss-backward pass produced finite values and gradients. No optimization or GPU computation was performed.

Observed minimum-reference hop histograms, weighted by scoring coordinate multiplicity:

```
train/rep000: {2:27, 3:107, 4:184, 5:312, 6:436, 7:484, 8:334, 9:164}
val/rep000:   {3:134, 4:541, 5:695, 6:530, 7:120, 8:28}
```

Ancestry label counts were `[37,1314,697]` in this training replicate and `[901,625,522]` in this validation replicate. This illustrates substantial between-replicate composition variation in these two examples, without establishing its distribution across the whole pilot. Fixed-class or cohort-level summaries can therefore vary in apparent accuracy between replicas. Do not interpret the near-constant learned predictions as a clean architecture comparison without stronger local controls and a broader optimization check.

Source tensor SHA256 values from the frozen manifest:

```
train/rep000: 91ffa8822e9b228f5a0ce2d66e0572eea2ba75c300965856f7feb3eac107c75e
val/rep000:   9799e8ac8915d70920df871c31350140abe8ffa49b6cc3a716ece9c927663874
```

Both probes explicitly declared reference columns `(0,1,2)`, time column `5`, branch column `6` and the recorded time scale `10000`. The result uses reference class labels and topology only. External ancestry labels enter only accuracy/NLL reporting and the loss-backward check.

A synthetic unit tree places its nearest reference three edges from the query and tests an exact 20-generation versus 50-generation TMRCA distinction. An intervention test swaps the two reference source labels while preserving topology and observed times. The two-layer independent GNN returns exactly unchanged query logits, while the whole-tree nearest classifier reverses its ancestry probabilities. This isolates the finite-hop exclusion mechanism. It does not measure the fraction of the pilot's full training failure caused by that mechanism.

Tests also cover tied nearest-reference votes, independent pairwise reconstruction, arbitrary node permutations and ID renaming, query/graph permutations, redundant interval refinement, finite MLP gradients, forbidden query/internal-node labels, malformed tree topology and equivalence of directed/bidirectional edge serialization.
