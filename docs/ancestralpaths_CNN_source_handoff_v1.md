# H014 AncestralPaths CNN source handoff v1

The frozen publication/source package does not support an exact native-CNN
comparison plan. Supplementary Note 3 calls the classifier a convolutional neural
network, but supplies no convolution-layer definition or CNN training hyperparameters.
Both the pinned GitHub source and the paper's Zenodo release define only the Dense
model already used by our completed native comparator. A guessed Conv2D model would
be a new experimental architecture.

Native ARGMix and native ARGformer remain the main competitors. This H014 extension
records the source discrepancy and data-reuse evidence; it authorizes no CNN
training, scoring, model download or sampler rerun.

## Checked publication and source

The paper is *The selection landscape and genetic legacy of ancient Eurasians*,
[DOI 10.1038/s41586-023-06705-1](https://doi.org/10.1038/s41586-023-06705-1),
PMC10781624. Supplementary Note 3 printed pp. 31-32 (PDF pages 32-33) says
“specifically a convolutional neural network.” It specifies five informative
ancestor rows, ancient-group fractions plus node age, and -15 padding if the root
is reached before five informative nodes. For this panel that means eight cohort
fractions plus age, compatible with the released 5x9 feature matrix.

The note describes three simulated 200 Mb training tree sequences, 10,000 pairs
per five admixed groups (50,000 total), and five further testing sequences with
10,000 GNNs each. It does not specify CNN channels, filters, kernels, strides,
pooling, padding, dense head, activations, optimizer, learning rate, epochs,
batch size, shuffle or random training seed. Figure S10 illustrates marginal-tree
GNN extraction and demographic paths; it is not a neural-network architecture.
The repository's Model_schematic.pdf also depicts population history, not a CNN.
These relevant PDF pages were rendered and visually inspected.

All 17 original files at commit
77db3cfae100641c35218b87e1451e14f3074666 match the existing frozen hashes and the
paper's accepted Zenodo archive byte for byte:
[GitHub source](https://github.com/AliPearson/AncestralPaths/tree/77db3cfae100641c35218b87e1451e14f3074666),
[Zenodo10.5281/zenodo.8319452](https://doi.org/10.5281/zenodo.8319452).
The GitHub archive SHA256 is
9624135ddb0d0ea3e2275a4ef463cbd0b62ecd13625cf2b2f3de653d778ba3a7;
the published ZIP SHA256 is
4d58adf9633d96ff9ef3b7f600d6192b4207c2e1551c1175367596e83b774e85.

## Exact architecture that is actually supplied

[training.py lines168-179](https://github.com/AliPearson/AncestralPaths/blob/77db3cfae100641c35218b87e1451e14f3074666/training.py#L168)
defines the following effective model:

| Property | Released model definition |
| --- | --- |
| Input |5x9; flattened 45 features |
| Layers |Flatten -> Dense(512, ReLU) -> Dropout(0.2) -> Dense(6, softmax) |
| Parameters |26,630 |
| Labels |Six native paths; labels-1 converted to six-class one-hot targets |
| Final effective loss |categorical_crossentropy; the preceding sparse-loss compile is overwritten |
| Optimizer |adam string; no explicit Adam arguments or dependency version pin |
| Fit |5 epochs; batch_size and shuffle omitted, hence Keras defaults 32/True |
| Seed |No training seed literal |
| Model output |Local model_<output>.h5 written by model.save |

Convolution2D and MaxPooling2D are imported in training/testing but never called.
No file in the 17-file release defines a convolutional model. The complete actual
Dense definition is training.py line 173; model.compile is at lines 175-176 and model.fit is at line 177.
The source feature order is Bronze_Age, BAA, Yam, Neo, WHG, EHG, Ana, CHG, age/1500.
The illustrative Figure S10 column order is not a substitute for a pretrained
model's verified preprocessing contract.

Our biological v2 adapter retains this Dense architecture and categorical loss,
uses the qualified TensorFlow CPU 2.15.1 / Keras 2.15.0 runtime, and makes defaults
explicit: 5 epochs, batch 32, shuffle=True, seeds 101/202/303, no VAL early stopping.
It preserves the separately disclosed v2 realized-row quota policy. Successful
execution of that released Dense architecture is not evidence that the paper CNN
was reproduced. Our two TRAIN / two VAL 2 Mb pilot and conditional physical-support
metrics do not reproduce the paper's 200 Mb simulation/sample protocol or numbers.

## Published pretrained checkpoint availability

No original pretrained CNN checkpoint was identified in the cited distribution.
The accepted ZIP contains exactly the 17 source/example/documentation files and
no .h5, .keras, checkpoint or trained-weight asset. README line 42 describes the .h5 as
an output of running local training, and painting takes a user-supplied model path.

Metadata-only public checks on 2026-10-10 found:

- Zenodo record 8319452 lists one file, ekirving/AncestralPaths-accepted.zip,
  96,075 bytes; it is the verified17-file source archive.
- The AliPearson/AncestralPaths GitHub releases endpoint lists zero releases/assets.

The request URLs, timestamps, statuses, response hashes and metadata responses are
preserved under this handoff's public_metadata directory. No model bytes were
downloaded. This establishes absence from the cited release and checked release
listing; it does not prove that no author-hosted checkpoint exists elsewhere.
Even a newly found .h5 must reveal its actual architecture and preprocessing
before it can be called the original pretrained paper CNN. Retraining a new model
on our frozen TRAIN rows would be a matched-training comparison, not evaluation
of published pretrained weights.

## Completed v2 TRAIN data preservation

The frozen v2 runner SHA256 49e90d848b7e7f7b6ac8d00488108f02e1906ee7c3df13115b87efa6cc5935ef
writes each TRAIN selection at lines 185-189 with ordered site_row, query_column,
positions_bp, query_sample_indices and native_labels_0based. Both selection files
exist as regular files (checked by stat only):

- checkpoints/ancestralpaths_biological_pilot_v2/training_sampler/bio_paths_pilot_train_000_training_selection.npz
- checkpoints/ancestralpaths_biological_pilot_v2/training_sampler/bio_paths_pilot_train_001_training_selection.npz

The qualified v1 TRAIN feature banks are preserved and already declared by SHA256
in the v2 config:

| Parent | Feature-bank path | Declared SHA256 |
| --- | --- | --- |
| TRAIN000 |checkpoints/ancestralpaths_biological_pilot_v1/feature_cache/bio_paths_pilot_train_000.npz |f3cc38a6a0cdd33568fa96e4d370f550042f383c9d152f5571e04b608910c454 |
| TRAIN001 |checkpoints/ancestralpaths_biological_pilot_v1/feature_cache/bio_paths_pilot_train_001.npz |c0b3d3b2b1016973fa9ee4fb7145cae065370696e26b3f1fab5a181c6fff63f6 |

The full concatenated X and one-hot target tensor are not written by the v2
runner: they are assembled in RAM at lines 257-258. The source nevertheless specifies
exact reconstruction from existing artifacts, without re-extracting features,
accessing truth or rerunning either sampler:

    X_parent = cache["features"][
        cache["site_feature_index"][selection["site_row"], selection["query_column"]]
    ]
    y_parent = selection["native_labels_0based"]
    X = concatenate([X_TRAIN000, X_TRAIN001])
    y = concatenate([y_TRAIN000, y_TRAIN001])
    targets = to_categorical(y, num_classes=6)

Preserve every stored row, order and replacement duplicate; do not deduplicate,
rebalance, pad, resample or change labels. The parent concatenation order follows
the frozen TRAIN partition and v2 run order. Root must first bind the current
selection-file hashes and verify their joins, shapes and existing feature-cache
hashes under a fresh reuse receipt. The original run did not record a fingerprint
of the complete concatenated X/targets, so there is no saved full-tensor fingerprint
against which to compare reconstruction. This audit read no selection/cache arrays
and made no claim about their numeric values.

## CNN plan decision

No exact compatible native-CNN comparison plan or implementation is issued:
the required native CNN architecture and its training hyperparameters are absent.
The evidence needed to change that decision is an author-supplied, versioned CNN
definition/configuration or an original checkpoint with a recoverable architecture,
plus verified feature ordering, output-path labels and training/preprocessing
settings. Root can then separately decide between a matched-training condition
using the preserved v2 rows and a published-pretrained-weight condition. Neither
condition automatically reproduces paper numbers.

This source-only handoff performed static reads/hashes, archive-member inspection,
PDF rendering, public release-metadata requests and file-existence/stat checks.
It opened no model, biological-array, truth or score artifact; performed no CNN
fit/scoring, GPU, installation or Git operation; and did not rerun v1/v2 sampling.
