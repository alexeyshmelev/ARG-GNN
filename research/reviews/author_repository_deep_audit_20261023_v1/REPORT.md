# Deep audit of all four author repositories

Audited on 23 October 2026. The earlier source review was incomplete. This pass inspected advertised public branches, tags, releases/assets, issues, complete current file trees, relevant source/config/test/data interfaces and reachable history. It checked9 ARGMix,12 ARGformer,27 AncestralPaths and18 GST commits, plus96 FLARE Java sources against a fresh official tag. Request receipts, exact hashes and file inventories are retained in each packet. Inaccessible/private or unreachable source is outside the claim.

## Corrections and recovered material

| Repository | New evidence | Consequence |
| --- | --- | --- |
| AncestralPaths | Historical CNN source and255064-byte example checkpoint recovered. HDF5 metadata verifies Conv32(5x1), channels-first input, Flatten, Dense64 and six outputs, Keras2.3.1/Adam. Current head explicitly replaces CNN with Dense512/dropout | Earlier CNN-unavailable statement was wrong. We can prepare an actual historical author-CNN comparator. Checkpoint paper-trained identity and complete paper-scale recipe remain unproven |
| ARGMix | Exported low-level APIs retain ancestor extraction and29-node geometry for query+25references+3ancestors | Ancestor components are recoverable. Supported archive/trainer/export/inference still reject them. A complete29-node route needs disclosed adaptation and qualification |
| ARGformer | Every reachable snapshot checked, including all tests/configs. No original PCA10 ancestry evaluator, simulator or checkpoint recovered. CLI references an absent defaults YAML; current wrapper supplies release YAML directly | Some gaps are genuine within the public history. Current native-release condition remains usable, with config/objective/evaluator differences disclosed |
| GST | Full166-file current tree and18-commit history inspected beyond16staged files. Deleted gst_v2.py and CIFAR scripts recovered | More original-task source is available than earlier wording implied. Historical/current versions differ. Current BH runner references an absent module, so historical restoration needs qualification |
| FLARE | All 96 local Java sources match official v0.6.0. Current source changes only a version banner in Java. Its EM semantics confirm gen10 is an initializer | No algorithm update or forced-ten-generation mismatch found in our source pin |

## Scientific implications

Recovering old files makes additional faithful comparisons practical. It does not identify their exact publication-run provenance. The AncestralPaths example VCF explicitly spans 2 Mb, not the supplement's 200 Mb. Historical CNN training requests five parents and 812,500 allocated rows, while the supplement says three 200 Mb parents and 50,000 examples. The recovered source can support a versioned historical-CNN comparison, but these recipe differences must be resolved or disclosed.

ARGMix's nine public commits start with a parentless August release. Its low-level ancestor components survive, yet the supported route has no complete 25+3 condition. The development repository referenced by .gitignore returns 404, which can mean private or nonexistent. No original European roster, map, genotype-error implementation, misspecification draws or checkpoint was recovered.

ARGformer still lacks its original Table 1 ancestry classifier/evaluator in all twelve reachable snapshots. Released temperature 0.30, uniform-weight fallback and 80/10/10 masking differ from paper descriptions. Its trained head-normalized embeddings and top1 retrieval are real native implementations, not recovered proof of every original table setting.

GST contains usable native synthetic/USPTO routes and public data pointers. The available saved USPTO 1,000-training-reaction run gives different results and uncertainty definitions from the printed paper table. Historical gst_v2 adds both local FFN/LayerScale and two gates, so it cannot silently replace current gst.py or be called exact paper code merely for having two gates. Historical CIFAR scripts also use different seed/set configurations from paper text.

## Next implementation boundaries

1. Prepare a separately named historical AncestralPaths CNN baseline from the recovered source and metadata. Keep the current Dense baseline/results intact. Qualify legacy layout/runtime, then use the same frozen observations, references, queries and evaluation support.
2. Keep current author-release ARGMix as one baseline. Any 29-node reconstruction must bind actual recovered extractor/model components and disclose all changes across archive geometry, training, export and inference. Do not present a25+0 configuration as 25+3.
3. Keep ARGformer release and any paper-equation/config correction as separate named conditions. Do not invent missing PCA clustering rules and call them original Table 1.
4. Freeze a coherent GST source/task condition, then implement the candidate's original-task adapter and paired evaluation. This is remaining project work, not simply unavailable upstream code.

This audit launched no training/tests/simulation/inference, constructed no models and changed no existing job, source pin or execution policy. The only checkpoint operation was exact download/checksum and HDF5 metadata reads without numerical weights, Keras loading or TensorFlow imports.

## Evidence packets

- [ARGMix](argmix/REPORT.md)
- [ARGformer and FLARE](argformer/REPORT.md)
- [AncestralPaths](ancestralpaths/REPORT.md)
- [GST](gst/REPORT.md)
- [Machine-readable synthesis](evidence.json)

The earlier 23 October scope audit remains historical evidence and is corrected by this packet. No new ancestry metric or superiority claim follows from source discovery.
