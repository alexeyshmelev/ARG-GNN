# Full canonical site data and execution report

The full registered TRAIN support is ready at `research/joint_model_20261029/data/cache_v1/manifest.json`. Its SHA256 is `a9a2839b450ee7bf12f90bc269a257bb759b79fdd6e8a93fa8e7eb4c0737e253`. All work ran on anogena under `/workspace-SR003.nfs2/ARG-GNN` through readable direct SSH and plaintext Python. This data task performed no GPU operations, package installation, job control, or Git operations.

| Support | Original TRAIN parents | Sites | Canonical targets |
|---|---|---:|---:|
| Internal fit | TRAIN000 + TRAIN001 | 830 | 33,334 |
| Internal development | TRAIN002 | 415 | 16,666 |
| Registered final full training | TRAIN000 + TRAIN001 + TRAIN002 | 1,245 | 50,000 |

Original registry splits remain `train`. Full training must be selected by a separate root run protocol after development selection. Native label order is exactly `Ana_nonBAA, CHG_nonBAA, WHG, EHG, Ana_viaBAA, CHG_viaBAA`; no class remapping, target thinning, observation changes, or label changes were performed.

## Observation and source contract

`DATA_PROTOCOL_v1.json` SHA256 `82d42d8e597ac385bccadfba720350ff689b50be8e37c8ebcec819a88717f387` and `site_plan_v1.json` SHA256 `bfebc589dc0fb5a2f5a658f488d28dda9fd7b3baa2eb6fb03da76ed7f14c3a11` register all sites and canonical row/sample mappings. The original protocol was preserved after its initial bounded-profile authorization; `cache_v1/authorization.json` records the root's later full-export authorization.

Each site uses the forced focal occurrence plus four IID uniform physical coordinates inside its clipped focal +/-50 kb domain. Seeds are `numpy.random.SeedSequence([20261029, parent_ordinal, native_site_id])`. The forced focal contributes zero source quadrature mass and each random occurrence contributes one quarter. All targets at a site share the identical context and encoded observations. Focal sample indices only select outputs.

The cache retains all five coordinate occurrences, including repeated containing trees. There are 356 windows with repeated containing native trees. Every selected native tree is complete, and all SNPs in its complete native interval are retained. In 132 windows the full native-tree extent exceeds the sampled context domain; this registered boundary behavior is recorded per window. A native panel contains 952 observed haplotypes, including 686 query haplotypes. Routing identity is namespaced by parent, sample ID and haplotype. Canonical targets per site range from 1 to 384.

`SOURCE_ASSESSMENT.json` records exact native source paths and hashes for all three original AP parents, derived from `.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json` SHA256 `fecd670f6f26332ab165648f4d65ad51131650d195e86df2a39a5545f2b724c1`. Observation construction uses authenticated small registries/panels, read-only Kastore column mappings, complete native-tree validation and bounded observed VCF seeks. Existing authenticated large-source references were reused; this task did not rehash entire multi-gigabyte native sources.

`observations.py` never opens a label array. `targets.py` authenticates and selects unchanged canonical six-class targets separately. Raw observation packets contain an exact observation-only whitelist. Native node/sample/age/cohort joins and original/local coordinate bindings are validated. The transitive reader/VCF helper `research/redesign_20261028/data/window_data.py` is archived in the prepared manifest with SHA256 `a1788ea0a33335450bfb43c689aa3c598070f07b28c40930be3fb086f42aa309`.

## Cache and execution measurements

Four CPU workers exported all observations in 38.47 seconds. Raw observation files occupy 93,892,552 bytes and separate label files occupy 559,360 bytes. Export source/provenance/manifest files bring the original export total to 100,086,420 bytes.

Four CPU workers then prepared every site in 306.36 seconds against frozen model SHA256 `ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4`. Plain tensor states occupy 9,011,935,132 bytes. They load through `torch.load(weights_only=True, map_location='cpu')`; no custom class objects are serialized. Every state was safely reloaded during preparation. Maximum preparation worker peak RSS was 653,197,312 bytes. Whole isolated scope at completion was 9,123,195,425 bytes, within the root-approved 16 GiB cap. A 16-site representative profile preceded full construction; all 1,245 sites completed without truncation or dropped failures.

The cache contains only immutable observations, graph routing, sparse maps and separate labels. Learned activations, weights and optimizer state are not cached. Native ARG/VCF reads and graph preparation occur before training.

`CachedSiteDataset(manifest_path, split='fit', cache_bytes=2*2**30, mode='structure')` supports all three registered splits and all three model modes. It exposes `site_count`, `target_count`, `site_keys` and deterministic `canonical_ids`. The latter is the full ordered tuple of `(parent_id, example_index)` for exact evaluation identity verification. Constructor checks bind the cache to the pinned model, registered plan, raw manifest and archived sources. Each prepared file and label file is authenticated, with filesystem fingerprints checked on each later use.

Each worker has a configurable LRU of prepared CPU `SiteItem` values, treated as immutable. The default is 2 GiB per worker; root may register a larger explicit cache cap in its run protocol. Cache hits avoid both NFS reads and tensor deserialization. Any changed source fingerprint evicts the item and triggers hash reauthentication. Cache charge uses full serialized state and label bytes, conservatively even for a smaller view.

For `mode='no_cross'`, the separately qualified `model/no_cross_view.py` SHA256 `0227f5cacbe4dc9a9fbd0bcca56a2f8582cd0d4a2c990b48a054e1a221f2b3e7` derives a complete focal-tree view once before the worker LRU. It preserves focal observations, all focal-tree SNPs/ancestors/leaves, labels, canonical row order and exact model outputs. The full five-tree disk cache and frozen core model remain reusable. Root must pin the current runtime loader and helper; `RUNTIME_DATA_RECEIPT_v1.json` records this additive integration while preserving the preparation-time loader source in the cache manifest.

## Training objective and loader qualification

One epoch is a uniform permutation of all sites in the registered support. For support size M sites and N canonical targets, an actual minibatch of S sites uses `sum(cross_entropy_per_target) * M / (S*N)`. The final partial minibatch uses its actual S. Dividing by the realized target count would change the stochastic objective when site sizes differ. Microbatches retain the original minibatch factor and share one optimizer update. Evaluation accumulates losses and counts across all N canonical rows before division.

`OBSERVATION_VERIFICATION.json` records exact unchanged labels and strict authenticated observation/context/sample mappings for every one of the 50,000 targets and 1,245 sites. On one actual window from each parent, every shared observation field matched the frozen `TrainWindowDataset` reader exactly after expanding compact focal rows. This audit took 11.30 seconds.

`SITE_CACHE_VERIFICATION.json` records passing safe-load, source mismatch, changed-cache rejection, LRU reuse/eviction, canonical alignment, target-mean weighting and microbatch checks. Collated versus separate real-site model outputs were exactly equal in all three arms, and the dataset's focal-only no_cross outputs were exactly equal to full-context no_cross outputs.

Complete shuffled fit epochs used 830 sites, 33,334 targets and 104 batches at eight sites per batch. With a 2 GiB worker cache, no pinning, and complete tensor/label/identity hashing included, the measured times were:

| Loader | Epoch | Wall seconds |
|---|---:|---:|
| Zero workers | First | 42.29 |
| Four persistent workers | First | 28.78 |
| Four persistent workers | Second | 6.93 |

The first zero-worker and four-worker epochs had the identical tensor/label/ordered-identity SHA256 `a9cd76bdcfbc49f1a57f0afcc25903628eedcf6f825925aacd1c1edaa0b9c5e0`. These are CPU loader qualification times with deliberate full hashing, not GPU training throughput. Root profiles the actual training loader, device transfers and model steps separately.

## Validation boundary

Independent VAL generation and observed VCF exist at `experiments/results/benchmark_campaign_20261018_workflow/release_v2/european_c4/argmix_release_ap200mb_validation_000/validation_generate/native/generation.json`. The inspected `release_v3/european_c4/argmix_release_ap200mb_validation_000/validation_infer/native` location had logs, inputs and pre-size artifacts but no completed post-size inferred tree. It is also an ARGmix/C4 endpoint without a verified compatible native six-class canonical registry. Root therefore registered TRAIN002 for internal development. Independent VAL was left unchanged. No TEST payload was opened, used for selection, or used for this cache.

This report certifies data/execution readiness. GPU qualification, fitting, model selection and final scientific performance claims belong to root's separately registered run protocol.
