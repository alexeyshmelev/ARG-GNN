# Prospective1611 native source adapters

Implemented additive source in this folder; frozen author repositories were not edited. This is a new synthetic published-counts comparison, not exact April Tables 1/2 reproduction.

- `panel_adapter.py` supplies the 1,611-diploid roster, exact haplotype-to-observed-node binding, NF/Yam dual roles, identity-based exclusion of both focal haplotypes, explicit native Rust requests, native flat-output decoding, AP ancestor features and four-founder labels/decoding.
- `native_runtime.py` supplies deferred extraction, fitting and prediction entrypoints. ARGMix invokes existing Rust extraction once per query with explicit allowed donor IDs; it imports the unchanged released transformer/BiGRU builders, chunker, loss and checkpoint functions. No Rust edit/rebuild is required.
- `truth_adapter.py` is a separate external-supervision stage. It joins simulation node IDs separately from observed Relate IDs, checks sample ages/populations, and requires the exact simulated ancestor at census350 and one of Ana/CHG/WHG/EHG. No cohort fallback or approximate-time match is allowed.
- `finite_fit_config.v1.json` fixes native architecture/training budgets and honest method names. The AP comparator is **AncestralPaths-derived C4 adapter**.
- `validate_source.py` runs only source/config and tiny deterministic metadata tests. Forty tests, five source syntax compiles and 68 frozen-author source pins passed.

## Exact implemented condition

The synthetic roster contains Ana22, CHG13, EHG42, WHG48, NF159, Yam13, Post-Neolithic654 and Present660 diploids; BAA0. Fixed integer generation ages are 260/300/250/250/180/160/135/0, with one NF proxy at185. These are explicit new simulation choices, not recovered real people or Iceman dates.

All six ancient reference cohorts remain eligible donors; NF/Yam also remain query cohorts. Both focal diploid haplotypes are removed before reference selection. Other NF/Yam samples keep their true observed cohort tokens. There are 594 donor candidates, 592 for an NF/Yam focal query after exclusion, and 2,972 eligible query haplotypes. Nodes need not be adjacent or ordered; the binding must cover all3,222 identities exactly.

ARGMix retains released15-reference + query extraction, zero ancestors, log2cap14, transformer hidden128/four heads/four layers, output feature16 and bidirectional one-layer GRU hidden128. Training has a20-epoch maximum, early stop2, AdamW2e-4/weight decay1e-4, batch16 and label smoothing.1. Native1024-row chunks/stride512 preserve overlap weighting. All observed intervals for requested queries are context; only common midpoint targets receive labels. Entirely unlabeled chunks are omitted from loss iterations.

The C4 AP adapter retains released5×9 ancestor features, Flatten→Dense512ReLU→Dropout.2→Dense6softmax, Adam and five epochs/batch32. It replaces fixed index ranges with the common allowed donor identities. Ancestor age uses the original training path's int(time)/1500 consistently; original paint uses floating time. Positive targets are only native heads0..3. Native BAA-specific heads4/5 receive no positive labels. Pool p0+p4 to Ana, p1+p5 to CHG, retain p2WHG/p3EHG, then use four-way argmax. This can differ from native six-way argmax; it is a changed prospective task with a retained classifier architecture.

Other disclosed changes: disable native drop-nearest TRAIN augmentation, include edge trees, apply exact common midpoint supervision, use existing registered fit seeds and select AP minimum external VAL loss within five epochs. ARGMix minimum-VAL selection remains native. Input/source hashes and exact common labels/truth definition appear in fit signatures; pairing checks reject mismatched supervision.

## Execution boundary

No native extraction, truth payload construction, fit, prediction, numerical-library import, real array/checkpoint read, job, guard/admission, install, Git or GPU/process probe was performed. The numerical entrypoints are executable source but remain unexercised. Existing interpreter paths were recovered from source constants only.

`ROOT_COMMANDS.md` gives source-only commands and separate future root entrypoints. `INPUT_CONTRACTS.md` states required observed mappings, common targets, external truth contracts and map binding. Source tests do not establish numerical ABI/runtime behavior or resource bounds; root owns the future execution wrapper and resource accounting.

No public map was needed for adapter implementation or tiny metadata tests, so none was downloaded. Earlier recovery recorded no authenticated local chr22 empirical map. Map binding checks local rebased bytes and exact[0,L]coverage to prevent native edge extrapolation; HTTPS/hash syntax does not authenticate public origin or the coordinate transform. Exact public GRCh37 chr22 map origin/build/interval/hash and derivation remain an explicit root-owned input precondition. No length is guessed.

The no-BAA marginal-law qualification remains limited to the pinned noncontributing terminal branch, with all other events/dates fixed; it promises neither identical seeded realizations nor paper-era behavior.
