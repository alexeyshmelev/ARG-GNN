# Inactive GST USPTO-15K native-data protocol v1

This is a source/metadata implementation checklist under H022. `active=false` in [the companion JSON](../configs/gst_uspto_native_data_protocol_v1.json). No ZIP was opened, no member was decoded or extracted, and no dataset row, raw array or model was executed. This artifact authorizes no extraction, parsing, installation, job or scoring. Root owns subsequent registration, storage admission and execution. The effective contract is immutable v1 plus corrected v3, whose source-only independent PASS already exists. That PASS does not establish empirical readiness.

## Exact pins and unresolved archive identity

The acquired archive is `data/paper_native_raw_acquisition_v1/gst_native_snapshots/USPTO15K_data.zip`, 3,446,572 bytes, SHA256 `66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723`, Git blob SHA1 `b60543892220e5b6517b6fd22580db7627f5077b`. Its upstream is `wengong-jin/nips17-rexgen` at `fb7dea369b0721b88cd0133a7d66348d244f65d3`, path `USPTO-15K/data.zip`. These hashes are inherited from the existing verified acquisition receipt, with only a current size stat here. Original GST author-used snapshot identity remains unproven.

GST source is `daenuprobst/gst-conference` at `6469d56aa131463325058b897b1e435c6d2edcd2`. The native runner `scripts/eval_rxn_center.py` has SHA256 `57fd803410d57c9b05d0719cdb5bd318d9d90b2cdebbc86a1260c37a3c3354b3`. Its actual import uses genuine `src/graph_set_transformer/models/gst.py`, SHA256 `bc08c15d55247b32bda98e4b03fcaee988ceb8f654423240a33ca632ce11ef72`. Both cached full-file bytes also match pinned Git blob identities. This protocol implements or aliases no missing `gst_v2`.

The runner proves three logical filenames: `train.txt`, `valid.txt`, `test.txt`. It does not prove their complete ZIP member paths. Prefixes, extra members, topology, member SHA256 hashes and raw/retained counts remain unknown because the archive was not opened. Never invent these fields from the dataset name or source comments. Archive-defined splits, original row order and all duplicates remain intact.

## Future decoder and member receipts

Before any future archive opening, reverify the complete raw hash and freeze a fresh contained destination and peak storage reservation. The central-directory inventory comes before payload extraction. Record every member in archive order, including ancillary members, with its exact name, type, flags, compression, compressed/uncompressed size and CRC. Bind the three logical split names only after unambiguous actual paths are frozen. Ambiguity fails admission.

The future decoder must avoid `extractall`, reject path escape, absolute/drive/UNC names, backslashes, NUL/control characters, dot components, special/link members, encryption, unsupported compression and normalized/case-folded collisions. Handle genuine directory entries explicitly. Resolve containment and create new files exclusively without following symlinks. No guessed flattening, overwrite or archive-derived permissions is allowed. Stream exact uncompressed bytes with full SHA256 and CRC verification, declared per-member/total limits, and preserved partial-failure receipts. The JSON proposes bounds of 512 members, 256 MB per member and 1 GB total. These bounds are not admitted estimates of actual archive size. Exceeding them requires a new reviewed budget decision, never selective member deletion.

Bind exact member hashes and bytes before text parsing. Preserve original line endings and record the runtime default text encoding/universal-newline behavior used by the unchanged author `open(path,"r")`. Any relocation required to present the source data-directory layout is a recorded mechanical mapping with byte and row parity.

## Native rows, labels and denominator

The native `rxn_id` is the zero-based raw file enumeration index, including rows that are dropped. `load_split` increments seen rows, parses each row, appends successful reactions, then tests its retained-example limit. `max_train=1000` therefore means the first 1,000 retained reactions, not the first 1,000 raw lines. Remaining rows are not parsed by that saved condition. They must be counted as `not_parsed_due_to_native_retained_limit`, not declared invalid.

The parser strips a row, uses its first two whitespace fields, takes reactants before the first `>>`, and labels reactant atoms by the union of lost/gained-H map numbers and deleted/added-bond endpoint maps. It preserves dot-separated reactant order and RDKit atom order. Inputs are the native 19 float32 atom features and molecular edges. Edit labels stay external to model features. No product validation, new map validation, deduplication or feature engineering may be added silently.

Record every original row by member hash, split, native ID, physical row number, byte offset/length/exact-row hash, status and reason. Native return-None reasons include blank/short rows, absent usable edits, unparseable/empty reactants and no matched positive atom. Malformed bond tokens are silently omitted by native code even when the row remains accepted, so record partial omissions separately. `_parse_atom_list` can instead raise an uncaught `ValueError` for malformed H-edit integers. Preserve that native exception as a fatal failure, with subsequent rows marked unreached. Converting it to a dropped row changes source behavior.

Require disjoint raw-row accounting across retained, native dropped, fatal, unreached and limit-tail statuses. Preserve accepted atom labels/features/edges/order and exceptions when adding instrumentation. Native pooled AP uses all atoms of native-retained TEST reactions. Also report raw reaction retention, partial-token omissions and each model's output coverage. Do not fabricate labels for dropped reactions or reduce the paired denominator to conceal candidate failures.

## Separate coverage and optimizer conditions

| Condition | TRAIN coverage | VALID/TEST | Native architecture |
| --- | --- | --- | --- |
| `released_source_saved1000` | First 1,000 retained reactions | Full native retained splits | Saved hidden 128, 1 layer, 4 heads, dropout 0.1, equalize smaller |
| `released_source_full_native_data_saved_architecture` | All retained reactions | Same full retained splits | Same saved architecture, explicitly changed TRAIN coverage |

The saved config hash is `ea02c7ab159b569e938a3ca9423287318bb986cc482998a64876d77e6b1eff9b`. It specifies five seeds 0–4, 20 epochs, batch 16, AdamW, LR 0.001 and weight decay 0.0001. Full data with the saved architecture is a named coverage extension. It is not proof of every original paper Table 3 cell. Default CLI hidden 64/layers 3 differs from this saved result.

Keep released-source common LR 0.001 for all four models separate from paper Section 3.1.2 AdamW LR 0.0001 for GST variants and 0.001 for DeepSets/SetTransformer. An unchanged all-model invocation has only one `--lr`; it cannot implement the paper rates. A new per-model driver is a scientific training change with its own diff, pins and qualification. USPTO source/paper both name AdamW. The BH source Adam/paper AdamW discrepancy belongs to the effective contract and does not apply to this runner.

Retain positive weighting from the first `min(TRAIN,1000)` retained training examples only, norm-1 gradient clipping, native epoch shuffles and cosine scheduler. Its `T_max` uses `epochs * max(1,len(TRAIN)//batch_size)` while every actual batch steps the scheduler. Replacing floor by ceiling is a protocol change. Strictly improving VALID pooled AP selects the checkpoint, preserving earliest ties and explicit final-weight fallback if no finite improving score occurs. TEST uses pooled per-atom sigmoid AP, with NaN preserved for fewer than two label classes.

Source `run` constructs a model before `train_one` seeds Torch/NumPy/Python. Parameter equalization also constructs models and consumes RNG. Seeds 0–4 alone therefore do not pin initialization. Preserve and capture process-entry and per-build RNG states. Moving seeds before construction or choosing a new entry seed is a disclosed change, not a silent reproducibility fix. The source gate differs from the paper two-gate equation and remains pinned source behavior.

The native runner loads and audits TEST before fitting and reports TEST scores after each selected fit. Freeze all conditions and tuning before this source-native access, record that access, and never use its TEST summaries for subsequent method selection. Runtime admission uses handmade fixtures, not TEST rows.

## Paired benchmark and implementation checklist

The genuine native model registry is GST cross-attention, GST broadcast, GCN+DeepSets and GCN+SetTransformer. Preserve all source width-search/build order, objectives and output heads. Saved parameter counts are author metadata rather than new measurements. A candidate native-task adapter remains unimplemented. Its molecular inputs, per-atom head, objective, selection and tuning budget must be explicit. Both sides must use identical accepted reaction IDs, atom identities, labels, splits and complete scoring support. Genealogy LAI results cannot replace this reaction-center endpoint.

The saved author environment is Python 3.14.4, Torch 2.10.0+cu128 and CPU. The package requires Python >=3.13, but RDKit/PyG/NumPy/sklearn versions and the full library closure are unpinned. A Python3.12/Torch2.10+cu129 reuse is a runtime port requiring a separate identity and qualification. No environment is selected or installed here.

1. Root registers the unfinished H022 branch, independent protocol/implementation review and fresh peak storage admission.
2. Implement bounded metadata inventory and freeze exact archive paths, then independently qualify path-safe extraction and complete member hashes.
3. Implement native row/partial-token/fatal-exception receipts and qualify exact parser parity on handmade fixtures.
4. Freeze the complete genuine GST execution copy/runtime and qualify model/optimizer/checkpoint/RNG behavior on handmade inputs.
5. Freeze coverage, source-versus-paper settings and the candidate adapter, then run the complete paired task with failure and denominator receipts.

Every checklist stage is unimplemented in the JSON. Activation belongs to a new root-owned execution contract. This artifact contains no empirical reproduction, accuracy or superiority claim. Evidence, source-function segments, author checks and SHA256 seal are under `research/reviews/gst_uspto_native_data_protocol_v1_author/`. No ledger/decision, existing source or Git file is modified by this authoring task.
