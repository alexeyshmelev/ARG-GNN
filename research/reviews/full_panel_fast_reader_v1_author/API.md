# Full-panel fast reader and codec v1

Frozen source `src/arggnn/mutation_routed_full_panel_fast_reader_v1.py` SHA256 `d6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03`. This helper must be pinned in the consumer source closure separately from the original scientific cache manifest source_refs.

Compatible consumer imports: `FullPanelParentDatasetV1`, `read_full_cache_index_v1`, `read_full_cache_parent_v1`, `make_parent_dataloader_v1`, `shutdown_parent_dataloader_v1`. Dataset arguments and complete-parent PyG output are unchanged. `get` invokes the original decode function in a private namespace with the fast item-validator callback; original scoring/provenance and byte/manifest source checks remain. Loader/drain APIs remain aliases to the frozen original.

Producer APIs: `check_item_v1(parent_id, item, entry_ref, roles)` and `encode_parent_v1(item, expected_haplotypes=952) -> (compressed_bytes, record)`. Both retain original code objects and bind only the new structural validator in independent namespaces. Original modules are never monkeypatched. `validate_inputs_v1(bank, expected_haplotypes=280)` and `validate_full_bank_v1(bank)` preserve topology, edge/sweep, chronology, root/leaf ownership, missingness, donor/panel, shape/dtype and exact full952 proxy checks through NumPy views and level-wise propagation.

Updated receipt.json: 28 small old/new acceptance cases matched; the full952 proxy contract matched. One unique authorized published TRAIN000 observation bank was decoded and re-encoded. Every NPZ field, dtype, shape and value matched, and compressed byte size remained 41,895,267. On its 1,828,783 nodes/961 trees: decode0.554s; two full validation calls0.397s; original compressed encoder7.376s. No old scalar validator was run on the large bank, no truth was read and no model/GPU qualification was run.

The five original validation/cache/PyG source hashes are pinned in SOURCE_PINS and checked before decode/codec entry points. New v5 CPU production uses the unchanged original full952 feature preparation, which contains no Torch bank validator internally, followed by fast item and encoding callbacks.
