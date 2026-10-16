# Native VAL000 full-state resume v7 independent delta review

Verdict: REQUEST_CHANGES_SOURCE_ONLY. One P1 finding blocks the actual entrypoint. Execution is not authorized.

The exact candidate manifest is research/reviews/native_VAL000_full_state_resume_source_20261016_v7/source_manifest.v1.json, SHA256 a3bd3b909bda7d0aaa190d75706eba333ba838174ccc98a68c46dd55c35bac8b. Its sealed handoff SHA256 is 13810086756ec0859472f2d2c4fce25290cdf804ec84ca75765f39050fb0efa8.

F1: execution_only_v1.py line7 still imports the v6 amendment contract. Its support imports and worker module string at line168 remain v6, as do the contract imports in direct_gpu0_v1.py line9, corpus_metadata_bridge_v1.py line9 and resume_scientific_bridge_v5.py line7. Actual v7 authority is therefore checked by the old schema and rejected. The new alias wrappers are also absent from the actual imported driver namespace. This is an actual routing defect; extracted-function fixtures do not cover it.

The independent fake reproduction executes the exact import node with a fake predecessor module and the actual v6 validation body with a fake v7 authority. It confirms the old namespace, absent new wrappers, old schema rejection and stale worker dispatch without importing project modules or probing actual authority/runtime state.

The rest of the narrow alias design passes source and tiny software checks. All13 real tiny normalizer cases and the actual extracted wrapper prep/consume/terminal sequence reproduce in the reviewer directory using original canonical/measure_scopes. The exact64KiB exception is explicit, all eight missing-field cases reject, and boundary=False uses zero planned writes. Independent whole-scope charge/projection checks precede durable exact metadata; dirfd-relative unlink touches only latest-rank0.pt after drain and identity revalidation; strict original boundary follows. Every regular checkpoint remains unchanged.

The intended call order repairs startup before first strict validation and emits genuine stdout waits before post-drain alias repair, then strict filesystem evidence. Full cleanup remains unchanged. The three science/full-state/capacity source files remain byte-identical to v6, including their defective old package bindings. Original checkpoint bytes, original7GiB charge, retained original plus two hard2GiB files, full227990 science, guards and deadlines are preserved by the intended functions.

All204 distinct packet/source/compact metadata refs, including172 inherited pins, passed hash verification and sealing recheck. The inherited broad scientific review was not repeated. Original actual aliases/checkpoints, prior v6 source and PASS_SOURCE_ONLY review, actual v6 authority and failed preparation lineage remain unchanged. No actual checkpoint payload, original namespace census, process/device probe, accounting/authority call, launch, Git or package action occurred.

Publish a fresh sealed successor with every dotted module import and worker command rebound, add a fake import/dispatch regression, and obtain independent review. ROOT separately owns actual authority, namespace repair, capacity/storage qualification and any launch; future runtime receipts are not used as source-review blockers.
