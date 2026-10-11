# Census biological fit v2 author seal

This bundle is author evidence for the minimal successor of preserved v1. It is not an independent review or an execution capability. The same agent previously reviewed v1 and authored this correction at root's explicit assignment. Root will review v2 independently.

`source_selection_checks.py` parses the sealed sources, proves the full byte/config/AST delta, verifies all 35 dependency pins and all prior author/reject artifact pins, and executes the exact source selection loop with bounded in-memory stand-ins. `checks_attempt01.json` and its stdout/stderr preserve the only run: 319 passing checks and seven fixture sequences, with zero actual training steps. `selection_checks.json` records every scalar fixture and its limits.

`source_complete_delta.diff` and `config_complete_delta.diff` are complete unified diffs. `exact_delta_proof.json` lists all permitted replacements and normalized AST hashes. The source differs only in its fit-owned version literals and the checkpoint predicate. The config differs only in kind and launcher module.

The source checker writes exclusive evidence files and is intentionally not a command to rerun over this sealed bundle. Fresh independent reviewers should put their own tests/results in their assigned independent review directory. The author validator reads the seal and writes its own exclusive validator receipt. Run it under the sourced project environment, existing `.runtime/env/bin/python`, disabled bytecode, empty CUDA visibility and one CPU thread. No packages, Git, device queries, live ledgers/locks, scientific arrays, actual models, optimizers or fits are needed.

The current condition remains Relate. The same-tsinfer consumer remains separate/unqualified. `budgeted_shared_pool_v2` still lacks fit_v2 in its finite allowlist, and its pinned v1 eligibility is not v2 authority. Root will address eligibility separately after source sealing. `registration_request.json` is an inactive request for root's H021 implementation-correction registration, not a ledger mutation.
