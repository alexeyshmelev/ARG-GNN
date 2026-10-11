# Remaining native ARGformer stages

The VAL000 fit r002 launcher passed its measured500GB check (289664155648bytes), but the native worker stopped before output creation because an unrelated root supplemental reservation changed the first-read ledger hash during the launcher’s four-minute du. Its own20GB reservation remained unchanged. Existing native source/config, scientific model/objectives/full4096 batches, TRAINfits/retrievals and VAL000probe are preserved.

This runner performs only:
1. VAL000 fit r003 using the passed VAL000 probe r002, then CPU retrieval r003.
2. VAL001 probe r003, fit r003 and CPU retrieval r003.

CPU retrieval can overlap the later GPU2 stages. Existing four20GB parent reservations remain intact; only the two VAL reservations are used by these remaining stages. Original admission_bindings computes fresh output/log/run IDs and the existing fixed per-parent short_tmp. No source/model/runtime installation, test suite, numeric gate or actual launch has been performed by this author.

Root activation:
1. Inspect run_remaining_stages.py and reused_receipts.json. Native source/config pins are unchanged.
2. Root may reuse the already active corrected GPU2 packet at research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v4/runtime_GPU2.active.json, SHA f0eb2a62f25206407decf3a4301d4f4b5f8b22277d7b595704ce177994ca1557. Alternatively copy runtime_GPU2.proposal.json to a new root path and set only active=true, then compute its SHA.
3. Keep root reservation/budget writes serialized until pending native/H025 workers pass their first Storage read. Stage storage metadata takes fresh current ledger hashes immediately before launcher entry. Later legitimate unrelated ledger changes remain allowed by the native guard.
4. Root launches:
    .runtime/env/bin/python -B research/reviews/native_argformer_tsinfer_remaining_v1_author/run_remaining_stages.py --execute --runtime research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v4/runtime_GPU2.active.json --expected-runtime-sha256 f0eb2a62f25206407decf3a4301d4f4b5f8b22277d7b595704ce177994ca1557

Default invocation prints an inactive source report and never launches. --execute creates fresh root queue/stage authority within this folder. The existing exclusive GPU launcher, actual inherited lease, same original scientific executor, source/config review, own reservation checks and watchdogs remain in force.
