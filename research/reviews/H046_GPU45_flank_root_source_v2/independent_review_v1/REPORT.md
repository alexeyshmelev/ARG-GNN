# H046 v2 independent integration review

PASS: 33 source, metadata and AST/text checks. The actual inherited launcher accepts arggnn.H046_GPU45_flank_worker_v2 and reserves src/arggnn/H046_GPU45_flank_worker_v2.py directly. Driver mandatory binding, input worker_cli_ref and launched module agree. The canonical wrapper rehashes the source closure and delegates the unchanged run_context_v1 API.

V1 config differs only in source_manifest_ref. All 77 original source pins remain exact, with four v2 integration sources added. No stale v1 dotted imports remain in the four new sources. Scientific model and worker hashes remain frozen. Fresh job/output names preserve the old failed launch lineage.

This review imported no project code and used no runtime, empirical arrays, checkpoints or GPUs. It does not establish runtime success, GPU fit or scientific outcome.
