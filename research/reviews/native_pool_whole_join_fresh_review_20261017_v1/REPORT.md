# Independent complete native/H058/downstream/pool review

**Verdict: SOURCE_BLOCKED.** One concrete provenance blocker remains in the sealed v1 packet. Existing jobs, recorded scores and runtime evidence are unaffected. This review grants no execution, admission, completion, close, storage credit or resource-fit authority.

## B001 — Bind the completed native fit to its waited input, manifest and consumed terminal

`benchmark_barrier_v1.completed_native_fit_bundle_v1` (lines 5–42) returns an authenticated completed-fit bundle without checking three necessary joins:

- The completion's `active_input_ref` (or `child_input_ref`) must equal `wait.input_ref`.
- `wait.adapter_manifest_ref` must equal the exact reviewed aggregate manifest.
- `outer.actual_terminal_consumed` must be true.

The independent RAM regression executes the exact extracted function. A valid baseline is accepted. Independently changing the completion input to a foreign input, the wait manifest to a foreign manifest, or the terminal-consumed field to false still returns a completed-fit bundle in all three cases. This contradicts the handoff's same-input/source/consumed-outer provenance promise. The separate downstream `pool_prerequisite_bundle_v1` already checks these conditions.

Repair these bindings in a new sealed successor and retain a valid positive plus all three rejection regressions. This helper calls `native_common.science_v1`, so its scope is the future fresh VAL001–004 pool fits. For those fits, bind `completed.active_input_ref` to `wait.input_ref`, the wait manifest to the new aggregate, and require the consumed-terminal flag.

Keep current VAL000 v12 distinct. Its existing legacy `_bundle_v1` path correctly prefers `completed.child_input_ref` when present and binds it to `wait.input_ref`. Its actual wait must retain the genuine reviewed v12 execution manifest and amendment, rather than the new pool aggregate. Preserve its real full-state resume and retained original `active_input_ref`/history. Do not impose the future fresh-fit optimizer/no-resume conditions on genuine current v12 completion, and do not accept either of two conflicting input fields merely because one matches.

Preserve actual pool-owned leader waits, original clocks/charges/science and every historical failure/unknown. The v1 packet was not edited.

## Complete source verification

All 104 sealed files (870,415 bytes) and 384 aggregate references match their hashes. All 301 aggregate Python sources parse. All 124 producer references, 21 exact file deltas and all 50 preserved prior-pool files verify. Forty literal reference occurrences were inventoried. The sole non-aggregate executable ref is bound by the pinned runtime JSON; executable bytes were not opened or qualified.

All five supplied final AST/RAM checks were independently rerun without writing the sealed packet. They pass. Seven extra RAM source-test groups pass: the complete 40-declaration roster and illegal declarations; both actual native/downstream capacity functions at the exact 80 GiB boundary and next MiB; owned title and UUID negatives; both actual direct-role union helpers at 64/65 and conflicting hashes; full 65 GiB/group-cap census; and the actual generated queue with one active job per kind and exact module dispatch. All Popen, paths, identity, inventory and lock surfaces in these tests are RAM fakes. The joined persistent-capacity regression confirms sticky abort, queue cancellation, continued direct/adopted waits and owned drain while unrelated external occupancy remains.

The entire admission/dispatcher/attachment path was read, including current explicit Relate child selection, existing mutation memo, real pre-reserve union64 checks, fresh scopes, finite capabilities, exact old-owner handover, writable exclusion proof, new physical lease ownership, borrowed-inode semantics, gate-before-science, GPU0 title/UUID, reservation/unfilled/external/headroom accounting, PID/start/pidfd lifecycle, bounded cleanup, genuine actual waits and late prerequisites. No other source blocker was found in these joins.

## Original science and barriers

H058 producer, observation and context ASTs match the sealed predecessor modulo namespace. Original 32 complete TRAIN parents, 30 passes, 960 updates per arm/1,920 total, FP32/seed101, independent AdamW resets, warmstart, matched arms and TRAIN-only truth remain. H058's 96-product barrier and C3's 30-product barrier remain separate.

Remaining VAL001–004 MLM uses each original complete parent and all six original corpus/order refs, original registered seeds/vocabulary plus four specials, genuine preparation/count direct0 and drain, read-only corpus certification and an unfinished-cell registration. Updates derive as five times floor(original TRAIN rows/512) at world1/drop_last. The original author loader is called once, checked for full cardinality, eight workers and batch512. The original Trainer must begin epoch/batch0 with empty optimizer state. Current VAL000 MLM replay, parent checkpoint transfer and fixed VAL000227990 budgets are rejected. Original model/objective/YAML, five epochs, seed17, BF16/FA2 and batch/micro512 remain.

All five original parents retain the full author downstream route: one epoch FP32 contrastive, batch/micro4096, seed36, temperature0.3, original class-weight fallback/dot/normalization; all20ba checkpoint events with disclosed keep1 retention; four separate complete batch256/head_norm/width768/max_samples0 embedding calls; unchanged train-then-val tensor/order joins in RAM; original author GPU top1 with q_batch1024/e_chunk100000 and unchanged tie arithmetic; truth-free seal; CPU metadata finalization after six actual scientific outcomes. Finalizer direct wait/drain and pool outer0 remain pending until genuinely observed. The barrier still requires all five original 10,000-focal/30 Mb native products and grants no truth access by itself.

Original native per-ARG30Mb/1600-haplotype BF16-MLM/FP32-contrastive conditions differ from H058's32TRAIN2Mb/952-haplotype FP32 protocol and saved C3 GNN training. These are declared pipeline conditions, not matched-budget architecture attribution. The original release's PCA evaluator is unavailable. Exact Table1/sample-unit reproduction and evaluator choice are not claimed.

## Resources and retained limitations

The original separate65 GiB downstream scope remains:8 GiB contrastive,56 GiB embeddings,1 GiB retrieval/finalizer. Original four full embedding tensors and all saved IDs/masks/labels/extant IDs remain. The VAL000 scalar payload forecast is49,979,628,096 bytes before metadata allowance, not measured serialization/RAM/device fit. No tensor compression, workload reduction, CPU retrieval substitution or old7 GiB scope extension is introduced.

The new controller128 MiB/registry32 MiB bounds, native7 GiB scopes, H058256 MiB scope,4096 inner/4097 pool-native identity ceilings,384 GiB native RSS and original64-role ceiling remain explicit failstops. Native34 GiB + H05830 GiB +4 GiB headroom is nominal68 GiB before external/driver usage. The74 GiB downstream allocator+2 GiB margin cannot coexist with H05830 GiB on80 GiB. The32 GiB profile requires four explicit ROOT acknowledgments that it is an unqualified full-workload attempt. No actual fit or current union result is inferred.

D001 is diagnostic, not a separate source authorization blocker: native/downstream direct adapters create bounded failing-capacity samples, but their supervisors currently stringify the exception and retain only the preceding successful sample. H058 already emits and records the failing sample before owned stop. Enforcement/drain still fail closed. A bounded truthful fix in the same successor would preserve efficient diagnosis.

Actual current VAL000 direct0/drain/outer0, complete old borrowers/controller closure and revocation, a new exclusive owner, fresh finite authorities, actual union<=64, sufficient storage, original remaining-parent counts/waits/certificates and full-workload resource fit all remain external execution prerequisites. No numerical datasets, checkpoints or private truth were read, no production module was imported, and no real guard/mutex/process/device/resource/runtime/admission/journal/policy/install/Git action was performed. Every project write is inside this review directory.

See `review.v1.json` for exact references and consumer fields; `provenance_regression_v1.py` reproduces B001 using only source and RAM fixtures.
