# Fresh GPU0 H050/C3 root admission draft

Core source `root_admission_draft_v1.py` is frozen at SHA256 `48bfab0e57aab5f97f761a16a62ec96a01bf19af8b59e352050f27822f9146c4`. `core_seal.v1.json` records its qualification. This worker did not bind, admit, launch, call a guard/state function, or change Git. Root owns every action below.

Use project root/src/scripts in PYTHONPATH and PYTHONDONTWRITEBYTECODE=1. The CLI is default-inert:

    .runtime/env/bin/python -B research/reviews/GPU0_shared_admission_draft_20261016_v1/root_admission_draft_v1.py bind --execute --record ACTIVATION --sha256 SHA
    .runtime/env/bin/python -B research/reviews/GPU0_shared_admission_draft_20261016_v1/root_admission_draft_v1.py admit-h050 --execute --record ACTIVATION --sha256 SHA
    .runtime/env/bin/python -B research/reviews/GPU0_shared_admission_draft_20261016_v1/root_admission_draft_v1.py admit-c3 --execute --record ACTIVATION --sha256 SHA
    .runtime/env/bin/python -B research/reviews/GPU0_shared_admission_draft_20261016_v1/root_admission_draft_v1.py assemble --execute --record ACTIVATION --sha256 SHA --h050-admitted H050_JSON --h050-sha256 SHA --c3-admitted C3_JSON --c3-sha256 SHA

Python APIs are `bind_metadata(activation_ref)`, `admit_h050(activation_ref)`, `admit_c3(activation_ref)` and `assemble(activation_ref,H050_admitted_ref,C3_admitted_ref,native_admitted_ref=None)`.

`bind_metadata` uses only JSON/source metadata. It imports no scientific module, installs no renewal hook and calls no guard/state API. It writes fresh metadata under `bound_metadata_v1`, preserves the prior H050 warmcheckpoint proxy unchanged, adds the actual GPU0 context/runtime/cache source hashes and records root's prior-input/status/repeat-reason/stop evidence. It copies C3's original scientific request/caps and changes only actual0/UUID and the fresh private TMP path. Root's concrete activationv2 was read-only validated during source qualification.

H050 admission compiles the exact frozen original H050 admit function with operational GPU0 eligibility, fresh root-specified packet/output/job, bounded expiry and new source/spec bindings. C3 admission compiles exact prior root_shared_admission_v2.admit_c3 with the corresponding0/namespace changes. Original source/science/caps/count-plan checks and guard reserve APIs remain. No accounting rebase, credit or reservation reuse exists.

H050's prospective direct union is64:43 original required references,8 original budget-only references,14 operational references with1 already-present context overlap. Only DOWN pool.staged.py uses the root-approved existing directly bound source manifest mechanism. Actual pool/runtime/context/guard/launcher/helper stay direct. Activation/request hashes are immutable input/manifest metadata without extra H050 consumer slots. The actual original admission still checks the64 ceiling before reserve. The initial conservative67-reference count and exact paths are retained for audit.

`assemble` creates the fresh32MiB controller plan/storage/capability via original guard flows, with the directly bound reviewed controller scope checker. It never starts the pool. Actual returned worker refs, canonical algorithm entries and caps populate the jobs. The pool independently revalidates the final capability.

Optional native assembly is isolated in `native_assembly_v1.py` so H050/C3 can bind without later core hash changes. Supply `--native-admitted PATH --native-sha256 SHA` to assemble, or the optional Python argument. This authenticates worker2's actual native admitted/input/storage/caps/entry refs, exact12-pool-reference native transitive source closure,40GiB cap and explicit root72h authority. It declares the native third job with at most2 concurrent workers, using the same inherited physical lease. It does not admit native, alter its science or create another pool.

The helper accepts `GPU0_fresh_H050_C3_root_activation_v1` with root active authority, policy/stop/renewal refs, root expiry/drain, jobs.H050/jobs.C3 fresh IDs/output/packet/prior/repeat fields and controller fields. Root's activationv2 retains that schema and adds optional-native authority. Old activationv1 and all prior jobs/charges/artifacts remain historical and untouched.

Source verification covered standard-library import, Python AST, compilation of both exact frozen admission transformations, read-only activation validation and the direct-reference calculation. Native assembly was compiled/imported only. No claim is made that actual reserve or runtime execution has passed.
