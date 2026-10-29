# Independent CNN4 queue review

Verdict: READY for root launch, subject to the existing live admission checks.

Reviewed plan: research/joint_evaluation_20261029/queue/cnn4_plan.json, SHA256 89969f1fd185c6ef03434d51a6d66e53df40f4e687f2323a7c689c8e5672cfda.

Independently authenticated all controller and task source references. Compared all fifteen concrete native calls, task identities, output paths, dependency edges, interpreter selections, disk caps and RAM caps to the original factory plan. All match. The only scientific task policy adaptation is the removal of the arbitrary wall limit. The plan runs one CPU lane, with a maximum 64 GiB task RAM reservation and no GPU task.

Read-only validation and admission succeeded in the native ancestralpaths_cpu_v1 runtime with the admission save function mocked. No jobs, claims or signals were issued. Total charged commitment is 1,117,673,015,226 bytes, below the owner's 1.2 TB limit. The CNN allocation remains 640 MiB within the existing 1 GiB combined evaluation scope. Available RAM at review was 1,729,470,464,000 bytes against the controller's required 1,717,986,918,400 bytes; this is a narrow live margin, correctly rechecked before each task.

Reviewed scheduler task completion, dependency resolution, output freshness checks, aggregate memory checks, shared evaluation cap and plaintext subprocess construction. No launch blocker found. This review does not reopen the completed European result audit or claim an exact published four-class numerical reproduction.

## Runtime correction and successor qualification

The initial source-only READY verdict missed an execution integration failure: all eight original preparation tasks failed because their authenticated inference JSON manifests are 293,096,357 to 294,354,022 bytes, exceeding the original 64 MiB reader default. Their original source, outputs, direct failed waits and controller state are preserved. The original controller exited naturally; no signal was sent.

The separate queue/cnn4_v2_workflow.py changes exactly one line in the original preparation function: its inference metadata read receives the exact measured bound for the pinned SHA256. Every other preparation operation is identical, and fit/predict/score delegate directly to the original workflow. The successor uses fresh outputs under the existing combined evaluation root and retains every original scientific argument, dependency and resource cap.

An actual full TRAIN parent preparation completed with exit code zero in 75.7457 seconds, producing all 13,334 original non-BAA rows. The original native load_features function accepts the saved result and confirms shape (13334, 5, 8), exact canonical identity and no query truth read. This is a preparation qualification, not an ancestry accuracy measurement. The output is retained in experiments/results/paper_ancestralpaths_four_20261027/cnn4_v2_prepare_smoke_train000.

All 68 focused successor checks passed. Evidence: review_finish/cnn4_v2_checks.json, SHA256 a1e924033a5a68a12ee3b82a5de85f223543243a2714d41939f2d5f55af63f30. Successor plan: queue/cnn4_v2_plan.json, SHA256 63d5084ab042fea644ad85ee408f225273bdd04b1df97192e837bcf13c035d15. Workflow SHA256: 65abf5c1d96c45f942e9a20c5e59dadd2ff466d4bc9ed4185043e033ee87c012.

Successor status: READY for root review and launch using the existing CPU scheduler and the new plan path. No production fitting was launched by this reviewer.
