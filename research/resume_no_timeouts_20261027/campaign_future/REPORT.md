# Future optional campaign deadlines — isolated preparation

The isolated candidate supports genuine JSON `null` for `max_seconds`: no total elapsed-job deadline is enforced by either the campaign runtime or controller. Positive integer limits remain explicitly supported; missing fields, booleans, zero, negatives, floats and strings are rejected. The current controller and workers are unchanged. This package is **not a globally unlimited or launch-ready campaign**.

`code/deadlines.py` supplies the shared policy. `code/runtime.py` and `code/scheduler.py` contain only the narrow validation/enforcement changes and dispatch to the isolated runtime. `source_changes.patch` shows the entire delta from current production files. Unchanged package-local copies of processes, repairs and continuation preserve relative-import behavior, including repair validation through the candidate scheduler. No original source was rewritten. Memory/storage guards, GPU authority, actual process waits, process-drain deadlines, telemetry polling timeouts, and loader stall detection remain intact.

Eight narrow tests passed in `test_receipt.json`: explicit optional validation, elapsed-boundary semantics, scheduler validation, no-limit progress reporting, finite-limit enforcement, storage enforcement, memory enforcement, and exact non-policy field preservation for the full template. Tests launched no workers.

`campaign_deadline_null_template.json` reconstructs the effective revision-5 plan and changes all 245 non-external task deadlines to null, plus controller source pins. All scientific task arguments, data, models, source lists, dependencies, outputs and budgets remain byte-for-byte equivalent as JSON values. Seven external evidence tasks have no execution deadline. The template deliberately retains the old records directory, continuation reference and conservative storage fields, so it must not be launched directly.

## Exact unresolved nested deadlines

The **150 ARGMix argv tasks bypass campaign.runtime** and still retain finite nested limits. `unresolved_tickets.json` pins each enforcing source and exact line locations; `activation_constraints.json` lists every affected task and flag value.

- 135 CLI tasks use `paper_argmix_features.ScopeGuard`, whose positive timeout requirement and elapsed deadline are reached through `paper_argmix_cli.py`.
- Fourteen bridge tasks and one paint-resume task use the same finite ScopeGuard through their explicit readable drivers.
- Native Relate uses total elapsed subprocess timeouts in `paper_argmix_relate.py:51,92`; the paint-resume driver also enforces one at line 182. The pinned recipe sets `native_timeout_seconds=86400`. The CLI's additional native helper has the same condition at line 35, although current native dispatch uses `run_parallel`.

These require separately reviewed isolated operational wrappers using an explicit no-duration policy and `subprocess` timeout `None`, with byte guards and unchanged native commands/seeds/order/provenance. Do not change the pinned recipe or author scientific code in place. This package records these tickets instead of silently declaring those tasks unlimited.

## Safe activation and storage accounting

Use a natural controller boundary with no owned workers running. The existing controller's exit handler terminates its owned workers, so stopping it now would lose healthy execution state. This package provides no hot-update or worker-migration mechanism. Refresh the effective plan and state at the boundary, prepare a new records directory and reviewed continuation, and preserve successful outputs and original immutable evidence. Successful task definitions must remain unchanged under the current continuation contract; apply null deadlines to future unresolved tasks in the activation plan and declare those replacements. The all-null template is a policy comparison artifact, not that final continuation.

The loaded plan still has a **995,000,000,000-byte ceiling** and **916,096,586,310-byte baseline** from before cleanup. The user requested a future **1,200,000,000,000-byte ceiling**. Neither value was changed in memory or fabricated in this preparation. The original and isolated scheduler still have the original 1 TB validator guard; a separate explicit storage-policy delta is required before a 1.2 TB plan can validate.

The preserved cleanup summary records 380,588,158,976 allocated bytes and 379,076,164,571 logical bytes removed, but explicitly says no fresh whole-project measurement was made. Its summary and both completion receipts are hash-bound in `activation_constraints.json`. At activation, reconcile fresh project usage, task scopes, auxiliary growth and archived-attempt accounting without double counting; do not blindly subtract allocated freed bytes from the logical reservation baseline. The currently waiting `paper_argmix_published_recon_train_002.native_relate` requires 25,769,803,776 auxiliary bytes plus 276,824,064 output bytes. Storage admission remains an unresolved operational issue, not a scientific change.
