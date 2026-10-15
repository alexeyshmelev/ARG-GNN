# Independent source-only audit

Verdict: **one material compatibility bug requires correction before using the documented PilotOutputV9 API**. No other material accounting defect was identified in the reviewed source under its explicit cooperative root-revocation assumptions. This is not activation approval or evidence that 30 GB of credit exists.

## Material finding

**[P1] Saving a payload over 32 MiB writes the file, then fails while returning its reference.** In completed_scope_credits_v1.py:415, PilotOutputV9.save returns _ref(self.path/name). _ref delegates to budget.v4.reference (line 32), which calls project_budget_v4.file_bytes. That metadata/source helper rejects every file larger than MAX_JSON = 32 * 1024 * 1024 (v4 lines 11, 51–63, 77–78). Consequently a valid, cap-compliant npz/torch/large JSON save above 33,554,432 bytes reaches the write/fsync at lines 412–413, increments used, then raises and preserves an output for which the caller received no reference. This contradicts HANDOFF.md:26's compatibility claim and can stop scientific consumers. The original V9 writer hashes the serialized data directly and has no such payload limit (project_budget_v9_consumer.py:140–143).

Fix: return the canonical relative path and hashlib.sha256(data).hexdigest(), as the original writer does, with the original post-write inode identity check retained if preserving the same save integrity contract. Do not route scientific payload references through the metadata-size-bounded helper.

## Source conclusions and limits

- The exact immutable capacity-v2 source SHA is 40c3e45fc148ffd07119bb0c90e5c7182c7fb84829f9df2b4bb4fa978af42281. The audited adapter SHA is f27da531d4322d998f191cd1e5f6c8cdaaaa201fd88be56c17c816b2673a2f2d.
- The frozen path reader is compatible with new unstored/stored V9 reservations: the original CPU reader selects successor_paths, which removes root_admission before scope_spec. The actual measure_scopes result matches _measurement exactly.
- The separate journal, frozen base projection, cap-minus-retained closure, prefix admission checks, permanent closed-key/path occupation and exclusion of already closed/malformed keys are internally consistent in the inspected source. Old active GPU4/5 scopes remain fully charged under unchanged guards and journals.
- Sound closure still requires honest complete root launcher/supervisor/worker rosters and actual revocation of every old/original/resume write capability. Old guards cannot learn successor closures. HANDOFF.md states that constraint.
- No live guard import/configuration, journal inspection/replay, PID/process inspection, scope inventory, admission, closure, job, install or Git operation was performed. Verification was source tracing, exact source hashes and AST parsing only. The audited implementation was not edited.
- Root must establish genuine unclosed completed scopes and measure their retained owned bytes before claiming available credit. No numerical capacity was certified.

See source_evidence.json for exact reviewed-source hashes.
