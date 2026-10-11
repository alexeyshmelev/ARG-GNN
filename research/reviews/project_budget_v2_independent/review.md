# Independent project budget v2 source review

Verdict: FAIL for c6d7bc9cb6693a5882a4342d28e66b82a7c9c790841f89fb73b88d3e005a90ef.

22 independently authored guarded handmade tests ran. 18 passed and 4 expected rejection cases failed, with no guard events. No live ledger or scientific product was read or written.

B1: Legacy jobs are checked only for exact duplicate output paths. An ancestor and descendant pair of legacy output scopes is accepted by reserve and current, despite required exclusive scopes across and between ledgers.

B2: Known-schema legacy ledger and entry field sets are not validated. An unrecognized top-level collection with 700GB reserved_bytes is ignored, and an extra job auxiliary_scopes overlapping the proposed supplemental output is ignored. Unknown accounting evolution therefore qualifies instead of failing closed.

B3: A legacy output equal to .runtime/storage is accepted even though it contains the supplemental ledger namespace. The incoming scope check protects supplemental scopes only.
