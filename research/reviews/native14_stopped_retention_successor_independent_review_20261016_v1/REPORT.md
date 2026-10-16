# Independent review: native14 stopped-retention successor

Verdict: **REQUEST_CHANGES_SOURCE_ONLY**. No activation, retirement, credit or execution authorized.

Candidate handoff SHA256:a725a81c8844f7ba4dbba7d19cf898ace170ca85bb3c5daefb0567df08be7b5e.
Candidate manifest SHA256:46ba3df4f7dcc217160bff80186e0730de253b77a9e6b0c00ee840b9951820e3.

## Blocking finding

**[P1] Socket pathname string inequality does not prove inactivity (native14_stopped_retention_v1.py:652–654).** The measurement helper supplies canonical absolute paths. The guard checks only that exact byte string against raw /proc/net/unix addresses. A live AF_UNIX filesystem socket can have been bound under a relative pathname or an equivalent noncanonical spelling/alias, which remains the recorded kernel address. The guard then reports the same scoped socket absent. An unretained historical descendant can therefore leave a live socket that the new all-node census accepts as inactive before conditional retention credit.

The independent fake-only check executes the exact source guard: a canonical absolute live-address fixture blocks, while equivalent relative, redundant-separator and dot-component fixtures each incorrectly return True. It does not read the real kernel catalog or probe an endpoint. This is particularly relevant to additional discovered sockets whose bind spelling is not authenticated by the24 historical endpoint records.

Publish a fresh sealed successor that fails closed on sockets without a justified exact kernel-address/identity mapping, or rejects additional/unqualified sockets. Any qualified alternative must cover relative paths, lexical equivalents and aliases while preserving the no-follow/read-only boundary and explicit incomplete-history limits. Add these negative fixtures.

## Passed checks and nonblocking note

All222 runtime source/compact metadata refs and17 authored files are exact;166 Python dependency refs parse. All89 prior audit refs are preserved. Independent unions reproduce150 PIDs with6 UNKNOWN starts,11 groups and24 exact endpoint/lstat records. The exact old14GiB reservation and all3 entire charged parents are preserved; current native7GiB and separate5MiB admin are excluded. Original expiry remains2026-10-18,with-15/-15/-15 and actual outer1 failure,unknown original v8 exit/optimizer/full descendant drain and no successful namespace03 receipt preserved.

All30 claimed ordinary API bodies and the ordinary reserve/completed replay branches are AST-identical to the pinned Census parent. Actual source-only ancestral role reconstruction gives7→13→15→17→23 parent refs; union with historical45 gives55,authored3 gives58,future policy/activation/revocation/review gives62,observation/operation gives64;65 rejects. No direct role dropped or ceiling increased.

All41 candidate tiny synthetic cases reproduce in this review directory,with54 regular fake payload bytes. Tests preserve synthetic symlinks,sockets and failed fixtures. The no-follow helper includes every node in all3 roots,uses the existing max(size,blocks*512) basis plus4096 per node,rejects unsafe nodes/races and performs2 full inventories plus final revalidation. Revocation chronology,stable identity/presence across certificate/pre/post measurement,UNKNOWN preservation,fresh known-target facts and final authority expiry are explicit in source. The socket-inactivity claim remains invalid because of F1.

Nonblocking documentation note D1: source_static_checks.v1.json:24 pins an obsolete direct_role_plan hashff9b1444... while the correct sealed/current hash is5a9a2b8d.... Handoff correctly binds the current plan,and the independent role reconstruction validates it. Correct this stale documentary pointer in the fresh successor packet.

## Actual ROOT work remains blocked

After fixing F1 and obtaining a passing fresh exact-manifest review,ROOT still needs genuine permanent scoped revocation/residual-history acceptance,actual frozen current parent prefix/projection and activation,complete64-role union,fresh known-target observations,accepted corrected all3-root retained measurement,and a current exact operation authority/journal prefix. No retained size,credit,current capacity,retirement eligibility or new admission is established.

Candidate unchanged. Only source/static/compact metadata reads and isolated tiny fixtures in the owned review directory were used. No actual process,filesystem payload census,GPU,accounting,authority probe,scientific import,Git or package action occurred. Structured evidence refs and blockers are in review.v1.json.
