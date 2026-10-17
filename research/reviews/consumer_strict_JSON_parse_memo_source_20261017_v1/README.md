# Inactive strict JSON parse memo component

This is a source-only component for a separately reviewed future consumer. It does not provide a launcher, authority, admission, retry, live patch or guard operation.

The live consumer memo already caches raw source and JSON bytes, so this component addresses a different repeated computation. It reuses the strict parsed value of explicitly allowed SHA-bound historical immutable JSON only within the same original scheduler mutex. It returns a fresh deepcopy each time. Every original reference/SHA comparison, caller and size check, schema/policy/proof/prefix/role predicate, scope inventory, clock and required guard boundary remains. Current controls, arbitrary raw JSON reads, ledgers, journals, observations, inventories, fingerprints and process/scope facts remain uncached.

The actual original v4.parse, bound and original guard reader shapes are derived from exact source/code pins. No API is guessed. The path-to-raw ownership check uses the exact unchanged old consumer frame and its current state['raw'] entry, requiring the same path, digest and raw object. Capacity bypass or absent ownership uses the original parser. The component never patches global parse. Its bound and selected _json/_completed_json seams exist only inside one original mutex and restore before the old memo's final namespace/leaf validation. An additional original uncached byte/digest check covers every cached parsed record before returning.

Eligibility needs an explicit historical ref list authenticated by frozen source manifests plus a complete current-control exclusion list. Guard policy/selector/repair refs and frozen manifests are excluded automatically. Control-like basenames and.runtime paths reject. The final actual control/guard/namespace/runtime/source join remains a mandatory whole-route review task.

Limits are256entries,4MiB raw inputs and32MiB conservative parsed-object charge. Original raw memo limits remain unchanged. These are failstop bounds, not a fit result.

Eighteen focused author checks pass using the original duplicate-key/nonfinite strict parser and genuine historical Census storage bytes. They cover constructor source/shape authentication, exact old raw ownership, fresh-copy isolation, current-control and historical-provenance rejection, conflicting SHA, raw tamper, final byte tamper, restoration, failure cleanup, cold next transaction and inert default. Operational lock/read/final-snapshot providers are RAM doubles. No real guard was imported/configured, no original mutex/state/resource/process operation ran, and no model or numerical fit ran.

An initial own RAM fixture omitted the original raw_hits counter and failed before parsing. The fixture was repaired and the diagnostic is retained in test_history.v1.json. A final source-signature tightening was followed by the same focused checks, with the earlier completed result preserved. No broad campaign was repeated.

This computation reuse changes parser invocation count. Existing source reviews do not authorize it. Independent component review and complete actual whole-route review are required before any fresh future ROOT admission. Copies may cost as much as parsing, so no speedup or training-throughput claim is made. Existing C3, FLARE, H058 and H060 sources/jobs/scopes/clocks/charges remain unchanged.
