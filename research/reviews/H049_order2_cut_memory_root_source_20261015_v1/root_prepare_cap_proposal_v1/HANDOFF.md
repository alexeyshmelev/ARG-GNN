# H049 prepare-only cap proposal

**Ready for root review; inactive.** The filled config and request use the genuine count/marker-preparation metadata. All eight preparation fields are populated. GPU/training fields remain null. No root approval, admission, execution or measured RSS is claimed.

- config_prepare.proposed.inactive.json
- admission_request.prepare.proposed.inactive.json
- plan_request.prepare.proposed.json
- cap_rationale_v1.json: per-parent inputs, formulas and named policy allowances
- readonly_plan_check_v1.json: actual public metadata planning CLI returned0

| Proposed field | Bytes | Basis |
|---|---:|---|
| CPU_bank_cap_bytes |232,783,872|Largest complete bank tensor upper232,374,932, rounded to1MiB|
| cut_preparation_cap_bytes |817,889,280|Largest named cut estimate817,610,496, rounded to1MiB|
| cut_recipe_archive_cap_bytes |6,291,456|Largest lossless NPZ upper6,068,175, rounded to1MiB|
| max_cpu_rss_bytes |4,026,531,840|3.75GiB fail-stop policy above3,890,941,586 named/policy components|
| log_cap_bytes |262,144|256KiB retained raw diagnostic tail for40 progress rows, startup and errors|
| fixed_metadata_cap_bytes |524,288|452,378-byte count/envelope policy subtotal, rounded to128KiB|
| qualification_allowance_bytes |8,192|Smallest fresh-output writer allowance; CPU prepare invokes no GPU qualification|
| output_cap_bytes |242,221,056|241,945,331-byte full forecast, rounded to1MiB|

The inactive request reserves242,221,056 bytes. This includes239,766,259 bytes for all40 compact recipe upper bounds,524,288 for fixed metadata, and1,654,784 for worst-case JSON-escaped raw-log/terminal planning. No dense int64 cut tables are saved.

## RSS is a stop limit, not a forecast guarantee

The bank formula is100*N+8*T*952-20*T+g*S*952+164*S+8*P+8*686+12*952. Genuine genotype width is2 bytes; the validated allele-channel bound is4. Node/sweep totals use the complete tree invariant: one root per hold andN-T nonroots.

The RSS policy includes1GiB for Python/Torch/PyG/runtime imports; two rounded bank/decode buffers; the largest44,764,818-byte compressed input; two named cut workspaces for preparation and lossless round-trip;128MiB for decoded JSON metadata;256MiB memo2 transaction capacity; and256MiB allocator-retention policy allowance. These allowances are explicit proposals, not observed usage or proved allocation bounds.

The frozen worker checks current RSS and process peak ru_maxrss at CPU guard boundaries. Before cut construction it checks current RSS plus the named estimate. The root supervisor polls owned-process-tree RSS about every0.25seconds and performs real wait/drain. This is fail-stop enforcement, not an OS allocation clamp or a guarantee that every instantaneous peak stays below the limit. A breach must preserve failure/terminal evidence; this proposal authorizes no cap increase or automatic retry.

## Static runtime bottleneck findings

No arrays were decoded, no timing pilot ran and no ETA is asserted.

1. The Python tree work spans72,881,094 native nodes across40 parents. Preparation builds adjacency and visits complete native trees using Python loops. It also performs per-hold sample/group dictionary work over38,298x952=36,459,696 rows.
2. Pair layouts are built in both directions during preparation and again during required round-trip reconstruction. WithG<=T this permits up to145,838,784 leaf/pair iterations across40 parents; actual distinct episodes remain unknown until preparation. Triple-dependence metadata adds dictionary work over distinct episode triples.
3. Input NPZ decoding reads1,672,000,826 compressed bytes across40 parents, with168–216 model members per parent. The loader decompresses/copies fields and performs vectorized full-bank structural checks. The reader retains byte/digest evidence, not every NPZ payload.
4. Fingerprinting scans native observation tensors during preparation and again on decode. Compact output compression and its required decode add work, but output storage remains bounded near240MB.
5. The main path saves80 per-parent files plus started/completed metadata:82 guarded writer boundaries, with additional setup/supervisor boundaries. Memo2 avoids repeated bytes within each transaction; it deliberately clears between operations. Boundary verification can therefore be material even when file sizes are small.

These findings support watching the actual first authorized preparation's progress and terminal resource evidence. No frozen reader, codec, model, guard, config or execution source was changed.

## Exact root review/activation path

The existing source-only planning call is runnable now:

~~~text
.runtime/env/bin/python -B -m research.reviews.H049_order2_cut_memory_root_source_20261015_v1.root_driver plan --record research/reviews/H049_order2_cut_memory_root_source_20261015_v1/root_prepare_cap_proposal_v1/plan_request.prepare.proposed.json --sha256 PLAN_REQUEST_SHA
~~~

The metadata CLI has already returned0 and reports resource_caps_reviewed=false. For actual execution root must make a separate active config copy, explicitly set execution_enabled and root_resource_caps_reviewed true after review, and create a separate active request with its exact config ref, status=active and root_authorizer=root_agent. The reserved-byte proposal and phase remain prepare-only. The frozen root_driver admit/launch --execute path then applies. This packet itself enables neither action.
