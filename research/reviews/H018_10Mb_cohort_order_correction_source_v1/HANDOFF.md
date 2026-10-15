# H018 frozen cohort/order failure and prospective correction

The cohort is correct. The exported subject TXT and TSV contain the same 20 unique ordinary subjects as the endpoint, with 8 references and 12 queries (R16/Q24). All roles/proxy memberships match. No IDs are missing, extra, duplicated or suffixed _rnd.

The failure is dictionary ordering. The frozen original endpoint and explicit TSV subject_order0 use Ana_proxy, CHG_proxy, WHG_proxy, EHG_proxy. The extension endpoint stores its reference map as Ana_proxy, CHG_proxy, EHG_proxy, WHG_proxy. Existing worker line 309 flattens the extension map's values and line 310 compares this to the unchanged exported roster. Thus positions 4–7 differ:

|0-based row|Actual TXT/TSV|Old worker expected|
|---|---|---|
|4|Loschbour|Sidelkino|
|5|Bichon|NEO88|
|6|Sidelkino|Loschbour|
|7|NEO88|Bichon|

Rows 0–3 and 8–19 are identical. The 20-subject membership is unchanged; the four middle references were only ordered differently by the map serialization.

The additive prepare_h018_r16_10mb_extension_v2.py derives the expected reference ordering from the already SHA-pinned ORIGINAL_REFS.original_endpoint. Exact subjects==expected, 20 unique and no_rnd checks remain intact. The exported roster and TSV are not reordered. The later exact metadata/header join remains unchanged. Only SOURCE identity and this expected-list construction differ from v1; prospective_worker_delta.patch records the entire delta.

Metadata qualification confirms the corrected expected list equals both frozen TXT order and TSV subject_order0, with exact 8/12 roles and proxy groups. All other worker functions, biological choices, endpoint, region, QC, AP rules, limits, native tooling and age joins are byte-identical.

The failed v1 worker/admission/log/wait remain unchanged and fully charged. Source places the failure before Output construction and RangeSource/network acquisition. No retry was launched. Root must prospectively register the correction, create a fresh operational packet/output/admission bound to the new worker SHA, and bind the launched failed attempt's real wait1/terminal/revocation evidence. A never-launched certificate is insufficient for this now-failed attempt. The unchanged config/runtime and frozen biological metadata refs remain usable.

No live guard/ledger/process operation, acquisition, genotype payload read, install or Git action was performed.
