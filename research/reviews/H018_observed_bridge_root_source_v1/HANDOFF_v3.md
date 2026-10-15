# Inactive H018 bridge v3 direct-binding correction

Fresh source only. Original v1/v2 drivers, manifests, configs, requests and the failed admission packet are preserved. Root owns the failed-admission evidence and actual accounting state. This task did not read that packet or state, reserve, admit, launch or run a worker.

The unchanged scope_spec accepts only .py/.sh/.json direct consumer bindings. V3 filters the candidates to exactly those suffixes, while keeping every original full source/metadata ref in the SHA-bound manifest and consumer JSON. The excluded four refs are conda-meta/history, planned.sample, planned.sample_ages.generations and qualified.regional.map. Their exact original refs remain in source_refs/metadata_refs and are additionally listed in transitive_only_refs. No asset ref, guard check or original runtime hash verification was removed.

The new direct_authority_refs helper checks the exact original suffix list, the complete excluded partition, and equality of the full actual consumer-closure source/metadata maps. Admission reads back that saved closure before using the helper. All original full SHA loops in verify_request remain AST-identical apart from adding the classification check. Worker admission validation checks the same full closure and requires its actual runtime/source refs, guard/policy/repair refs and stored budget components directly. The fresh root state view bindings are all mandatory direct authorities: filtering any such authority causes an error. The original64-direct-ref guard remains. Nineteen source/metadata closure refs are eligible directly before the state-dependent runtime/root/guard additions and deduplication. Actual final guard acceptance/count has not been checked here.

Fresh files:

- root_bridge_driver_v3.py uses SOURCE v3, MANIFEST source_manifest_v3.json, CONFIG config_v3.json and PACKET admission_v2.
- config_v3.json differs from the original config only in packet_directory=admission_v2. Every budget, source/runtime/guard/policy/repair pin, output and auxiliary directory is unchanged.
- source_manifest_v3.json retains all original source/metadata mappings except the fresh driver/config refs, adds the direct/transitive classification, and binds independent_serialization_verifier at research/reviews/H018_bridge_serialization_readback_source_v1/serialization_readback.py, exact SHA eb32b6ad22b81276d88437ffe3c0bcef36e0ccb773788fe8fc19a4b11e12b6ff.
- root_request.inactive_v3.json updates only driver/config/manifest refs and packet_directory. It remains inactive, with root authority/registration absent.
- driver_v2_to_v3.diff contains the exact source changes. source_checks_v3.py is rerunnable and source_checks_v3.json records22 passing source/metadata checks.

The worker, bounds, scientific preparation, exact ten-file data output, payload rules, process limits, direct waits, draining, failure retention and supervision functions are AST-identical to v2. The ROOT plus ROOT/src worker PYTHONPATH correction is retained in both locations. The new verifier is source-bound only and is not invoked by the worker. Root can separately perform actual readback after genuine worker wait/drain, with its compact guarded receipt in worker_aux under the original50MB scope. No readback or auxiliary write was done here.

This successor reuses the original output/auxiliary names because root reports they were never created. It uses the new admission_v2 packet because admission_v1 already contains failed preparation metadata. This task does not establish fresh-path availability or any ledger eligibility; the unchanged driver/guard must establish those during a later root-authorized attempt. The original metadata-controller5MB scope still has unknown OS exit and remains fully charged, with no replay or credit authorization. V3 introduces no close/credit path.

Verification executed only standard-library source/JSON analysis. All14 source refs were rehashed, including the requested verifier. The9 metadata refs were compared exactly to the old manifest without opening their target files. Full original global budget/guard/output/aux constants and every worker/science/wait/supervision function were compared by AST. The full original request verification and SHA-validation AST was preserved after removing only the new classification call. No driver, common, guard, accounting or scientific module was imported during this v3 work. No payload, failed packet, process, GPU or live accounting state was opened.

Repeatable source checks, without a receipt overwrite:

    /usr/bin/python3 -B research/reviews/H018_observed_bridge_root_source_v1/source_checks_v3.py

Root command shapes, after separately producing a genuine new active request and confirming actual readiness:

    .runtime/env/bin/python3.11 -B -m research.reviews.H018_observed_bridge_root_source_v1.root_bridge_driver_v3 admit --execute --record ROOT_NEW_ACTIVE_REQUEST --sha256 ROOT_ACTIVE_REQUEST_SHA256
    .runtime/env/bin/python3.11 -B -m research.reviews.H018_observed_bridge_root_source_v1.root_bridge_driver_v3 launch --execute --record research/reviews/H018_observed_bridge_root_source_v1/admission_v2/admitted.json --sha256 ACTUAL_NEW_ADMISSION_SHA256

These are unexecuted command shapes. Do not activate the inactive preview directly or reuse failed admission_v1 metadata. No admission or runtime success is claimed by the static checks.

Final refs:

- Driver b36af1a0bacf85bd46acb73a0350a923c1de0974e1b7f9eb21efd72736c2991b
- Config960205388a47a9a9a9ee1e809baf5747d59dd1b74013bd367fb2f62e89e841bb
- Manifest513c5dd292740e519c269b1979dd5be3713fad9751ebcbba7f021733606ff6d4
- Inactive request f02f00455ff76b4d9f3f82ec6ce222e9026084ff730630fb8b1d80bb17e32d74
- Exact diff38de495d533e6fb9b903918b93372ca3a98707138ab844eccdab4e88aaba54c9
