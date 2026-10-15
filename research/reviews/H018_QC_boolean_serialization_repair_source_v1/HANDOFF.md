# H018 exact QC Boolean and counter repair source package

Status: source-only, synthetic checks passed, inactive and not admitted.

The immutable root erratum is research/reviews/H018_QC_count_erratum_root_20261015_v1/actual_QC_site_count_erratum_v2.json (SHA256 47db1404927848d2dfa836f5760ded106fe50f868e699616ff339ccc493b1f78). It records the stable, SHA-verified original TSV masks: 2,074 passing rows among 94,892 extension candidates and 217 among 13,312 original 2 Mb candidates. The unchanged threshold is 1,000. The extension still passes and the original 2 Mb gate still fails. No row is added, rescued, dropped or intersected.

The pinned upstream writer emits exact 0/1 strings at line 438. Its lines 429 and 431 add each passing Boolean twice to the same aggregate counter. Reported extension 4,148 and original 434 counts remain historical incorrect reports. Original QC, VCF, acquisition reports, counters, bridge/wrapper/verifier sources and all failed scope evidence remain unchanged. This package does not repair those historical artifacts.

## Fresh root-owned scope

- Driver: root_bridge_driver.py.
- Config: config.json.
- Manifest: source_manifest.json.
- Request: root_request.inactive.json, with null root authorizer and registration.
- Job: H018-observed-baseline-GT-preparation-v2.
- Key: H018/observed_baseline_GT_preparation_v2.
- New output: data/H018_R16_observed_bridge_v2.
- New packet: research/reviews/H018_QC_boolean_serialization_repair_source_v1/admission_v1.
- New auxiliary scope: research/reviews/H018_QC_boolean_serialization_repair_source_v1/worker_aux_v2.

No packet, output or auxiliary directory was created. Root owns registration, actual guard verification, admission and execution. Preserve the original failed preparation's full 50,000,000-byte charge. The new scope also reserves 50,000,000 bytes. No capacity credit or release is supplied.

## Production code changes

bridge_core_v2.py accepts only original 0/1 spellings and fixes the immutable retained count to 2,074. It binds the root erratum and upstream writer in constants and its report. Polarization, every haplotype nucleotide and phase, source candidate order, coordinates, fractional .17g ages, and all monomorphic rows remain unchanged.

The driver changes only fresh identities, the bridge pin, correct counts, root erratum validation and explicit history/provenance fields. The unchanged v3 supervision, direct wait/drain, admission, suffix partition, guards, runtime identity, ROOT-plus-src PYTHONPATH and budgets are retained. The three changed driver functions reverse exactly to the prior AST after removing enumerated repair additions.

readback_verifier_v2.py provides the fixed public entry verify_h018_serialized_v2(basis_bytes, actual_output_bytes). It still accepts no contract or count override. Root must supply the original nine byte-basis entries plus the additional tenth entry qc_count_erratum, containing the exact root erratum bytes. ORIGINAL_BASIS_REFS lists all ten exact refs. All nine existing basis refs remain unchanged. The verifier checks actual output bytes against the fixed 94,892/2,074 contract and new v2 admission/output identities.

Core report, completed artifact, provenance and independent verifier receipt bind the erratum and upstream writer, preserve historically reported 4,148/434, and preserve the old 50 MB charge. Original 2 Mb provenance records actual 217 and reported 434 separately. Native execution remains separately gated, including retained actual-40 monomorphic sites and unresolved fractional native age serialization. No HS, GPU, model, ancestry truth, native chronology or inference qualification is supplied.

The manifest has 16 source refs and nine unchanged metadata refs, with 21 eligible direct refs. The same four non-direct refs remain fully bound transitively. bridge_source_manifest remains the unchanged original historical source evidence. The new producer authority is the fresh config/driver/core/verifier plus the exact root erratum. Actual dynamic guard binding count and acceptance have not been tested.

## Checks completed

- source_checks.py / source_checks.json: 27 passing source/hash/AST/budget checks, no driver/guard/producer import.
- serialization_fixtures.py / synthetic_checks.json: 26 passing synthetic writer/parser/binding checks.
- synthetic_readback_checks.py, included in that receipt: 24 passing independent roundtrip/corruption checks, with and without retained monomorphic rows.
- Three exact diffs document original-to-successor changes.

Fixtures execute only tiny in-memory bytes and exact extracted serialization/counter/parser AST nodes. They demonstrate the writer's exact 0/1 output, the old parser failure, correct new decisions/order, rejection of alternate spellings, double accumulation, count/erratum/writer/history/charge corruption rejection and the public fixed-contract rejection of a tiny synthetic basis. Tiny fixture constants exist only in isolated test namespaces, never as a production override.

No original genotype/QC/tree/target/model payload was read during this source task. No failed packet or live-state data was read, no worker/guard/admission was executed, and no project files were written locally. All package files are on anogena inside the owned repository folder. Generic Relate support work remains paused.
