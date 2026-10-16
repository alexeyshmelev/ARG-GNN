# Pure H056 chr20 native-v2 binding constructor

construct_native_chr20_v2_v1.py is an inert, standard-library-only metadata helper. It performs no filesystem or network IO, imports no project/scientific module, launches nothing and implements no guard or admission. Root supplies real metadata bytes and remains responsible for every write, admission, activation and actual execution.

The constructor accepts exactly six JSON metadata records under bundle keys completed,readback,actual_wait,counts,map_qualification,roster. Each record is exactly {"ref":{"path":"relative.json","sha256":"..."},"raw":actual_bytes}. It authenticates the supplied bytes and rejects duplicate keys/nonfinite JSON. No VCF,QC,haps,tree or checkpoint payload is accepted or read.

The completed acquisition/readback/rootwait/count/map/roster joins are checked against the actual native-v2 reader schema. Worker-only readback and oldwait1 cannot substitute for the genuine new root-sealed readback and root externalwait0/drain. All seven rich data artifacts must come from one immutable bridge folder and agree with serialized-readback hashes. The map must match the original GLIMPSE chr20 commit/path/blob/build/units/boundaries. The actual36-row roster must match the frozen canonical roster bytes and H36/R16/Q20 subject/class/date semantics.

Function:

```python
packet = construct_native_chr20_v2_v1(
    bundle,
    actual_runtime_identity,
    original_bindings,
    max_chunks=root_finite_J,
    created_at_utc=current_root_issuance,
    expires_at_utc=current_root_absolute_expiry,
)
```

actual_runtime_identity has exactly python_binary_ref,python_version,native_binary_refs,license_ref and must equal the pinned original installation identity. It describes the actual target native interpreter/runtime. The real native caller rechecks the executable/version/binaries/packages at launch. original_bindings is root's complete actual original guard source/state map, not a subset.

The unchanged derive_caps_v1 and ceil_mib functions are copied AST-identically from native_h056_v2.py. Their input is the authenticated actual acquisition C/S/K plus root's finiteJ. They remain abort ceilings, not measured usage forecasts. The constructor never imports or invokes the native source.

The fixed future namespaces are:

- Job H056-native-chr20-recovered-pre-size-v2
- Key H056/native-chr20-recovered-pre-size-v2
- Output experiments/results/H056_chr20_H36_Relate_pre_size_CPU_v2
- Aux experiments/results/H056_chr20_H36_Relate_pre_size_CPU_v2_aux
- Packet exactly output/runtime
- Root metadata research/reviews/H056_chr20_native_actual_root_20261016_v2

The supplied bridge folder is derived from actual completed artifacts. All intended scopes must be disjoint. This is pure path validation, not a claim that the actual paths are canonical/nonexistent. Root and the unchanged native caller must check actual freshness/canonical identity before launch. This helper is specific to chr20 and the reviewed native-v2 source family.

Returned packet:

- documents: exact config.json,runtime.json,source_manifest.json,input.inactive.json dictionaries
- encoded_documents: their exact canonical bytes
- document_refs: canonical path/SHA refs for those bytes
- consumer_bindings: the complete exact original+sixsources+sevenmetadata+fixedPY/SH/JSON+config/runtime/manifest union, at most64
- reservation_expectations: exact key/output/aux/cap/consumer bindings for root's authentic admission
- storage_record_expectations: exact job/output/expiry/cap/policy fields
- native_argv_template: exact native-v2 CLI with future active input path
- root_pending: explicit remaining root actions

Root must write the returned config/runtime/source_manifest bytes exactly through its existing metadata writer. Do not reformat them after hashing. input.inactive.json deliberately has statusinactive,execution_authorizedfalse,storage_refnull. It is not an execution input or part of the directconsumer union.

After genuine admission, root creates a separate fresh input.json from the returned inactive input dictionary, fills its actual storage_ref, sets statusactive/execution_authorizedtrue, computes that actual active inputSHA and invokes native-v2. There is no activate/reserve/write/launch function here. The storage/input hash cycle is avoided because storage is not included in its own consumer union and the active input is separately pinned by the real outer root command.

The actual native caller still verifies all original bindings, source metadata, map reparse, same H36/R16/Q20 pre-size N30000/mu1.25e-8/28years/seed414855340/memory2GB/no size iterations, exact original CLI and child waits/drains, seven-count topology census, strictGT and independent graph-only facts, final postwrite guard and real root outerwait0. This helper does not replace any of them or establish runtime fit.

No constructor invocation, fixture, payload access, model/native execution, guard/admission or Git action was performed to author this source. Its static checks parse/compile source only and compare copied formulas without executing them. Independent exact-source review is required.
