# Independent source review: REJECT

Exact source: e9d681be39f3038e6c5258e22554fa65bc3554d0d036260f3e6cea6961b4a449.
Exact config: 656650f5bb44465fe8964fa8803e53004416973e82c22fe97d1c366eb74855cd.
Exact author receipt: b10be6e3948bd3b4110e309b7551537070958a96cb6d4e540bf5c387b3bc33e2.

The intended v2 repairs are substantially verified by 134 fresh source and
bounded handmade checks. Empty/malformed runtime inventories, incomplete
distribution/module/import-byte manifests, wrong scientific distributions,
actual child module-version/origin changes and unrecorded executable bytecode
are rejected. Actual-runtime comparisons were exercised only against a tiny
contained fake prefix and handmade import objects. No genuine native inventory
was captured.

Helper source/version/wheel/directory authority is exact. Extra executable
imports are rejected even when the handmade directory manifest includes their
bytes, including nested pyc, pyo, extensions, pth and dll. Direct exact-source
helper loading preserves sys.path and uses the discovered source version.
Wrong helper version fails. The real root-prepared threadpoolctl3.7.0 artifact
references were byte-checked without importing the actual helper. No generic
helper path was injected.

The real execute() function now preserves failed receipts and exit_status1 for
owned root-admission, malformed child exec-admission, runtime-preflight and
stage exceptions. Existing terminal evidence is immutable. A malformed/large
terminal context is reduced below the64KiB failure bound.

One remaining P2 blocker prevents PASS:

**Owned child configuration-policy failures still bypass terminal evidence at
the CLI entrypoint.** At source line884, main() calls policy() before the call
to execute() at line886. A configuration byte mismatch, malformed JSON or
configuration read failure therefore occurs before the owned-output handler.
The fresh counterexample exercises real main(argv --execute --native) and real
policy(): only the configuration path and its expected digest constants are
relocated into the review fixture. The handmade configuration initially passes
policy. After bootstrap ownership is represented by a fresh stage directory
and exec_admission.json, one newline changes the handmade config bytes.
main() raises the exact policy error, and neither receipt.json nor exit_status
is created. Original source/config are untouched. The earlier injected-policy
counterexample is also retained, but the final handmade report records the
stronger real-policy byte-drift case.

This is the requested owned-child preflight evidence contract, not a new
scientific or runtime gate. A successor must extend terminal failure coverage
across the entrypoint policy boundary while continuing to prove ownership and
preserve existing/foreign terminal evidence. The sealed v3 bundle must remain
unchanged.

Scientific source conclusions are supported. All30 native scientific ASTs
equal the sealed v2 receipt. The five fitting/retrieval wrapper ASTs also remain
unchanged. The same native recipe, source commit, source tokenizer/full-path
dedup, preparation bindings, vocabulary namespaces, model initialization,
optimizer/scheduler/loss and retrieval rules are retained. MLM is5 epochs,
logical512/micro64, seed17. Contrastive is1 epoch, true full4096, seed36.
No gradient-cache fallback or query-target admission is granted. Primary k1
and sensitivity k5 use all pure-reference train-plus-val full paths. The
auxiliary local mean readout remains separate from primary native retrieval.

A fresh small fixture executes the original native seal/kernel definitions
on280 handmade full paths, all280 nonconsecutive observed leaf IDs, all96
pure-reference paths, all120 query paths,768 dimensions and11,520 exhaustive
pairs. This fixture contains one path per observed leaf and one retained
interval. It verifies full-reference IDs, duplicate embedding rows, one zero
row, cross-class exact ties, original-ID tie order, chunk merging, raw k1/k5
votes, all common query support, and missing/cross-ARG/query-label rejection.
It has no native-fit success wiring: its fit-reference metadata explicitly
states actual_native_fit=false. It is not a real fit, full4096 qualification,
or biological prediction evidence.

Four fresh20GB parent scopes, short TMP namespaces, exact unchanged budgetv4
source/config and separate inactive probe/fit/retrieval templates remain
consistent. Actual root capture of the complete native inventory and actual
same-device full4096 CUDA optimizer-memory qualification are pending. Source
review does not activate the scientific recipe, reserve storage, schedule
devices or supply runtime authority.

Review limits: no actual prepared scientific arrays, native model, true
ancestry, actual native retrieval, genuine native package capture, GPU
inventory/context, live ledger, live device lock, installation, native-prefix
mutation, Git operation or authority activation occurred. Only contained
handmade arrays and fake guard resources were used. No4096 scaling fixture was
repeated. Native source/config, author receipts and preserved v2 REJECT remain
exact.
