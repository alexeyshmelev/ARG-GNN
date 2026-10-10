# Separate handmade qualification with one rescaling interval v2

This harness is source-only by default. Root requested it after the sealed v3 failure diagnosis. It imports the exact frozen v3 producer as standard-library definitions and invokes producer.pipeline directly with a deep copy of its config. The only scientific difference is explicitly pinned qualification_dating.rescaling_intervals=1, instead of biological1000. The fixture is the exact frozen six-site/eight-haplotype handmade table. No global date binding, producer helper or native module is monkeypatched. The frozen producer/config/docs/tests are unchanged.

Source-only and native modes are the only entry points. There is no infer mode or observation asset loader call. The separate manifest kind, condition, reduced-certification wording, config and harness hashes prevent this result from being accepted as the original1000-interval qualification. A passed native run would certify only the single-interval fixture. There has been no native run of this harness.

The harness validates the original biological config against the frozen producer's exact1000 contract, then validates its own exact qualification copy. It preserves every other native inference, dating, genotype, date, site, strict-time, identity and exact-support check by reusing the frozen pipeline and qualification_summary.

Storage uses the same immutable runtime receipt and exact existing8GB reservation. A fresh root-issued storage receipt must bind this harness/config, the previous failed scope pins, and all four write scopes: preserved v3 tree/log and fresh v2 tree/log. A StorageAccounting subclass delegates the original atomic shared-ledger/all-reservations/external-reserve/free-space checks and adds the preserved v3 local footprint to the new v2 footprint before enforcing the same combined1GB local allowance and8GB installation-plus-local reservation. It never mutates the live ledger or base storage/date bindings. The complete old tiny scope is hash-verified at admission and successful completion. All new TMP/TEMP/JIT/XDG writes live under the fresh v2 output. Local caps remain stage-boundary sampled footprint guards, not OS quotas or continuous monitors.

Native admission requires an exact separate root-admission kind, authorizer=root_agent, source/config/admission SHA256 opt-in, empty CUDA_VISIBLE_DEVICES, the same isolated Python/runtime versions and complete runtime inventory, plus unchanged previous artifacts. Runtime inventory is checked once at admission and once at successful completion. Root must register the numerical fixture correction, obtain independent source review, fill the inactive admission/storage templates, renew the live initial ledger SHA, and authorize any native run. No prior v3 admission can authorize v2. Both native failure and storage failures preserve partial manifests through the reused stage journal.

Safe source-only command:
~~~sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" research/reviews/biological_paths_tsinfer_native_failure_v1/qualify_handmade_rescaling_v2.py
~~~

Root-only native invocation would use the isolated interpreter and --mode qualify-cpu, --expected-config-sha256, --expected-source-sha256, --admission, --expected-admission-sha256. The templates are inert, no completed CPU authorization has been emitted by the author.

qualification_v2_static_checks.json and qualification_v2_author_receipt.json record source-only/fake-context verification, exact pins and remaining limits. The sealed diagnosis_receipt.json and its document remain unchanged. All failed v3 outputs remain pinned. Actual biological observations, native inference/date reruns, package installs, live ledger edits, Git and GPU work are outside author execution.
