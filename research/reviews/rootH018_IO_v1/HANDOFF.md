# Root H018 actual980GB CPU IO and launch

root_io_v1.py is a minimal root-only admission, injected IO context and outer CPU launcher for the finalized H018/H045 inference_v2. It does not modify any science, model, checkpoint, bank, FLARE context/build/stage, guard or policy. Import is inert. All actual admission and execution remain root-owned.

Pinned actual guard: research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py SHA fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56. Actual active policy: activation_policy.root_active_v1.json SHA eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018. Root configure_v2 uses its exact repair authority. Admission uses original unmemoized v4 mutex/state/reserve and directly binds actual v4, marker, repair/helper, capacity ancestors, current guard/policy, runtime, producer, IO/launcher and all actual science/metadata sources. No memo, old admission route, guard monkeypatch or policy rewrite is installed.

The unchanged scientific producer is research/reviews/H018_H045_empirical_application_source_v1/inference_v2.py SHA a8223979ee775d0aae83f576d48d3cbec482eb03022397acf1d684a08d50f227. Config is config.active_template_v2.json SHA97ab5f019a672147acf15a84e77bf4dd8cb690300991d88313cb105caee7d8a0; science manifest SHA0dc5a040f018afe43b1e34b5ec6c985d2c07994398c6fdfd010b4440497aad65.

From /workspace-SR003.nfs2/ARG-GNN, use explicit fresh paths/key and future expiry:

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.rootH018_IO_v1.root_io_v1 --admit --packet [fresh-root-packet] --output [fresh-prediction-output] --job-id [fresh-job-key] --expires-at-utc [future-UTC-expiry]

The output is status admitted20MB_not_launched plus exact input_ref path/SHA and actual consumer binding count. Then:

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.rootH018_IO_v1.root_io_v1 --launch --input [reported-input-path] --expected-sha256 [reported-input-SHA]

Direct file invocation also works because the module adds ROOT and ROOT/src to sys.path. The default invocation without a mode exits through argparse; it never admits or executes implicitly. Never replay into existing output/worker_aux scopes. A concurrent new journal event can invalidate the exact prospective prefix; preserve failed packet metadata and have root choose a fresh packet before explicit retry.

Admission creates runtime.json, IO_source_manifest.json, reserve_authority.json, storage.json, input.json and reserve_receipt.json in the fresh root packet. The exact20,000,000B reservation owns the fresh prediction output plus packet/worker_aux. Precharge for parent seed/wait/log metadata is524288B; retained outer stdout is capped262144B. The complete actual closure reconstructed with the root's existing980GB admission has37 source files and55 direct refs, below64. Source enforces64 before reserve and again before launch. Direct bindings use only .py/.sh/.json. Exact selected checkpoint bytes remain descriptor/config pinned and are read only by the admitted worker.

The worker API supplies every inference_v2 context field: actual input/source/config/runtime/cfg/consumer_bindings/io_context_ref/resource_binding_refs, read-only Reader, writer(name,bytes) and before_stage(stage,plan). Reader only accepts exact admitted source/metadata refs plus the two exact existing bank/selected-checkpoint payloads, verifies project-local regular singly linked identity, byte limits and SHA, and rejects arbitrary payload capabilities. Bank cap32MiB and selected checkpoint cap1MiB are read allowances, not new copies or storage reservations.

The raw writer creates a fresh output directory through an actual guard boundary, restricts names to basenames, checks planned bytes with the real980GB guard, writes exclusively with O_EXCL/O_NOFOLLOW, fsyncs and rereads the actual path/SHA. before_stage verifies real worker peak RSS/deadline/expiry, actual charged reservation and planned growth through the same unchanged guard. Final science outputs plus a compact root_IO_worker_receipt.json stay within the20MB charge.

The outer root Popen is CPU-only, CUDA_VISIBLE_DEVICES empty, threads4. It creates an EMPTY inherited handshake pipe. Child blocks before guard configuration, tensor imports, payload reads or scientific compute. Parent first proves the child's real PID/start ticks/PGID/session, seeds owned process/group identities, measures aggregate child plus monitor RSS and fsyncs worker_aux/launch_monitor_seed.json with child_compute_released=false. Only then does parent write the65-byte release ticket. This fixes an empty owned-roster observer or prefilled-pipe race before child compute.

Root monitor polls observed owned descendants across groups every.25s, with2,000,000,000B aggregate RSS and1800s deadline,30s termination grace. This is a polling soft monitoring bound, not a kernel quota; between-poll peaks/processes and exit/task races are explicitly reported. Only proven owned identities/groups are signalled; unrelated live jobs are never scanned or changed. Parent performs no scope guard scan while child writes.

After the real proc.wait, the FIRST postwait write is exclusive fsynced worker_aux/actual_wait_minimal.json with actual_direct_wait=true and actual returncode. Group-drain checks, final guards, stdout and outer_wait.json follow. Thus real wait evidence survives later risky guard/log failure. Unresolved observed groups block final guard/log writes. Popen failure records actual_direct_wait=false; no wait is inferred. No automatic completion credit or cleanup/deletion is issued.

30 source/static/synthetic reader-capability checks passed; source_static_receipt.json records hashes and the exact existing980GB binding roster used for the55-ref preview. All21 science sources and the reused proven monitor helper remain unchanged. No guard configuration/admission, launcher Popen, actual model/bank/checkpoint/GPU/native or Git execution occurred during this source task. Root can now admit and run actual descriptive empirical predictions; query ancestry truth and accuracy remain unavailable.
