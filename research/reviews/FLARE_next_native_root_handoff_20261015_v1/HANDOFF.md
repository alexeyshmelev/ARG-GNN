# Next native FLARE CPU run: exact root handoff

Source-ready for root’s existing native admission and launch after its pending
conversion close returns enough ledger headroom. No wrapper, new science,
rebuild, package installation, GPU work or additional audit is needed. This
packet is source/preparation only and is not an admission.

## Actual predecessors verified

- TRAIN000 conversion is genuinely completed: `data/FLARE_train000_convert_980_root_v1/completed.json`, SHA256 `1994ab932cf2f533577bfd09eddf5b083330b5a32f16a5cf8ee73e07dbd4e6b6`.
- Its worker direct wait0 is bound by SHA256 `3c29af3e968cdc2342e19f83d0cfab41282ed7375cad77332d7e1b920c927d54`. Its outer wait0/no-error/group-ESRCH receipt is bound by SHA256 `5bf54cb4047f7e52f325563b80c81425d99b1ce3e63e54f55c0adbcd14e25fa8`.
- The unchanged driver’s `target_v1` and native `config_v1` both accepted actual target, conversion, completed installed-JDK build, binary provenance and original actual build/tool waits. No JAR, executable, VCF, marker array, tree, focal payload or truth was read.
- All 27 source-closure pins in the existing handoff match. The proposed native packet/output scopes were absent when inspected.

The conversion preserves all 226185 supported input sites, with 26320 expected
native retained markers under the original author filters, 600 reference and
1000 query haplotypes, and the same 10000 focal records. This is
`TRAIN_full30Mb_inference_canary_development_not_heldout`. Scoring is pending.
No ancestry-accuracy result is asserted.

## Exact commands for root

Run on anogena in `/workspace-SR003.nfs2/ARG-GNN`. Wait for root’s already active
accounting call to return before issuing another accounting operation. The
existing admission performs its own fresh prospective980 state, source/runtime,
closure and capacity checks. Do not manually draft or edit guard records.

```bash
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver \
  --admit --stage native \
  --target research/reviews/FLARE_scientific_root_stage_source_v1/train000_target.json \
  --expected-target-sha256 a944251c7e5b0d7cc311192b2e953aac4f2d193235f4e8834c5f7745406e954b \
  --conversion data/FLARE_train000_convert_980_root_v1/completed.json \
  --expected-conversion-sha256 1994ab932cf2f533577bfd09eddf5b083330b5a32f16a5cf8ee73e07dbd4e6b6 \
  --packet research/reviews/FLARE_train000_native_980_root_v1 \
  --output data/FLARE_train000_native_980_root_v1 \
  --job-id FLARE/TRAIN000-native-980-root-v1 \
  --expires-at-utc 2026-10-17T18:00:00Z
```

After that command actually returns `admitted_native_not_launched`, launch with
the returned `admission_ref.sha256`. The following equivalent command hashes
only the exact newly created admission file:

```bash
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver \
  --launch \
  --admission research/reviews/FLARE_train000_native_980_root_v1/admission.json \
  --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_train000_native_980_root_v1/admission.json | cut -d ' ' -f 1)"
```

Do not invoke launch after a failed/partial admission. If either proposed scope
has since been created, choose a fresh matching suffix for packet/output/job
before admission. Preserve the actual conversion reference unchanged. Do not
rerun conversion or rebuild FLARE.

## Preserved native command and limits

The existing producer constructs Java’s native command and byte-verifies the
JAR/inputs at execution. No direct Java bypass is supplied. Original source
commit is `ec0ad295aeebfacf10d8cbfb6486ab78981de1b1`, reported version
`flare version 0.6.0 [04ad73fd0 31-Oct-2025]`, heap 16 GiB, four CPU threads,
14400-second native and outer deadlines, 24-GiB aggregate RSS polling soft
bound, and 9000000000-byte reservation including the existing auxiliary allowance.
The original max Java temporary space is 1 GiB and native VCF ceiling is
7373083377 bytes. Retain all original JVM placement/home/tmp flags.

Author options remain exactly:

```text
probs=true em=true update-p=false array=false min-maf=0.005 min-mac=50
 gen=10 panel-probs=false nthreads=4 seed=-99999
```

All four original output products remain: `.anc.vcf.gz`, `.model`,
`.global.anc.gz`, `.log`. No model or query-ancestry truth is supplied to FLARE.

## Closure status and completion boundary

See [close snapshot](close_snapshot.json) and [subsequent root status](root_status_update.json).
Root subsequently admitted the 16-GB VAL native-path stage and reports a ledger
projection of 971525618721 bytes. This leaves 8474381279 bytes below 980 GB,
525618721 bytes short of the 9000000000-byte native FLARE reservation. Root’s
existing conversion-close call (session86706) is still running. **Wait for that
call to return successful sufficient headroom, then run the admission above.**
No new audit or wrapper is needed. These are projected ledger charges, not
measured physical disk usage. Actual prospective admission remains the live
capacity decision, and the 950-GB warning remains active.

Preserve native `native_actual_wait.json`, actual `completed.json`, outer
`worker_aux/actual_wait_minimal.json`, `worker_aux/outer_wait.json`, the launch
seed and all admission/storage/reserve records. The native direct wait0,
original outputs, worker wait0 and outer no-error/group-ESRCH must genuinely
exist before root requests sealing or completion accounting. Do not manufacture
terminal receipts or repeat a consumed launch. Native sealing/scoring is a
later stage and no heldout/scoring authority is created here.

**Source/predecessor blockers: none. Capacity blocker: the currently reported
ledger headroom is 525618721 bytes short until the existing conversion close
returns sufficient credit.** Capacity/binding count and exact runtime are
checked by the existing root admission. A rejection there is an actual root-stage failure to
inspect, not a reason to create another wrapper or weaken the guard.

[Metadata qualification](metadata_qualification.json) records the read-only
checks and exact refs. [Activation packet](activation_packet.json) contains
structured argv and pins. This worker performed no guard configuration/state
query, PID probing, admission, accounting, native/scientific execution, GPU or
Git action. Only this dedicated handoff folder was written.
