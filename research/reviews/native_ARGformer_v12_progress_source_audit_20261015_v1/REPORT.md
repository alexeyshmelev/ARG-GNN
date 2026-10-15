# Native ARGformer v12: real training progress, final optimizer counters pending

**V12 has real training progress.** Root's actual session48073 emitted unchanged Composer epoch1 banners through700/44,955 and rolling throughput about10.2–10.55 batches/s, or5,231–5,402 samples/s. These are training batches after optimizer calls, not a disposable qualification or merely CUDA initialization. This worker inspected source/metadata; the raw session observations are root-reported and are identified separately in observation_v1.json.

An independent exact-directory metadata observation found ep0-ba2500-rank0.pt, a regular2,014,523,270-byte file, and latest-rank0.pt, a19-byte symlink. No checkpoint contents were opened. The current bounded snapshot is saved alongside that earlier observation; native retention may replace its filename. A named checkpoint is progress/save metadata, not verification of its serialized Adam counters.

## Counter semantics

Installed Composer trainer.py:2685 calls _train_batch before advancing timestamp at2727 and running BATCH_END at2733. _train_batch calls optimizer.step at2905/2916. ConsoleLogger.batch_end reads that advanced timestamp; its progress string uses epoch+1 and batch_in_epoch. With the actual512 global batch,512 microbatch, one rank and amp_bf16, the observed banners therefore follow ordinary optimizer execution. They do not inspect every parameter's Adam step or prove final saved state.

The v12 child deliberately performs strict counter verification **after** module.main(config, return_trainer=True) returns (native_child_v1.py:204–219). Author main blocks in trainer.fit and only then returns Trainer. The child subsequently requires epoch5/global batch224,775, checks actual in-memory optimizer state steps, loads/validates the final saved checkpoint, and writes completed.json. This worker did none of those payload operations. No such final receipt is expected midway through epoch1. Five epochs ×44,955 batches is the frozen224,775-update budget; an epoch1 banner is not five-epoch completion.

Resolved actual YAML retains5ep, batch512/microbatch512, amp_bf16, seed17, train drop_last, eight loader workers, evaluation every500 batches for10 eval batches, and checkpoint saving every500 batches with one retained checkpoint. loggers={} disables configured W&B; log_to_console=true and100ba console interval remain. The throughput monitor has a10-batch rolling window and excludes evaluation in its timestamp accounting. Its instantaneous rates are not an end-to-end ETA or ancestry accuracy.

## Why the console can go quiet

The frozen progress_v8.py forwards only the **first eight actual batch banners**, plus the first banner from each new epoch. Composer logs its first batch as well as100-batch intervals, so those eight can be1,100,…,700. Root's observed700 may already exhaust epoch1 banner forwarding. It also forwards at most24 throughput lines, with a48-total-notice cap. RootRelay forwards only those prefixed notices, with a65,536-byte cap.

Thus later epoch1 console silence is expected even while native Composer continues logging into the supervisor's bounded RAM tail. It is not evidence of a stall or lack of optimizer calls. No relay repair is needed to explain it, and no replacement relay implementation is supplied.

The preallocated native.tail/outer.tail files are persisted on actual child/launcher wait and terminal drain, not continuously. Their creation-time mtime or zero prefix while running is expected. The bounded current prefix/header observations in observation_v1.json are not a full live-log scan. No unseen traceback is ruled out. No exception/traceback was supplied in root's actual progress notice.

## Safe observation route

OBSERVE_COMMANDS.md contains exact bounded, read-only commands for checkpoint names/lstat, completion presence and terminal-frame prefixes. Do not deserialize a live checkpoint, follow latest into its contents, signal the worker, read unrelated /proc state or import the execution/guard modules. Root can continue polling its existing session48073 for any relayed notices; this worker did not consume that session.

The current state should be reported as **native full30Mb MLM training has genuine Composer batch progress and checkpoint-save metadata; final live+saved optimizer-state counters and five-epoch completion remain pending**. Do not reduce that to “only CUDA context,” and do not promote filenames to verified Adam steps. The absolute timing remains the original authority's; no restart/resume or additional run is proposed.

source_refs_v1.json pins actual v12 source, author main/YAML, installed Composer logger/trainer/monitor source, and active input/admitted metadata. Only this owned folder was written. No corpus/checkpoint/model payload, GPU query/job, process signal, live-source edit, guard/accounting call, Git, installation or test was performed.
