# Queue paused by request, 31 October 2026 (Moscow)

The user requested: “Pause current queue and release GPUs.”

All eight identified campaign controllers are suspended, including the American continuation, finishing and review controllers. Seven CPU reconstruction processes are suspended in place. The ARGformer handoff watcher had already exited with `opportunity_closed`; it cannot launch a replacement.

The ARGformer TEST009 MLM worker was interrupted with SIGINT. Remaining members of its verified process family were terminated to release resources. GPUs 3, 4 and 5 each reported 0 MiB used after release. Other projects were not signalled. No GPU reset was used.

Retained periodic checkpoint: `ep0-ba32500-rank0.pt`. Its ZIP directory is readable; the serialized optimizer/training state has not been loaded in this operation. Updates after the saved checkpoint are not preserved. Completed results, input data and existing checkpoints remain intact. This operation does not mark the interrupted fit as complete.

## Resuming

Wait for the user's instruction. Reverify process IDs and start ticks from `controllers.json` and `workers.json`. CPU reconstruction can be continued in place while its suspended processes survive. ARGformer needs an explicit checkpoint continuation: the existing adapter disables automatic resume and rejects an existing output directory. Do not simply resume every controller, overwrite outputs, or relabel the interrupted task as successful. The original ARGformer controller has not yet collected its interrupted child's exit status.

`PAUSE.json` records the current hold. `pre_pause_status.json` preserves the previous publication. Low-level queue files retain their original task records; their old `running` labels are not evidence that work continues during this hold. The operational hold is enforced by suspended controllers, not by a newly implemented scheduler flag.
