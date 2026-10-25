# Benchmark recovery and scalability, 25 October 2026

The healthy American TreeLoom and native ARGformer fits are preserved. The European TreeLoom final fit is complete. New ancestry comparisons still await predictions and common scoring.

| Issue | Evidence | Action and current state |
| --- | --- | --- |
| American training | 237/600 complete-parent updates, median 9.74 minutes over the last ten updates | Continue the frozen fit on GPU 3. Training loss is not held-out accuracy. |
| European prediction | Completed epoch-30 checkpoint has all 49 optimizer counters at 90. Failed prediction used raw full-length GRU instead of the adapter already used in fitting. | Prediction-only repair prepared. CPU check of the saved model and a small CUDA recurrent check passed. No retraining. An active deferred submitter will publish the repair after the exact full-parent profile child exits. |
| ARGMix support | Failed at 5.97 GB PSS under a 4 GiB allowance. Exact failed leaf contained only the 1,522-byte wait receipt. | Capacity-only repair accepted at campaign revision 6. Support then completed in 52.88 seconds at 8.21 GB observed PSS. Completed generation, Relate inputs, native Relate and conversion are preserved. TRAIN001 generation and Relate inputs also completed. The next native run is currently waiting for storage reservation. |
| GST data loading and recovery | Timed out. Three parent updates are durably recorded. No model checkpoint exists. Further unrecorded progress cannot be established. | Persistent tree cursor and checkpoints between parent updates passed synthetic window/update/resume checks and an observed topology probe. Fit restart has not been submitted. |
| Full-parent profile | First attempt was admitted, then yielded when the campaign needed host capacity. No timing result was produced. | A fresh unchanged retry is preparing the complete TRAIN000 observation for GPU 5. Full-parent execution and numerical results remain pending. |

## Decisions

- Preserve every original simulated and inferred observation, sample identity, evaluation position, model tensor and training schedule for the execution repairs.
- Reuse completed expensive native stages. Archive only exact failed task output scopes, with the original evidence retained.
- Keep reference-method changes limited to the documented execution or resource repairs. The GST condition remains an adapted LAI baseline.
- Do not interpret the 128-tree GPU speed ratio of 1.63 as a full-parent speedup. Do not interpret pretraining accuracy or training loss as ancestry accuracy.
- Current host headroom can fit the European prediction alongside current jobs, but the next native Relate task raises its reservation from 16 to 256 GiB. Therefore prediction is deferred until the opportunistic profile exits. This changes scheduling only.
- No low-rank architectural replacement is justified yet. The full-parent measurements will determine the next computational target.

## Remaining work

1. Finish the full-parent paired profile and inspect total time, memory and numerical agreement.
2. Resume European predictions from the completed checkpoint after the profile releases its capacity.
3. Review the prepared GST retry, then schedule its complete fit with existing resource admission.
4. Continue American training, native ARGformer stages and ARGMix generation/inference/training.
5. Seal predictions and run common ancestry scoring before publishing new comparative metrics.

Detailed receipts live beside this report. All execution and Git operations are on anogena, inside the project. GPUs 3, 4 and 5 remain the only eligible devices.

## GST loading measurement

The observed TRAIN000 sequence contains 95,625 trees. Its complete load took 29.75 seconds and the probe reached 14.53 GB RSS. On eight identical interior windows, initial seek time was unchanged at about 3.69 seconds. Seven subsequent original fresh seeks took 27.022 seconds in total, whereas advancing the retained cursor took 0.0456 seconds. Active node identities, parents and intervals matched. This isolates topology traversal, without neural encoding, transfer or training, and is not an end-to-end speedup.

The synthetic check additionally compared actual four-worker persistent-loader output, optimizer updates with dropout, and checkpoint continuation. All passed. The checkpoint is 625,710 bytes and is replaced atomically after each complete parent update.

## ARGMix storage admission

After successful preprocessing and the planned TRAIN000 scratch release, the next native Relate run requires 24,020,545,375 additional reserved bytes. The operating ceiling leaves 23,577,282,448 bytes, a shortage of 443,262,927 bytes. The 40 GiB ARGMix scope is not the blocker. The independent American pipeline retains a protected allowance until its complete retirement chain finishes. Individual file retirement may therefore not release net campaign reservation immediately.

A ceiling of 991 GB would remain inside the owner-approved 1 TB and cover this request, but the running controller has no supported live ceiling update. Editing its sealed plan would neither update loaded policy nor preserve the seal. No cap was reduced without supporting peak evidence, no accounting credit was invented, and no historical or native outputs were deleted. The existing queue can eventually release the protected reservation, so this is a wait rather than a dependency deadlock. Safe scheduler policy evolution remains an operational follow-up.
