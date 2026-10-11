# Census device profile v1

This separate inactive module measures target-free CPU float32 versus CUDA float32 outputs and active parameter VJPs for the unchanged complete Census whole-tree model. It uses the original fixed float32 tolerances: output rtol 0.0002 and atol 0.00002, VJP rtol 0.001 and atol 0.00001. No threshold is selected after seeing GPU results.

The two cells are the minimum and maximum original tree counts among TRAIN tiles, with parent/tile tie breaking. Their complete externally frozen certificates and initialization are reused. All five controls run with identical initial weights. Each device performs a fixed mean-logit objective and one disposable clipped AdamW update. These are numerical measurements without ancestry targets, scientific training, checkpoint selection, or optimizer-update parity claims. Input identities and complete certificates remain checked.

The default CLI reports source/config hashes without importing Torch or opening observations. Actual execution requires a fresh independent source/config review, the completed original all32 numerical terminal chain, exact original observation/runtime bytes, the actual project interpreter and live Python/Torch/NumPy/tskit versions, a held UUID device lease and the prescribed title verified through the original CUDA/NVTOP qualifier. The logical job must match the admission under both exclusive and shared launch routes. Admission, lease, memory receipt bytes/status/lifetime, and the supplemental v4 reservation are rechecked at resource boundaries. Root must supervise the actual profile with an external watchdog.

The planned per-worker reservation is 16,384 MiB on one GPU plus 64 GB CPU and 1 GB output. This is an admission bound, not a measured allocation. Both min/max full forwards and backwards must stay inside the bound. Existing co-workers keep their own reservations, and the controller keeps separate device headroom. Root may only create an accepted fitting gate from complete measured evidence. A profile alone does not launch training.

Timings report synchronized wall time, including host validation and transfers. They do not establish sustained utilization, kernel-only time, caching benefit, or a speedup. The original model's CPU topology checks remain intact. This module does not change geometry, graph sampling, biological features, coordinate precision, any model layer, or scoring.

Author verification uses only handmade CPU examples. Four initial tests passed, including the full default two-tree/280-leaf/five-control model, exact duplicated-CPU outputs and VJPs, and deliberately changed output/VJP rejection. Later metadata checks cover live interpreter/version mismatch, memory expiry/lifetime and exclusive/shared logical-job binding. No actual observation arrays, target arrays, CUDA context or live budget ledger were accessed during author tests.

Actual command, under a qualified root launcher and a fresh admitted output:

```sh
python -m arggnn.census_biological_device_profile_v1 --execute \
  --admission ROOT_PROFILE_ADMISSION \
  --expected-admission-sha256 ROOT_PROFILE_ADMISSION_SHA256
```

The inactive template is proposal metadata. Actual GPU profile and predictive benefit remain unestablished.
