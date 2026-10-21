# TreeLoom full-training operational successor

The preserved October 20 fit exited by owned SIGTERM with genuine supervisor
wait -15 after the initial 128 GiB RSS cap was exceeded. Its last measured peak
was 140453486592 bytes (130.81 GiB), at allocation of the full CPU context
adjoint after the complete context forward. The first original parent prepared
in 2102.79 seconds and its context forward took 697.79 seconds. No optimizer
update completed, no qualification was written, and the checkpoint directory is
empty. Therefore this successor starts at update 1 of the original 600.

`fit_v1.fit_factory` accepts the original task's `preparation_refs`,
`generation_refs`, `output_directory`, `max_rss_bytes`, and `qualification_ref`.
It also accepts an optional exact `resume_checkpoint_ref`. It authenticates the
original job, observed barrier, all 20 original input bindings, and the genuine
historical full-parent qualification. Its descriptor uses the existing project
Python and the ordinary `run_fit(spec, resource_check=...)` queue interface.
The return value preserves the maintained completion and `completion_ref`
interface for downstream predictions.

The queue must admit 512 GiB RSS from the start, a 16 GiB CUDA allocator cap,
physical GPU5 UUID `GPU-3db85505-d061-e45d-d154-e7e8c1cad497`, and the prescribed
process title. Logical CUDA remains `cuda:0`. The fit verifies physical identity,
visibility, title and allocator admission before its first CUDA initialization.
ROOT owns queue state, shared GPU admission, process supervision and launches.
There are no standalone GPU or continuation gates in this successor.

The unchanged static numerical code retains every original parent, tree, site,
haplotype and 10000-label update, full recurrent backpropagation, 49 parameter
tensors, FP32, AdamW settings and the original 30-epoch/600-update schedule. The
compact RAM pool retains each needed observed parent after one preparation per
process, within its 352 GiB cache budget and 128 GiB working reserve. No disk
observation cache or partial-parent update is introduced.

A complete optimizer update is followed immediately by an atomic rolling
`checkpoints/latest.pt` and a small `latest_checkpoint.json` pointer. The full
model, all 49 AdamW states, exact completed schedule position and Python, NumPy,
Torch CPU and Torch CUDA random states are saved. Scheduled checkpoints at
epochs 5, 10, 15, 20, 25 and 30 are retained. A future fresh scope may resume
from an explicitly hashed completed checkpoint and runs only the remaining
schedule. It preserves the input checkpoint and previous scheduled references.
A process lost between its optimizer update and durable checkpoint cannot
provide proof of that unsaved update; it must resume from the last actual saved
state rather than invent progress.

One CPU integration smoke interrupted the actual new fitter after seven saved
synthetic updates, then resumed 593 updates. Its final model, every AdamW moment
and all CPU random states matched an uninterrupted 600-update run exactly. It
used all 49 tensors/32132 parameters, the actual compact RAM codec on tiny
synthetic forests, and the maintained saved-checkpoint validator. It initialized
no CUDA and loaded no scientific observations or truth. A separate metadata-only
factory smoke resolved the original task arguments from the adopted queue state
and accepted the genuine qualification without creating an output scope.

All old source files, failed output and receipts remain unchanged. This repair
changes admission and recovery bookkeeping, not the scientific model or budget.
