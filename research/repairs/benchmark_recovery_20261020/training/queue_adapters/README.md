# Ordinary source queue adapters

These are in-worker callables. The maintained scheduler owns admission, actual
process waits, resource bounds and shared physical GPU 0 reservations. This
source package launches no process and changes no historical source or output.

## Routes

Package prefix:
`research.repairs.benchmark_recovery_20261020.training.queue_adapters`

| Module | Factory | Worker |
| --- | --- | --- |
| `european_stage` | `qualification_factory` | `run_qualification` |
| `european_stage` | `fit_factory` | `run_fit` |
| `european_stage` | `prediction_factory` | `run_predictions` |
| `american_prediction` | `prediction_factory` | `run_predictions` |
| `empirical` | `factory` | `run` |

European factories preserve the maintained V9 keyword arguments. Fit and
prediction require the new same-source CPU `qualification_ref`. The
qualification receipt schema is
`european_readable_complete_parent_CPU_qualification_v1`.

The European fit callable accepts
`run_fit(spec, *, resource_check, acquire_gpu0)`. Before importing the numerical
package, the worker must have fixed physical GPU 0 UUID visibility and
`PYTORCH_NVML_BASED_CUDA_CHECK=1`, without an acquired shared reservation.
Complete observed preparation stays in the same process. All original three
TRAIN observation seals precede the late callback and every target read. The
callback supplies the actual shared reservation descriptor and finite allocator
cap. It must not claim an exclusive device lease.

American prediction accepts the unchanged maintained final epoch 30 / 600
update completion and checkpoint schemas. The standalone candidate fit can be
adopted after its genuine final completion and supervisor wait. No American fit
or qualification is rerun by this package.

## Preserved scientific behavior

- AP native six and current C4 remain independent fits with their original
  target schemas, heads and output classes. The C4 registry view changes class
  declarations only, preserving coordinate, record and sample identities.
- Every native tree, node, site and sample remains in the observed bank. Sampled
  repeated rows remain repeated loss and emission contributions.
- Original two shared tree passes, structural adapters, stable sample memory,
  learned memory scale, full genomic context and original 49 parameter tensors
  remain unchanged.
- The bidirectional European GRU is evaluated in bounded chunks through
  ordinary imported views sharing the original parameter objects. Hidden states
  stay connected across chunks. Decoder tiles retain the complete sequence and
  accumulate the complete context VJP before one upstream backward pass. FP32
  reduction order can differ from a monolithic call.
- European training retains 30 passes, 90 complete parent updates, original
  AdamW and clipping, six scheduled checkpoints and fixed final epoch selection.
  Each completed update writes one compact progress receipt.
- Empirical application retains the three original H36 observed windows, donor
  and query identities, genuine physical intervals and original span and person
  pooling. It supplies descriptive predictions without accuracy labels.

## Scope of verification

`test_adapters.py` uses only handmade or random synthetic CPU tensors and fake
stage callbacks. It covers recurrent outputs and gradients, complete synthetic
European trunk logits and parameter gradients, tiled decoder VJPs, repeated
rows, empty tiles, fixed update schedule, checkpoint counters, separate target
schemas, the late GPU callback barrier and empirical pooling. It is not a real
benchmark, full parent memory qualification or GPU throughput measurement.

No source execution loader, AST compiler, bytecode cloning or encoded executable
payload is used in the operational adapter call paths. Ordinary source parsing
appears only in the static test that checks those paths.
