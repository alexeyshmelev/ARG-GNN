# TreeLoom training recovery source

This is an inactive numerical replacement. It does not launch processes, attach
to the live worker, signal a process, admit resources, open scientific datasets,
or modify existing historical sources. ROOT owns activation and GPU execution.

## Verified diagnosis

The train000 observed plan has 77,472 native holds and 227,474 sites. The old
executor uses 4,842 tree blocks of 16 holds, two tree sweeps, checkpoint
recomputation during backward, and 200 full-sequence GRU tiles of eight
haplotypes. Only the 1,000 query haplotypes can contain focal labels.

The constructor repeatedly computes parent height keys and scans all nodes
separately for each height/depth. `layout.py` replaces this with stable grouping
while preserving exact ascending node order in every index family. The normal
bundle and cut validation functions mainly check immutable tensor stamps and
small hold/focal arrays; they do not rescan the entire feature bank.

The old decoder checkpoints every GRU tile, including tiles with no focal
labels, then recomputes these recurrent passes during the later backward. The
new decoder computes sum CE / all 10,000 focals per nonempty tile, immediately
accumulates its parameter gradients, and saves the complete CPU context VJP.
One subsequent context backward propagates that exact VJP through the full tree
and history graph. One AdamW step remains after every complete parent.
FP32 reduction order can differ; model equations, inputs, seeds, history,
all-focal objective, clipping and optimizer schedule are preserved.

## Integration

After constructing the static `ReadableTreeLoom` runtime, before any parent
update, use ordinary imports:

```python
from research.repairs.benchmark_recovery_20261020.training.step import install_execution

install_execution(
    runtime,
    status_path=output_directory / "training_live.json",
    holds_per_block=128,
    haplotypes_per_gru=32,
)
```

The existing `runtime.train_step` call remains unchanged. The bound callback
retains the runtime's exact frozen class namespace and preserves its prediction
API. `status_path` must have an existing admitted project output directory.
The reporting thread uses at most a 64 KiB latest snapshot and a 64 KiB atomic
replacement. It records stage times, active-stage duration, progress marks,
main-thread Python stack and peak RSS every 15 seconds. It never invokes CUDA.
Every numerical resource callback remains active.

The source permits 1–256 holds per tree block and 1–128 haplotypes per decoder
tile. Start GPU qualification at 128 holds and 32 haplotypes under ROOT's
16 GiB allocator cap. A 64-haplotype tile is a measured follow-up, not a proven
16 GiB fit. Do not automatically retry an OOM or advance the optimizer after
an incomplete parent.

For train000, the complete CPU context is 15,866,265,600 bytes. Its VJP needs
that much additional CPU memory. One 32-haplotype decoder input is 317,325,312
bytes; one 64-haplotype input is 634,650,624 bytes. Forward/reverse activations,
cuDNN reserve, gradients, tree recomputation and allocator retention add memory.
These shape estimates are not a GPU peak measurement or a full-parent ETA.
`plan_shapes.result.json` records all 20 parents without loading tensors/truth.

## CPU verification

Run from the repository on anogena after reading `scripts/project_env.sh`:

```bash
. scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -B -m research.repairs.benchmark_recovery_20261020.training.smoke_cpu
PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -B -m research.repairs.benchmark_recovery_20261020.training.sweep_cpu
PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -B -m research.repairs.benchmark_recovery_20261020.training.plan_shape experiments/results/benchmark_campaign_20261018_workflow/release_v8/american/shared/AM_treeloom_fit/native/plans
```

`smoke_cpu.result.json` compares the original checkpointed decoder with the
staged decoder on synthetic CPU data, including full trunk gradients, one
AdamW update and empty focal tiles. It also checks all four layout index
families for exact equality. `sweep_cpu.result.json` compares the unchanged PyG
upward/downward cavity equations and gradients with 16 versus 128 hold blocks.
These are phase tests. They do not establish full-parent GPU speed or capacity.

## Provenance and review

`execution.py` is a visible static copy of the pinned GPU-history executor
SHA256 `8198f589aa7e7ad6b26cb530992ed44be2481ecde2c6ef0a7c98d396408de7e4`.
It preserves site chunks, tree equations and stable-leaf history. Changes are
explicit frozen-namespace injection, stable CPU layout grouping, larger bounded
tree partitions and returning complete context at the decoder boundary.
There are no dynamic source loaders in this repair.

The readable runtime binding and static preparation modules were independently
reviewed for the intended namespace binding and v8 integer/root-hoist changes.
The binding bypasses the old source-executing constructor. Its source integrity
verification reads and hashes files; it does not execute their source text.
The live historical process and its prepared RAM were not modified.
