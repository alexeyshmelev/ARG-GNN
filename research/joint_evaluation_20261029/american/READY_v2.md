# American TreeLoom joint benchmark successor

The v2 runner is independently reviewed and CPU qualified. Historical v1 files,
original American competitors, the frozen European model/evaluation, and the
joint model source remain unchanged. Root owns production launches and Git.

Freeze: `research/joint_evaluation_20261029/american/freeze_v2.json`
SHA256: `a203156e0de57b9a536d176542b84c1fea934f70f8200cb3a57a35129af060c7`.
The runtime independently verified all 19 pinned source references after sealing.

## Scientific contract

All 20 original TRAIN parents, 200,000 canonical midpoint targets, full native
inferred trees, exact missing focal genotype treatment, K=5 coordinate context,
and 50 kb radius remain fixed. There are 13 complete epochs and no held-out model
or checkpoint selection. Classes remain EUR/EAS/AFR. Every completed epoch retains
its model, optimizer, and RNG states; only the final epoch-13 checkpoint authorizes
predictions on the five VAL and ten TEST parents. All 15 predictions must seal
before original private truth is resolved by the unchanged native3 common scorer.

The corrected summed-CE multiplier is
`W_parent * B_total / (B_parent * S * N)`, where
`B_parent = ceil(W_parent / 16)` and `S` is the actual batch window count.
Uniformly shuffling each parent's windows gives every canonical target expected
coefficient `1/N` after averaging all `B_total` updates, including partial tails.
Analytic checks include parents of 1, 15, 16, 17, 31, and 33 windows and actual-sized
parents. They expose the old global-M formula's bias with unequal partial tails.

## Exact cache and measurements

The cache preserves all tensors and entries. Integer tensors use exact differences
and narrower storage, then restore their original dtype. Floating tensors undergo
byte rearrangement only. The packed state uses zstd level 9. Sparse CSR row indices,
column indices, values, shape, and dtype are retained. No model approximation,
context change, quantization, hidden-state reuse, or target reduction is used.

A seeded uniform sample of 128 TRAIN windows across all 20 parents passed exact
roundtrips. It measured 8,589,191 raw bytes/window, 389,302 packed bytes/window,
1.076 seconds preparation/window, 0.0331 seconds compression/window, and 0.00837
seconds decode plus tensor restoration/window. The 16-worker probe observed
13,790,904,320 bytes total process-family RSS.

Projected payload is 61,694,958,133 bytes. Including retained probe evidence,
headers, and 4096 bytes conservative extra charge per file gives 62,721,441,059
bytes against the 67,645,734,912-byte (63 GiB) hard cap. The margin is 4,924,293,853
bytes. This is a sample projection; the producer measures the complete cache.
A flock-protected counter reserves each write before it occurs and fails before
the cap. Cache keys authenticate exact public source bindings, parent, context,
focal row ownership, and codec namespace; each payload has a checksum.

Ideal preparation plus compression is 3.05 hours with 16 workers. Thirteen epochs
require about 4.79 CPU hours of serial decode work. These estimates exclude GPU
compute, source startup, and NFS contention. The probe also compared several
codecs, so its complete 36-second wall time is not production throughput. GPU
runtime and reservation remain subject to actual qualification.

The CPU producer seals all 158,476 TRAIN windows before training can start.
Training only reads the complete cache; a missing or corrupted window fails.
Held-out windows are prepared once and are not added to the persistent cache.
The producer records the global largest 16 windows by node count, SNP row count,
and prepared tensor bytes for full-minibatch GPU stress tests after completion.

## Evidence

- `smoke_result_final_v2.json`: final CPU integration PASS, persistent cache reuse,
  two optimizer steps, checkpoint/logit parity, original common scoring, expected
  tail weighting, unsafe integer bound rejection, and exact float NaN payload,
  signed zero, Inf, dtype and shape recovery.
- `../review_finish/american_v2_checks.json`: independent fresh-preparation versus
  cache byte/dtype/shape comparison and learned-checkpoint prediction parity;
  complete original20-parent target identities and rational weighting PASS.
- `RESOURCE_ASSESSMENT_v2.json`: measured numbers, estimates, and limitations.
- `experiments/results/joint_american_20261029/compression_probe_v5.json`: all128
  unbiased probe records. Earlier probe evidence is preserved, including two
  worker-initialization failures fixed before the passing probes.

## Root execution commands

All commands run on anogena inside `/workspace-SR003.nfs2/ARG-GNN` after sourcing
`scripts/project_env.sh`. Root provides its directly owned stage supervisor,
actual wait receipts, and GPU allocation. The American stages have exactly one
results writer and execute sequentially. Root holds its shared GPU lock across
the complete GPU training/prediction chain. Do not alter live queue sources.

CPU cache preparation (no CUDA and no private truth):

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARGGNN_PYTHON" -B -m research.joint_evaluation_20261029.american.pipeline_v2 \
  prepare-cache --freeze research/joint_evaluation_20261029/american/freeze_v2.json
```

After its complete cache product and actual successful wait, expose the single
root-assigned GPU UUID through `CUDA_VISIBLE_DEVICES`. The example below uses
physical GPU3, with its UUID supplied by root's verified assignment:

```bash
"$ARGGNN_PYTHON" -B -m research.joint_evaluation_20261029.american.qualify_gpu_v2 \
  --gpu 3 --freeze research/joint_evaluation_20261029/american/freeze_v2.json
```

This records actual allocated/reserved GPU peaks and two real AdamW updates for
each of three full16-window stress batches. Temporary diagnostic parameters are
never saved as a benchmark checkpoint. Root consumes these measurements before
admitting the production GPU fit:

```bash
"$ARGGNN_PYTHON" -B -m research.joint_evaluation_20261029.american.pipeline_v2 \
  train --gpu 3 --freeze research/joint_evaluation_20261029/american/freeze_v2.json

for parent in argformer_american_validation_{000..004} argformer_american_test_{000..009}; do
  "$ARGGNN_PYTHON" -B -m research.joint_evaluation_20261029.american.pipeline_v2 \
    predict --gpu 3 --parent "$parent" \
    --freeze research/joint_evaluation_20261029/american/freeze_v2.json || exit "$?"
done

CUDA_VISIBLE_DEVICES= "$ARGGNN_PYTHON" -B -m research.joint_evaluation_20261029.american.pipeline_v2 \
  score --freeze research/joint_evaluation_20261029/american/freeze_v2.json
```

Root's supervisor must stop the chain on any nonzero return and only score after
all 15 successful prediction waits. The scorer itself also verifies all15 seals
before reading any held-out truth. These are visible commands for root integration;
this preparation agent has launched no production cache, fit, prediction, or score.
MD'