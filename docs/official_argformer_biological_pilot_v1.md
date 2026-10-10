# Native ARGformer biological pilot v1

This runner trains the published ARGformer implementation separately within each
observed ARG, then exports an explicit local C4 ancestry readout. It is a biological
adaptation of native MLM and supervised contrastive retrieval; it does not claim
reproduction of the paper's original task or an exact author dependency-lock
environment. No query ancestry, oracle topology, census labels or migration labels
are input. No 48-feature surrogate or reduced encoder is used.

The four development parents are `bio_paths_pilot_train_000`,
`bio_paths_pilot_train_001`, `bio_paths_pilot_val_000`, and
`bio_paths_pilot_val_001`. Each receives its own vocabulary, MLM checkpoint,
contrastive checkpoint and embeddings. Integer node tokens carry meaning only
inside that parent/checkpoint namespace; this is transductive within-ARG fitting,
including observed query paths in unsupervised MLM. No cross-ARG token or encoder
transfer is claimed.

## Frozen native source and environment

The unchanged author source is `AI-sandbox/ARGformer` commit
`f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596`, under
`.runtime/author_sources/argformer_f7545be7bca0`. Its 60-file tree SHA256 is
`2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b`.
The native interpreter is
`.runtime/conda/envs/argformer_native_v1/bin/python`: Python 3.11.9,
Torch 2.4.0/CUDA 12.4, Transformers 4.44.1, Composer 0.24.1,
Streaming 0.8.0, Triton 3.0.0, and NumPy 1.26.4. The existing environment recipes
identify packaging repairs (tskit 0.6.4, multiprocess 0.70.16, ninja 1.13.0)
and the official FA2 2.6.3 prebuilt wheel. The runner verifies the frozen
qualification helper and lease guard by SHA256, all author bytes, admitted
distribution versions, actual loaded module origins, full backbone dimensions,
RoPE implementation and actual FA2 backend.

The earlier qualification receipts established actual CUDA forward/backward for
both full native backbones on handmade inputs. They do not establish biological
model training, source4096 optimizer memory feasibility, GradCache parity, or
ancestry accuracy.

## Completed producer and coordinate admission

Preparation defaults to policy-only inspection. Actual preparation requires
`--execute`, explicit parent/output paths, and root-reviewed config and completed
manifest hashes. The only graph is `inferred_tree.path` from
`data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json`.
Admission requires `kind=biological_paths_relate_v1`, top status
`four_development_parents_inferred_pre_and_post_five_iterations`, all four unique
completed `replicates`, `status=inferred_both_conditions`, and primary condition
`post_five_size_iterations_primary`. Raw converter trees and the separately
preserved pre-size graph are not used.

The graph, post-size derived roster, original observed roster and completion
provenance are independently hashed. The derived roster must contain 280 ordered
sample indexes and the same explicit unique mapping as
`sample_index_to_inferred_tree_node_id`. Both `inferred_tree_node_id` and its
`observed_node_id` alias must identify actual inferred samples.
`source_observed_node_id` remains provenance only. There are 96 pure references
(24 each ANA/CHG/WHG/EHG), 64 mixed reference-only haplotypes and 120 queries.
Only pure references may have observed class integers 0..3. The graph's sample
flags and ages are checked against the observed roster; population, migration,
individual, provenance and non-sample flag annotations are rejected.

Inferred trees use local coordinates. Original center x maps to
`x - local_to_original_offset_bp`; tree length must equal
`coverage_right - coverage_left`. The hashed grid must have NPZ key
`positions_bp` equal to `arange(50, 2000000, 100)`. Only the declared integer,
ordered `retained_fixed100bp_indexes` are joined to path occurrences.
The physical parent interval is [0, 2000000), normalization stays 2 Mb, and
uncovered tails receive no extrapolated predictions.

## Source-default training and explicit adaptations

Full leaf-to-root paths are deduplicated before extant-token exclusion or
truncation. Vocabulary construction includes all observed paths before a seeded
90/10 split of unique full paths (seed 42). Pure-reference paths retain their
positions in that split for contrastive training. Native `ARGDataset` handles
node tokens, skip-extant behavior, CLS/separator insertion, truncation and padding
to length 128. Distinct full paths that become identical after skip/truncation
are not silently deduplicated again.

Both phases keep 12 layers, hidden width 768, intermediate width 3072, 12 heads
and the source optimizer/scheduler. MLM uses seed 17, five epochs, logical batch
512, bf16 and microbatch 64 (an explicit memory adaptation from source microbatch
512). The source Transformers MLM collator applies training masking probability
0.30 and validation probability 0.15 with its native special-token exclusion and
80/10/10 policy. Source gradient clipping remains enabled. MLM uses its source
window 128/global-every-three setting, unpadded embeddings and fused FA2 CE.

Contrastive training uses seed 36, one epoch, fp32, logical batch and Composer
microbatch 4096, temperature 0.3, normalized author mean-head embeddings, and the
separate source all-global/unpad-embeddings-false defaults. Ordinary accumulation
of losses from smaller contrastive batches is prohibited because it changes the
negative set. The source dataset, collator and distributed sampler are retained;
the actual sampler index stream and global full-path IDs are recorded once per
iterator call without consuming a second shuffled stream. Per-update label
counts, source loss, finite gradients, learning rates, model state and native
optimizer state are audited. Source drop-last requires at least 4096 unique
pure-reference training paths.

The released factory receives `cfg.model`, while its class-weight branch attempts
to access `cfg.train_loader`. Its caught exception and uniform-weight fallback
are preserved. `num_hard_negatives=2` is present in YAML but not used by the
selected supervised loader route; the runner does not repair that mismatch.
The factory's three-output classifier is untouched and unused by contrastive
forward/loss/readout. Its gradients must remain None and its weights unchanged.
The new C4 grouping uses pure observed reference labels in the source contrastive
loss, which accepts arbitrary integer class groups. This is an explicit task
adaptation, not a claim that the native classifier predicts C4 ancestry.

The actual source checkpoint factory initializes contrastive fitting from the
trained MLM Composer checkpoint. Every `bert.*` encoder key must be present and
exactly equal after loading. Source overlap between MLM dense-head keys and the
mean pooling head is retained and reported, as are extra/new checkpoint keys.
MLM and contrastive model defaults are not merged.

## Full4096 memory probe and admitted GradCache retry

Root must run an actual full4096 author forward/backward plus native optimizer
step/state allocation using the real prepared corpus. A probe is explicitly not
a biological fit: temporary updated weights and optimizer state are hashed,
model weights are restored and checked. An OOM preserves actual runtime,
attempt flags, traceback, artifacts and exit status. A failed full-batch fit is
also preserved; no output directory is reused or overwritten.

Optional GradCache follows Gao et al., *Scaling Deep Contrastive Learning Batch
Size under Memory Limited Setup*, arXiv:2101.06983. It runs the real author
encoder in chunks of 64, forms one full4096 representation-level similarity
matrix/loss, then replays chunks and accumulates encoder parameter gradients.
Each chunk saves CPU/CUDA RNG state before encoding; backward restores it and
requires bitwise identical replayed embeddings, then restores post-forward RNG.
Chunk dropout therefore matches the sampled chunk computation.

The source has no loss-only API, so the inline native contrastive algebra is
transcribed while calling the real author `_similarity` method. Cache fitting is
blocked until root produces same-corpus, same-code, actual-CUDA full-author versus
cache loss/parameter-gradient parity on 16 real paths with chunk 4 in eval mode
(dropout disabled). Required tolerances are absolute loss error <=1e-5 and global
gradient relative L2 <=0.01. This is not a training-dropout full-batch equivalence
claim. Both parity and actual full4096 optimizer-memory receipts and their
reviewed hashes are mandatory for cache fit. A non-memory source failure cannot
be silently remedied by cache.

## Local C4 downstream readout

After real fitting, the trained native mean head embeds every unique full path
(batch 128). At each retained local center, query and the 96 pure-reference path
occurrences are joined using actual inferred sample IDs. For class c:

`logit_c = log(mean_{24 refs in c}(exp(cosine(query, ref) / 0.3)))`

Softmax across ANA, CHG, WHG and EHG yields uncalibrated probabilities. This is an
explicit downstream reference-similarity adapter. Coordinate chunking (32)
changes only evaluation memory. The output preserves original/local positions,
fixed-grid indexes, node identities and path joins. Native classifier outputs,
query targets and uncovered tails are unused. External biological VAL
calibration and scientific performance validation remain pending; probabilities
are not presented as calibrated.

## Root execution

Run from `/workspace-SR003.nfs2/ARG-GNN` after reviewing/freezing the four NEW
files. Root owns all actual preparations and GPU actions. Source project env
before every project command. Targets must be fresh and have an existing parent.
Keep outer logs/return codes and the generated exit-status files.

Policy-only CPU check:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/prepare_official_argformer_biological_paths.py
```

Preparation for each parent, only after all four Relate parents complete:

```bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/prepare_official_argformer_biological_paths.py   --execute --manifest data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json   --parent-id <parent> --output-dir <fresh-prepared-directory>   --expected-manifest-sha256 <completed-manifest-hash> --expected-config-sha256 <policy-hash>
```

Use the existing `scripts/launch_gpu.py` scheduler for every native action.
It admits only free GPUs 2/3/4/5, holds an inherited UUID lease and limits project
GPU concurrency to two. The bootstrap execs the fixed native interpreter,
contains caches under the fresh run, enforces existing 450/500 GB storage limits
with 5 GB headroom, and sets exact title `AI Auto Research Pipeline Test`.
Actual current-context UUID and NVTOP evidence are saved.

```bash
.runtime/env/bin/python scripts/launch_gpu.py --gpu <eligible-free-gpu> --job <unique-job>   arggnn.train_official_argformer_biological --action <probe-memory|cache-parity|fit>   --prepared-dir <prepared-directory> --expected-prepared-sha256 <prepared-manifest-hash>   --expected-config-sha256 <policy-hash> --output-dir <fresh-run-directory>
```

For a reviewed cache fit add `--contrastive-execution gradient_cache`,
`--cache-parity-receipt <receipt>`, `--expected-cache-parity-sha256 <hash>`,
`--full-batch-memory-receipt <receipt>`, and
`--expected-memory-receipt-sha256 <hash>`. Full-batch execution is default.

Native receipts distinguish failed attempts, passed probes/parity and completed
fits. Fits retain full Composer checkpoints, a source-compatible final backbone
checkpoint, native path embeddings, local logits/probabilities and hashes.
Source/API/CPU algebra checks are implementation checks only. This implementation
agent has not prepared real paths, launched CUDA, fit models, installed packages,
opened truth or used Git.
