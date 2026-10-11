# Biological bank cache v1 author handoff

Inactive CPU/GPU execution wrapper for unchanged H020 mechanisms. Source-only handmade CPU work has completed. Fresh independent review, root registration and actual target-free one-tile profile remain required. No actual scientific arrays, CUDA context, GPU profile or scientific fit were opened by this author.

Owned files are `src/arggnn/biological_bank_cache_v1.py`, `configs/biological_bank_cache_v1.json`, `docs/biological_bank_cache_v1.md` and this author directory. Original pinned model/config/receipts and the previous Census fitting bundle remain unchanged. The root owns Git, ledger updates, scheduling, memory/lease admission and actual execution.

The cache retains the complete exact original loader-returned prepared object. It leaves full raw serialized archives unchanged, hashes each bank on every request including hits, retains float64 coordinates, and only replaces the TRAIN/VAL loader aliases inside a restoring context. It does not reformulate operators, select fields, change precision, or change scientific behavior.

The final cache caps are 64,000,000,000 CPU and 32,000,000,000 GPU resident tensor bytes. Separate original loader/model/optimizer/autograd/runtime, allocator and caller-reference headroom is required. Both baseline and cached banks are held for one-tile parity, so the profile needs that additional headroom. The one-tile profile cannot establish sustained full-loop GPU utilization or full-32-bank residency. Required SHA hashing remains in all timings.

The exact inactive root schemas are `profile_admission.template.json`, `memory_reservation.template.json`, `profile_gate.template.json` and `fit_admission.template.json`. `docs/biological_bank_cache_v1.md` explains the required reference roles, source map, TRAIN selection, runtime guards and commands. Templates do not activate capabilities. `registration_request.json` requests an unfinished H020 execution-optimization condition and makes no ledger edit.

Tests have 27 distinct final cases and 31 passing executions across three preserved attempts. Attempt01 records 22 fixture setup errors, attempt02 21 passes and one mistaken test-only expanded-stride assertion, attempt03 10 corrected/new/changed passes with 17 deselected. The two fixture issues were corrected without changing science or tolerances. All seven unchanged default models have exact full-field/logit/VJP/clipped-AdamW/RNG equality on handmade full280 fixtures. Explicit CPU-only fake GPU transfer tests establish bookkeeping only. The actual callable target-free profile was exercised only through handmade CPU adapters.

Review should inspect the actual source and recorded limits independently. The author receipt status deliberately cannot satisfy the independent review gate. In particular, PyTorch mutation versions cover ordinary tensor mutation by the pinned read-only consumers, not arbitrary `.data`/external-storage mutation. Evicted tensor memory can remain allocator-reserved. Original loader cold misses hash twice. No speed or accuracy benefit is claimed.

Bounded test command, only when a new source change or review concern justifies execution:

```sh
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/env/bin/python -m pytest -q research/reviews/biological_bank_cache_v1_author/test_cache_v1.py \
-p no:cacheprovider --basetemp research/reviews/NEW_REVIEW_OWNED_DIRECTORY/pytest_tmp
```
