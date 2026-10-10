# Synthetic development CUDA admission

This smoke is a runtime feasibility and exact replay check. It never loads a cohort, accesses actual labels or held-out data, simulates genealogies, creates an optimizer, updates parameters, selects a checkpoint, or measures ancestry performance. Artificial cyclic categories exist only outside model inputs to drive one backward pass and check raw NLL replay. A successful smoke cannot establish predictive quality, reproduction fidelity, full experiment feasibility, or a scientific claim.

## Launch ownership and prerequisites

Root owns GPU scheduling. Other agents must not launch this smoke. Root must first verify that an eligible physical GPU (2, 3, 4 or 5) is free and that fewer than two project GPUs are active. Use the existing project launcher and a new output directory. The wrapper is necessary because the launcher only accepts arggnn.* modules.

Example for root after verifying physical GPU 4 is eligible and free:

```sh
source /workspace-SR003.nfs2/ARG-GNN/scripts/project_env.sh
cd /workspace-SR003.nfs2/ARG-GNN
"$ARGGNN_PYTHON" scripts/launch_gpu.py --gpu 4 --job synthetic-development-admission-v1 arggnn.check_development_cuda_admission --output experiments/results/synthetic_cuda_admission_v1
```

The script sets and verifies the process title `AI Auto Research Pipeline Test`, then applies the strict development execution policy. Only afterward does it validate the inherited lease and inspect the CUDA device. It verifies the eligible physical assignment, exact lease path, descriptor inode/device, held flock, job metadata, one visible launcher UUID, one logical CUDA device and the runtime UUID when supplied by PyTorch. Root must verify the process title and assignment in NVTOP while the process is live. The receipt does not claim that NVTOP was verified automatically.

Do not retry into an existing output directory. Preserve failed runs. Use a fresh suffix for a retry after fixing a cause. The source snapshot is captured at import and checked before and after each case and at completion. Any source or execution-policy drift rejects admission. Constructors and pure observed preprocessing are called directly, with no production admission bypass flags.

## Closed case matrix

The default smoke contains 44 cases:

| Family | Settings | Cases |
|---|---|---:|
| Fixed-anchor candidate | plain/age geometry × kernel depth 1/2, width 32, no dropout | 4 |
| Same-anchor GRU and Transformer | each width 32/64 × dropout 0/.1, source depth 2 and temporal depth 2 | 8 |
| All-record boundary GRU and Transformer | each width 32/64 × dropout 0/.1, source depth 2 and temporal depth 2 | 8 |
| Canonical observed-tree GNN-GRU | width 32/64 × dropout 0/.1, graph and GRU depths 2 | 4 |
| ARGMix full tree | width 32/64 × dropout 0/.1, depth 2 | 4 |
| GST span cross-attention | width 32/64 × dropout 0/.1, depth 2, paper gate | 4 |
| GraphSSM exact ZOH | width 32/64 × dropout 0/.1, depth 2, bidirectional shared blocks | 4 |
| Observed-reference coordinate smoother | bandwidth 10k/30k/100k/300k bp × pseudocount .001/.01 | 8 |

The candidate and same-anchor controls use 32 physical anchors, latent width 8, three branches and initial coupling rates 5/20/80. The candidate uses full coupling, log rate parameters and initial damping 1. No learning-rate dimension is relevant because no optimization occurs. The same-anchor Transformer retains its 256 MiB dense-score estimate guard per layer; that estimate excludes activation, backward and workspace memory. The smoke records actual CUDA allocator peaks separately.

## Handmade observed input

Three balanced trees carry 64 persistent query leaves and 24 persistent references, six references per observed class. Each tree has 175 nodes. Pairing order changes across physical holds [0,.31), [.31,.72) and [.72,1). Query features contain no class labels. Reference one-hots and descendant clade presence indicators encode only observed reference roles. Query ages are zero, reference ages are positive, all parent branches are positive, and internal vertices use anonymous identity -1. Leaf IDs serve correspondence only. Query order is deliberately reversed.

The existing observed reference adapter generates raw F48 inputs. The existing leaf interface removes branch column 7 to produce F47, and its exact query/reference partition supplies plain and age operators at scales 0/350/1400. All spectral work runs on CPU before neural forwards. The canonical comparator uses its existing observed canonicalization and fresh adapter. Eight independent fixed readout positions cover all three holds. Coordinates retain float64 precision; neural features/operators use float32.

This is a small structural smoke. Three holds and 175 nodes per tree do not cover the record count, depth, topology, memory demand or numerical extremes of production cohorts. Strict deterministic CUDA execution may reject a family whose backend operation has no deterministic implementation. Such rejection is a preserved failure, not an invitation to relax the policy.

## Checks and records

Each trainable model performs one training-mode forward and backward without any parameter update. Logits and raw float64 artificial NLL must be finite. Every trainable parameter must have a finite connected gradient. Each parameter-module group must contain an active gradient, with candidate/source rate tensors reported separately. The only declared expected zero groups are the first gate affine layers in GST's two blocks, because each gate's final affine layer starts at zero. Finite nonzero observed-feature gradients are also required.

Each model then runs twice in evaluation mode, saves state, reconstructs the same pure model, reloads state strictly, and runs again. Both evaluation logits and raw NLL must match exactly. The deterministic smoother checks forward and serialization replay and reports backward as inapplicable because it has no parameters.

Files are created exclusively and never replaced:

- `run_started.json`: immutable source snapshot, scope and all required cases.
- `runtime.json`: applied policy, process title, launcher lease, physical/logical device, UUID, software/build versions and storage usage.
- `fixture.json`: observed fixture dimensions, coordinates, roles, spectral audit and tensor hashes, with a separate artificial-target hash.
- `<case_id>/started.json` and `result.json`: every attempted case, stage, success/failure, gradient evidence, exact replay, elapsed runtime, source/policy admission and CUDA peak allocated/reserved bytes.
- `<case_id>/state.pt` and `eval_logits.pt`: exclusive checkpoint and original evaluation logits, hashed in the case receipt when replay reaches completion.
- `run_failure.json`: any fatal startup or outer failure, including partially completed runtime admission steps.
- `completion.json`: all successes, failures and uncompleted cases. `cuda_admitted` is true only when all 44 cases complete successfully on CUDA.

Failures remain visible and other cases continue where their guards permit execution. A drift failure prevents subsequent model construction. An interrupted run preserves its begun case and reports uncompleted cases. Source pins describe the actual runtime snapshot; no historical proposal pins are substituted.

## CPU verification

The CPU tests exercise the default fixture contract and closed 44-case matrix, plus smaller direct helper fixtures for every neural family and the smoother. They cover finite backward groups, exact checkpoint/logit/NLL replay, expected GST initialization zeros, title/policy/lease/device ordering, lease rejections, immutable outputs, source drift and failure accounting. Mocks cover CUDA admission without using a GPU.

```sh
source /workspace-SR003.nfs2/ARG-GNN/scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /workspace-SR003.nfs2/ARG-GNN
"$ARGGNN_PYTHON" -m pytest tests/test_development_cuda_admission.py --basetemp=data/synthetic_admission_cpu_tests_fresh
```

`--device cpu` is an explicit whole-matrix diagnostic and requires `CUDA_VISIBLE_DEVICES=''`. CPU completion always sets `cuda_admitted=false`. Neither CPU tests nor a CPU diagnostic establish CUDA admission.
