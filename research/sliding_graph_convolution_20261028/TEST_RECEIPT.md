# CPU qualification receipt

Final qualification: 2026-10-28T11:46:21.371347+00:00. All 18 checks passed. The post-review execution completed with directly observed outer exit 0 (tool session 44840, completion chunk dd74d0).

Canonical readable command on anogena:

```bash
cd /workspace-SR003.nfs2/ARG-GNN
bash research/sliding_graph_convolution_20261028/run_cpu_tests.sh
```

The launcher sources the reviewed project environment, hides CUDA devices, sets two CPU threads, disables bytecode writes and routes temporary files inside this owned directory. Source hashes are in test_receipt.json. Environment: Python 3.11.17, PyTorch 2.6.0+cu124, PyG 2.7.0.

The tests check exact sparse-vs-independent-dense overlap, raw coupling transpose, identical-tree diagonal mass, substochastic coverage, ancient sample timing, zero branch behavior, missing old-age support, explicit stable sample-ID mismatch rejection, native-node relabeling, consistent sample permutation, disjoint-batch equivalence and isolation, fixed-coordinate serialization invariance, core/halo logits and parameter/input gradients, finite-difference input gradients, date-sensitive overlap and finite synthetic backward derivatives.

The sample-ID contract was strengthened after independent review. Ages and array positions alone could previously accept incorrectly ordered same-age individuals. The final Tree carries explicit stable sample IDs, checks uniqueness and the same ordered roster, and uses them only as routing metadata. Correct reordering must be performed by the adapter before building the graph. No ancestry truth enters the operator.

## One final synthetic timing observation

Five independently randomized balanced binary trees, 952 terminal samples each, 9,515 nodes, 109778 directed relation edges, three layers, width 32, two CPU threads. Two warmups preceded one measured forward/backward.

- Preparation: 0.4212 s.
- Forward: 0.0461 s.
- Backward: 0.0444 s.
- Process peak RSS: 668.4 MiB, including imports, prior checks and allocator state.

These are toy operator measurements, not real-tree sparsity, actual loader throughput, accelerator performance, training speed or ancestry accuracy. Preparation includes Python validation and overlap construction. The timing excludes a complete-tree initializer, observed SNP encoding, output decoder and loss. The bare layer returns branch coverage but does not consume it. The independent review also identified asymmetric source-clade containment, conditional-kernel distance cancellation, O(E d) gather/scatter temporaries, and branch-constant hidden states across ancestral ages. These are stated limits of the tested design, not subsequently repaired numerical properties.

## Execution history and scope

The initial run wrote its complete 17-check receipt but its inline shell wrapper returned 1 after a post-run status assignment. That outer result is preserved rather than relabelled. A reviewed standalone Bash launcher then obtained a directly observed exit 0 on the same initial sources (session 36726, completion chunk 9db4cb). The final review-driven revision added explicit sample IDs and the eighteenth check, and obtained its own observed exit 0. Initial source copies, initial/pre-review receipts and all logs are retained.

This work read only instructions, existing source/reports and public papers/repositories. No real tree/genotype/ancestry payload, training fit, GPU computation, baseline mutation, job control, dependency installation or Git operation occurred. Project outputs are confined to this research directory. Root owns integration and any later real-observation qualification or registered learning comparison.
