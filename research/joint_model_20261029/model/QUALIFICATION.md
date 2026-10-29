# CPU qualification and integration handoff

Core model SHA256:
`ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4`.
Focal-only no_cross helper SHA256:
`0227f5cacbe4dc9a9fbd0bcca56a2f8582cd0d4a2c990b48a054e1a221f2b3e7`.
The qualification was run through readable plaintext SSH on anogena using the
project Python runtime, with CUDA_VISIBLE_DEVICES empty, two CPU threads and no
labels, fitting, optimizer, GPU allocation, package installation or job control.

`qualification_receipt.json` is the machine-readable core receipt.
`no_cross_view_receipt.json` qualifies the separate focal-only helper.
`../review/SHARED_SITE_MODEL_REVIEW.json` records the independent complete
shared-query and batch-isolation output/input/parameter VJP audit.

All core checks passed. On the real packet, fused CSR versus the independent
materialized-scatter implementation produced identical logits for every arm;
maximum parameter VJP error was 2.98e-8. Sample-identity mismatch and label keys
are rejected. Stable sample/internal-node relabeling, complete ancestry maps,
query gathering, disjoint batching, numeric pathologies and cache roundtrips were
checked. The deep foreign-branch test has a gradient norm of 1.884e-7 under
structure and exactly zero under leaf_only while native propagation is disabled.

The first real context has five complete tree occurrences, 952 samples/tree,
9,515 nodes, mean leaf-path length 15.14 and maximum 32. Its eight immutable maps
have 72,066 / 72,066 / 9,510 / 9,510 / 9,520 / 11,424 / 36,591 / 44,946 nonzeros
(descendant/readout/up/down/leaf-left/leaf-right/branch-left/branch-right), using
3,758,580 bytes of CSR storage. Its witness-step upper bound is 485,208. A
separate 163-node leaf path remains complete; an intentionally too-small
execution bound fails explicitly without truncation.

The following is one warmed width64, three-layer invocation on the same first
real observation packet with all 14 targets, on two CPU threads. It is not a
full-epoch or GPU benchmark and does not establish accuracy or general speedup.
The new model has 133,202 parameters; the prior bounded model has 88,590.

| Computation | Preparation, s | Forward, s | Backward, s |
|---|---:|---:|---:|
| New no_cross, full context | 0.96793 shared | 0.05108 | 0.12378 |
| New leaf_only | 0.96793 shared | 0.07859 | 0.14537 |
| New structure | 0.96793 shared | 0.07417 | 0.13111 |
| Prior bounded v1 | 0.05374 | 0.28860 | 0.26627 |

Including uncached preparation plus one structure forward/backward, the new
model takes 1.1732 seconds versus 0.6086 seconds for the prior model. With maps
already cached, those forward/backward measurements are 0.2053 and 0.5549
seconds. The prior API encodes its focal SNP again for each target row; the new
API shares the unique focal row. This comparison is useful for diagnosing costs,
not for attributing speed to a single architectural component.

The separate focal-only optimization passed complete width64/L3 no_cross logits
and every parameter VJP on two real packets and their disjoint batch. Logits were
identical; the worst parameter VJP relative L2 error was 7.40e-8 and absolute error
4.77e-7. It also passed target-order, collate-before/after-view and tensor-state
roundtrip checks. On the first packet it keeps 1,903 of 9,515 nodes and 14 of 90
SNP occurrence rows, with a one-time 0.00787 second view construction. Median of
three warmed full forward/backward measurements was 0.18184 seconds for the
full-context no_cross and 0.02753 seconds for the exact view. The two-site batch
measured 0.43083 versus 0.06214 seconds. Full loading/transfer/optimizer epoch
profiling remains the parent training owner's responsibility.

Historical test harness failures are retained. The first core run used bitwise
query-gather equality despite different classifier batch shapes; measured error
was 5.96e-8, and the corrected check records a 3e-6 FP32 absolute threshold. The
first no_cross harness referenced a nonexistent profile site000001; the saved
profile IDs are spaced, and the corrected test uses site000000 and site000027.
The core model did not change during these corrections. A shell-quoting error in
an attempted harness text substitution also exited without executing the
qualification; the correction used a visible quoted plaintext heredoc.

Full training remains conditional on the dataset/export integrity receipts,
complete prepared-cache qualification, device/full-epoch smoke checks and the
registered unbiased site-sampling loss. Those are integration responsibilities,
not evidence supplied by these CPU-only receipts. No TEST inputs were accessed.
