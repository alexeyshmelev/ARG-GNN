# Inactive C3 feature v3 stability correction

V3 preserves the standalone CPU-only contract of v2. It grants no runtime, storage, GPU, dataset, model, optimizer, guard or source-lineage authority. The same original C3 encoder code object runs with only private layout and branch-support helper bindings. Every existing live and v2 file remains unchanged.

The independent v2 review is BLOCKED. Its valid full-bank case has4,096repeated triallelic sites [0,1,2] on a three-leaf star with site_tile1. Plain FP32 partial accumulation produced mean discrepancy0.22221678495407104 versus original0.2222222238779068, absolute error5.4389e-6. This exceeded the declared tolerance. Default tile32 passing did not justify accepting tile1 or relabeling the whole component.

The only v3 mathematical change is compensated FP32 addition across site-tile/allele partial sums for the first three branch-summary channels. Integer PyG descendant propagation, per-site discrepancy, allele eligibility, tie weights, split counts, normalization and all other features are unchanged. Compensation occupies12bytes per original vertex. The conservative named tensor working allowance adds32bytes per vertex. Neither the tolerance nor the original configured cap is enlarged. No FP64 computation is introduced. Original reference code still has its own inherited accumulator.

All12other existing functions/classes have identical whole ASTs to v2. Only branch_support_batched_v1 changes, and one small _compensated_add_ helper is added. This is a preprocessing numerical-stability repair, with no ML novelty or ancestry-accuracy claim.

## Bounded evidence

The revised author test reuses the preserved original21-field fixture, the independent exact4,096site full-bank counterexample, and the independent1,600haplotype/3,199vertex/3SNP fixture with its original generator and seed. No original scientific payload or truth is read.

The exact counterexample now matches all21 bank fields. Its tile1 mean discrepancy is bit-identical to the original FP32 output. Tiles17,32and256 have maximum full-bank differences1.49e-8,1.49e-8and2.98e-8. The1,600sample case matches all21fields with maximum absolute difference2.98e-8. The earlier32haplotype fixture, zero/degenerate/permuted sample-axis cases and six cache/cap/mutation rejection cases also pass. All floating bank tensors are FP32, integer tensors and masks compare exactly, and rtol=atol=2e-6 remains unchanged.

No new microtiming is performed. Earlier v2 timing used129SNPs/tree, while the admitted full VAL000 plan averages about2.99SNPs/tree. That timing cannot establish whole-parent speed or select an operational route. V3 establishes only these bounded input-tensor comparisons. Full-parent memory fit, source-lineage integration, runtime throughput and independent v3 acceptance remain unproved.
