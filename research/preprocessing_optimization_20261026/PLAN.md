# Exact preprocessing optimization

Owner request: accelerate the benchmark while preserving its scientific logic, numerical outputs and native competitor implementations. GPU authority remains physical 3, 4 and 5. All files and computation stay inside this repository, with the 1 TB project limit.

## Work packages

- Tree preparation: compiled exact layout, forest, clade and cut operations. Compare both current American and European reference routes. Preserve node/episode order and floating reductions.
- Observed loading: vectorized VCF parsing with original fallback and unchanged validations. Compare complete retained-site joins on real observed TRAIN data.
- Independent review: adversarial cases, scientific fairness, source identity, memory and rollout assessment.
- Integration: full observed-parent comparison, logits/gradients/optimizer equivalence, measured end-to-end timing, source-pinned adapters and checkpoint-preserving deployment where qualified.

## Acceptance gates

1. Original files, authors' models, simulations, loss definitions, class order and evaluation support remain unchanged.
2. Every deterministic preprocessing array must match byte for byte. Error behavior, empty/isolated/unary/multifurcating trees, exact clade continuity, missing/multiallelic/phased alleles, field order and physical boundaries are covered.
3. Identical prepared inputs enter the unchanged model. Check finite logits, loss, gradients, clipping and optimizer state from the same saved state and RNG. Any floating tolerance must be stated, not presented as bitwise identity.
4. Time and memory are measured on real complete observations. Small-case speedups alone do not justify claims about full training throughput.
5. Independent review resolves material findings before rollout. Reject changes that weaken baseline methods or alter the biological problem.
6. Live author runs stay intact. Our worker can migrate only using genuine owned-process completion/cancellation evidence and a durable optimizer checkpoint. No fabricated task states or edits to an in-memory controller plan.

## Status

2026-10-26: all three agents working. Running benchmark is preserved. No candidate is deployed. Component parity tests are underway. Full-parent and integration tests remain open.

## Qualification progress

- Closed: exact VCF conversion. 11,173 genotype cases, 1,750 VCF cases, 17 loader cases and 2,586 independent adversarial VCF cases passed. Full actual TRAIN000 retained-site arrays are byte-identical, 422.10 s versus 46.41 s.
- Closed: compiled serial structural kernels. 782 layouts, 210 episode sequences, 210 cut plans and 25 encoded banks passed. Independent complete-sequence and malformed-level checks passed after rejection guards were restored. Initial failure evidence is retained.
- Closed: unchanged-model numerical check. All inputs match exactly. FP32 logits, gradients, clipping and AdamW state agree at atol 1e-6 and rtol 1e-5. An unchanged-baseline repeat exhibits comparable rounding differences. Bitwise GPU training identity is not claimed.
- Rejected: threaded tree encoding. It was slower than the compiled serial path.
- Closed: bounded process encoding component checks for ordered output, RNG preservation, exception cleanup, shrinking transport chunks and complete-tree serial fallback. A full-parent startup exposed inherited ru_maxrss rather than actual worker RSS. That resource guard was corrected and the failed test remains recorded.
- Open: complete-parent reference/serial/process comparison. These are actual registered TRAIN inputs, with no truth loaded and no scientific training in the qualification.
- Open: independent review of release authentication, history restoration and actual checkpoint migration.
- Open: integrate accepted worker variant, verify real training advances, then publish measured deployment status.

The European parser and one-parent prediction lookahead wrapper are prepared and qualified for orchestration. Its currently healthy prediction worker remains intact. The live scheduler cannot replace healthy pending tasks, so this wrapper awaits a legitimate repair or future continuation. This limitation must remain explicit rather than silently rewriting the live plan.

The source-pin audit of the four live workers passed: GST 71 references, European TreeLoom 171, native ARGformer 15 and American TreeLoom 145. No native competitor model or simulation code was changed. Faster preprocessing is an execution result, not a new ancestry accuracy result or a guarantee that GPU utilization will be permanently 100 percent.
