# C3 prediction efficiency: source audit and action plan

The five pinned routes do **not** rebuild or encode a whole tree for each focal. They encode complete planned forest blocks, scan the parent in both directions, and gather the original focal rows. The important difference is serial versus already-interleaved checkpoint scheduling.

| Original30Mb parent | Trees | Sites | Blocks | Effective route | Raw block constructions, all six states | Learned encoder block calls |
|---|---:|---:|---:|---|---:|---:|
| american_c3_val_000 | 76527 | 228477 | 2392 | serial | 28704 | 28704 |
| american_c3_val_001 | 76453 | 227316 | 2390 | serial | 28680 | 28680 |
| american_c3_val_002 | 78151 | 229909 | 2443 | serial | 29316 | 29316 |
| american_c3_val_003 | 76423 | 227361 | 2389 | already interleaved | 4778 | 28668 |
| american_c3_val_004 | 77050 | 229643 | 2408 | already interleaved | 4816 | 28896 |

These are static full-successful-route counts from pinned source/plans, **not observed progress or timings**. Each candidate retains10,000 focals; original epochs are5/10/15/20/25/30. The historical inventory is dated2026-10-17T00:59:13.515601+00:00 and was not repolled.

## Existing amortization

VAL000–002 instantiate forward/reverse datasets for each checkpoint:12 observation loads and raw-feature traversals per parent. Dataset state retains sanitized tree/metadata handles and a genotype mmap once per instance/PID, not once per focal or block.

VAL003–004 already reuse one forward/reverse loader pair and share each current observation bank across six distinct models. Keep this existing optimization. The separately pinned single-parent device cache belongs to training and is not called in the inspected prediction routes; a pin alone is not activation evidence.

Both implementations rebuild native layouts, branch support, donor features and tensors on the reverse traversal. Within each connected-tree construction, layout is also computed twice: directly by encode_block, then by branch_concordance_support_v1. Learned contexts remain independent per checkpoint and are recomputed in reverse; only focal forward states are retained,1.28MB per checkpoint/7.68MB for six, excluding other state.

## Torch/PyG batching and readout

- The zero-worker loader emits one forest with batch_size=None. Native node offsets and height/depth buckets already batch up to32 full trees, not a PyG Batch per focal.
- Site pooling visits all sites and1600 haplotypes in256-site chunks. Each model repeats learned gather/MLP/index-add/scatter-reduce work in both directions; evaluation does not invoke activation-checkpoint recomputation.
- Sequential upward/downward levels trigger gathers/copies, small linear/normalization operations and PyG propagation. Routing arange/stack tensors are recreated per call; downward sum has one summand. These are source-visible allocation/dispatch candidates, without a measured dominance ranking.
- GRUs consume [trees,1600,35], hidden32. Haplotype columns are independent batch elements.600 known-reference columns are scanned even though all focal readout columns belong to1000 queries.
- Classifier/softmax readout is already sparse. Only focal rows are processed and scattered to CPU NumPy output; there is no dense all-tree/all-sample classifier output to remove.

## Narrow successor

1. Reuse the effective six-candidate scheduler and scheduled-checkpoint metadata handling for future replacements of serial routes. VAL003–004 already do this. Across all five complete planned routes, raw constructions change from96,294 to24,044; learned encoder calls remain144,264. This implies no measured speed ratio and does not justify interrupting current jobs or rerunning seals.
2. Project each **full1600-leaf graph context** onto the dataset's exact persistent query_column vector before the GRUs. Carry six independent states, invert the query map for every focal gather, and retain all original trees/focals. Reference leaves continue influencing query graph contexts. The training efficient step already uses this separation. Column independence is source-backed; changed FP32 batch shapes may change kernel roundoff. Successor receipts must truthfully say graph1600/query recurrence1000 instead of retaining current all1600-recurrent claims.
3. Pass the already-computed connected-tree layout into an internal branch-support helper, preserving the public function and all split/tie equations. This removes duplicate layout without a persistent parent-wide cache.
4. Defer broader aggregation/kernel changes. Current-bank routing reuse may remove repeated PyG allocation, but changing message/summation order requires separate justification. Add no full-parent learned-context cache, guard, bridge or admission layer.

Preserve all five original30Mb parents, every tree/site and1600 graph leaves, all six original saved states, all10,000 focals per cell, original identities/classes, full chronological carry, global-neighbor geometry and true endpoint resets. Keep **all30 prediction seals before any VAL truth/scoring**, no partial-score gate, and TEST deferred until complete original-five-VAL selection.

This is planning only: no source/runtime changes, numerical imports, inference, tests, payload reads or resource operations. The saved600 family trained on20x2Mb TRAIN; this is not final TreeLoom or matched full30Mb training. audit.v1.json carries exact input/plan/source SHA-256 identities and line-level evidence.
