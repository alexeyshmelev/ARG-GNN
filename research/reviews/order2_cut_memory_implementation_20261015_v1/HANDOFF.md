# Order-2 cut-memory implementation handoff

**Status:** importable Torch/PyG source candidate and final bounded synthetic CPU check complete. This is an inactive, unregistered source packet. Root owns the fresh full-budget execution integration. No fits, optimizer updates, GPU use, scientific payload/checkpoint reads, admission, process inspection, Git or live/frozen-source edits occurred.

## Delivered files

| File | Role |
|---|---|
| order2_cut_memory_v1.py | Actual full native-tree PyG model, memory operator, CPU cut preparation, bound bundle transfer, three-arm initializer and named raw-table bounds |
| integration_hooks_v1.py | Importable inert cache/family/evaluation/count/effect entrypoints for a fresh engine |
| training_config.inactive.json | Matched 30-pass, 960-updates-per-arm scientific proposal; no active identity or authority |
| ENGINE_INTEGRATION.md | Concrete minimal callable substitutions and producer/root-driver changes |
| fresh_engine_integration_map_v1.json | 33 exact source/line integration rows plus pinned original literal/label audit |
| check_CPU_v1.py / CPU_check_capture_v3.log / CPU_check_direct_wait_v3.json / CPU_check_receipt_v1.json | Final bounded synthetic forward/backward and memory-witness evidence |
| source_manifest_v1.json / receipt_v1.json | New packet and inherited source pins, final verification scope |

Remote repository: /workspace-SR003.nfs2/ARG-GNN.
Packet: research/reviews/order2_cut_memory_implementation_20261015_v1/.
Use the remote pinned .runtime/env/bin/python -B; add repository root and src to its normal module path. These files are source artifacts, with no launcher or CLI authority.

## Model contract and public entrypoints

~~~python
from research.reviews.order2_cut_memory_implementation_20261015_v1.order2_cut_memory_v1 import (
    ARMS, CutMemoryV1, Order2CutMarkerGridV1,
    prepare_cut_plan_v1, prepare_order2_bundle_v1, make_matched_family_v1,
    logical_bounds_v1,
)
~~~

**prepare_cut_plan_v1(bank)** consumes only caller-supplied CPU native parent/date/sample/span tensors. It verifies complete packed trees, the declared sampled leaves, actual sample ages, root ownership, monotone dates and finite positive widths. Production preparation requires all 952 leaves and 686 outputs. Samples at age <=350 use their highest existing ancestor at or below 350; samples older than 350 retain their actual sample node as a singleton. A root already at or below 350 is a real root-saturated cut representative.

The selected groups form a disjoint partition. Canonical stable-leaf membership, rather than internal node IDs, identifies unchanged consecutive cuts. An unchanged native hold integrates its source/width while retaining the same pair layout. Pair layouts index the two most recent distinct cut partitions. Preparation records observation-only group sizes, distinct-cut events, older/root masks, pair-cell counts, conditional triple information and source fingerprints. It uses neither query ancestry nor targets.

**prepare_order2_bundle_v1(original_h048_grid)** adds the cut plan to the caller's unchanged original H048 CPU grid bundle. **Order2CutBundleV1** validates its exact source tensor/version binding, exposes the original engine's bank/marker/features hooks, and transfers the native grid and cut plan together. It does not rebuild bins, markers, coordinates, query sets or score support.

**Order2CutMarkerGridV1(arm=..., query_block=32, checkpoint_chain=True, checkpoint_memory=True)** runs the actual H042 first complete-tree PyG sweep, inherited local clade adapter, new compact cut injection, second complete-tree PyG sweep, inherited leaf context and full native-hold BiGRU. It directly calls the existing H048 decode_block_v1 method for the original full-grid graph_to_transition chain. All three arms use that same decoder placement and all 686 outputs. The new backbone parameter prefix is common.backbone.

**make_matched_family_v1(payload, checkpoint_ref=..., completion_ref=..., device=...)** receives an authenticated original H042 pass-30 payload from root. It reads no checkpoint file. The same seed-101 model initializes all three independent parameter sets; all common new heads, U, zero O and ell are identical. Each arm receives a fresh independent AdamW. The initializer is usable before any H048-trained checkpoint exists.

The generic complete-parent loss/update remains the original H046/H048 exact-span method. See the integration document for the concrete fresh cache/family/evaluation hooks and source closure.

## Three matched memory arms

| Arm | Event carry |
|---|---|
| triple_support_pair_memory | Lift old pair cells to stable leaves, then count-average into new pair cells: n_ABC / n_BC |
| adjacent_marginal_null | Lift/pool old cells within B, then copy that mean to each outgoing B,C cell: n_AB / n_B |
| stable_leaf_memory | Preserve rank-8 state at each observed leaf; regroup/read by the same current cut |

Each arm uses the same bias-free U:32->8 source from first-sweep upward+outside states at actual cut representatives, shared positive genomic scale ell in [100,100000] bp initialized at 5000 bp, width-aware exponential recurrence, and shared parameters in both directions. Boundary state initializes to its current forcing. Source integration uses -expm1 for accuracy at short widths. The current-cut read averages the left/right rank-8 states over observed leaves, subtracts current forcing and applies zero-initialized O:8->32. The compact injection is added at those representatives before the second tree sweep.

PyG SumAggregation performs the actual parameter-free reductions. Cut plans are prepared once and shared across the three sequential arm steps. There are no node-pair matrices or per-forward tree rebuilds. Tree traversal and grouping are linear passes over the native nodes and hold/leaf tables; uniqueness validation may sort. Rank-8 directional histories use O(T*H*8) storage and arithmetic, plus event reductions. Python native-hold loops remain; full-grid throughput has not been qualified. Memory and chain nonreentrant checkpointing are enabled by default.

## Parameter count and full-budget protocol

Per arm: original H042 body 30,788 + common H048 grid head/chain 1,145 =31,933; U 256 + O 256 + ell 1 =513 added memory scalars; **32,446 total**. PyG aggregation adds no learned parameters. All three models have this same count.

The inactive configuration fixes:

- 30 passes, the same 32 complete original TRAIN parents in the same per-pass shuffled order, 960 optimizer updates per arm and 2,880 overall.
- Full FP32, seed 101, independent reset AdamW, lr 0.001, weight decay 0.0001, betas 0.9/0.999, eps 1e-8, gradient clip 1.
- Checkpoints after passes 5/10/15/20/25/30: six groups and **18 actual checkpoint files**.
- The same eight original VAL parents, original exact-span scoring and common earliest-best selection rule. **All 144 complete original-grid probability products are sealed before any VAL truth opens.**
- Original 100-bp [P,686,4] support, no missing bins or tail extrapolation, no purity filter, no probability floor, no durable dense prediction arrays.
- No early stopping or VAL-driven optimization. Reused VAL is development evidence for one optimizer seed. Registered outer/TEST support remains unopened and unchanged.

The condition changes historical latent pooling while decoder placement is fixed. Generic operator priority is no-go under the earlier literature audit: this is incidence lift/project on an expanded memory state. Domain-specific pooling/denoising utility remains an empirical possibility. Stable full-leaf history is richer; the candidate carries no expressivity, information-dominance, ancestry-identification or biological-rate claim.

## Final bounded CPU evidence and retained failures

The final captured subprocess returned 0 in 12.14 seconds. It used one handmade seven-hold, 952-leaf, 686-query forest with seven synthetic SNPs, six distinct cut episodes A,B,B,A,B,C,A, and one actual sample aged 360 generations. It used real PyG complete-tree sweeps, the native BiGRU and inherited decoder, with 14 synthetic grid bins.

The final run verified identical [14,686,4] normalized finite outputs across all arms at O=0; one full-model CPU backward after opening O deterministically; connected finite U/ell/O and every existing parameter gradient; empty optimizer states; and source-version mutation rejection. Additional direct scalar witnesses within that same fixture showed positive triple conditional information and the intended true/null event difference, then later pair/leaf pooling difference.

Earlier fixture-development records remain intact. The first direct SSH attempt exited 1 without diagnostics (unknown cause). Captured attempt v1 reached full backward but its zero-SNP fixture did not connect the inherited site MLP; the synthetic input was corrected to include SNPs and v2 passed. The final fixture was strengthened because the earlier passing example had zero triple conditional information. Its v3 success supersedes the final receipt while all prior logs, waits, source snapshot and correction records remain. Each recorded full-model run had zero optimizer updates. There was no training pilot or equivalence campaign.

This verifies source behavior on the bounded fixture. Actual production observation bundles, authenticated real warm-start payload, bound full-grid device transfer, 20,000-bin runtime/allocator/VRAM, and full fits remain unqualified.

## Root completion path

Create fresh producer/context/driver/registration and bind the actual new/inherited source closure. Use the 33-row integration map: it includes the hard two-arm comparison unpack, common.backbone prefix, all parameter-count literals, all 96-seal fields, root 12-to-18 prediction/checkpoint forecasts, and all count-plan consumers. Do not mutate original H048 globals or sources.

Review new resource/recipe/output caps using actual observation counts. Raw cut-plan upper is89*T*952+952 bytes; preparation work estimates do not certify total RSS. The complete all-144 raw validation matrix is18*sum(P_parent)*686*16 bytes (31,610,880,000 bytes if every parent has 20,000 bins), excluding all other workspaces. First actual scheduled training steps provide the prescribed qualification without extra updates. Register the two planned paired contrasts and decision criteria before their full-budget evaluation. Then execute the authorized full schedule under root's actual guards and provenance.
