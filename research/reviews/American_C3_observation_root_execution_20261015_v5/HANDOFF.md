# C3 explicit unary-collapse reconstruction: observe_v3 / root_v5

Source-only implementation of root's newly authorized reconstruction policy. Root must register this condition before actual execution, then run one100kb qualifier and proceed to full30Mb on verified success. No new biological inference, dataset, no-op/model test, GPU, guard configuration, overlay state read, policy activation, credit or admission was executed by this agent.

## Scientific correction and source basis

The corrected handle-lifetime/eight-thread condition reached tsdate, whose installed variational default rejects unary nodes. The earlier v3 short root run exited1 with `Tree sequence contains unary nodes, simplify first`; its store, raw inferred tree, output and wait remain preserved.

Installed tsdate/variational.py explicitly recommends simplification. tsdate.preprocess_ts also simplifies, but its defaults erase flanks, delete1Mb gaps and split disjoint nodes. Those changes are outside root's policy. The implementation therefore uses exactly the authorized direct tskit1.0.3 call:

    collapsed,node_map=inferred.simplify(
        samples=inferred.samples(),keep_unary=False,filter_sites=False,
        filter_individuals=False,filter_populations=False,map_nodes=True)

No allow_unary=True bypass, flank trimming, gap deletion or splitting is used. The node map has an entry for every raw inferred node; removed nodes have the documented-1 value. All sample IDs are joined through that map in original phased genotype-column order.

The unsimplified inferred topology is retained as `observed/tsinfer_sanitized.trees` before simplification. Metadata sanitization preserves its node IDs/topology/times/sites/genotypes. The old-to-new map is retained as `observed/tsinfer_unary_old_to_new_node_map.npy`. A versioned `observed/unary_preprocessing.json` records policy, raw tree/map refs, raw/collapsed sample-node vectors, counts and all preservation flags. No third tree file is needed.

Before the unchanged tsdate.date defaults run, the worker verifies identical sample count/order, sequence length, all original site positions and every phased genotype by allele string. The original complete verification repeats after dating/sanitization. Original observed input files, source-reference labels, simulation roster and private truth remain unchanged. The typed SampleData reader/explicit-close correction and tsinfer num_threads8 remain in force.

Policy ID is `direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1`. Configs, output manifest and completion record it. This is a declared reconstructed recipe; exact paper inference options remain unresolved.

## Root CLI

Worker: `src/arggnn/american_c3_observe_v3.py`.

Root driver: `research/reviews/American_C3_observation_root_execution_20261015_v5/root_observation_driver.py`.

Only root invokes admission/execution. From `/workspace-SR003.nfs2/ARG-GNN`:

    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v5/root_observation_driver.py --preflight qualification100kb --policy research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json --policy-sha a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85
    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v5/root_observation_driver.py --admit qualification100kb --policy research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json --policy-sha a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85
    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v5/root_observation_driver.py --launch qualification100kb --policy research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json --policy-sha a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85

After actual short success, repeat this sequence with full30mb. APIs are preflight_v5(stage,policy_ref), admit_v5(stage,policy_ref), launch_v5(stage,policy_ref). No-flag/--source-report is inert. Preflight validates metadata/hashes without guard configuration, journal replay, arrays or inference.

Fresh packets are admitted_qualification100kb_v5 and admitted_full30mb_v5; each reserves an initially absent worker_aux for logs/cache/temp. Fresh outputs are experiments/results/American_C3_observed_STAGE_20261015_v5. All older packets, journals, reservations, failures and outputs remain unchanged.

## Reviewed resource overlay

The driver binds `concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py`, exact SHA256 `ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef`, and the exact active policy ref above. Root already owns policy activation and genuine completed-scope credits. This source task does not establish any credit or available capacity.

Only future root admit/launch configure the adapter via configure_v2(repair_ref,policy_ref=...). Admission uses its seven-part _state_locked return, all state[3] bindings, operation_draft_v1 reserve request and actual reserve receipt. Storage binds completed_overlay_policy_ref and keeps the receipt's initial_ledger_pins containing its own reserve-event prefix. It does not rewrite those pins after unrelated overlay appends.

The unchanged common worker receives the exact JSON policy record in ARGGNN_COMPLETED_CREDITS_POLICY_REF; commoncli's one-argument configure_v2 therefore configures the same overlay/policy. The source/config/runtime, guard/policy/marker/repair closure, driver/helper and input metadata are reserved directly as .py/.sh/.json refs only. NPY/JSONL inputs and map/tree products remain transitive data. Operation/root metadata stay within64KiB.

Both stages are CPU-only with CUDA hidden. Policy's GPU4/5 restriction remains unchanged and no device is assigned here. Existing active GPU45 scopes keep their original guards/caps.

## Bounds and gate

| Stage | Original observed data | Store cap | Each tree-table cap | Node-map cap | Total planned bytes | Reservation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qualification100kb |244 sites ×64 haplotypes /100kb |250,000,000 |250,000,000 |250,000,000 |1,202,375,168 |1,250,000,000 |
| full30mb |226,185 sites ×1,600 haplotypes /30Mb |4,000,000,000 |2,000,000,000 |2,000,000,000 |10,202,375,168 |12,500,000,000 |

Serialization formula is store +2*(tables+16MiB) +node_map_cap +16MiB; total planning adds128MiB private cache/temp,16MiB log and1MiB metadata allowance. SampleData max_file_size and explicit source/table/map/write checks remain enforced. Private temporary/cache growth is monitored with sampled enforcement. RAM remains unmeasured. Qualification walltime600s, full75,600s and fresh22h authority remain the original operational settings.

Full admission requires this exact v5 driver/source/config/runtime's short direct worker wait exit0, no operational failure, actual completed inferred identity, original64-column roster, preserved input refs and complete declared preprocessing receipt. Failed earlier qualification conditions cannot satisfy the gate. Full execution introduces no second scientific qualifier.

## Source verification

Both new sources compiled. Static symbol checks found no undefined global references. The source qualifier verifies the exact requested simplify keyword set and unchanged tsdate.date keyword set. Read-only preflights validate all four original input-file hashes and NPY headers. Short metadata prerequisites are ready under the exact policy; full remains gated on actual new short success.

No actual simplify, infer, date, model test or data rewrite was performed. The root's next short run establishes operational validity of this new condition.
