# Reusable API and integration boundaries

Remote module: `/workspace-SR003.nfs2/ARG-GNN/research/reviews/H051_conditional_innovation_design_20261016_v1/conditional_innovation_v1.py`. This is not installed into arggnn. Load by path and insert the module into sys.modules before exec_module for dataclasses.

## Dimensions and controls

- `a,b,c`: aligned CPU integer vectors [H], one observed membership/persistent witness. Anonymous cut-key renaming is allowed. Permute leaves, state and optional witnesses together.
- `make_plan_v1(a,b,c,provenance=record)` builds occupied AB/B/BC cells and ABC intersections. Returned unique-cell order is authoritative.
- `state`: [K_AB,*F] with `basis="ab"`, or [H,*F] with `basis="leaf"`. Arbitrary trailing dimensions, including scalar features, are allowed.
- Output: [K_BC,*F]. Broadcast to persistent leaves with `out.index_select(0,plan.leaf_bc)`.
- `components` returns `triple,marginal,innovation` on BC cells. Leaf-expanded innovation equals J on the leaf input.
- `batch_plans_v1(plans)` concatenates disjoint ragged events. Concatenate states in event order. No padding/cross-parent mixing. Each event_sizes entry is (H,K_AB,K_B,K_BC,Q).
- Build plans on CPU. Transfer explicitly with `plan.to(state.device)`. Cooperative callers keep tensors immutable. Stamps detect ordinary in-place changes, not malicious data manipulation/fabricated constructors.
- FP32 is the diagnostic feature dtype. Source also accepts float64/FP16/BF16, with FP32 accumulation for low precision. The receipt verifies FP32 features, not accelerator mixed precision.

The exact identity-free source record is:

    provenance = {
        "schema": "H051_observation_only_cut_source_v1",
        "observed_trees_sha256": authenticated_observation_sha256,
        "builder_sha256": reviewed_builder_sha256,
        "targets_or_outcomes_used": False,
        "identity_namespace_sha256": None,
    }
    plan = module.make_plan_v1(a,b,c,provenance=provenance)
    triple = module.ConditionalInnovationTransportV1(gate="fixed",alpha=1.0)
    marginal = module.ConditionalInnovationTransportV1(gate="fixed",alpha=0.0)
    fixed = module.ConditionalInnovationTransportV1(gate="fixed",alpha=0.5)
    count = module.ConditionalInnovationTransportV1(gate="count",prior_count=16.0)
    parts = triple.components(state,plan,basis="ab")
    out = count(state,plan,basis="ab")

Use `backend="pyg"` for native SumAggregation. No learned parameters exist. Fixed/count reject hidden endpoint witnesses. The transport is unit-free and reusable with any fixed witness set. A country diagnostic needs its own causal observation builder/protocol. A forecast must not derive c, membership or features from current/future target edges or same-year labels. This package contains no such data loader/run.

## Optional identity gate

Set an authenticated identity namespace in each event contract. Pass `EdgeWitnessV1(before,after,namespace_sha256)` with int64 endpoint tensors [sumH,2] containing (parent global key,child global key), and a tuple with one namespace/event. Call `ConditionalInnovationTransportV1(gate="edge_continuity",prior_count=16.0)` with witness=sidecar.

Pairs are actual cut-crossing edges in last native tree before and first after B-to-C. Root/no-edge is exactly(-1,-1). Builder lineage must verify native interval coverage and joins. Global bijective renaming preserves output. No edge-row IDs, calibrated ancestry-confidence claim, clade/time matching or packed-index aliases are accepted. The API cannot verify tskit interval coverage from key tensors. Default Relate plus the current packed bank does not establish useful shared internal identity.

## Chain and clocks

At a genuine cut change, old AB memory moves to BC before a common feature update/readout. Bootstrap the first two cut episodes identically across controls. Unchanged-cut native holds retain their distinct observed forcing/durations, without an extra projection solely because another tree record exists.

`relax_interval_v1(state,forcing,distance,length_scale,distance_unit="bp",scale_unit="bp")` needs equal state/forcing full shape/device/dtype. Distance is an observed nonnegative scalar or [rows] vector, scale a positive scalar optionally differentiable. No observed-distance gradient is allowed. `new_cell_event_distances_v1(plan,distances)` expands one event distance to [sumK_BC]. Relax only on a shared carrier.

`interval_morgan_distances_v1` requires CPU float64 map bp/cumulative Morgans and [M,2] contained bp intervals, with no extrapolation. The prospective gate experiment stays in bp. No implicit scan, data-window policy, reset, detach, truncation, readout, optimizer, warmstart reader or label resolver is implemented. Future wrappers must declare these and preserve complete native trees/vertices/leaves.
