# Complete one honest temporal-graph benchmark for the current TreeLoom core

2026-10-17. **Use the existing Hungarian Chickenpox task for a new, explicitly adapted current-core comparison. Decline an ordered-memory USPTO adaptation.** This packet supplies callable Torch/PyG model, data, training and heldout-prediction interfaces. It is source-only, unregistered and not an execution authority. No numerical module, payload or model was executed while authoring it.

## Decision from the existing record

H010 already answered the old anchor/boundary applicability pilot negatively. Its ordinary GRU selected validation MAE is16.281982 cases, versus anchor-full depth1 16.83561, depth2 17.24698 and boundary-full18.019365. Those are historical development results, not current TreeLoom results or TEST scores. The old seed101 GRU selected epoch16 and has a retained checkpoint digest in its compact receipt. The retained graph-GP validation MAEs are17.619417,17.675065 and17.702193 cases for nonseparable, separable and temporal variants. This proposal does not replay any of those fits.

The new condition is the final core's **two weight-shared graph stages, rank-eight stable-identity memory between them, and recurrent readout**, versus its matched NoMem version, under a fixed external-forecast adaptation. Neither current arm is present in the inspected Chickenpox pilots. Three predeclared initialization seeds and completion of the reserved79-week TEST endpoint are also new. Register it as a new condition linked to H010, not as an overwrite or reversal of that negative result.

H022 completed four genuine native GST modes on the first1000 retained USPTO TRAIN reactions and all1495 VALID reactions. Native gsc-bc pooled-atom AP is.65378329, versus GCN-DS.63945129 and GCN-ST.63823432, one native seed0. The latest ledger also records full TRAIN retention10439/10500, but not a full new fit or our candidate score. Those native results and original source must remain separate from earlier independently reconstructed GST-style biological controls.

USPTO reactant components form an unordered collection. Atom map IDs are reaction-local correspondence keys, not persistent physical entities across reactions. Making a component order or carrying atom states across arbitrary reaction rows would manufacture a temporal problem. A mapped product graph is an outcome rather than an allowed native reactant input, so it cannot supply a leakage-free second snapshot. The tree-specific model also lacks a fair molecular input/head/interface. This packet supplies no USPTO score, atom-identity fiction, “native GST” reimplementation or claim that Chickenpox discharges H022's original-paper molecular milestone. The genuine author GST branch stays unchanged. GST is not inserted as a native forecasting competitor because its released molecular task supplies no native county forecasting protocol.

## Dataset, features, target and split compatibility

Retain the pinned UCI CC BY4.0 archive, SHA256 dc306b1845c86616d2a72c202deeefac8fd1fcf0a34b3067cf6721e377d9da44, original county-name join,522 actual weekly rows and20 county identities. The source page's521-instance discrepancy remains disclosed. Use the existing loader rather than a new archive decoder. The original graph has102 directed records, including20 self loops and41 undirected non-self county links.

For each target week t, input exactly counts[t-52:t]. Standardize each county using mean and population SD from rows[0:365] only, with zero SD replaced by1. County identifiers are fixed column metadata, not learned numeric features. There is one count feature per node. No ancestry, reference class, inferred age, tree cut, clade, label, future count, weather or population feature is invented.

| Purpose | Zero-based target indices, exclusive right endpoint | Forecasts |
|---|---|---:|
| TRAIN | [52,365) |313|
| VALID | [365,443) |78|
| Reserved TEST | [443,522) |79|

Development preparation makes rows443 onward NaN and read-only before any training/scaling calculation. Each model input separately records every numeric content week and rejects content at/after its target. Per-window state resets. Earlier validation/test counts may enter later rolling-origin histories after their observation date, with frozen weights/scaler. This is not a79-week open-loop forecast.

The scalar target is next-week raw cases. Training uses standardized MSE, while selection/scoring invert the fixed scaler. No softmax, discretization, positivity clipping or target graph is used. Every county and target week remains in the denominator.

## The exact missing interface, and the disclosed replacement

The final ancestry class requires a complete dated parent forest, sampled-node indices, cut representatives, clade flags, episode metadata and a fixed class/reference layout. Geographic graphs contain cycles and none of those ancestral objects. Calling that class with a fabricated forest would change the observations and conceal the mismatch.

The new callable class is TreeLoomGenericForecastV1 in generic_core_v1.py. It is a clean fixed-entity operator port, not a wrapper claiming the ancestry class accepts counties.

| Current final stage | This fixed-geography port |
|---|---|
| Genotype/reference/date-to-node features | Linear/tanh projection of the one training-standardized count feature |
| Complete upward/outside ancestral traversal | A declared two-field generic PyG mean-neighbor block. It uses every county and geographic edge, but has finite local spatial reach rather than a full ancestral traversal. |
| Two graph stages with shared weights | The exact same graph_pass module is called twice, first on base features and then on base plus residuals. |
| Own-state rank-eight residual | Same algebra O(Uh·Vh+Wh),1024 weights and zero output initialization. Repeated county identity replaces the eligibility condition. |
| Gained/lost clade flags | Omitted because fixed geography has no supplied clades. No zero-valued fake genealogy is promoted to a biological feature. |
| Stable-leaf memory,513 parameters | Same U32→8/O8→32 plus one length parameter, same forward/reverse exponential identity carry, average and current-forcing centering. Singleton groups are physical counties. |
| Genomic distance clock | Weekly durations; length initial5weeks, bounded.1–100weeks, chosen prospectively without current-core scores. No demographic-rate interpretation. |
| One recurrent layer per direction and local/recurrent concatenation | Same widths/input35 and one layer per direction. Both terminal states encode the entire already observed52-week history. |
| Per-position ancestry classifier | Scalar external forecast head: last local context plus both observed-history terminal states. |

Native self loops are removed from the neighbor list because the generic block has an explicit self route. All82 non-self directed records remain. The original102-record representation is retained separately for the legacy comparator. No node or edge is sampled, and numeric county IDs never enter a projection.

For first-stage fields u_t,o_t, the stable-entity forcing is f_t=U(u_t+o_t). In either direction, the first state equals its forcing and subsequent states obey

    m_t = exp(-Δt/ell)m_previous + (1-exp(-Δt/ell))f_t,
    delta_t = O((m_forward,t + m_reverse,t)/2 - f_t).

Reverse processing sees only weeks preceding the external forecast target. It is a completed-history encoder, not a claim of online causality at an earlier internal week. There is no carried state between forecast windows.

NoMem has identical common weights, architecture, inputs and head, with only this513-weight memory module absent. Its common initialization is copied from Full after seed initialization. Both initially coincide because the new memory output matrix is zero. Analytic parameter counts are24,865 for NoMem and25,378 for Full, to verify by future instantiation. No ancestry weights or validation-selected H050 checkpoint are transferred.

These adaptations preserve the specified core mechanism, while changing the task-specific representation, graph kernel, clock and head openly. A result concerns this generic port, not literal unchanged-model transfer, full ancestral-cavity computation or changing-topology performance.

## Callable completion path

completion_v1.py has a default metadata-only CLI with no execute option and no eager Torch/NumPy/dataset/guard imports. ROOT's separately admitted caller can use:

1. load_development_v1(archive_path): verify through the existing pinned loader, mask TEST and fit TRAIN-only scaling.
2. fit_development_pair_v1(prepared, seed=..., boundary=..., checkpoint_sink=...): fit both arms for one fixed seed, returning selected metadata through ROOT's mandatory bounded sink.
3. After all six selections/configurations are frozen, load_test_v1(..., release_selections=...).
4. Load each selected state through ROOT's authenticated checkpoint path and call predict_test_v1. Its numeric input builder reads only[t-52:t].
5. Seal all six complete chronological prediction products, then call score_test_v1(..., release_prediction_seals=...).
6. Run the already frozen classical/legacy comparators and chronology diagnostic without retuning.

Callbacks carry an explicit integration boundary. Their small metadata checks are not authority or storage guards. ROOT must authenticate real references, enforce current policy/resources and provide the bounded sinks. This packet neither imports a guard nor supplies a new lease/accounting policy or process launcher. No training CLI can bypass those callbacks.

The model interface accepts ObservedHistoryV1: FP32[B,52,20,1], local non-self edge_index, positive weekly durations, fixed county names and target/content-index receipts. Targets remain external. All52 snapshots are packed for PyG processing, with no graph pooling that discards per-county outputs. A CPU-only consumer is proposed for this small graph, one Torch thread and no CUDA discovery. No CPU duration, memory fit or parameter count was measured.

## Frozen matched recipe and practical comparators

Use seeds101,202,303. Each arm completes40 epochs, batch16, all313 TRAIN forecasts,20 updates/epoch and800 updates/arm/seed. This is6 new fits and4800 updates. Fresh AdamW(.001, weight_decay.0001), norm1 gradient clipping, FP32 and no early stop. Random(seed+epoch) gives the same batch order to both arms. Select minimum raw-unit VALID MAE every epoch, with earliest exact tie. No TEST or boundary/shuffle selection is allowed. Failure is recorded, not replaced with a smaller history, fewer counties or a cheaper model.

This batch/fixed-epoch recipe differs from the old one-window/early-stopped pilot. Historical validation scores are therefore descriptive practical references, not equal-optimization causal controls. Full-versus-NoMem is the matched mechanism comparison.

Last-value and lag52 controls use identical allowed observations. The ridge helper reconstructs TRAIN-only coefficients with alpha1000, already selected in the old validation pilot at MAE21.040020 cases. It performs no new alpha search. Reconstructing unretained deterministic coefficients for the previously unopened TEST endpoint must be registered as completion, not presented as a new fit result.

The old selected GRU checkpoint is the strongest inspected ordinary practical reference. Its existing metadata pins SHA2560a311637a0e011a60e691a53aff2088e5d296b3c512b0b2dd77545a4a57b10bc. Reuse its accepted weights with no optimization only after independent constructor compatibility authentication. The current study_models.py hash differs from the old training audit, so this packet does not assert that an unreviewed current constructor is identical. predict_verified_legacy_gru_v1 accepts ROOT's independently verified old model and uses the exact original GraphSequence/102-edge representation. A strict state load alone is insufficient authentication. If that source lineage cannot be established, report the comparator unavailable rather than silently refitting or substituting it. The matched core completion can still be reported, with no broad-superiority claim.

## Decision, uncertainty and limits

Primary endpoint is raw-unit TEST MAE across79weeks/all20counties. Also report RMSE, county/week errors and the mean/sample SD over three initialization seeds. Seeds are not independent populations, and counties/overlapping histories are not independent replicates.

The fixed utility rule is Full mean TEST MAE at most.98×NoMem, upper95% paired block interval for Full-minus-NoMem MAE below0, and mean RMSE no worse. The supplied interval callable averages paired errors across seeds/counties, then uses10,000 common circular13-week block draws, seed7721. It describes conditional temporal uncertainty on one time series, not broad deployment certainty. Also show all comparators and all seed outcomes even if the gate fails.

The optional frozen chronology diagnostic permutes the first51 known weeks using Random(9031+t), holding the last known week fixed, identically across methods. Its multiset, physical county identities and available observations stay unchanged. Apply only after selections are fixed and report reliance on chronology, not universal necessity or an arbitrary-order advantage.

Failure ends this condition and completes a negative benchmark. Passing supports fixed-topology forecasting applicability and a stable-identity-memory utility finding conditional on this port. It establishes neither new operator priority, arbitrary graph-set superiority, biological transfer nor a changing-topology result.

## What is complete, and what remains before execution

Source contains the generic model, observed-history builder, two-arm optimizer/selection loop, separate TEST release/prediction/scoring interfaces, fixed classical/ridge controls, verified-legacy hook, chronology intervention and paired block interval. A qualification checklist records the required synthetic numerical checks. The source was parsed/compiled without importing its numerical module. No numerical qualification, raw archive access, checkpoint loading, prediction or score was performed.

Before any numerical work, ROOT needs independent source review, explicit new-condition registration, full runtime/source closure, synthetic qualification and current bounded CPU/storage admission. Only current GPU0 eligibility and the current1TB project limit may inform any later resource choice. Old scheduling/storage sections in inherited docs do not supply authority. No H059/H060, native GST, current job, guard, README, ledger/decision or Git file is changed.
