# Exact root commands

The following source-only commands are safe within this packet's scope and were used for metadata verification. Rerunning validation without--receipt creates no receipt overwrite.

```sh
cd /workspace-SR003.nfs2/ARG-GNN
P=research/reviews/ARGMix_prospective1611_native_adapters_20261016_v1
.runtime/env/bin/python -B "$P/validate_source.py"
.runtime/env/bin/python -B "$P/panel_adapter.py" verify-sources
.runtime/env/bin/python -B "$P/panel_adapter.py" summary --roster "$P/prospective_roster.unbound.v1.json" --unbound
```

The commands below are **future entrypoints, not executed or authorized here**. Root supplies a sealed common-data bank, complete mapping/targets/demography/map receipts, an admitted resource wrapper and fresh output paths. Numeric runtime/source ABI qualification is still external. No install or scheduler command is provided.

The existing interpreter locations below come from source constants, not runtime probes. AP fitting/prediction uses the existing qualified CPU TensorFlow2.15.1/Keras2.15.0 prefix. Root carries the qualified ARGMix library environment from its established launcher; the displayedLD_LIBRARY_PATH is the source-recorded wheel/prefix path.

```sh
RUN=experiments/ARGMix_prospective1611_native_v1
PARENT=published_recon_train_000
ARGMIX_PY=.runtime/official_argmix_abi210_v1/cu129/env/bin/python
AP_PY=.runtime/conda/envs/ancestralpaths_cpu_v1/bin/python
ARGMIX_PREFIX=/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/env
```

Bind the actual observed node mapping for each parent:

```sh
.runtime/env/bin/python -B "$P/panel_adapter.py" bind --roster "$P/prospective_roster.unbound.v1.json" --mapping "$RUN/inputs/$PARENT/observed_node_mapping.json" --output "$RUN/inputs/$PARENT/roster.bound.json"
.runtime/env/bin/python -B "$P/panel_adapter.py" validate-targets --roster "$RUN/inputs/$PARENT/roster.bound.json" --targets "$RUN/inputs/$PARENT/targets.jsonl"
```

Extract each native representation from identical observations. Repeat the same command shape for every registered TRAIN/VAL/IID/OOD parent:

```sh
LD_LIBRARY_PATH="$ARGMIX_PREFIX/lib:$ARGMIX_PREFIX/lib/python3.12/site-packages/torch/lib" "$ARGMIX_PY" -B "$P/native_runtime.py" extract --method argmix --roster "$RUN/inputs/$PARENT/roster.bound.json" --targets "$RUN/inputs/$PARENT/targets.jsonl" --trees "$RUN/inputs/$PARENT/observed.trees" --parent "$PARENT" --output "$RUN/inputs/$PARENT/argmix.features.jsonl"
"$AP_PY" -B "$P/native_runtime.py" extract --method ap --roster "$RUN/inputs/$PARENT/roster.bound.json" --targets "$RUN/inputs/$PARENT/targets.jsonl" --trees "$RUN/inputs/$PARENT/observed.trees" --parent "$PARENT" --output "$RUN/inputs/$PARENT/ap.features.jsonl"
.runtime/env/bin/python -B "$P/panel_adapter.py" paired-metadata --argmix-metadata "$RUN/inputs/$PARENT/argmix.features.jsonl.metadata.json" --ap-metadata "$RUN/inputs/$PARENT/ap.features.jsonl.metadata.json"
```

Construct external common biological labels once; both fit bundles reference the same output and contract. This separate command reads simulation truth and must not be embedded in feature extraction:

```sh
"$AP_PY" -B "$P/truth_adapter.py" --roster "$RUN/inputs/$PARENT/roster.bound.json" --targets "$RUN/inputs/$PARENT/targets.jsonl" --simulation-trees "$RUN/truth/$PARENT/simulation.trees" --simulation-mapping "$RUN/truth/$PARENT/simulation_node_mapping.json" --population-mapping "$RUN/truth/$PARENT/population_mapping.json" --demography "$RUN/truth/$PARENT/demography.json" --observed-metadata "$RUN/inputs/$PARENT/ap.features.jsonl.metadata.json" --parent "$PARENT" --output "$RUN/truth/$PARENT/labels.jsonl"
```

After all fiveTRAIN/oneVAL inputs exist, place completed bundles at the paths below (templates are included; adjust root-owned paths before freezing). The examples useCPU device strings so no GPU selection is implicit:

```sh
LD_LIBRARY_PATH="$ARGMIX_PREFIX/lib:$ARGMIX_PREFIX/lib/python3.12/site-packages/torch/lib" "$ARGMIX_PY" -B "$P/native_runtime.py" fit --method argmix --train-inputs "$RUN/train.argmix.inputs.json" --val-inputs "$RUN/val.argmix.inputs.json" --map-binding "$RUN/map_binding.json" --device cpu --output-dir "$RUN/argmix_fit"
"$AP_PY" -B "$P/native_runtime.py" fit --method ap --train-inputs "$RUN/train.ap.inputs.json" --val-inputs "$RUN/val.ap.inputs.json" --device /CPU:0 --output-dir "$RUN/ap_fit"
.runtime/env/bin/python -B "$P/panel_adapter.py" paired-fit-summaries --argmix-summary "$RUN/argmix_fit/summary.json" --ap-summary "$RUN/ap_fit/summary.json"
```

Predict identical threeIID target sets using label-free bundles; later OOD uses the same interface:

```sh
LD_LIBRARY_PATH="$ARGMIX_PREFIX/lib:$ARGMIX_PREFIX/lib/python3.12/site-packages/torch/lib" "$ARGMIX_PY" -B "$P/native_runtime.py" predict --method argmix --inputs "$RUN/iid.argmix.inputs.json" --checkpoint "$RUN/argmix_fit/best_model.pt" --fit-summary "$RUN/argmix_fit/summary.json" --map-binding "$RUN/map_binding.json" --device cpu --output "$RUN/argmix.iid.probabilities.jsonl"
"$AP_PY" -B "$P/native_runtime.py" predict --method ap --inputs "$RUN/iid.ap.inputs.json" --checkpoint "$RUN/ap_fit/best_model.h5" --fit-summary "$RUN/ap_fit/summary.json" --device /CPU:0 --output "$RUN/ap.iid.probabilities.jsonl"
```

Root's common scorer must join predictions and external labels byrow_id, retain every shared target, apply four-wayargmax, report correct/total per cohort/parent and paired method differences. An unavailable prediction must not reduce the denominator. Do not label these results unchanged-native AP or exact published-table reproduction.
