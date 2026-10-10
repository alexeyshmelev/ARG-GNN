# Current-release ARGMix biological pilot v1

This is a mechanical integration of author commit c2481248eb0de5fda659d68f59c51e0e27becd20 on the four completed development Relate parents. It is a current-release comparison, separate from April-paper reproduction. The immutable 86-file author tree has SHA256 4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8. Author model, trainer, Rust extraction, checkpoint selection and inference remain unchanged.

New files are src/arggnn/official_argmix_biological.py, scripts/train_official_argmix_biological_v1.py, configs/official_argmix_biological_pilot_v1.json and this document. Default invocation checks source/policy hashes only. This source handoff does not claim completed qualification, biological extraction, fitting or scores. Root owns execution and supplies factual build/CPU/GPU receipt paths/hashes.

## Observed frame and labels

Use bio_paths_pilot_train_000, bio_paths_pilot_train_001, bio_paths_pilot_val_000 and bio_paths_pilot_val_001 in the declared two-TRAIN/two-VAL partition, with post_five_size_iterations_primary trees only. Inferred manifest SHA: 6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28. Dataset manifest SHA: d404f4308466b09d7fa9a296b12bb76932bf84142bebbd03503cb2d2445665ea. Original/derived rosters, trees, coordinates and root manifests are rechecked across stages.

All 280 haplotypes retain actual inferred IDs and observed times. Only 96 pure references receive ANA/CHG/WHG/EHG tokens, 24 per class. The 120 queries receive generic QUERY metadata and stock <SAMPLE> input tokens. The 64 mixed_reference_only haplotypes receive UNLABELED_REF and are excluded from nearest-reference selection. Ancestor inputs and ancestry/population/individual graph annotations are forbidden. Source-observed node IDs are used only for external query-label correspondence.

The stock truth resolver assumes node ID = 2 * Relate.sample row + hap. Preparation verifies actual mappings, aliases, diploid order and tree samples against IDs 0 through 279, rejecting mismatch without reindexing. The native .sample includes all 140 diploids in verified order.

Clipped trees use local bp0, with original = local + coverage_left. External labels join at original observed SNPs; native SNP positions are exact local int64 bp0. Inference projects onto declared retained original 100bp centers; bin edges are clipped to coverage, physical normalization remains 2Mb, and uncovered tails are not extrapolated. The observed uniform 1cM/Mb map is translated to local origin: header "position Genetic_Map(cM)", then "0 0" and "2000000 2". Native span/delta features are invariant to that translation.

External C4 truth capabilities are TRAIN gradients, native development VAL selection without gradients, and descriptive external VAL scores after both prediction products are sealed. Labels never enter tokens, graph annotations, reference/query eligibility or fitted normalizers. Native first_snp labels and no-purity-threshold behavior remain unchanged; native supervised extraction may omit intervals without usable SNP labels.

## Native behavior and scores

C4 output order is 0 ANA, 1 CHG, 2 WHG, 3 EHG. Stock sorted vocabulary ordering and class remapping are separate. Defaults are 15 references, hidden128, four layers/four heads, FFN multiplier4, final-feature16, transformer dropout0.25, attention dropout0.1, relative distance14 and stock log TMRCA scaling. The removed num_ancs option is forbidden.

Chunks use 1024 rows/stride512. GRU hidden128 is bidirectional, one layer, configured dropout0.1 and actual recurrent dropout0. Archives store two map channels; default GRU consumes only raw cM-span (include_delta_cm_prev=false). No map normalization is fitted. TRAIN requests16 nearest refs, removes the first and keeps15; VAL/inference drops0. TRAIN/selection excludes edge trees, uses fast backend and disables merging. Source integer age truncation and genealogical time/distance transforms are retained.

Unchanged training seeds50, batch16, up to20 epochs, AdamW LR2e-4/decay1e-4, smoothing0.1, weighting none and patience2. Native selection averages masked smoothed losses within VAL batches, then averages batch means without valid-row weighting; overlap rows count repeatedly. Strict improvement selects a state, restored and saved after the loop. External full-support scores do not replace native selection.

Joint training calls transformer forward_features and GRU classification. The standalone transformer classifier receives no gradients and stays unchanged; base probabilities are not a trained ancestry result. The trained GRU output is primary. The bridge checks changed selected model weights and unchanged unused base classifier.

Stock export uses logits_and_final. Rust HDF5 base probabilities/logits/features are uint16 half-float bits, decoded by source loader. GRU inference retains this quantization. Native refinement averages per-chunk softmax probabilities across overlaps. The bridge saves original FP32 refined per-chunk logits, indexes/masks and requires source-helper replay error <=1e-6 against streamed stock refinement. It never calls log averaged probabilities original logits. Source standalone feature chunking stops when a full chunk exactly reaches group end, while source streamed refinement emits a further 512-row trailing overlap for groups of length >=1024 divisible by512. Replay adds that exact native trailing geometry using unchanged helpers and restores the stored delta channel across chunk boundaries. Input order must match native chronological grouping. The default still consumes only span; the probability parity gate remains mandatory.

Inference alone uses exclude_edge_trees=false to request full support; TRAIN/selection defaults stay unchanged. Gaps are preserved. Both VAL grids, stock HDF5 products and chunk logits are hashed/sealed before scoring. NLL, Brier, accuracy,15-bin ECE and C4 confusion use exact physical bin/truth intersections for present64, ancient56 and all120. No purity/eligibility filter, probability floor or denominator reduction is used. Any missing query-bp leaves full-support metrics unavailable while reporting full denominator/missing mass. Zero true-class probability is explicitly infinite NLL with affected mass. Parents are independent units; these two VAL parents support development descriptions, not final TEST or confirmatory claims.

## Root launch commands

Run remotely from /workspace-SR003.nfs2/ARG-GNN. Review source-only audit and actual passed build/CPU FFI receipts. Use the successful build phase receipt, not a pending sequence completion. v5 is root's current attempt; a later reviewed completed attempt can supply its own factual paths.

~~~bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/train_official_argmix_biological_v1.py

ARGMIX_BIO_CONFIG_SHA=CONFIG_SHA_FROM_SOURCE_AUDIT
ARGMIX_BIO_BUILD=experiments/results/argmix_native_abi210_execution_v1/v5/build/receipt.json
ARGMIX_BIO_BUILD_SHA=ACTUAL_COMPLETED_BUILD_RECEIPT_SHA
ARGMIX_BIO_CPU=.runtime/official_argmix_abi210_v1/cu129/admission/v5/receipt.json
ARGMIX_BIO_CPU_SHA=ACTUAL_PASSED_CPU_RECEIPT_SHA
ARGMIX_BIO_RUN=experiments/results/official_argmix_biological_pilot_v1
ARGMIX_BIO_GPU="$ARGMIX_BIO_RUN/gpu_qualification/receipt.json"
ARGMIX_BIO_GPU_SHA=''
mkdir -p "$ARGMIX_BIO_RUN"

argmix_bio_gpu() {
  ARGMIX_BIO_JOB="$1"
  shift
  PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/launch_gpu.py \
    --job "$ARGMIX_BIO_JOB" arggnn.official_argmix_biological \
    --execute --expected-config-sha256 "$ARGMIX_BIO_CONFIG_SHA" \
    --build-receipt "$ARGMIX_BIO_BUILD" --expected-build-sha256 "$ARGMIX_BIO_BUILD_SHA" \
    --cpu-admission "$ARGMIX_BIO_CPU" --expected-cpu-admission-sha256 "$ARGMIX_BIO_CPU_SHA" \
    --gpu-admission "$ARGMIX_BIO_GPU" --expected-gpu-admission-sha256 "$ARGMIX_BIO_GPU_SHA" "$@"
}
argmix_bio_cpu() {
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
    scripts/train_official_argmix_biological_v1.py \
    --execute --expected-config-sha256 "$ARGMIX_BIO_CONFIG_SHA" \
    --build-receipt "$ARGMIX_BIO_BUILD" --expected-build-sha256 "$ARGMIX_BIO_BUILD_SHA" \
    --cpu-admission "$ARGMIX_BIO_CPU" --expected-cpu-admission-sha256 "$ARGMIX_BIO_CPU_SHA" \
    --gpu-admission "$ARGMIX_BIO_GPU" --expected-gpu-admission-sha256 "$ARGMIX_BIO_GPU_SHA" "$@"
}
argmix_bio_gpu argmix-bio-v1-qualification --phase qualify-gpu \
  --output "$ARGMIX_BIO_RUN/gpu_qualification"
~~~

Qualification ignores the empty prior GPU hash. It performs one synthetic full source joint forward/backward with finite gradients, no optimizer/weight change, then UNTRAINED stock checkpoint/export and actual Rust TorchScript GPU fixture/device comparison. It is not biological or a fit. After review, fill the actual passed GPU hash before subsequent stages:

~~~bash
ARGMIX_BIO_GPU_SHA=ACTUAL_PASSED_GPU_RECEIPT_SHA
argmix_bio_cpu --phase prepare-observed --output "$ARGMIX_BIO_RUN/observed" \
  --expected-observed-manifest-sha256 6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28 \
  --expected-dataset-manifest-sha256 d404f4308466b09d7fa9a296b12bb76932bf84142bebbd03503cb2d2445665ea
~~~

Pass the stage product observed/observed_frames.json, not enclosing receipt.json. CPU native preparation opens external TRAIN/VAL supervision, adapts stock SNP NPZ labels and runs unchanged build_training for train/val. Native archives are NPZ format2 despite .pt names:

~~~bash
ARGMIX_BIO_FRAMES_SHA=ACTUAL_OBSERVED_FRAMES_SHA
argmix_bio_cpu --phase prepare-native --output "$ARGMIX_BIO_RUN/native" \
  --input-receipt "$ARGMIX_BIO_RUN/observed/observed_frames.json" \
  --expected-input-sha256 "$ARGMIX_BIO_FRAMES_SHA"
~~~

After reviewing native/native_prepared.json, fit unchanged source. Checkpoint epoch/loss live under joint_model.epoch and joint_model.best_val_loss:

~~~bash
ARGMIX_BIO_NATIVE_SHA=ACTUAL_NATIVE_PREPARED_SHA
argmix_bio_gpu argmix-bio-v1-stock-fit --phase train --output "$ARGMIX_BIO_RUN/fit" \
  --input-receipt "$ARGMIX_BIO_RUN/native/native_prepared.json" \
  --expected-input-sha256 "$ARGMIX_BIO_NATIVE_SHA"
~~~

After reviewing fit/stock_fit.json, run stock inference, seal both VAL products, then score:

~~~bash
ARGMIX_BIO_FIT_SHA=ACTUAL_STOCK_FIT_SHA
argmix_bio_gpu argmix-bio-v1-stock-inference --phase infer-and-score \
  --output "$ARGMIX_BIO_RUN/inference" \
  --input-receipt "$ARGMIX_BIO_RUN/fit/stock_fit.json" \
  --expected-input-sha256 "$ARGMIX_BIO_FIT_SHA"
~~~

Native execution uses isolated Python3.12/Torch2.10+cu129/torchvision0.25. Actual loaded ancpath mappings match hashed build artifacts; source-copy bytes and libtorch/libc10 wheel paths are checked. GPU phases verify inherited scheduler lease, current-driver UUID, logical GPU, exact "AI Auto Research Pipeline Test" title and own NVTOP compute PID. The frozen CUDA12.4 UUID helper is not reused. All IPC TMP/cache/log/output paths are repository-contained. The runner installs nothing and rechecks source/policy/native/lease bindings after stages.

Inspect actual receipts before advancing. Fresh outputs retain commands, hashes, failure traces and exit statuses; retries use fresh names. Support gaps, export/replay mismatches and missing files remain visible. No final freeze, held-out preparation or final TEST read is performed.
