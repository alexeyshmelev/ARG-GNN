# Native ARGMix biological v1 — independent source audit

Verdict: **passed source review; no blocking static integration finding**. GPU qualification, biological preparation/extraction, fitting, stream probability parity and scores remain pending actual root execution.

Scope: source/config/document/producer-audit inspection, AST/in-memory compilation and source-byte reconciliation; actual v6 build/CPU admission JSON metadata was read. No helper, native or data stage executed; no biological/model/score/truth arrays were opened; no GPU, installation or Git operation occurred. Producer files remain untouched.

## Frozen identity and unchanged author behavior

All four requested source/config/doc SHA256 values match the producer receipt. The original86-file author inventory hashes to4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8; every corresponding execution-copy file matches. Frozen lease/NVTOP helper hashes also match. Full evidence is source_reconciliation.json.

The bridge constructs the current author's released template model and joint-training settings:15 references,128 hidden width,4 layers/heads, final feature16, original dropout/relative-distance/log-TMRCA behavior,1024/512 chunk geometry and bidirectional one-layer GRU. Template argmix.yaml sets20 epochs, batch16, LR2e-4, decay1e-4, smoothing0.1 and patience2; the source function's fallback epoch value is not substituted for that declared template. One raw span channel is active; two channels are stored. No median normalization is fitted. Train nearest-reference augmentation drops one of16 to retain15, while VAL/inference drop0.

run_author executes the exact source script main, without replacing trainer/model functions. Source train_joint_model seeds50, uses TRAIN gradients and no_grad VAL selection, averages masked smoothed losses within batches then averages batch means, counts overlapping rows repeatedly, uses strict improvement/patience2, restores best_state and saves its own checkpoint. Bridge acceptance requires changed selected weights, finite nested joint_model.best_val_loss/epoch and unchanged unused standalone base classifier. External full-support scoring never replaces that native selection (bridge323–390,696–738; author train_joint_model467–588).

## Observation and supervision boundary

frame_prepare requires the exact four frozen completed parents and splits, ordered280 actual inferred/sample IDs, Q120/R96/mixed64,24 pure references per class, null query/mixed source classes, positive observed times, exact diploid pairing and unique140 subject IDs. It rejects incompatible node IDs instead of reindexing: stock NPZ truth mapping uses2*Relate.sample row+hap. Tree samples/times/local length and coverage must agree; ancestry/population/individual graph annotations are forbidden.

Pure references alone receive C4 source tokens; queries receive QUERY/<SAMPLE>; mixed64 receive UNLABELED_REF and are outside the selected reference classes. Numeric source-observed IDs are only external target joins. Coordinate inputs preserve original positions through local+coverage_left, the declared100bp centers, clipped bin edges and original2Mb normalization. The uniform observed map is translated in origin without changing its span/delta features (bridge393–520).

external_truth is separate from observed features and checks full2Mb contiguous C4 intervals plus exact ordered120 raw query-node identities. TRAIN labels are joined at observed original SNPs for gradients; VAL labels are used by unchanged source checkpoint selection without gradients. They do not choose feature tokens/reference/query eligibility or normalizers. Stock first_snp labeling/no-purity-threshold and its omissions remain native behavior. This regime legitimately reads development VAL supervision before later descriptive scoring; it is not an untouched holdout (bridge523–632; author npz_truth163–186,265–299).

## Exact stock streamed replay and half bits

The author standalone build_joint_feature_chunks stops when a full chunk reaches the group end. The stock HDF5 streamer emits full chunks while available, advances by512, and flushes the remaining buffered tail. Therefore groups n>=1024 with n divisible by512 need the additional final512 rows. The bridge adds precisely that trailing chunk using unchanged source helpers, remaps its row indexes and rejects nonchronological group order. Other lengths already have the helper's equivalent padded tail. It restores the second stored delta channel at chunk starts from the prior row in the same group; the native default consumes raw span only (bridge741–785; author joint_interval_model1339–1406,1577–1699).

Both the author eager HDF5 loader and streamed loader reinterpret uint16 via view(float16) before conversion tofloat32; they do not numerically cast bit integers. Original quantized source final-layer features are preserved (author joint_support195–203; joint_interval_model1464–1472). Replay uses loaded eval-mode source weights, original FP32 per-chunk GRU logits, per-chunk softmax and the source overlap-probability aggregator. Native row IDs/intervals/probability shape/finiteness/normalization are checked, and maximum absolute replay error must be<=1e-6 with unchanged model state. Raw chunk logits are retained separately; log of averaged probabilities is never called original logits (bridge788–912). Actual parity remains a runtime gate.

## Seals, support and scoring

infer_and_seal completes both VAL parents' grid/HDF5/chunk-logit products and individual seals, then writes all_VAL_predictions_sealed.json and returns. The runner calls external_score only after this return. The inference pair files contain observed tree/reference paths; the CLI override prevents the original supervised training/selection pair files from being used as inference pairs.

Inference alone changes exclude_edge_trees=false; TRAIN/selection stay at released defaults. Duplicate output intervals for one query are rejected. Unsupported cells remain NaN plus an explicit mask. Exact prediction-bin/full-truth intersections retain full query-bp denominators for present64,ancient56 and all120. Any missing query-bp makes the group's full-support metrics unavailable while preserving total denominator/missing mass. Supported zeros yield explicit infinite NLL mass/null, without floor, purity filtering or recalibration. This review evaluates those source rules, not coverage or scores (bridge915–996; runner phase dispatch).

## Same-code GPU/native admission

The default path checks policy/source only. Actual stages require a reviewed code/config hash and fresh output. Bootstrap validates the inherited held scheduler descriptor/UUID and exact process title before scientific imports, preserves the descriptor across isolated interpreter exec, and uses candidate-contained environment/library/TMP/cache paths. Candidate Torch2.10+cu129/vision0.25, absence of version bypass, source module paths and actual loaded ancpath/Torch mappings must match hashed completed build/CPU evidence.

gpu_runtime obtains the actual current driver-context UUID from its loaded libcuda, compares it with the inherited lease, requires the one logical device and uses frozen NVTOP verification of own compute PID/physical index/title. It does not call the ARGformer CUDA12.4 UUID qualifier. qualify_gpu exercises a synthetic full source joint forward/backward without optimizer/weight change and actual stock Rust TorchScript CUDA fixture/device comparison. Later stages require that same code/config qualification, and lease/source/native bindings are rechecked (bridge195–320,644–693,999–1055; runner admission/final guards). None was executed by this auditor.

The v6 build and CPU receipt metadata match the bridge's expected schema and extension paths. Their SHA256 values are fa1898dc4f55638d4e106a2da9b6fb7df139f2719b50b664b433c7955e0dd0dd and592561fe97bd74e8678ba2ebb1bf749c0c09124c13eda03d0ded98d120c99a39. This establishes recorded prior stages only, not missing GPU/biological outcomes.

Nonblocking execution note: the document's launch example names historical v5. Root should supply the completed v6 build/CPU paths and the above actual receipt hashes; the source accepts explicit factual later-attempt bindings. No producer-file edit was made.

No April-paper reproduction, completed GPU/scientific qualification, fit, biological accuracy or final TEST outcome is certified.
