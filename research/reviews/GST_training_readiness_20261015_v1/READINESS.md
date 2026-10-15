# GST training readiness, 2026-10-15

GST is **not ready for a GPU4 launch from the current native artifacts**. Exact acquired data and genuine author sources exist, but the recorded GST environment is CPU-only and its wrapper explicitly rejects CUDA. A meaningful new full-TRAIN condition is viable after runtime, count and resource qualification. GPU4 availability and GPU5 H043 activity are root-reported, not queried here.

## Completed evidence

The existing payload-copy acceptance confirms12,710,870 exact uncompressed bytes, member CRCs and hashes, original names/order and the train/valid/test split files under data/gst_uspto_native_payload_v2_r001/data/. Reuse these artifacts; no repeat acquisition, archive inventory or extraction is needed. The archive SHA is66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723 from wengong-jin/nips17-rexgen@fb7dea369b0721b88cd0133a7d66348d244f65d3. This provenance does not prove the GST paper authors used those exact bytes.

Sixteen retained author source/config/initializer copies still match their receipt for GST commit6469d56aa131463325058b897b1e435c6d2edcd2. USPTO uses genuine GraphSetConv; missing gst_v2 concerns other task scripts and is not this runner's blocker. Retain its source gate and FFN multiplier4. The project genealogy GST control uses a different independent adaptation.

H022 already completed the native four-model saved1000 condition on CPU: first1000 retained TRAIN reactions, all1495 retained VALID reactions,20epochs, batch16 and seed0. TEST was withheld by passing fullVALID into both evaluation slots of unchanged train_one. Selected VALID APs were .63791939 GST-ca, .65378329 GST-bc, .63945129 GCN-DS and .63823432 GCN-ST. Native train_one CPU times sum to145.06seconds. These are historical one-seed validation observations, not a GPU/full-data timing forecast or a candidate/final-TEST comparison. An unchanged seed0 repeat is already answered.

## Exact native task and budget

Each reaction is a set of reactant-molecule graphs. Each atom has the original19 float32 features and a separate binary edit-map reaction-center label. Preserve molecular edges, component/atom order, archive splits and duplicates. The endpoint is pooled per-atom sigmoid AP over all retained reactions, not reaction-mean accuracy.

| Condition | TRAIN | Seeds/models | Epochs/batch | Meaning |
| --- | --- | --- | --- | --- |
| Author saved result | First1000 retained |0–4 × four modes |20/16 | Saved release setting; fullVALID/TEST |
| Existing CPU result | Same1000 |Seed0 × four modes |20/16 | Completed; VALID-only invocation |
| Full-TRAIN saved-architecture extension | All retained N, count unresolved |0–4 × four modes |20/16 | Representative new coverage condition |
| Raw CLI defaults | All retained |Five seeds/four modes |20/16 | Hidden64/layers3/no equalization differs from saved setting |
| --quick |1500 retained, VALID/TEST200 |One seed/four modes |2/16 | Reduced condition; not recommended as representative outcome |

Saved architecture is hidden128/layers1/heads4/dropout.1 with original smaller-parameter equalization. Existing receipts measure GST-ca hidden76/130798parameters, GST-bc84/130622, GCN-DS128/134913 and GCN-ST128/267777. Full author scheduling is20fits and400model-epochs.

For1000 TRAIN reactions each fit has20×ceil(1000/16)=1260 actual updates, while unchanged cosine T_max is20×floor(1000/16)=1240. For full TRAIN use20×ceil(N/16) updates and20×max(1,floor(N/16)) scheduler T_max. Retain final partial batches. Do not repair the floor/ceiling difference silently.

Training is AdamW, common LR.001, weight decay.0001, native positive BCE weight from firstmin(N,1000) retained TRAIN examples, norm1 gradient clipping, native shuffles and strict improving VALID AP checkpoint selection with earliest ties and explicit final-weight fallback. The paper-stated branch uses GST LR.0001 and controls.001. It is a separate scientific condition; one unchanged all-model --lr call cannot implement both.

Source builds/equalizes before train_one reseeds. Record process-entry and per-build CPU/CUDA RNG state and preserve original model-major/seed order. Seeds0–4 alone do not reproduce initialization. Do not silently move reseeding before construction.

## Readiness gaps

The native environment is Python3.14.3/Torch2.10.0+cpu, PyG2.7.0, NumPy2.4.1, SciPy1.17.0, sklearn1.8.0 and RDKit2025.9.3. Its receipt says torch_GPU_backend=false. The author saved result used Python3.14.4/Torch2.10+cu128 with CPU execution; the package requires Python>=3.13. Another GPU environment can be a disclosed port after actual qualification, but these receipts establish no current GST CUDA readiness. Preserve the completed CPU prefix.

The existing scripts/train_native_gst_uspto_val_only_v1.py enforces cpu, emptyCUDA visibility, Python3.14.3/Torch+cpu, TRAIN1000 and seed0. It needs a fresh guarded GPU invocation successor. Genuine USPTO author functions need no demonstrated model patch; runtime/device/coverage/supervision changes must be explicit and source-pinned.

Full retained TRAIN count and full raw-row/drop/fatal/tail accounting remain unresolved. Existing parser instrumentation is a source-only plan, while existing fit receipts give aggregate1000TRAIN/1495VALID audits. Preserve native raw-line enumeration including drops, retained-limit tails, fatal integer-edit exceptions and partial malformed-bond omissions. Actual CUDA operations, gradients, optimizer/checkpoint/RNG behavior, memory bounds and current resource admission are unqualified.

The candidate molecular adapter is absent in the bounded src/scripts search and explicitly unimplemented in the protocol. It must use the same19features, raw reaction/atom IDs, labels, splits, objective, selection and tuning budget with meaningful set/permutation semantics. Arbitrary genomic intervals/order cannot be invented for reactant sets. Baseline training alone does not finish the paired comparison.

Original author-used snapshot, entry initialization state and full paper Table3 configuration remain unproven. Full TRAIN plus saved architecture is an explicitly named coverage extension, not automatic exact reproduction.

## Shortest valid path

1. Reuse accepted assets and unchanged author copies. Root registers the unfinished full-TRAIN/runtime-port H022 condition and obtains actual all-TRAIN counts/row support with native parser parity, unchanged fullVALID and TEST withheld.
2. Root qualifies an actual CUDA GST environment on physicalGPU4 under current lease/title/guard rules. Reuse a compatible GPU prefix if qualified; do not assume one exists or mutate the CPU prefix. Qualify the exact import closure and native forward/backward/optimizer/checkpoint/RNG behavior and actual caps.
3. A new guarded root wrapper imports unchanged full author modules, explicitly discloses CUDA runtime/device/output and max_train=None, and preserves all20epochs, four modes, five seeds and native build/model-major order.
4. Run one continuous full-budget suite. The first completed20epoch GST-ca seed0 fit is useful progress inside the committed suite, not a smaller final budget. No short epoch/data/seed schedule substitutes for representative training.
5. Freeze the candidate molecular adapter and all tuning before separately admitted complete paired TEST evaluation. Preserve every retained atom/reaction denominator and failure receipt. Keep source-common-LR and paper-stated-LR branches distinct.

This assessment read source and existing metadata only. It did not open raw archives, payload rows, arrays or checkpoints, rerun archive inspection, query runtime/process/GPU state, read guards, install, admit, launch or use Git. All writes are in this fresh folder. Exact evidence and structured blockers are in evidence_refs.json and readiness.json.
