# American paper-oriented benchmark implementation

The corrected ARGformer route is implemented in `src/arggnn/benchmarking/paper_american/`. It preserves the author encoder, optimizer, scheduler, seeds, complete native path corpus, five MLM epochs and one contrastive epoch. The released recipe remains a separate control. No running source, job or checkpoint was modified.

The corrected route selects 30% of nonspecial tokens for MLM in both internal splits and replaces every selected token with MASK. It retains the extant token and uses its hidden state at position one through the native projection, activation, normalization and dropout. The same representation is used for contrastive learning and prediction. Contrastive temperature is 0.05 and actual TRAIN reference labels determine inverse-frequency anchor weights. The weight calculation is vectorized, preserving the native normalized weighted loss formula and its gradients. The contrastive batch remains 4096 because splitting it would change the negative set.

The original PCA10 cluster classifier is unavailable. The frozen study choice fits centered, unscaled PCA on every reference path only and classifies projected query vectors by Euclidean distance to known reference class centroids. Each reference path has equal weight. Ties use EUR, EAS, AFR order. The encoder is normalized as in the native contrastive model. The ten PCA coordinates are not whitened. Query labels never enter fitting.

The selected executable DAG is `expanded_delivery/task_dag.json`. It contains 58 tasks for the actual current 15-parent cohort: 13 CPU full-path preparations and 45 corrected GPU stages. TEST000 and TEST001 have usable existing path corpora and concrete ready MLM specifications. The five validation preparation receipts are historically successful, but their raw corpora were already retired. They therefore receive fresh preparation tasks from the same observed genealogy and focal registry. TEST002 through TEST009 still need their first complete path preparation. Earlier expansion directories are preserved development evidence and are not selected.

All 15 completed FLARE prediction seals are reused. Their conversion observation and focal identities match the current common cohort. No new simulation, inferred genealogy, genotype conversion or native FLARE run is needed. The original AN1/AN2 calls and rounded posterior handling are preserved. Prior release MLM and contrastive products can serve as controls, but none initializes the corrected arm because the masking and extant representation changed.

The worker exports `run_task(request, resource_check=...)`, so ROOT can resolve real predecessor completion files after each DAG dependency succeeds. New preparation nodes call the existing bounded native path preparation. GPU nodes use the isolated native ARGformer interpreter. ROOT selects a single device UUID on physical GPU 3, 4 or 5. With `ARGGNN_GPU_MEMORY_POLICY=full_device`, no allocator fraction is imposed. ROOT retains scheduling, storage admission, process supervision and actual wait ownership.

The common evaluator bridge is `paper_american.scoring.score_factory`. It accepts the corrected PCA hard-call seal explicitly and delegates the complete 45-prediction barrier, canonical focal registry, separate private truth join, equal-parent summaries, sample SD and paired metrics to the maintained American scorer. PCA one-hot values carry hard calls and receive no NLL or Brier score. The maintained TreeLoom seal checks remain in force; a new TreeLoom training variant needs its own truthful scorer integration.

The known current parents have about 224,000 to 228,000 MLM updates and about 2,100 contrastive updates each. ARGformer fits each evaluation ARG using its observed paths and known source references. Training another 20 ARGformer models on the global TreeLoom TRAIN parents would produce unused models and is omitted. Supported persistent workers, pinned batches and prefetching remain enabled. Trailing padding is cropped only in the corrected arm. The existing unpadded attention already avoids part of the padding cost, so no speed multiplier is claimed.

Prediction streams reference embeddings into a 768 by 768 covariance matrix and three centroid accumulators. It avoids writing roughly 30 GB of reference embedding archives per parent. Only compact PCA arrays and the common query predictions are saved. Final checkpoint discovery reuses a native final-budget checkpoint when present rather than writing a duplicate. Eight GiB per training phase is a conservative output planning allowance, not measured occupancy. New path preparation retains the existing 20 GiB serialization bound and 256 GiB RSS failstop. Training requests 128 GiB CPU RSS. GPU fit and wall-clock forecasts remain unmeasured.

CPU checks passed for the weighted loss and gradients, masking, extant pooling, streaming covariance and identity joins. A second smoke used the actual author ARG loader and tokenizer on two synthetic paths and confirmed retained leaf targets, literal MASK replacement, special-token preservation and padding crop. Actual native author modules imported successfully with CPU-only visibility. No model was constructed or trained in those checks. Full native GPU qualification belongs to the first ROOT-admitted task.

Planning CLI:

```bash
source scripts/project_env.sh
.runtime/env/bin/python -B -m arggnn.benchmarking.paper_american.cli plan --stage mlm --preparation PREPARATION_COMPLETION.json --output-directory experiments/results/NEW_PARENT/mlm --task-file research/benchmark_rebuild_20261024/american/NEW_TASK.json
```

The task specification binds author sources, adapter sources and the reviewed abstract IPC helper. The latter is installed before native multiprocessing so eight fork workers do not attempt filesystem sockets on NFS. It does not invoke historical GPU selection or accounting code.

ROOT should pin the two reusable path corpora against retirement until corrected prediction completes, register the new scientific condition in the experiment ledger, integrate the selected DAG and regenerate current queue guards. The sources and reports are ready for that integration. No GPU work or Git operation was performed here.
