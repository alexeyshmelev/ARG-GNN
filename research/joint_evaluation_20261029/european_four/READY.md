The direct European four-output condition is CPU-qualified and frozen. Root owns GPU admission and launch.

Frozen files: freeze_v1.json, protocol_v1.json, smoke_result_v1.json. The smoke checks every one of the40,000 cached labels against the original CNN4 TRAIN targets. It also checks three parents for unchanged context tensors and exact direct-four focal logits, two real optimizer updates, two persistent-worker epochs, clean worker exits, and bitwise checkpoint recovery. It opens no TEST payloads and initializes no CUDA.

The dataset contains531 retained sites,177 per parent, and10,000 targets from each of present_day, Bronze_Age, Neo and Yam. Training uses fresh seed101, width64, three layers, four outputs, structure mode, original AdamW settings, batch16, four workers, and fixed final epoch50. This is1,700 updates and2,000,000 target presentations. The objective is M/(S*N)*sum CE, using M=531,N=40000 and actual minibatch S. The complete existing cache is reused unchanged; only focal_site and focal_sample readout routing are sliced. Full observed native trees and genotype context retain BAA query leaves with unknown ancestry. No BAA target labels or outputs are used. This distinction from CNN4 features must accompany the paired results.

All new source, metadata, logs and outputs fit a128MiB allocation within the shared1GiB evaluation cap. Every process family is capped at64GiB; no copied persistent prepared cache is created. Root must keep these stages sequential and reserve the full family while they run. TRAIN qualification chooses16 windows globally by each of nodes, SNP rows and prepared bytes, performs two updates per group, and discards those temporary weights. Training requires a passing same-freeze and same-GPU qualification.

Readable root commands, executed from /workspace-SR003.nfs2/ARG-GNN after admission:

```sh
env CUDA_VISIBLE_DEVICES=GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71 PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python3.11 -B -s -m research.joint_evaluation_20261029.european_four.train qualify --gpu 4
env CUDA_VISIBLE_DEVICES=GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71 PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python3.11 -B -s -m research.joint_evaluation_20261029.european_four.train train --gpu 4
env CUDA_VISIBLE_DEVICES=GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71 PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python3.11 -B -s -m research.joint_evaluation_20261029.european_four.predict --parent ancestralpaths_european_test_000 --gpu 4
```

Repeat the prediction command for TEST001 through TEST004, only after the final50 fit and owned-loader shutdown succeed. Output products are outputs/predictions/<parent>/product.json and satisfy the original ap_four_predictions_v1 contract. The model condition is treeloom_joint_european_direct_four_no_baa_v1. Softmax uses the four direct logits; no six-output checkpoint or merged probabilities enter this condition.

After all five CNN4 and all five direct4 predictions exist, create a JSON mapping of the five parent IDs to the authenticated CNN4 prediction product references. Run the original common paired scorer through:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python3.11 -B -s -m research.joint_evaluation_20261029.european_four.score --cnn-predictions /absolute/path/to/authenticated_cnn4_products.json
```

The scorer validates every prediction before opening any TEST truth, enforces identical original registries and non-BAA target IDs, and retains original parent metrics and sample-SD summaries. A separate disclosure records the observed-context difference. The completed six-class condition is unchanged.

The initial draft CPU run failed on an optional pynvml import inherited from the old trainer. That draft source and failure log remain in draft_attempt1. loader.py copies the original loader/evaluation functions exactly while isolating them from optional GPU telemetry. The successful smoke uses the final source hashes.
