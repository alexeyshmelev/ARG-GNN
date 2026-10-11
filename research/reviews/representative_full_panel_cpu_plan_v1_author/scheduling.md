# Representative full952 CPU data scheduling

Source-only measurements and scheduling handoff. No generation/inference job, GPU, package installation, Git operation, project-wide scan or budget change ran. Only four explicit completed parent directories were stat'ed and their compact JSON receipts/source were read. Target/genotype/tree arrays were not opened.

## Measured raw output scope

The four existing author-panel parents are2Mb each,952 haplotypes/476 diploids,686 queries,266 pure-source references including34 older proxies. Regular-file bytes are:

| Existing parent | Retained file bytes | End-to-end seconds |
|---|---:|---:|
| ap_full_train_000 |1,013,559,932 |521.930 |
| ap_full_train_001 |1,009,261,153 |503.634 |
| ap_full_train_002 |994,244,208 |531.284 |
| ap_full_discovery_holdout_000 |983,960,775 |545.870 |

Mean1,000,256,517 bytes, observed max1,013,559,932. The separate du apparent-byte readings include directory metadata and are~32KB larger per parent. These are retained outputs, not measured peak writes or RAM.

The dominant stock output is relate/post_size.pairwise.bin at663,418,192 bytes in every parent. Cleaned post-size observed trees are83.06–91.49MB, uncleaned native trees76.10–83.82MB, pre_size.anc55.33–61.13MB, and original phased VCF25.10–26.68MB. Total observed files are26.87–28.53MB, external truth~1.93–1.97MB, and native Relate products955.15–983.24MB. Retain the existing products/receipts; no removal or altered author output is proposed.

For **32TRAIN +8VAL +8untouched outer =48 parents**, the measured-size extrapolation is:

| Split | Mean-based retained bytes | Inherited4GB/parent transient output ceilings |
|---|---:|---:|
| TRAIN32 |32,008,208,544 |128,000,000,000 |
| VAL8 |8,002,052,136 |32,000,000,000 |
| Outer8 |8,002,052,136 |32,000,000,000 |
| Total48 |48,012,312,816 |192,000,000,000 |

At the largest observed parent,48 retained scopes would occupy48,650,876,736 bytes. These extrapolations exclude PyG banks, mutation preprocessing, native AP/Rust training archives, model checkpoints, predictions and root launcher logs. The1TB user allowance does not turn this into a claim about presently available project space. Root keeps the existing4GB parent ceiling and1GB phase headroom contract rather than adopting a new budget system.

## Qualified environment and preserved generator

Use the exact completed-parent environment:

~~~bash
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
~~~

Interpreter: .runtime/env/bin/python, Python3.11. Original source runtime tuple is msprime1.3.4, tskit1.0.3, NumPy2.4.6. No native-baseline TensorFlow/ABI210 environment is needed for generation. The existing static Relate1.2.4 tools are under .runtime/tools/relate/relate_v1.2.4_x86_64_static, with the recorded installation/tool hashes retained.

Mutation-support's source adapter is prepare_expanded_full_panel_v1.py. It freezes32train/8val/8outer parent IDs ap_expanded_<split>_<NNN> under data/expanded_full_panel_v1 and reuses the original generate_parent()/infer_parent() functions. Preserve author demography, full original age/order roster,2Mb,Hudson,JC69,r1e-8,mu1.25e-8 and the original five-iteration inference chain. The original commands use RelateParallel --threads8 --memory2 and EstimatePopulationSize --threads8 --num_iter5 --threshold0.5 --years_per_gen28 --noplot, including final branch re-estimation. --memory2 is a native chunk option, **not a bound on total process RSS**.

The source author scripts provided no seeds. Existing project streams use SHA256 of a frozen root101/split/replicate/purpose domain, first8 bytes big endian, %(2**32-1)+1 for ancestry/mutations and %(2**31-100)+1 for Relate. Size iterations use base+1..5 and final branch re-estimation intentionally reuses base+5. The expanded adapter uses a distinct namespace with the same mechanism and audits collisions against existing frozen streams. Freeze the entire registry before the first parent writes truth. Do not select or replace streams using resulting accuracy.

Outer generation necessarily creates its separately stored truth files; “untouched outer” means no downstream target-array read, tuning, fitting or scoring. Source-only metadata/stat inspection does not open their ancestry arrays.

## Minimal concurrent CPU recipe

Use one **all-stage process per parent**, so each worker simulates then runs the full native inference chain sequentially inside its own output scope. The producer already uses per-parent relate/tmp, native working directories, explicit output exclusivity and receipt guards. Its native command groups and failure preservation remain intact.

Start with4 concurrent parent workers if root has assigned32 CPU threads and adequate RAM/I/O. Each parent requests8 native threads. This is a scheduling choice, not a measured host-capacity claim. /usr/bin/xargs is present, so no extra queue package is needed. Root supplies one bound per-parent execution_admission.json and separate launch.log scope, then dispatches the frozen registry with GNUxargs -P4 -n1. The adapter author has finalized the CLI and source pin below. Root supplies the new per-parent admissions and existing storage authority.

The source-only proposed config freezes registry SHA25621cff73e89c174e33d1993beee6d8063bc593bacc257ae86138a39a82a4d6871. If root places byte-identical config at another path, update REP_CONFIG only. The minimal template is:

~~~bash
# Run after root freezes config/source hashes and creates all parent admissions.
export REP_RUN_ROOT="$ARGGNN_ROOT/research/reviews/expanded_full_panel_v1_root"
export REP_CONFIG="$ARGGNN_ROOT/research/reviews/expanded_full_panel_v1/proposed_config.json"
export REP_CONFIG_SHA='662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08'
export REP_SOURCE_SHA='2ff091677986d5cfec2936c2acf53eeed71c957192f9f048cb47f59d3351913e'

# registry_parent_ids.txt contains precisely the frozen32/8/8 parent IDs.
xargs -P4 -n1 bash -c '
  set -euo pipefail
  set -C
  parent_id=$1
  .runtime/env/bin/python -B -m arggnn.prepare_expanded_full_panel_v1 \
    --execute --stage all --parent "$parent_id" \
    --config "$REP_CONFIG" \
    --expected-config-sha256 "$REP_CONFIG_SHA" \
    --expected-source-sha256 "$REP_SOURCE_SHA" \
    --admission "$REP_RUN_ROOT/$parent_id/execution_admission.json" \
    >"$REP_RUN_ROOT/$parent_id/launch.log" 2>&1
' _ <"$REP_RUN_ROOT/registry_parent_ids.txt"
~~~

This is a recipe, not an executed command. Parent IDs and hashes come from the adapter's frozen registry/root admission, not shell-generated replacement seeds. Child GPU visibility stays empty. Native output and temp paths are parent-local, and shell noclobber avoids overwriting an existing launch log. Producer rejects pre-existing parent output for all/simulate and preserves partial failures. No automatic retry, deletion or favorable-parent substitution is proposed.

Mean observed full-parent time is525.679s (8.76min), range8.39–9.10min. Initial Relate accounts for98.55–111.82s, the five size iterations215.74–229.60s, and conversion~4s. The remaining161–221s includes simulation, cleaning, export, hashing and I/O; it is not an isolated simulation benchmark. Idealized48-parent time is7.01h serial or1.75h at4 workers, before shared CPU/NFS contention. This is a forecast, not a concurrency result.

Minimal success tracking is each fixed parent's all_completion.json plus existing Relate return codes/inference receipt. Root can consume completed TRAIN parents while the fixed queue continues, keeping VAL for selection and the8 outer parents sealed. No extra accuracy/FP64/parity validation matrix is proposed before representative generation.
