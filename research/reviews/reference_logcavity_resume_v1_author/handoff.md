# H030 recovery source handoff

Narrow operational recovery only. Root owns review, unfinished-cell registration, active admission, runtime/reservation/lease and execution. Original H030/H032 bytes and predecessor outputs stay read-only.

| Path | SHA256 |
|---|---|
| `src/arggnn/train_reference_logcavity_resume_v1.py` | `4f7c9b55eb58282eb974ea86580bd37b79492b0e0f9f60dc035b34f6768371ec` |
| `configs/reference_logcavity_resume_v1.json` | `8b5d9e95f2a954f636a62f659e58a23035450ae2fd798b1aee869abb8f97b573` |
| `docs/reference_logcavity_resume_v1.md` | `4a94d788364b3fdd253769d5b63c780ca92965ec6a2385f87914d382892ffd88` |

Model remains original `src/arggnn/reference_logcavity_gnn_v1.py`, SHA256 `8970aa2e82a43b45d1ec10b2eaae6e2203dab12356a4b968bea3c65b2e21fc0b`. Retry adapter is `src/arggnn/project_budget_v9_consumer_retry_v2.py`, SHA256 `75fd586fc8e0733027451a6d9485b7e49f10c3d8d7cf69c09cdec4b9f1b7ba1e`.

## Commands

From `/workspace-SR003.nfs2/ARG-GNN`:

```sh
source scripts/project_env.sh
.runtime/env/bin/python -B -m arggnn.train_reference_logcavity_resume_v1
```

Default reports source only. Later root execution inside its new reviewed held lease:

```sh
.runtime/env/bin/python -B -m arggnn.train_reference_logcavity_resume_v1 \
  --execute --input "$H030_RESUME_INPUT_PATH" \
  --expected-input-sha256 "$H030_RESUME_INPUT_SHA256"
```

## Input and storage

Schema `reference_logcavity_resume_input_v1`; root authorizer, fit_authorized=true, active status/expiry, canonical NEW logical job matching lease, seed101/CUDA/10epochs, NEW fresh repository-relative output and root CPU/CUDA caps. Pin new driver and original model in source_sha256. `refs.config` is new recovery config; `refs.runtime` and `refs.storage` are root's new records. Every other scientific ref and observed-parent binding must equal immutable original input exactly. The config pins original failed record/input/scientific config, source files and author resume manifest; root need not republish those predecessor records.

The storage uses the same V9 schema with retry_v2 as its exact consumer_adapter. Root supplies current core/launcher pins and real2GB reservation/initial ledger pins; bind new driver/config/runtime plus adapter/core/launcher, with new job/output. Include BOTH original V9 consumer provenance and retry_v2 actual consumer_adapter in reservation consumer_bindings; unchanged core successor_paths requires the original pin. Existing output parent must exist; fresh output cannot equal predecessor output. Author storage template is inactive and creates no authority.

Kernel/static start epoch5 from epoch4; recurrent arms start epoch4 from epoch3. Restore model AND optimizer states; only528 new updates. Count800 final logical trajectory updates separately from848 original-plus-recovery physical updates, including48 discarded/replayed unsealed updates. Preserve original minimum-NLL history and selected refs. Unsealed recurrent epoch4 work (48 updates) is explicitly replayed, never treated as a new arm. Final logical endpoint160/arm, complete H030 history and disclosed failure/recovery lineage. H025 baseline metadata reused; H026 remains frozen comparator. New writer prunes only its own files.

All detailed recovery limits, first-update qualification and deferred H032 recovered-static comparison policy are in `docs/reference_logcavity_resume_v1.md`. Original completed.json is never fabricated. Source-only metadata fixtures/default CLI verify integration without actual data/checkpoint payload reads or GPU use. GPU resumed-state validity and first actual update remain root qualification.
