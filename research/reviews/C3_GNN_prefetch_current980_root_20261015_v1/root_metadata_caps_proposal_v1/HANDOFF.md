# Root metadata and cap proposal for the600-update C3 GNN fit

Metadata-only snapshot. No activation, admission, guard operation, model, array/checkpoint, GPU job or Git action occurred. The frozen executable successor and its installed shim are unchanged.

## Available bindings

`snapshot_v1/TRAIN_bindings.available.v1.json` contains the exact four mappings required by the executable TRAIN request. `snapshot_v1/request_train.available.proposed.inactive.json` is an inactive request with those mappings and a proposed caps_template_ref already inserted.

- TRAIN000–017:18 genuine completed simulation/config/inferred-observation bindings with direct wait0/drain lineage.
- TRAIN000–016:17 genuine completed focal/block plans and their successful actual waits, including000's real retry2 and016's continuation.
- TRAIN017:simulation/inference ready; final focal/plan completion or successful direct wait was still pending at snapshot time. All three plan-map fields remain null.
- TRAIN018–019:all four mapping entries remain null/pending. No active producer artifact was promoted to completion.
- Physical GPU/UUID and active caps_ref remain null; active=false. Root still needs genuine remaining plan/input evidence before bind.

`TRAIN_metadata_evidence.v1.json` preserves per-parent exact refs and lineage. Private truth paths were checked as metadata only. No truth/focal/tree/genotype/checkpoint payload was opened or hashed. The available plan counts sum to `2629` blocks, `83838` trees and `258308` sites; these are preparation metadata, not training or ancestry accuracy.

## Explicit proposal

`snapshot_v1/caps.train.proposed.inactive.json` is a concrete inactive proposal for root review.

| Prefetch cap | Proposed value |
|---|---:|
| Persistent workers / maximum in-flight tasks |2 /2 |
| max_block_bytes |134,217,728 (128MiB) |
| worker_rss_cap_bytes |8,589,934,592 (8GiB) each |
| aggregate_rss_cap_bytes |128,849,018,880 (120GiB) |
| task_timeout_seconds |900 |
| drain_timeout_seconds |30 |

The unchanged training config's CPU cache is536,870,912B (512MiB). Every available plan has source-declared feature bound73,240,576B (69.85MiB), from:

`256*250000 +32*32*1600 +(1600+256)*4096 =73,240,576`.

The128MiB block cap is1.83times that declared bound. The largest available parent has162blocks, so summing that conservative per-block bound gives11,864,973,312B (11.05GiB). This sum is a metadata planning diagnostic; features are not all retained in the512MiB cache and the formula does not bound autograd/native allocation peaks.

Two workers limit duplicated dataset state, NFS reads and shared-host contention before any throughput result exists. Each worker has one task/result slot; there is one persistent pool, no feeder queue, restart, retry or fallback. The required cache/transport/worker planning sum is:

`512MiB +4*128MiB +2*8GiB =17GiB (18,253,611,008B) <=120GiB`.

The generous120GiB aggregate cap is an explicit fail-stop proposal. Root-owned RSS is capped separately at128GiB, leaving8GiB between those bounds for launcher/helper/supervisor overhead. Historical peak-RSS summation is conservative and does not fall when parent dataset state is released.

| Baseline operational cap | Proposal |
|---|---:|
| Reservation |2GiB |
| Product writes |1GiB |
| TMP/cache allowance |512MiB |
| Root auxiliary allowance |128MiB |
| Root packet overhead |4MiB |
| Child log RAM / retained outer log |4MiB /4MiB |
| Root-owned CPU RSS |128GiB |
| CUDA reserved memory |40GiB |
| Full fixed-condition deadline |86,400seconds |

Product+TMP+aux+packet sum to1,749,024,768B, below the2,147,483,648B reservation. These retain the baseline storage proposal while making CPU/GPU/deadline choices explicit. Storage admission remains the unchanged guard's decision. CUDA40GiB is below the successor's50,000,000,000B ceiling and must be reconciled with root's actual physical4or5 device, UUID, free memory and existing leases.

The byte formula, actual plan dimensions, metadata identities and successful wait refs are verified components. The8GiB/worker,120/128GiB RSS,40GiB GPU, storage allowances and900/30/86400second limits are **prospective fail-stop limits**, not measured full-fit peaks, timing guarantees or proof that600updates will finish. The earlier tiny actual PyG fixture qualifies only the abstract shared-tensor mechanism; persistent lifecycle, actual Dataset transport, gradient equivalence and throughput remain unexecuted.

## Closed original VAL000 plan

`snapshot_v1/VAL000_closed_plan_reuse.v1.json` contains ready=true and exact old plan, native preparation, preparation wait and plan-stage evidence refs. The actual original plan completion, wait0/drain, close receipt, root terminal consumption and launch/write revocation were checked. This can populate the successor predict_val_000 bindings after new TRAIN genuinely completes. Its training field remains null. It does not introduce nativeMLM or old-model completion as a dependency.

## Root finish

Supply genuine017 plan/wait plus018019 complete input/plan/wait refs, recheck every final plan's declared block bound, choose actual device/UUID and review/adopt or revise the explicit operational proposal in a new root-owned copy. The existing frozen executable bind/admit routes then perform their full checks. Do not activate this partial snapshot.

`collect_metadata_proposal_v1.py` defaults inert. Its optional metadata snapshot mode writes only a fresh immediate child directory here, reads bounded JSON/source metadata, and never activates or invokes a guard. This version intentionally leaves018019 null; root owns final all20 binding. `metadata_source_inventory.v1.json` pins the observed metadata and `validation_receipt.v1.json` records actual checks.

No frozen successor, old current980 file, scientific engine, existing config, shim, guard or live/completed plan was changed.
