H049 canonical GPU execution successor — root handoff

Status: source-only, inactive, ready for root review. The canonical launcher, admission, guards, leases, GPU execution and scientific payloads have not been exercised by this packet. The live original CPU preparation has not been touched.

The confirmed launcher contract is in scripts/launch_reserved_gpu_full_storage_v1.py: lines 99–100 require an arggnn module; lines 112–115 derive src/arggnn/<module>.py and require its exact source reference in the reservation. The original research-namespace producer cannot pass this gate. The sole new canonical source is src/arggnn/H049_order2_cut_memory_exec_v1.py, launched as arggnn.H049_order2_cut_memory_exec_v1.

The additive root_driver.py retains the original root_common_v1.HERE, frozen context, producer and preparation source. Its small delta is captured exactly in minimal_driver_delta.diff. It permits TRAIN/EVALUATE only, adds the canonical dispatcher to mandatory reservation bindings, pins the actual dispatcher in config and input, and selects that canonical module for the existing launcher. It does not modify original sources or use global patches.

The inherited producer_ref, main_source_ref and worker_cli_ref deliberately name the underlying frozen scientific producer_v1.py because the frozen context requires their equality. The added canonical_dispatcher_ref honestly binds the actual outer module that the launcher executes. The dispatcher verifies its own config/input pin, the successor driver pin, original producer/context pins, active GPU capability and source-map entries, then calls the unchanged frozen producer CLI on the same authenticated input. The producer retains the full original validation path.

Original preparation lineage remains accepted. The frozen completed_cut_preparation_v1 checks the exact original preparation producer, registration, observation lineage, successful direct wait/drain and all40 lossless compact products; it does not require the preparation source manifest to equal the GPU successor manifest. Runtime admissions intentionally remain under the original packet's admitted_train/ and admitted_evaluate/ folders, so frozen downstream TRAIN lineage checks retain their original locations. The original admitted_prepare/ folder and CPU output path remain owned by the live original preparation.

The scientific protocol is unchanged: complete 952 observed leaves and 686 outputs, original 100-bp scoring support, the three registered arms, common H048 decoder, original H042 pass30 warm start, independent AdamW, 30 passes × 32 TRAIN parents, 960 updates per arm, 18 checkpoints and all 144 VAL products sealed before truth. TRAIN/EVALUATE configs differ only in declared RESOURCE_CAPS. Relative to the completed fit-cap proposal, only status, source_manifest_ref and the two explicit dispatcher/driver pins differ.

Static and inert checks passed:
- Both new Python sources compile without bytecode writes.
- All 121 source pins match; the inherited 119 source pins are unchanged.
- Ten frozen original packet references match, including all six original executable sources, original source manifest/config, registration snapshot and original receipt.
- The stored unified diff matches the exact original/new driver bytes.
- Default canonical dispatcher returned zero with Torch_imported=false.
- Default successor driver and both public metadata-only plans returned zero.
- TRAIN output upper bound is 61,522,688 bytes under 61,865,984 reserved bytes; EVALUATE is 19,371,776 under 19,922,944 bytes.
- Evaluation's raw all144 probability RAM is 31,607,126,208 bytes under the 31,608,274,944-byte gate. TRAIN correctly declares zero evaluation prediction RAM.
- Actual full-model/allocator peaks remain unqualified. The first scheduled update per arm qualifies peaks; no disposable pilot has been run.

The completed fit-cap proposal remains in ../H049_order2_cut_memory_root_source_20261015_v1/root_fit_cap_proposal_v1/. Its receipt is pinned in receipt_v1.json. Runtime caps are policy workspaces/headroom, including 37 GiB TRAIN RSS, 60 GiB EVALUATE RSS and 21 GiB CUDA reservation stop; canonical 50 GB + 4 GB authority eligibility remains unchanged.

The existing transaction_memo_adapter_v2.py binding (c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364) is retained through frozen original root_common_v1. No future namespace-memo successor has been integrated. Any such integration requires a separately authenticated successor with updated exact source/config bindings; it must not be folded into these sealed proposed bytes.

Remaining root bindings before scientific execution:
- Review the cap rationale and this successor, then create reviewed active config/request copies without changing scientific fields.
- Bind actual successful original cut-preparation completion and actual direct wait/drain references in each active GPU config.
- Choose the physical GPU (4 or 5), bind the exact active config reference and reservation bytes, and explicitly activate the root admission request.
- For EVALUATE, bind the actual successful full TRAIN completion, launch and actual wait/drain references.
- Perform root admission, canonical launch and actual wait/drain; root owns scheduler eligibility, resource authority, terminal close and Git.

Proposed inactive config/request and public plan request paths are pinned in receipt_v1.json. Metadata plan rechecks can use:
~~~
.runtime/env/bin/python -B research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v1/root_driver.py plan --record research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v1/plan_request.train.proposed.json --sha256 4ecbf3511decd28f7334174c8e5d15de0e0ffd0e4f3bb3020d0be6e4e82ba209
~~~
The EVALUATE plan command and all exact checked commands/stdout are retained in checks_static_inert_v1.json. Do not invoke the canonical launcher as a help-only check: its outer wrapper configures resource guards before dispatch.
