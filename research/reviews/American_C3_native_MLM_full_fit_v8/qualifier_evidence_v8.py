"""Historical CLOSED/READONLY v7 qualification evidence; no rerun or lease."""
import json,math
FIXED={'qualifier_source': {'path': 'research/reviews/American_C3_native_MLM_amendment_v7/qualifier_v1.py', 'sha256': 'fbb745c1c8a838eba8916811e85d1897ae175d5ddfa8182cdc2c54e428c6d322'}, 'qualifier_input': {'path': 'research/reviews/American_C3_native_MLM_amendment_v7/input.json', 'sha256': '3eb960bc8d63f1bcd965d3b3e5586209e154620c40871b421b83702cf8777fcd'}, 'qualifier_source_closure': {'path': 'research/reviews/American_C3_native_MLM_amendment_v7/execution_source_closure.json', 'sha256': 'fdeb69e48591f6514a09457cb7d424c4cf46cd2963ee933de47e8827fb08a119'}, 'qualifier_producer': {'path': 'research/reviews/American_C3_native_MLM_amendment_v7/entry_v7.py', 'sha256': '15269460f1ebeca3074abd11ba2fd38521ed59e686238698328ec4fb14d50b2c'}, 'qualification': {'path': 'experiments/results/American_C3_native_MLM_full30mb_qualify_mlm_v4/completed.json', 'sha256': 'ce132a5d2a9e8c63611ee45e7c87f7855dcef126ba455bded5dc557dc9b64290'}, 'qualification_wait': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/worker_aux/amended_v7_attempt1/actual_wait.json', 'sha256': '736cfd5d9940393221aa3216837a3d0ef6e927570bd03df4dc1997638d86fa4b'}, 'qualification_outer_wait': {'path': 'research/reviews/American_C3_native_MLM_amendment_v7/actual_outer_direct_wait_root_v7.json', 'sha256': '32f0b37721e2042b98be73c7dda79393cc061fd9672ff1da8ad858ffd6a1e14a'}, 'qualification_close': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/worker_aux/amended_v7_attempt1/closed/close_receipt.json', 'sha256': '314dddd2b543a5d0428c1a3ae2f57777dd18ea5ab867bb8c2e69c19f4280a9d8'}}
def require(ok,message):
    if not ok:raise ValueError(message)
def validate_qualifier_and_timing(req,caps,runtime,*,read,reference):
    refs=req["refs"];waits=req["upstream_wait_refs"]
    require(refs["qualification"]==FIXED["qualification"] and refs["qualification_close"]==FIXED["qualification_close"]
        and waits["qualification"]==FIXED["qualification_wait"]
        and waits["qualification_outer"]==FIXED["qualification_outer_wait"],"Exact genuine completed/closed v7 qualification lineage")
    q=read(refs["qualification"]);wait=read(waits["qualification"]);outer=read(waits["qualification_outer"])
    require(q["status"]=="completed_C3_native_MLM_qualification_v1" and q["parent_id"]=="american_c3_train_000"
        and q["preparation_ref"]==refs["preparation"] and q["runtime_ref"]==refs["runtime"]
        and q["actual_producer_ref"]==FIXED["qualifier_producer"],"Same TRAIN000 corpus/runtime and actual v7 producer")
    require(wait["returncode"]==0 and wait["actual_direct_launcher_wait"] is True
        and wait["all_owned_processes_and_groups_drained"] is True and wait["operational_failure"] is None
        and wait["completed_ref"]==refs["qualification"] and wait["input_ref"]==FIXED["qualifier_input"],
        "Real completed qualifier direct wait0/drain")
    require(outer["actual_direct_outer_supervisor_wait"] is True and outer["returncode"]==0
        and outer["pid"]==wait["supervisor_pid"] and outer["outer_process_group_absent_after_wait"] is True,
        "Real consumed qualifier outer wait0/group absence")
    closed=read(refs["qualification_close"]);value=closed["value"]
    require(closed["kind"]=="concurrent_completed_scope_close_receipt_v1"
        and closed["key"]=="H041/American-C3-native-MLM-full30mb-qualify_mlm-v2"
        and value["future_writes_and_relaunch_revoked"] is True
        and value["request"]["completed_ref"]==refs["qualification"]
        and value["request"]["direct_wait_ref"]==waits["qualification"],"Permanently CLOSED/READONLY original qualifier scope")
    child=read(q["child_completion_ref"]);proof=child["qualification"]
    require(child["active_input_ref"]==FIXED["qualifier_input"] and child["physical_gpu"]==4
        and child["runtime_ref"]==refs["runtime"] and child["initializer_checkpoint_ref"] is None,
        "Exact qualified native child, no learned initializer")
    qspec=read(FIXED["qualifier_input"]);closure=read(FIXED["qualifier_source_closure"])
    source=FIXED["qualifier_source"]
    require(qspec["transitive_source_closure_ref"]==FIXED["qualifier_source_closure"]
        and qspec["source_sha256"]==closure["source_sha256"]
        and qspec["source_sha256"][source["path"]]==source["sha256"]
        and reference(source["path"])==source,"Actual source that produced the v7 qualifier")
    exact=dict(status="passed_synthetic_MLM_resource_qualification",batch=512,microbatch=512,padded_length=128,
        true_tokens_per_path=30,vocabulary=106776,parameters=167655192,optimizer_steps=3,
        precision="amp_bf16",seed=17,layers=12,heads=12,hidden_size=768,intermediate_size=3072,
        scientific_training_or_checkpoint_claimed=False,dataloader_corpus_RAM_qualified=False)
    require(all(proof.get(k)==v for k,v in exact.items()) and proof["losses_finite"] is True
        and proof["actual_context_uuid"]==runtime["target_GPU_UUID"].removeprefix("GPU-")
        and 0<proof["peak_reserved_bytes"]<=caps["max_cuda_reserved_bytes"]
        and len(proof["step_seconds"])==3 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in proof["step_seconds"]),
        "Exactly three synthetic optimizer steps at batch512; no corpus training/throughput claim")
    timing=read(refs["timing_review"])
    require(timing["root_authorizer"]=="root_agent" and timing["active"] is True
        and timing["qualification_ref"]==refs["qualification"]
        and timing["actual_qualifier_step_seconds"]==proof["step_seconds"]
        and timing["full_optimizer_updates"]==224775
        and timing["operational_deadline_seconds"]==caps["operational_deadline_seconds"]
        and timing["root_accepts_unproven_full_budget_completion"] is True
        and timing["synthetic_step_rate_is_full_corpus_rate"] is False and timing["full_budget_ETA_claimed"] is False,
        "Explicit source-based deadline acceptance; synthetic timings are not full-corpus throughput")
    validate_nvml_acceptance(read(refs["nvml_abi_acceptance"]),runtime,read=read,reference=reference)
    return proof

def validate_nvml_acceptance(acceptance,runtime,*,read,reference):
    monitor="research/reviews/American_C3_native_MLM_full_fit_v8/gpu_monitor_v8.py"
    provenance="research/reviews/American_C3_native_MLM_full_fit_v8/nvml_abi_provenance.json"
    require(acceptance["schema"]=="native_MLM_NVML_ABI_root_acceptance_v8"
        and acceptance["root_authorizer"]=="root_agent" and acceptance["active"] is True
        and acceptance["root_accepts_actual_UUID_memory_comparison"] is True
        and acceptance["root_accepts_observed_live_memory_differences"] is True
        and acceptance["root_confirms_probe_PIDs_are_owned"] is True,
        "Root accepts actual readonly NVML ABI/UUID and owned live-memory comparisons")
    records=acceptance["actual_probe_refs"]
    require(isinstance(records,list) and 1<=len(records)<=4
        and len({r["path"] for r in records})==len(records),"Bounded distinct actual root preflight receipts")
    target=False;owned=False
    for record in records:
        envelope=read(record)
        require(envelope["command_readonly"] is True and envelope["returncode"]==0
            and envelope["no_model_or_CUDA_allocation"] is True
            and envelope["source_sha256"]==reference(monitor)["sha256"]
            and isinstance(envelope["stdout"],str),"Actual readonly root probe exit0/source envelope")
        probe=json.loads(envelope["stdout"])
        require(probe["schema"]=="native_MLM_readonly_NVML_SMI_probe_v8"
            and probe["status"]=="completed_readonly_NVML_SMI_comparison"
            and probe["source_ref"]==reference(monitor) and probe["abi_provenance_ref"]==reference(provenance)
            and probe["physical_gpu"]==envelope["physical_gpu"] and probe["physical_gpu"] in (4,5)
            and isinstance(probe["GPU_UUID"],str) and probe["GPU_UUID"].startswith("GPU-")
            and probe["CUDA_context_created"] is False and probe["GPU_allocations_or_training"] is False
            and probe["accounting_or_reservation"] is False
            and probe["non_owned_process_details_recorded"] is False,
            "Actual frozen helper/provenance readonly UUID result without unrelated process details")
        if probe["physical_gpu"]==4:
            require(probe["GPU_UUID"]==runtime["target_GPU_UUID"],"Actual target GPU4 UUID equals exact native runtime")
            target=True
        rows=probe["own_process_memory"]
        require(isinstance(rows,list) and len(rows)<=256,"Bounded root-provided owned memory records")
        for row in rows:
            require(type(row["pid"]) is int and row["pid"]>0
                and type(row["nvml_bytes"]) is int and 0<=row["nvml_bytes"]<2**64-1
                and type(row["smi_bytes"]) is int and 0<=row["smi_bytes"]<2**64-1
                and type(row["difference_bytes"]) is int
                and row["difference_bytes"]==row["nvml_bytes"]-row["smi_bytes"],
                "Both actual NVML and SMI readings exist for every selected owned PID")
            owned=True
    require(target and owned,"Actual GPU4 UUID and at least one real owned-process memory comparison required")
