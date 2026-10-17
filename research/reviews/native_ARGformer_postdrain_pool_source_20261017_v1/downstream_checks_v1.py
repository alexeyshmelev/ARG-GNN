"""Actual metadata bridge/profile/forecast RAM checks. No scientific imports."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,hashlib
HERE=Path(__file__).parent;ROOT=HERE.parents[2]
def need(x,message):
    if not x:raise ValueError(message)
def extract(path,names,ns):
    nodes=[copy.deepcopy(n)for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    need({n.name for n in nodes}==set(names),"Exact actual source definitions")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<RAM exact metadata functions>","exec"),ns);return ns
def rejected(call):
    try:call()
    except (ValueError,KeyError,TypeError,AssertionError):return True
    return False
def run_v1():
    registry=json.loads((HERE/"parent_registry.v1.json").read_text())
    original_counts=json.loads((ROOT/"experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json").read_text())
    contract=extract(HERE/"pool_contract_v1.py",["slot_spec_v1","validate_native_roster_v1"],dict(require=need,read=lambda _:registry,ref=lambda x:x,h=SimpleNamespace(HERE="fake"),SLOTS={"H058":dict(cuda_cap_bytes=28*2**30,margin_mib=2048,peak_mib=30720),"native":dict(cuda_cap_bytes=32*2**30,margin_mib=2048,peak_mib=34816)}))
    slot=contract["slot_spec_v1"];validate=contract["validate_native_roster_v1"]
    native74=slot(dict(worker_kind="native",native_phase="contrastive",resource_profile="original_downstream_74GiB_proposal"))
    native32=slot(dict(worker_kind="native",native_phase="contrastive",resource_profile="explicit_unqualified_full_stage_32GiB"))
    cpu=slot(dict(worker_kind="native",native_phase="finalize"))
    need(native74["cuda_cap_bytes"]==74*2**30 and native74["peak_mib"]==77824,"Prior74GiB downstream proposal preserved explicitly")
    need(native32["cuda_cap_bytes"]==32*2**30 and native32["peak_mib"]==34816 and cpu["peak_mib"]==0,"Explicit unqualified32GiB full workload or original CPU metadata finalizer")
    need(native74["peak_mib"]+30720+4096>81920,"74GiB proposal cannot silently coexist with H058 on80GiB")
    need(native32["peak_mib"]+30720+4096==69632,"Nominal shared slot68GiB before all external/driver usage")
    jobs={"H058":dict(worker_kind="H058"),"CL0":dict(worker_kind="native",parent_id="american_c3_val_000",native_phase="contrastive"),"MLM1":dict(worker_kind="native",parent_id="american_c3_val_001",native_phase="mlm")}
    validate(jobs)
    bad=copy.deepcopy(jobs);bad["CL0"]["native_phase"]="mlm"
    need(rejected(lambda:validate(bad)),"Current VAL000 MLM cannot be replayed")
    bad=copy.deepcopy(jobs);bad["repeatCL"]=dict(jobs["CL0"])
    need(rejected(lambda:validate(bad)),"Duplicate original parent/phase declaration rejected")
    # The actual bridge source imports only stdlib when defining original
    # stage/IO wrappers. No author/model/production entry is called by this test.
    counts=copy.deepcopy(original_counts)
    records={"counts":counts}
    c=SimpleNamespace(require=need,COUNTS_REF="counts",PARENT="american_c3_val_001",PREP_REF=dict(path="RAM_preparation1",sha256="a"*64),PREP_CONFIG_REF=dict(path="RAM_config1",sha256="b"*64),PREP_OUTPUT="RAM_original_parent1",EXPECTED_CONFIG=dict(scientific_protocol=dict(contrastive=dict(updates=2),retrieval=dict(reference_rows=12288))),read=lambda r:records[r] if isinstance(r,str)else json.loads((ROOT/r["path"]).read_text()),raw=lambda r:(ROOT/r["path"]).read_bytes(),local=lambda r:ROOT/r)
    counts["MLM_optimizer_updates"]=10
    counts["datasets"]["contrastive/train"]["rows"]=8192;counts["datasets"]["contrastive/val"]["rows"]=4096
    namespace=extract(HERE/"downstream_context_v1.py",["contextual_module_v1"],dict(c=c,ast=ast,types=__import__("types")))
    stages_ref=dict(path="research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/scientific_stages_v1.py",sha256="455704b88e9247021c09566609801838f88a118420cde856e487aaef4cee30da")
    stages=namespace["contextual_module_v1"](stages_ref,"_RAM_defined_original_stage_wrappers")
    need(stages.REFS["preparation"]==c.PREP_REF and stages.ROW_COUNTS["contrastive/train"]==8192,"Original metadata/path/count bridge is parent-specific")
    checkpoint=dict(path="RAM_parent1_full_MLM.pt",sha256="1"*64);complete_ref=dict(path="RAM_completed",sha256="2"*64);wait_ref=dict(path="RAM_wait",sha256="3"*64)
    complete=dict(stage="mlm",parent_id=c.PARENT,completed_epochs=5,actual_optimizer_updates=10,observed_counters_verified=True,trainer_optimizer_counters=dict(unique_step_values=[10]),checkpoint_optimizer_counters=dict(unique_step_values=[10]),initializer_checkpoint_ref=None,checkpoint_ref=checkpoint,preparation_ref=c.PREP_REF,author_source_manifest_ref=stages.REFS["author_manifest"],query_ancestry_truth_opened=False)
    wait=dict(returncode=0,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,completed_ref=complete_ref,operational_failure=None)
    outer=dict(actual_exit_code=0,actual_terminal_consumed=True,actual_wait_ref=wait_ref)
    need(stages.full_mlm_metadata_prerequisite_v1(complete,wait,outer,completion_ref=complete_ref,wait_ref=wait_ref,checkpoint_ref=checkpoint)==checkpoint,"Actual generic original metadata check accepts only full same-parent completed budget")
    wrong=copy.deepcopy(complete);wrong["parent_id"]="american_c3_val_000"
    need(rejected(lambda:stages.full_mlm_metadata_prerequisite_v1(wrong,wait,outer,completion_ref=complete_ref,wait_ref=wait_ref,checkpoint_ref=checkpoint)),"Cross-parent checkpoint/preparation rejected")
    failed=dict(wait,returncode=1)
    need(rejected(lambda:stages.full_mlm_metadata_prerequisite_v1(complete,failed,outer,completion_ref=complete_ref,wait_ref=wait_ref,checkpoint_ref=checkpoint)),"Original failed/unknown MLM wait cannot become prerequisite")
    io=namespace["contextual_module_v1"](dict(path="research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/efficient_downstream_v1.py",sha256="bab7057a958e9cfd51f503b5680034c0067226a661b820eb42bb16c7133a94b9"),"_RAM_defined_original_IO_wrappers")
    need(io.REFS["preparation"]==c.PREP_REF,"Original full author I/O option bound to exact same-parent preparation")
    forecast_ns={};exec(compile((HERE/"downstream_storage_forecast_v1.py").read_text(),"<scalar forecast>","exec"),forecast_ns)
    forecast=forecast_ns["forecast_v1"](original_counts)
    need(forecast["all_saved_tensor_payload_upper_bytes"]==49979628096 and forecast["retained_whole_downstream_scope_cap_bytes"]==65*2**30,"Full four-save payload and original65GiB storage requirement retained")
    need(forecast["original_four_full_author_saves_preserved"] and not forecast["CPU_retrieval_substitution"],"No tensor/workload/device algorithm reduction")
    context_ast=ast.parse((HERE/"downstream_context_v1.py").read_text())
    fn=next(n for n in context_ast.body if isinstance(n,ast.FunctionDef)and n.name=="source_IO_v1")
    need("return source_stages_v1()"in ast.unparse(fn),"Original I/O wrapper uses actual parent-bound stages, no stale VAL000 factory")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=str(p))
    return dict(schema="native_full_downstream_postdrain_author_RAM_checks_v1",status="PASS_STATIC_RAM_ONLY",original_complete_same_parent_MLM_metadata_positive=True,cross_parent_and_failed_MLM_wait_rejected=True,original_contextual_stage_and_IO_metadata_only=True,original_four_embeddings_normalization_pooling_top1_batch_chunk_retained=True,original65GiB_whole_storage_requirement_retained=True,prior74GiB_GPU_proposal_not_claimed_fit=True,new32GiB_attempt_requires_explicit_unqualified_ROOT_condition=True,native74_blocks_H058_concurrency=True,current_VAL000_MLM_no_replay=True,CPU_retrieval_substitution=False,scientific_model_or_numerical_payloads=0,actual_author_stage_calls=0,real_process_guard_probe_device_admission_journal_Git_operations=0,fresh_independent_whole_join_review_required=True)
if __name__=="__main__":
    result=run_v1();out=HERE/"downstream_checks.final.v1.json"
    with out.open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps(dict(status=result["status"],result_path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),sort_keys=True))
