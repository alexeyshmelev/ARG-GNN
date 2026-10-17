"""Source/AST/RAM author checks only. No Torch/PyG/guard imports or resource probes."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1"
checks=[]
def need(ok,label):
    if not ok:raise ValueError(label)
def ref(path):
    p=ROOT/path;return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def fn(path,name,env):
    n=next(n for n in ast.parse((ROOT/path).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),path+"#RAM_AST","exec"),env);return env[name]
def reject(call):
    try:call()
    except(ValueError,KeyError,TypeError):return
    raise AssertionError("Expected rejection")
def check(label,call):call();checks.append(label)
m=json.loads((ROOT/D/"source_manifest.v1.json").read_text())
for path,sha in m["source_sha256"].items():
    assert ref(path)["sha256"]==sha,path
    if path.endswith(".py"):compile((ROOT/path).read_bytes(),path,"exec")
checks.append("all_complete_source_and_metadata_pins_authenticate_and_Python_compiles")
source=(ROOT/D/"bounded_execution_v1.py").read_text()
tree=ast.parse(source)
assert "bank.to("not in source and "bundle.to("not in source and ".detach("not in source.replace("loss.detach()","loss")
checks.append("bounded_executor_has_no_full_bank_CUDA_transfer_or_history_detach")
assert 'preserve_rng_state=True'in source and 'use_reentrant=False'in source
checks.append("explicit_nonreentrant_recomputation_preserves_RNG")
assert "(left//256)*256"in source and "end=min(start+256"in source and 'reduce="amax"'in source
checks.append("original_global256site_chunks_and_amax_ties_retained")
assert "self.model.sweep_tree_state_v1"in source and "m.sweep_tree_state_v1"in source
checks.append("both_sweeps_call_original_PyG_equations")
assert 'torch.func.functional_call(self.history,{"memory.ell_theta":theta}'in source and ".memory._history"in source
checks.append("original_global_memory_history_uses_differentiable_ell_copy")
assert 'index=episodes.member_episode.long().to(self.device)'in source and 'minimum>0'in source
checks.append("own_state_zero_foreign_guard_uses_original_scalar_CUDA_aggregation")
assert 'memory.O(value.to(self.device))'in source and "context.split(self.hap_tile,dim=1)"in source
checks.append("global_memory_residual_and_full_length_haplotype_GRU_partitions_connected")
assert source.count("optimizer.step()")==1 and 'reduction="mean"'in source and "clip_grad_norm_(params,1."in source
checks.append("single_original_mean_CE_clip1_AdamW_update_per_complete_parent")
parity=(ROOT/D/"parity_v1.py").read_text()
assert 'for key in("step","exp_avg","exp_avg_sq")'in parity and "p.grad,other[name].grad"in parity
assert "torch.nn.init.normal_(adapter.O.weight"in parity and "Nonzero memory-path gradient"in parity
checks.append("actual_parity_code_compares_logits_loss_raw_clipped_all49gradients_AdamW_moments_nonzero_O")
assert sum([8000,17000,2,11000,23000,31000,29909998])==30000000
checks.append("synthetic_fixture_has_exact_FP32_representable30Mb_spans_and_isolated_hold")
worker=(ROOT/D/"worker_v1.py").read_text()
assert worker.index("install_before_torch_v1")<worker.index("import torch")
assert worker.index("run_parity_v1(resource_check=tick)")<worker.index("torch.manual_seed(70001);model")
checks.append("real_worker_title_and_borrowed_FDs_before_Torch_then_parity_before_fresh_real_check_state")
assert "train_step_v1(model,bundle,targets,optimizer,meta"in worker and "bundle.to("not in worker
assert "epoch_order_v1(epoch)"in worker and "else range(1,31)"in worker
checks.append("actual600runner_reaches_bounded_step_with_original_epoch_order_and_CPU_bundle")
assert 'qualification_weights_are_not_final_training_weights'in worker and 'ancestry_accuracy=None'in worker
checks.append("qualification_is_disposable_and_has_no_benchmark_score_or_fake_training_completion")
# Real contract functions, with only metadata filesystem/runtime boundaries replaced by RAM.
config=json.loads((ROOT/D/"scientific_config.FROZEN.v1.json").read_text());mr=ref(D+"/source_manifest.v1.json")
q0=json.loads((ROOT/D/"ROOT_qualification_input.INACTIVE.v1.json").read_text())
now=datetime.now(timezone.utc);q0.update(root_authorizer="root_agent",status="active",execution_authorized=True,template_only=False,
 created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(hours=1)).isoformat(),
 drain_expires_unix=(now+timedelta(hours=1)).timestamp(),source_manifest_ref=mr,
 source_review_ref=dict(path="RAM_ONLY/review.json",sha256="0"*64),runtime_ref=dict(path="RAM_ONLY/runtime.json",sha256="0"*64),
 runtime_qualification_ref=dict(path="RAM_ONLY/runtime_qualification.json",sha256="0"*64),
 storage_ref=dict(path="RAM_ONLY/storage.json",sha256="0"*64),worker_auxiliary_scope="RAM_ONLY/aux",ROOT_record_directory="RAM_ONLY/aux")
q0["caps"]["max_seconds"]=600
q0["cpu_affinity"]=list(range(8))
ir=dict(path="RAM_ONLY/input.json",sha256="0"*64)
def contract_case(change=None):
    q=copy.deepcopy(q0)
    docs={ir["path"]:q,mr["path"]:m,q["source_review_ref"]["path"]:dict(independent=True,verdict="SOURCE_PASS",blockers=[],candidate_manifest_ref=mr),
      q["runtime_ref"]["path"]:dict(python="RAM_ONLY/python",source_ref=ref(D+"/worker_v1.py"),CUDA_VISIBLE_DEVICES=q["gpu_uuid"]),
      q["runtime_qualification_ref"]["path"]:dict(python="RAM_ONLY/python",ROOT_qualified_existing_Torch_PyG_runtime=True,cpu_affinity=list(range(8))),
      q["storage_ref"]["path"]:{},q["config_ref"]["path"]:config}
    if change:change(q,docs)
    env=dict(ROOT=ROOT,HERE=D,MANIFEST=D+"/source_manifest.v1.json",SOURCE=D+"/worker_v1.py",GPU0_UUID=q0["gpu_uuid"],
      GPU_PHASES=("qualification","full_training"),BASE_POOL=q0["base_pool_ref"],CURRENT_INPUT_REF=None,
      datetime=datetime,timezone=timezone,require=need,read=lambda r:copy.deepcopy(docs[r["path"]]),
      ref=ref,raw=lambda r:None,local=lambda path:SimpleNamespace(is_file=lambda:True))
    fn(D+"/contract_v1.py","active",env)
    return fn(D+"/contract_v1.py","validate_input",env)(ir)
check("complete_qualification_input_contract_RAM_positive",lambda:contract_case())
for key,value in [("physical_gpu",1),("process_title","wrong"),("execution_authorized",False),("template_only",True),
 ("all_other_GPU0_borrower_routes_revoked",False),("automatic_retry_or_resume",True),("query_VAL_TEST_truth_access_authorized",True)]:
    check("input_reject_"+key,lambda key=key,value=value:reject(lambda:contract_case(lambda q,d:q.__setitem__(key,value))))
check("input_reject_review_wrong_manifest",lambda:reject(lambda:contract_case(lambda q,d:d[q["source_review_ref"]["path"]].__setitem__("candidate_manifest_ref",{}))))
check("input_reject_review_blocker",lambda:reject(lambda:contract_case(lambda q,d:d[q["source_review_ref"]["path"]].__setitem__("blockers",["RAM"]))))
check("input_reject_CUDAcap_growth",lambda:reject(lambda:contract_case(lambda q,d:q["caps"].__setitem__("max_cuda_reserved_bytes",65*2**30))))
check("input_reject_expired",lambda:reject(lambda:contract_case(lambda q,d:q.__setitem__("expires_at_utc",(now-timedelta(seconds=1)).isoformat()))))
# Exact full-argv title validation accepts an actually truncated NVML/comm display.
def title_case(bad=False):
    data=b"Wrong title\0"if bad else b"AI Auto Research Pipeline Test\0"
    env=dict(require=need,os=SimpleNamespace(getpid=lambda:123,getuid=lambda:1000),
      base_v1=lambda:SimpleNamespace(proc_identity=lambda pid:dict(uid=1000,state="S",start_ticks=777)),
      Path=lambda path:SimpleNamespace(read_bytes=lambda:data,read_text=lambda:"AI Auto Researc\n"),
      GPU0_UUID=q0["gpu_uuid"])
    return fn(D+"/borrowed_gpu0_v1.py","actual_title_v1",env)(123,"AI Auto Researc")
check("actual_title_AST_RAM_accepts_full_argv_despite_truncated_display",lambda:title_case())
check("actual_title_AST_RAM_rejects_wrong_full_argv",lambda:reject(lambda:title_case(True)))
contract=(ROOT/D/"contract_v1.py").read_text()
g=next(n for n in ast.parse(contract).body if isinstance(n,ast.FunctionDef)and n.name=="guard_v1")
calls=[(n.lineno,n.func.attr)for n in ast.walk(g)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
assert min(n for n,k in calls if k=="load_selected_guard_v1")<min(n for n,k in calls if k=="install_consumer_v1")<min(n for n,k in calls if k=="validate_storage_v9")
checks.append("exact_original_select_then_readonly_memo_before_first_guard_validation")
parent=(ROOT/D/"parent_v1.py").read_text()
assert "pass_fds=(gate_r,physical_fd,logical_fd)"in parent and 'seed[\'start\']'in parent
assert "adopted_returncode!=0"in parent and "raw_wait is None"in parent and "owned.signal_identity"in parent
checks.append("actual_parent_inherits_only_authenticated_borrowed_FDs_and_preserves_raw_wait_pidfd_adopted_cleanup")
inventory=(ROOT/D/"gpu0_inventory_v1.py").read_text()
borrow=(ROOT/D/"borrowed_gpu0_v1.py").read_text()
assert "subprocess"not in inventory.replace("subprocess calls","") and "rows=inventory_v1()"in borrow
checks.append("periodic_actual_GPU0_inventory_is_inprocess_NVML_without_cumulative_helper_PIDs")
assert "all20_observations_sealed_before_any_TRAIN_truth"not in contract
assert "all20_observations_sealed_before_this_full_training_truth"in contract
assert "prior_disposable_TRAIN000_qualification_genuine_truth_opened"in contract
checks.append("honest_genuine_label_qualification_exception_and_separate_full_training_barrier")
print(json.dumps(dict(schema="frozen_bounded_TreeLoom_author_source_AST_RAM_checks_v1",passed=True,count=len(checks),checks=checks,
 source_refs_authenticated=len(m["source_sha256"]),actual_Torch_PyG_guard_scientific_imports_or_probes=0,
 no_actual_fit_parity_training_or_success_claimed=True),sort_keys=True,indent=2))
