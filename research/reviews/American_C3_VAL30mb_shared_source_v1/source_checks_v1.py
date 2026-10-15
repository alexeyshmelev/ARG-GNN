"""Pure metadata and tiny mocked/source-extracted CPU checks. No science/guard imports."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,copy,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/American_C3_VAL30mb_shared_source_v1"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(name,call,kind=ValueError):
    try:call()
    except kind:checks.append(name);return
    raise AssertionError(name+" accepted")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def take(src,names,ns):
    nodes=[n for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<source-helpers-only>","exec"),ns)
    return ns
def require(ok,msg):
    if not ok:raise ValueError(msg)
loader=importlib.util.spec_from_file_location("_VAL_packet_metadata_checks",HERE/"root_packet_v1.py")
packet=importlib.util.module_from_spec(loader);loader.loader.exec_module(packet)
pins=packet.frozen()
check("all original source/runtime and contract refs exact",all(packet.ref(p)==v for p,v in pins.items()))
check("no guard/scientific module imported by packet",all(s not in __import__("sys").modules for s in ["numpy","tskit","msprime","stdpopsim","torch","tsinfer","tsdate","_C3_VAL_root_current_guard"]))
for stage in packet.STAGES:
    plan=packet.stage_plan(stage)
    check("full30Mb original VAL-only plan "+stage,plan["parent_id"]=="american_c3_val_000" and plan["split"]=="val" and plan["sequence_length_bp"]==30000000)
    check("no truth TEST GNN training or transfer "+stage,all(plan[k] is False for k in ["TEST_access","GNN_VAL_optimization","query_ancestry_truth_payload_access","native_model_transfer_from_TRAIN000"]))
    check("exact common.cli source closure "+stage,plan["source_closure_refs"][0]==pins[packet.COMMON] and plan["source_ref"]==pins[packet.PRODUCERS[stage]])
reg=read(ROOT/"research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json")
check("original20TRAIN5VAL10TEST registry counts",[(sp,sum(r["split"]==sp for r in reg["parents"])) for sp in ["train","val","test"]]==[("train",20),("val",5),("test",10)])
row=next(r for r in reg["parents"] if r["parent_id"]=="american_c3_val_000")
check("exact original VAL000 seeds",row==dict(parent_id="american_c3_val_000",split="val",ancestry_seed=31001,mutation_seed=1031001,path_sampling_seed=93001))
oldsim=read(ROOT/"research/reviews/American_C3_root_execution_20261015_v1/admitted_full30mb_v2/config.json")
newsim=packet.configuration("simulate")
allowed={"execution_enabled","parent_id","split","ancestry_seed","mutation_seed"}
check("simulation options unchanged outside explicit heldout identity", {k:v for k,v in oldsim.items() if k not in allowed}=={k:v for k,v in newsim.items() if k not in allowed})
check("simulation inactive and exact full extent",newsim["execution_enabled"] is False and newsim["sequence_length_bp"]==30000000)
oldinf=read(ROOT/"research/reviews/American_C3_observation_supervisor_repair_source_v1/admitted_full30mb_repair_v1/config.json")
newinf=read(HERE/"inference.inactive.json")
allowed={"execution_enabled","observation_ref"}
check("inference options exactly completed original full30Mb", {k:v for k,v in oldinf.items() if k not in allowed}=={k:v for k,v in newinf.items() if k not in allowed})
check("inference all-site unary policy and8threads",newinf["num_threads"]==8 and newinf["unary_preprocessing_policy"]=="direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1")
paths=read(HERE/"native_paths.inactive.json");forecast=read(HERE/"storage_forecast.json")
src=(ROOT/"src/arggnn/american_c3_paths_v1.py").read_text()
f=take(src,{"storage_bound"},{})
check("original native bound exact and within16GB",f["storage_bound"](paths)==forecast["native_original_storage_bound_bytes"]==15218324396 and forecast["native_total_planning_bytes"]==15370367916<paths["output_cap_bytes"])
check("count ceilings honest planning and nontruncating",forecast["VAL_counts_unknown"] is True and forecast["scientific_payloads_read"] is False and paths["max_raw_path_nodes"]==1024 and paths["path_sampling_seed"]==93001 and paths["query_focal_count"]==10000)
check("all native count ceilings25percent above TRAIN000",all(paths["max_"+key]==(5*forecast["baseline_counts"][key]+3)//4 for key in ["unique_paths","path_tokens","path_occurrences","vocabulary_nodes"]))
check("all templates inactive",all(read(HERE/n)["execution_enabled"] is False for n in ["simulation.inactive.json","inference.inactive.json","native_paths.inactive.json","shared_contract.inactive.json"]))
# Source-extracted metadata lineage validator; read only from in-memory dictionary.
def fixture(stage):
    prior="simulate" if stage=="infer" else "infer"
    complete_ref=dict(path=packet.identity(prior)[1]+"/completed.json",sha256="fake-metadata-sha")
    wait_ref=dict(path="review-only-fake/wait.json",sha256="fake-metadata-sha")
    obs_ref=dict(path=packet.identity(prior)[1]+"/observed/manifest.json",sha256="fake-metadata-sha")
    completion=dict(status="completed_C3_simulation_reconstruction" if stage=="infer" else "completed_current_C3_inference",observation_ref=obs_ref)
    wait=dict(returncode=0,actual_direct_worker_wait=True,completed_ref=complete_ref,owned_worker_group_ESRCH=True,operational_failure=None)
    m=dict(parent_id=packet.PARENT,split="val",sequence_length_bp=30000000,sample_unit="haploid",class_order=["EUR","EAS","AFR"],
      counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,
      genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",sites_and_sample_identities_verified=True)
    objects={complete_ref["path"]:completion,wait_ref["path"]:wait,obs_ref["path"]:m}
    touched=[]
    def fake_read(r):touched.append(r["path"]);return objects[r["path"]]
    ns={"require":require,"identity":packet.identity,"PARENT":packet.PARENT,"read":fake_read}
    take((HERE/"root_packet_v1.py").read_text(),{"upstream_observation"},ns)
    return ns,complete_ref,wait_ref,m,wait,touched
for stage in ["infer","native_paths"]:
    ns,cr,wr,m,w,touched=fixture(stage)
    check("mock valid genuine prior completion accepted "+stage,ns["upstream_observation"](stage,cr,wr)["path"].endswith("/observed/manifest.json"))
    check("no truth payload followed by metadata linker "+stage,len(touched)==3 and all("/truth/" not in p for p in touched))
for field,value in [("parent_id","american_c3_train_000"),("split","test"),("sequence_length_bp",2000000),("query_ancestry_truth_in_inputs",True)]:
    ns,cr,wr,m,w,_=fixture("infer");m[field]=value
    reject("mock wrong upstream "+field+" blocked",lambda ns=ns,cr=cr,wr=wr:ns["upstream_observation"]("infer",cr,wr))
for field,value in [("returncode",1),("actual_direct_worker_wait",False),("owned_worker_group_ESRCH",False),("operational_failure","failed")]:
    ns,cr,wr,m,w,_=fixture("infer");w[field]=value
    reject("mock failed or undrained prior "+field+" blocked",lambda ns=ns,cr=cr,wr=wr:ns["upstream_observation"]("infer",cr,wr))
# Tiny native vocabulary checks execute only the original pure Python function.
a=ROOT/".runtime/author_sources/argformer_f7545be7bca0"
text=(a/"src/data/prepare_data_pretrain.py").read_text()
env=take(text,{"build_vocab_from_sequences"},{"SPECIAL_OFFSET":4,"SPECIAL_TOKENS":{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3}})
va=env["build_vocab_from_sequences"]([[0,10],[1,10]])
vb=env["build_vocab_from_sequences"]([[0,11],[1,11]])
check("author vocab exact raw node IDs after special offset",va["node_id_tokens"]=={"node_0":4,"node_1":5,"node_10":6})
check("same token rank can refer to different independentARG raw nodes",va["node_id_tokens"]["node_10"]==vb["node_id_tokens"]["node_11"])
s=(a/"src/arg_data.py").read_text();tree=ast.parse(s)
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_tokenize_raw_sequence_from_vocab")
env={"List":list};exec(compile(ast.Module(body=[method],type_ignores=[]),"<author-pure-tokenizer>","exec"),env)
check("same namespace tokenizer retains raw-ID lookup",env["_tokenize_raw_sequence_from_vocab"](NS(_node_id_token_map=va["node_id_tokens"]),[0,10])==[4,6])
reject("foreign absent raw node is KeyError not alignment",lambda:env["_tokenize_raw_sequence_from_vocab"](NS(_node_id_token_map=va["node_id_tokens"]),[0,11]),KeyError)
contract=read(HERE/"shared_contract.inactive.json")
check("separate transductive/native and inductive/GNN roles explicit",contract["GNN_VAL_optimization_forbidden"] is True and contract["native_protocol"]["per_target_transductive_MLM_on_observed_paths"] is True and contract["native_protocol"]["cross_ARG_frozen_checkpoint_transfer_allowed"] is False)
check("private query truth and TEST access forbidden",contract["query_ancestry_truth_for_training_forbidden_for_every_method"] is True and contract["TEST_access_forbidden"] is True)
check("all-method seals and original GNN gate retained",contract["all_method_prediction_seals_required_before_common_truth_scoring"] is True and "all six checkpoint x five original VAL" in contract["original_GNN_selection_protocol_not_relaxed"])
# Static assurance that only explicit root-admit mode imports/calls the guard.
packet_source=(HERE/"root_packet_v1.py").read_text();pt=ast.parse(packet_source)
top_imports=[n for n in pt.body if isinstance(n,(ast.Import,ast.ImportFrom))]
check("packet source compiles without runtime project imports",all(not (getattr(n,"module","") or "").startswith(("arggnn","research")) for n in top_imports))
check("no worker launch or close/credit operation implemented","subprocess" not in packet_source and "apply_close_completed" not in packet_source and 'operation_draft_v1("reserve"' in packet_source)
for p in HERE.glob("*.py"):compile(p.read_text(),str(p),"exec")
check("all packet Python sources compile",True)
result={"schema":"American_C3_VAL30mb_pure_source_metadata_checks_v1","status":"PASS","checks_passed":len(checks),"checks":checks,
"scientific_or_guard_runtime_imports":False,"scientific_arrays_or_payloads_read":False,"actual_simulation_inference_native_training_admission_process_GPU_or_Git":False,
"source_refs":{p:sha(ROOT/p) for p in pins},"root_packet_sha256":sha(HERE/"root_packet_v1.py")}
(HERE/"checks_result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":"PASS","checks_passed":len(checks),"checks_sha256":sha(HERE/"checks_result.json")}))
