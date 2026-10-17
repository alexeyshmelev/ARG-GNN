"""Independent narrow source/AST/RAM additive outer2 review, no production module imports."""
from pathlib import Path
from types import SimpleNamespace,FunctionType,CodeType
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OLD="research/reviews/frozen_TRAIN001_observe_ready_20261017_v1";NEW=OLD[:-1]+"2"
W=NEW+"/worker_entry_v1.py";M=NEW+"/source_manifest.v1.json"
PREV="research/reviews/frozen_TRAIN001_observation_independent_20261017_v1/review.observation.v1.json"
checks=[]
def need(ok,msg="Independent RAM assertion"):
 if not ok:raise ValueError(msg)
def yes(name,ok=True):need(ok,name);checks.append(name)
def deny(name,call):
 try:call()
 except (ValueError,KeyError,TypeError):yes(name);return
 raise AssertionError("Expected rejection "+name)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def doc(p):return json.loads((ROOT/p).read_text())
old=(ROOT/OLD/"worker_entry_v1.py").read_text();new=(ROOT/W).read_text()
prev=doc(PREV);m=doc(M);oldm=doc(OLD+"/source_manifest.v1.json");h=doc(NEW+"/HANDOFF.v2.json")
yes("previous66check_independent_SOURCE_PASS_exact",ref(PREV)["sha256"]=="a39a3cfe7403d0c6e6809490ee9355d3f1fb57efdf8b8ed7e02542db88f31b5d"and prev["independent"]is True and prev["verdict"]=="SOURCE_PASS"and prev["blockers"]==[])
yes("previous_review_exact_source_and_manifest",prev["source_ref"]==ref(OLD+"/worker_entry_v1.py")and prev["source_manifest_ref"]==ref(OLD+"/source_manifest.v1.json"))
yes("sealed_handoff_exact",ref(NEW+"/HANDOFF.v2.json")["sha256"]=="5f60ce174248775d48b4143a3093df5eecd3bd8ebbc2a8d5865092af050d1f18")
yes("sealed_v2_worker_exact",ref(W)["sha256"]=="08fe20361a2294ecc00afa8c36d780cd620faf75562685b664102ae33d6c09c2"and h["source_ref"]==ref(W))
yes("sealed_v2_manifest_exact",ref(M)["sha256"]=="4ff3ac5b8e18faca841fd235e4203ca7f7569096169906e7eafa5c435ebb9530"and h["source_manifest_ref"]==ref(M))
normalized=new.replace(NEW,OLD).replace(NEW.replace("/","."),OLD.replace("/",".")).replace("outer_terminal.root_actual_v2.json","outer_terminal.root_actual_v1.json")
yes("entire_AST_only_namespace_and_outer_filename_delta",ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False))
yes("one_outer2_literal_no_outer1_literal",new.count('"outer_terminal.root_actual_v2.json"')==1 and '"outer_terminal.root_actual_v1.json"'not in new)
yes("entire_condition_including_original_science_caps_unchanged",m["condition"]==oldm["condition"])
for rr in m["refs"]:need(ref(rr["path"])==rr)
yes("all13_source_refs_authenticated_and12transitive_unchanged",len(m["refs"])==13 and m["refs"][1:]==oldm["refs"][1:])
yes("all47_roles64ceiling_no_retry_preserved",m["full_original_current_guard_roles"]==oldm["full_original_current_guard_roles"]==47 and m["original_consumer_ceiling"]==oldm["original_consumer_ceiling"]==64 and m["no_automatic_retry"]is True)
for k in("ROOT_recipe_ref","ROOT_request_template_ref","source_delta_ref"):yes("handoff_authenticates_"+k,h[k]==ref(h[k]["path"]))
code=compile(new,W,"exec")
def function(name,env):return FunctionType(next(c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name),env,name)
def fixture(outer_version=2):
 rows={};readpaths=[];condition=m["condition"];packet=NEW+"/RAM_simulation_packet";output="experiments/results/RAM_simulation_TRAIN001"
 def rref(path):return dict(path=str(path),sha256="0"*64)
 def put(path,value):rows[str(path)]=copy.deepcopy(value);return rref(path)
 config=put(packet+"/config.json",dict(condition["simulation_config_inactive"],execution_enabled=True))
 inp=put(packet+"/input.json",dict(schema="frozen_TRAIN001_current_Census_simulator_input_v1",root_authorizer="root_agent",execution_authorized=True,GPU_authorized=False,parent_id="american_c3_train_001",stage="simulate",phase="simulate_train_001",root_driver_ref=condition["simulation_stage"]["worker_source_ref"],main_source_ref=condition["simulation_stage"]["worker_source_ref"],source_sha256=condition["simulation_stage"]["source_sha256"],refs=dict(config=config),worker_auxiliary_scope=packet+"/worker_aux",output_directory=output))
 samples=[]
 for pop,n,cls in(("AFR",200,2),("EUR",200,0),("ASIA",200,1),("ADMIX",1000,None)):
  start=len(samples)
  samples.extend(dict(sample_index=start+j,observed_node_id=start+j,sample_id=pop+"_%05d"%(j//2),haplotype=j%2,age_generations=0,role="query"if pop=="ADMIX"else"pure_source_reference",observed_reference_class=cls)for j in range(n))
 roster=put(output+"/observed/roster.json",dict(parent_id="american_c3_train_001",class_order=["EUR","EAS","AFR"],samples=samples))
 ob=dict(schema="american_c3_observed_v1",parent_id="american_c3_train_001",split="train",sequence_length_bp=30000000,sample_unit="haploid",class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,genealogy_kind="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG",migrations_and_internal_population_labels_removed=True,roster_ref=roster)
 for k,n in(("genotypes_ref","genotypes_int8.npy"),("positions_ref","positions_bp0.npy"),("alleles_ref","site_alleles.jsonl")):ob[k]=rref(output+"/observed/"+n)
 observation=put(output+"/observed/manifest.json",ob)
 completed=put(output+"/completed.json",dict(status="completed_C3_simulation_reconstruction",parent_id="american_c3_train_001",sample_unit="haploid",class_order=["EUR","EAS","AFR"],observation_ref=observation,sites=100))
 wait=put(packet+"/worker_aux/actual_wait.json",dict(schema="American_C3_full_original_CPU_actual_wait_v1",input_ref=inp,completed_ref=completed,parent_id="american_c3_train_001",phase="simulate_train_001",actual_direct_worker_wait=True,actual_direct_launcher_wait=True,actual_OS_wait_status=0,returncode=0,all_owned_processes_and_groups_drained=True,operational_failure=None,cleanup_errors=[],adopted_waits=[],group_observations=[dict(pgid=123,absent=True)]))
 outer=put(packet+"/outer_terminal.root_actual_v%d.json"%outer_version,dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,input_ref=inp,completed_ref=completed,direct_wait_ref=wait,all_prior_launch_resume_write_capabilities_revoked=True,simulation_outputs_frozen=True,
  ROOT_additive_revocation_provenance=dict(previous_immutable_outer_ref=rref(packet+"/outer_terminal.root_actual_v1.json"),no_historical_file_rewrite=True)))
 prior=dict(input_ref=inp,config_ref=config,completion_ref=completed,wait_ref=wait,outer_ref=outer,observation_ref=observation,roster_ref=roster)
 def read(r):readpaths.append(r["path"]);return copy.deepcopy(rows[r["path"]])
 e=dict(c=SimpleNamespace(read=read,ref=rref),Path=Path,PARENT="american_c3_train_001",need=need)
 e["observation_metadata"]=function("observation_metadata",e)
 return SimpleNamespace(join=function("simulation_prerequisites",e),prior=prior,rows=rows,reads=readpaths,condition=condition)
f=fixture();yes("exact_prerequisites_accept_additive_outer2_and_provenance",f.join(f.prior,f.condition)==f.prior)
yes("RAM_prerequisites_read_only_seven_metadata_no_payload",set(f.reads)=={r["path"]for r in f.prior.values()}and not any(p.endswith((".npy",".jsonl",".trees"))for p in f.reads))
f=fixture(1);deny("old_immutable_outer1_cannot_substitute",lambda:f.join(f.prior,f.condition))
for name,key,field,value in[
 ("outer_unconsumed","outer_ref","actual_terminal_consumed",False),("outer_nonzero","outer_ref","actual_exit_code",1),
 ("outer_unrevoked","outer_ref","all_prior_launch_resume_write_capabilities_revoked",False),("outputs_unfrozen","outer_ref","simulation_outputs_frozen",False),
 ("outer_wrong_input","outer_ref","input_ref",{}),("worker_unknown","wait_ref","actual_direct_worker_wait",False),
 ("worker_nonzero","wait_ref","actual_OS_wait_status",256),("worker_undrained","wait_ref","all_owned_processes_and_groups_drained",False),
 ("truth_in_inputs","observation_ref","query_ancestry_truth_in_inputs",True)]:
 f=fixture();f.rows[f.prior[key]["path"]][field]=value
 deny("unchanged_actual_prerequisite_rejects_"+name,lambda:f.join(f.prior,f.condition))
f=fixture();f.prior.pop("roster_ref");deny("all_seven_prior_roles_still_required",lambda:f.join(f.prior,f.condition))
print(json.dumps(dict(schema="TRAIN001_observation_outer2_independent_delta_checks_v2",status="INDEPENDENT_PASS",count=len(checks),checks=checks,source_ref=ref(W),source_manifest_ref=ref(M),previous_independent_review_ref=ref(PREV),
 actual_guard_imports_runtime_configurations_resource_process_GPU_probes_admissions_launches_closes=0,actual_outer_records_read_or_written=False,scientific_payload_truth_checkpoint_reads_or_hashes=0),sort_keys=True,indent=2))
