"""Source/JSON/RAM assessment only: full original VAL002..004 converter route."""
from pathlib import Path
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");D=ROOT/"research/reviews/FLARE_original_VAL002004_conversion_route_source_20261017_v1"
reads={};ram={};checks=[]
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 assert p.suffix in(".json",".py",".md",".txt",".patch",".sh")and p.is_file()and not p.is_symlink()and p.stat().st_size<=2**20
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 if (r["path"],r["sha256"])in ram:return copy.deepcopy(ram[(r["path"],r["sha256"])])
 assert Path(r["path"]).suffix==".json"and ref(r["path"])==r;reads[r["path"]]=r
 return json.loads((ROOT/r["path"]).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def check(label,ok=True):require(ok,label);checks.append(label)
def put_ram(label,value):
 r=dict(path="RAM_ONLY_NOT_ACTUAL/"+label+".json",sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest());ram[(r["path"],r["sha256"])]=copy.deepcopy(value);return r
p=ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py"
core=types.ModuleType("_original_full_context_and_config_AST_only");basis=read(ref(p.parent/"VAL000_basis.v1.json"))
core.__dict__.update(HERE=p.parent,ROOT=ROOT,copy=copy,json=json,MISSING=tuple("american_c3_val_%03d"%i for i in range(1,5)),require=require,OPTIONS=basis["configurations"]["native_config"]["native_options"],COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1",CLASSES=("EUR","EAS","AFR"),_tools=lambda:(read,lambda *a:(_ for _ in()).throw(AssertionError("No payloads")),ref))
names=("_local","_local_ref","_sources","_context","_gzip_bound","_conversion_bound","_native_output_bound","inactive_stage_configs_v1")
nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(nodes)==len(names)
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<unchanged source metadata functions>","exec"),core.__dict__)
known=read(ref(p.parent/"original_remaining_inputs.v1.json"))
ctx=read(ref("research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_3h_setupv4_20261017_v1/american_c3_val_001.FLARE_context.json"))
readiness=read(ref("research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v3/readiness.v1.json"))
sim2=read(ref("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/dependencies.v1.json"))["ORIGINAL_SIMULATION_CONFIGS"]["american_c3_val_002"]
rows={};summary={}
for index in(2,3,4):
 parent="american_c3_val_%03d"%index
 if index<4:row=copy.deepcopy(known["known_inputs"][parent])
 else:
  obs=ref("experiments/results/American_C3_val004_full30mb_infer_native14v3_v3/observed/manifest.json")
  prep_ref=ref("experiments/results/American_C3_val004_full30mb_native_paths_fresh7500_v1/completed.json");prep=read(prep_ref)
  row=dict(parent_id=parent,observation_ref=obs,native_preparation_ref=prep_ref,focal_registry_ref=prep["focals_ref"],plan_ref=ref("experiments/results/American_C3_val004_full30mb_plan_fresh7500_v1/american_c3_val_004_block_index.json"),haplotypes=1600,input_sites=229643,count_source_kind="completed_original_C3_plan_JSON_metadata_no_array_read")
 row["simulation_config_ref"]=sim2 if index==2 else readiness["parents"][parent]["simulation_config_ref"]
 row["simulation_completion_ref"]=known["known_simulation_completion_refs"][parent]
 plan=read(row["plan_ref"]);prep=read(row["native_preparation_ref"])
 check(parent+"_full_original_plan_and_focal_support",plan.get("parent_id")==parent and plan.get("haplotypes")==1600 and plan.get("sequence_length_bp")==30000000 and plan.get("sites")==row["input_sites"]and plan.get("focal_records")==10000 and plan.get("all_trees_and_samples_retained")is True and plan.get("targets_opened")is False and plan.get("observation_ref")==row["observation_ref"]and plan.get("focal_registry_ref")==row["focal_registry_ref"]and prep.get("query_focal_count")==10000)
 count=dict(schema="American_C3_full30Mb_observation_count_root_v1",source_observation_ref=row["observation_ref"],num_sites=row["input_sites"],num_samples=1600,sequence_length=30000000,truth_opened=False,count_source_ref=row["plan_ref"],parent_id=parent)
 row["count_manifest_ref"]=put_ram(parent+"_count",count);ctx["targets"][parent]=row;rows[parent]=row
ctx_ref=put_ram("all_four_context",ctx)
core._sources();core._context(ctx_ref)
check("all_four_original_core_context_predicates_pass_with_RAM_counts_no_payloads")
for parent,row in rows.items():
 cfg=core.inactive_stage_configs_v1(ctx_ref,parent)["conversion"]
 frozen=copy.deepcopy(basis["configurations"]["conversion_config"])
 changed={k for k in set(cfg)|set(frozen)if cfg.get(k)!=frozen.get(k)}
 expected={"execution_enabled","registration_ref","count_manifest_ref","focal_registry_ref","max_sites_bound","external_truth_ref","observation_ref","simulation_config_ref","max_sites","output_cap_bytes"}
 check(parent+"_inactive_conversion_changes_only_original_target_fields",changed<=expected and cfg["execution_enabled"]is False and cfg["external_truth_ref"]is None and cfg["native_options"]==core.OPTIONS and cfg["max_sites"]==row["input_sites"])
 summary[parent]=dict(input_sites=row["input_sites"],haplotypes=1600,reference_haplotypes=600,query_haplotypes=1000,focal_records=10000,sequence_length_bp=30000000,conversion_output_cap_bytes=cfg["output_cap_bytes"],whole_scope_reservation_bytes=cfg["output_cap_bytes"]+8*2**20,cpu_threads=4,rss_bytes=4*2**30,target_row_for_later_ROOT_binding=dict(row,count_manifest_ref=None),original_core_context_passed_in_RAM=True)
converter=ref("src/arggnn/american_c3_flare_convert_v1.py")
check("exact_proven_converter_source_unchanged",converter["sha256"]=="6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5")
def cpu_groups(groups):
 require(set(groups)==set(rows),"Exactly three named original targets")
 all_ids=[]
 for ids in groups.values():
  require(type(ids)is list and len(ids)==len(set(ids))==4 and all(type(i)is int and i>=0 for i in ids),"Explicit CPU4 group")
  all_ids.extend(ids)
 require(len(set(all_ids))==12,"Three disjoint CPU4 groups required; no CPU reuse")
 return True
check("disjoint_three_CPU4_groups_RAM_only",cpu_groups({p:list(range(i*4,i*4+4))for i,p in enumerate(rows)}))
try:cpu_groups({p:[0,1,2,3]for p in rows})
except ValueError:check("overlapping_CPU_groups_fail_closed")
else:raise AssertionError("Overlapping CPUs accepted")
check("no_scientific_or_guard_runtime_imports",not any(n=="torch"or n.startswith(("numpy","arggnn","composer"))for n in sys.modules))
plan=dict(schema="FLARE_original_VAL002004_conversion_source_route_assessment_v1",status="FULL_ORIGINAL_METADATA_ROUTE_VALIDATED_SUCCESSOR_REQUIRED",source_only=True,active=False,execution_authorized=False,converter_ref=converter,unchanged_algorithm=True,original_options=core.OPTIONS,original_multiallelic_policy=basis["configurations"]["conversion_config"]["multiallelic_policy"],unsupported_site_policy=basis["configurations"]["conversion_config"]["unsupported_site_policy"],parent_routes=summary,scientific_seconds=10800,setup_seconds=7200,wall_seconds=11400,combined_seconds=18600,evidence_reserve_seconds=300,authority_margin_seconds=600,CPU_parallel_shape=dict(target_count=3,CPU_threads_per_target=4,total_distinct_CPU_ids_required=12,actual_CPU_ids=None,aggregate_RSS_bound_bytes=12*2**30,aggregate_reservation_bound_bytes=sum(v["whole_scope_reservation_bytes"]for v in summary.values()),actual_fit_or_eligibility_proven=False),current_v4_direct_reuse=False,required_successor_changes=["Explicit original parent002/003/004 allowlist and distinct new names/keys; retain original input/output freshness and no retry","Count-derived per-parent output cap from unchanged original conversion bound; do not reuse fixedVAL001 cap","Keep v4 full3h scientific7200setup/11400wall18600combined timing and unchanged owned wait/drain/completion lifecycle","Preserve all historical VAL001 failures and completed conversion without transplanting its condition/failure acceptance onto a new parent","Bind exact genuine ROOT count/context/registration and independently reviewed source closure per target","Use a single explicit actual Census child selector and source after its reviewed activation; reject absent/Relate64/mixed selector, no fallback","Retain complete inherited source/metadata closure, selected-child direct role, original budget checks and existing consumer memo; validate full current role unions","ROOT separately determines capacity and disjoint actual CPU4 groups; serialize state mutations even if scientific workers overlap"],future_Census_selector_ref=None,future_Census_selector_required_explicit=True,old_policy_fallback_forbidden=True,no_truth_read=True,no_payload_read=True,actual_guard_resource_process_admission_launch_or_Git=False)
with (D/"route_assessment.v1.json").open("x")as f:json.dump(plan,f,sort_keys=True,indent=2);f.write("\n")
result=dict(schema="FLARE_original_VAL002004_source_route_narrow_checks_v1",status="PASS_SOURCE_JSON_RAM_ONLY",checks=len(checks),cases=checks,actual_JSON_metadata_refs=list(reads.values()),RAM_count_and_context_not_written_as_ROOT_evidence=True,scientific_payloads_not_opened=True,guard_resource_process_admission_launch_or_Git=False)
with (D/"source_checks.result.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(checks=len(checks),caps={p:{k:v for k,v in row.items()if k.endswith("bytes")}for p,row in summary.items()},actual_metadata_refs=len(reads)),sort_keys=True))
