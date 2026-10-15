"""Static source and exact plan-metadata audit; no project module import or guard."""
import pathlib,json,ast,hashlib,datetime
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN");P=ROOT/"research/reviews/C3_GNN_immutable_RAM_cache_source_20261015_v1"
OLD="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
m=json.loads((P/"staged_source_install_manifest.v1.json").read_text());NEW=m["future_root_packet"]
mapping={v["intended_ref"]["path"]:ROOT/v["staged_ref"]["path"] for v in m["files"]};checks=[];compiled=0
def ck(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
for row in m["files"]:
 ck("staged_SHA:"+row["intended_ref"]["path"],sha(ROOT/row["staged_ref"]["path"])==row["intended_ref"]["sha256"])
 ck("target_absent:"+row["intended_ref"]["path"],not (ROOT/row["intended_ref"]["path"]).exists())
 if row["original_ref"]:
  v=row["original_ref"];ck("old_source_preserved:"+v["path"],sha(ROOT/v["path"])==v["sha256"])
 p=ROOT/row["staged_ref"]["path"]
 if p.suffix==".py":compile(p.read_text(),str(p),"exec");compiled+=1
preserved=[]
owner=ROOT/"research/reviews/C3_isolated_boundary_forest_repair_20261015_v1/staged_source_install_manifest.v1.json"
for row in json.loads(owner.read_text())["files"]:
 v=row["intended_ref"]
 if v["path"].endswith(".py"):ck("current_isolated_source_preserved:"+v["path"],sha(ROOT/v["path"])==v["sha256"]);preserved.append(v)
cfg=json.loads(mapping[NEW+"/training_config.inactive.json"].read_text());oldcfg=json.loads((ROOT/OLD/"training_config.inactive.json").read_text())
ck("only_cache_config_changes",{k for k in cfg if cfg[k]!=oldcfg[k]}=={"CPU_observation_cache_bytes"} and cfg["CPU_observation_cache_bytes"]==2*2**30)
caps=json.loads(mapping[NEW+"/caps_train.proposed.inactive.json"].read_text());oldcaps=json.loads((ROOT/OLD/"caps_train.root_active_v1.json").read_text())
ck("same_root_GPU_storage_workers_RSS_caps",all(caps[k]==oldcaps[k] for k in oldcaps if k not in {"active","schema"}) and caps["active"] is False)
req=json.loads(mapping[NEW+"/request_train.proposed.inactive.json"].read_text());oldreq=json.loads((ROOT/OLD/"request_train.root_active_v1.json").read_text())
ck("all20_bindings_unchanged",req["bindings"]==oldreq["bindings"])
ck("same_scientific_trainer_hooks",req["scientific_train_source_ref"]==oldreq["scientific_train_source_ref"] and req["hook_ref"]==oldreq["hook_ref"])
ck("inactive_GPU4_UUID40GiB_2GiB",req["active"] is False and "caps_ref" not in req and req["physical_gpu"]==4 and req["GPU_UUID"]==oldreq["GPU_UUID"] and caps["max_cuda_reserved_bytes"]==40*2**30 and caps["reservation_cap_bytes"]==2*2**30)
rows=[]
for parent,rec in req["bindings"]["parent_plan_refs"].items():
 p=ROOT/rec["path"];ck("actual_plan_SHA:"+parent,sha(p)==rec["sha256"]);plan=json.loads(p.read_text())
 n=sum(v["nodes"] for v in plan["blocks"]);t=plan["trees"];s=plan["sites"];b=len(plan["blocks"])
 bound=100*n+8*t*1600+1730*s+24*t+27200*b+240000
 ck("whole_parent_fits2GiB:"+parent,bound<=2*2**30)
 rows.append(dict(parent=parent,plan_ref=rec,blocks=b,trees=t,nodes=n,sites=s,tensor_byte_upper_bound=bound))
forecast=dict(schema="C3_immutable_RAM_cache_memory_forecast_v1",method="current tensor dtypes/shapes plus authenticated exact plan counts; no payload run",
 tensor_bound_formula="100*N + 8*T*1600 + 1730*S + 24*T + 27200*B + 24*10000",
 formula_components=dict(node_tensors_max_bytes_per_node=100,sample_index_bytes_per_tree=12800,
  site_max_bytes=1730,hold_geometry_bytes_per_tree=24,query_reference_date_bytes_per_block=27200,
  three_focal_indices_bytes_per_focal=24),rows=rows,
 total_tensor_byte_upper_bound=sum(v["tensor_byte_upper_bound"] for v in rows),
 maximum_parent_tensor_byte_upper_bound=max(v["tensor_byte_upper_bound"] for v in rows),
 logical_parent_cap_bytes=2*2**30,logical_all20_cap_bytes=40*2**30,
 main_RSS_hard_cap_bytes=88*2**30,main_noncache_included_planning_allowance_bytes=48*2**30,
 main_duplicate_block_staging_included_allowance_bytes=2**30,main_other_runtime_graph_allocator_metadata_included_allowance_bytes=47*2**30,
 worker_count=2,worker_RSS_cap_each_bytes=8*2**30,main_plus_workers_hard_maximum_bytes=104*2**30,
 existing_aggregate_RSS_cap_bytes=120*2**30,aggregate_headroom_bytes=16*2**30,
 existing_root_scope_RSS_cap_bytes=128*2**30,root_scope_margin_above_main_plus_workers_bytes=24*2**30,
 duplicate_and_ownership_notes=[
 "Cache ownership counts each complete CPU tensor bank once per block, conservatively counting tensor aliases separately.",
 "Each worker's 8 GiB whole-process RSS cap includes source observations, feature construction, outgoing bank and original hash-copy buffers.",
 "The 1 GiB block-staging allowance is included within the 48 GiB main-process allowance: eight 128 MiB block slots for two incoming results, IPC/serialization bookkeeping buffers, the clone before CUDA and transfer/read scratch margin.",
 "Retention keeps the received CPU object; it makes no second bank copy. Shared IPC pages are conservatively counted in the main and worker RSS bounds.",
 "The remaining 47 GiB main-process allowance covers Python/PyG objects, runtime, CPU autograd boundary graphs, pinned transfer/driver allocations and allocator fragmentation.",
 "The 48 GiB noncache allowance is prospective, not measured fit evidence. Whole-main-process peak RSS is hard-limited to 88 GiB by the new entry's original fast boundaries and cache accesses.",
 "Main plus two workers is hard-bounded by 88+8+8=104 GiB, leaving 16 GiB under the existing 120 GiB aggregate bound. Root/launcher/supervisory overhead remains subject to the unchanged 128 GiB whole-owned-tree root cap with 24 GiB above this main/worker bound.",
 "CUDA tensors and model-dependent encoder/GRU outputs are not cached; the original 40 GiB CUDA cap and original checkpointed FP32 computations remain.",
 "No disk spill or additional data product is introduced; all retained banks are explicitly cleared after final worker drain."],
 measured_RSS_or_timing=False,no_disk_spill=True,storage_cap_bytes=2*2**30)
ck("all20_tensor_bound_within40GiB",forecast["total_tensor_byte_upper_bound"]<=40*2**30)
ck("exact_original_inventory",sum(v["blocks"] for v in rows)==3089 and sum(v["trees"] for v in rows)==98516 and sum(v["nodes"] for v in rows)==254141098)
cp=mapping[NEW+"/root_contract_v1.py"];tree=ast.parse(cp.read_text())
ns=dict(__builtins__={},Path=pathlib.Path,ROOT=ROOT,dict=dict,tuple=tuple,range=range,frozenset=frozenset,str=str)
for node in tree.body:
 if isinstance(node,ast.Assign):
  try:exec(compile(ast.Module(body=[node],type_ignores=[]),str(cp),"exec"),ns)
  except (NameError,AttributeError,TypeError):pass
for n in ast.parse((ROOT/ns["BASE_HELPER"]["path"]).read_text()).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PROC_HELPER" for t in n.targets):ns["PROC_HELPER"]=ast.literal_eval(n.value)
ns["read"]=lambda rec:json.loads((ROOT/rec["path"]).read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="source_paths")
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(cp),"exec"),ns)
paths=ns["source_paths"]()
for path in paths:ck("closure_resolves:"+path,mapping.get(path,ROOT/path).is_file())
for path,h in ns["PREFETCH_SOURCE_PINS"].items():ck("closure_pin:"+path,sha(mapping.get(path,ROOT/path))==h)
ck("old_trainer_hooks_still_bound",ns["HOOK_REF"]==oldreq["hook_ref"] and ns["TRAIN_SOURCE_REF"]==oldreq["scientific_train_source_ref"])
stream=(ROOT/"src/arggnn/american_c3_gnn_stream_train_isolated_v2.py").read_text()
ck("unchanged_clone_before_CUDA",'block=provider.get(physical).clone().to(actual)' in stream)
ck("unchanged_600step20seal_trainer",sha(ROOT/OLD/"trainer_prefetch_v1.py")=="b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c")
ck("unchanged_620_release_factory",'len(releases)==620' in (ROOT/OLD/"persistent_provider_v1.py").read_text())
fixture=json.loads((P/"focused_cache_fixture_receipt.v2.json").read_text())
ck("focused_fixture_current_source_PASS",fixture["status"]=="PASS" and fixture["RAM_provider_source_sha256"]==sha(P/"ram_cache_provider_v1.staged.py"))
fp=P/"memory_forecast.v1.json";fp.write_text(json.dumps(forecast,indent=2,sort_keys=True)+"\n")
receipt=dict(schema="C3_immutable_RAM_cache_source_static_receipt_v1",status="PASS",observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 check_count=len(checks),checks=checks,python_sources_syntax_compiled=compiled,staged_files=len(m["files"]),source_closure_entries=len(paths),
 current_original_sources_preserved=preserved,focused_fixture_ref=ref(P/"focused_cache_fixture_receipt.v2.json"),memory_forecast_ref=ref(fp),
 canonical_install=False,admission=False,scientific_execution=False,scientific_payload_reads=False,GPU_or_process_probe=False,guard_mutation=False,Git=False)
p=P/"source_static_receipt.v1.json";p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="PASS",check_count=len(checks),compiled=compiled,source_closure_entries=len(paths),total_tensor_bound=forecast["total_tensor_byte_upper_bound"],maximum_parent_tensor_bound=forecast["maximum_parent_tensor_byte_upper_bound"]),indent=2))
