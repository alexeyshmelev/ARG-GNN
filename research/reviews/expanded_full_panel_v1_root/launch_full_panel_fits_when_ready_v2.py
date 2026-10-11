"""Root dispatch of the two frozen H038 FP32 PyG fits after real corpus/cache completion."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,os,subprocess,sys,time,traceback
R=Path("/workspace-SR003.nfs2/ARG-GNN");sys.path.insert(0,str(R))
P=R/"research/reviews/expanded_full_panel_v1_root"
sp=importlib.util.spec_from_file_location("root_h038_dispatch",P/"close_completed_orphan_VAL_when_ready_v1.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.accounting=m.m;m.budget=m.m.budget;m.HELPER=m.m.SOURCE
m.CONFIG="configs/expanded_full_panel_v1.json";m.REGISTRATION="research/reviews/expanded_full_panel_v1_root/registration.json"
def run():
 done=R/"data/expanded_full_panel_pyg_cache_v4/completed.json"
 all_data=P/"all40_development_completed_v1.json"
 deadline=time.monotonic()+72000
 while not (done.exists() and all_data.exists()):
  if (R/"data/expanded_full_panel_pyg_cache_v4/failed.json").exists():raise RuntimeError("Preserved cache preparation failure, no automatic replay")
  if time.monotonic()>deadline:raise TimeoutError("Full development corpus/cache not complete")
  time.sleep(10)
 data=json.loads(done.read_text())
 assert data["parent_count"]==40 and not data["targets_read"] and not data["outer_included"] and data["owned_workers_drained"]
 from arggnn import train_mutation_routed_full_panel_pyg_v1 as train
 config=m.read(train.CONFIG)
 config.update(execution_guard_ready=True,output_cap_bytes=5000000000,output_cap_status="root_admitted_5GB_per_arm_raw_last_and_best_seals")
 for ref in config["source_refs"]+list(config["helper_refs"].values()):ref["sha256"]=m.ref(ref["path"])["sha256"]
 (R/train.CONFIG).write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
 train.validate_protocol_v1(config)
 source_sha={p:m.ref(p)["sha256"] for p in train.ALLOWED_SOURCE_PATHS}
 cache=m.read(data["cache_manifest_ref"]["path"]);entries=cache["entry_refs"]
 generation=m.read(m.CONFIG);roles={r["parent_id"]:r["split"] for r in generation["parent_registry"] if r["split"]!="outer"}
 assert set(entries)==set(roles)
 packet=P/"full_panel_pyg_fits_v2";packet.mkdir(exist_ok=False)
 targets={}
 for parent,split in roles.items():
  genpath=generation["output_root"]+"/"+parent+"/generation.json";gen=m.read(genpath)
  targets[parent]=dict(split=split,generation_receipt=m.ref(genpath),truth_npz=dict(path=generation["output_root"]+"/"+parent+"/truth/full_truth_intervals.npz",sha256=gen["files"]["truth/full_truth_intervals.npz"]["sha256"]))
 target_ref=m.save(packet,"target_bindings.json",dict(parents=targets,outer_included=False,targets_opened=False))
 items=[]
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat()
 for gpu,arm in [(2,"aligned"),(3,"zero_concordance")]:
  tag=packet/arm;job="mutation-routed-expanded-v1-"+arm.replace("_","-")+"-seed101"
  output="experiments/results/mutation_routed_full_panel_pyg_v1_"+arm+"_seed101_GPU"+str(gpu)
  rt=m.save(tag,"runtime.json",dict(status="active",device="cuda",physical_GPU=gpu,process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,job_id=job,output_directory=output,torch="2.6.0+cu124",torch_geometric="2.7.0"))
  refs=dict(config=m.ref(train.CONFIG),runtime=rt,registration=m.ref(m.REGISTRATION),generation_config=m.ref(m.CONFIG),observation_cache=data["cache_manifest_ref"],target_bindings=target_ref)
  consumers=[dict(path=p,sha256=h) for p,h in source_sha.items()]
  consumers += list(refs.values())+[m.ref("src/arggnn/project_budget_v9_consumer.py"),config["base_fit_config"]]+list(entries.values())
  a=m.authority("reserve")
  a["reservation"]=dict(key="H038/"+job,output=str(R/output),auxiliary_scopes=[],reserved_bytes=5000000000,consumer_bindings=consumers)
  ar=m.save(tag,"reserve_authority.json",a);res=m.accounting.apply_reserve_v1(ar,expected_source_sha256=m.CORE,expected_launcher_sha256=m.BASE)
  m.save(tag,"reserve_receipt.json",res)
  items.append(dict(gpu=gpu,arm=arm,tag=tag,job=job,output=output,refs=refs,reservation=res["value"]))
 view=m.inspect();running=[]
 for item in items:
  storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.accounting.MODE,job_id=item["job"],output_directory=item["output"],reserved_new_bytes=5000000000,reservation=item["reservation"],expires_at_utc=expiry,initial_ledger_pins=m.pins(view),budget_components=dict(implementation=m.ref(m.budget.SOURCE_REL),launcher=m.ref(m.budget.LAUNCHER_REL),consumer_adapter=m.ref(m.HELPER)))
  sr=m.save(item["tag"],"storage.json",storage);refs=dict(item["refs"],storage=sr)
  spec=dict(schema="mutation_routed_full_panel_pyg_input_v1",root_authorizer="root_agent",fit_authorized=True,pyg_authorized=True,status="active",arm=item["arm"],seed=101,device="cuda",job_id=item["job"],output_directory=item["output"],created_at_utc=now.isoformat(),expires_at_utc=expiry,max_cpu_rss_bytes=512000000000,max_cuda_reserved_bytes=75000000000,refs=refs,source_sha256=source_sha,observation_entry_refs=entries,observation_cache_source_refs=cache["source_refs"])
  train.validated_source_records_v1(spec,config);train.validated_input_refs_v1(spec)
  ir=m.save(item["tag"],"input.json",spec)
  cmd=[str(R/".runtime/env/bin/python"),"-B","scripts/launch_reserved_gpu_terminal_scopes_v1.py","--gpu",str(item["gpu"]),"--job",item["job"],"--storage-record",sr["path"],"--expected-storage-sha256",sr["sha256"],"--expected-budget-source-sha256",m.CORE,"--expected-launcher-sha256",m.BASE,"arggnn.train_mutation_routed_full_panel_pyg_v1","--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
  with (item["tag"]/"launch.log").open("xb") as log:child=subprocess.Popen(cmd,cwd=R,env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4"),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launch=dict(pid=child.pid,gpu=item["gpu"],arm=item["arm"],input_ref=ir,command=cmd,full_parent_updates_maximum=960,TRAIN_parents=32,VAL_parents=8,precision="float32",framework="torch_and_pyg")
  m.save(item["tag"],"launch.json",launch);print(json.dumps(dict(stage="representative_PyG_fit_launched",**launch)),flush=True)
  running.append((item,child))
 failures=[]
 while running:
  for item,child in list(running):
   rc=child.poll()
   if rc is None:continue
   cp=R/item["output"]/"completed.json";record=dict(arm=item["arm"],launcher_pid=child.pid,returncode=rc,completed_ref=m.ref(cp) if cp.exists() else None)
   m.save(item["tag"],"root_wait.json",record);running.remove((item,child))
   if rc!=0 or not cp.exists():failures.append(record)
   print(json.dumps(dict(stage="representative_fit_exited",**record)),flush=True)
  if running:time.sleep(10)
 m.save(packet,"dispatch_complete.json",dict(failures=failures,models_completed=2-len(failures),outer_evaluated=False,top1_claim=False,original_paper_reproduction=False))
if __name__=="__main__":
 try:run()
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
