"""Inert H060 staging assembler: stdlib metadata only, never guard/admit/execute.

No-argument invocation reads/writes no project metadata. ROOT must separately
supply an exact SHA-bound staging instruction after its own current observations.
All metadata is serialized and validated in memory before the first write.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,ast,copy,hashlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H060_isolated_CPU_ROOT_metadata_source_20261017_v1"
WRAPPER="research/reviews/H060_isolated_CPU_runtime_source_20261017_v1"
SOURCE=HERE+"/assemble_root_metadata_v1.py"
FLAGS=dict(backend="cpu",intraop_threads=32,interop_threads=1,default_dtype="torch.float32",CUDA_initialized=False,
 torch_version="2.6.0+cu124",numpy_version="2.4.6",PyG_version="2.7.0")
DIRECTIVE_FIELDS=("preparation_directory","source_manifest_ref","source_review_ref","scientific_review_ref","registration_ref",
 "registration_decisions_ref","registration_ledger_ref","training_context_ref","runtime_ref","runtime_qualification_ref",
 "scientific_config","caps","cpu_ids","created_at_utc","expires_at_utc")
EXCLUSIVE_SUFFIXES=(".scientific_config.json",".config.json",".source_and_input_closure.json",".reserve_authority.json")
def require(ok,message):
 if not ok:raise ValueError(message)
def local(name):
 p=Path(name);p=p if p.is_absolute()else ROOT/p
 require(p!=ROOT and ROOT in p.parents and ".."not in p.parts and p.resolve()==p,"Canonical contained no-link metadata path")
 return p
def raw_path(name):
 p=local(name);s=p.lstat()
 require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20
  and p.suffix in(".json",".py",".md",".patch",".sh"),"Bounded regular source/metadata only")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:
  actual=os.fstat(f.fileno());data=f.read(32*2**20+1)
 after=p.lstat();fields=lambda v:(v.st_dev,v.st_ino,v.st_mode,v.st_nlink,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
 require(fields(s)==fields(actual)==fields(after)and len(data)==s.st_size,"Metadata changed during bounded read")
 return data
def reference(name):
 p=local(name);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw_path(p)).hexdigest())
def raw(record):
 require(isinstance(record,dict)and set(record)=={"path","sha256"},"Explicit exact metadata path/SHA")
 data=raw_path(record["path"]);require(hashlib.sha256(data).hexdigest()==record["sha256"],"Exact metadata SHA")
 return data
def read(record):return json.loads(raw(record))
def template(name):return json.loads(raw_path(WRAPPER+"/"+name))
def encoded(value):return(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def absent(paths):
 return not any(p.exists()or p.is_symlink()for p in paths)
def source_packet(instruction):
 require(instruction["assembler_source_manifest_ref"]==reference(HERE+"/source_manifest.v1.json"),"Exact ROOT-reviewed assembler manifest")
 manifest=read(instruction["assembler_source_manifest_ref"])
 for path,sha in manifest["files_sha256"].items():raw(dict(path=path,sha256=sha))
 deps=json.loads(raw_path(HERE+"/dependencies.v1.json"))
 for path,sha in deps["pins"].items():raw(dict(path=path,sha256=sha))
 review=read(deps["wrapper_review_ref"]);b=template("scientific_binding.v1.json")
 require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_ref")==reference(WRAPPER+"/root_driver_v1.py")
  and review.get("source_manifest_ref")==reference(WRAPPER+"/source_manifest.v1.json")
  and review.get("source_handoff_ref")==reference(WRAPPER+"/source_handoff.v1.json")
  and review.get("runner_source_ref")==b["runner_source_ref"]and review.get("protocol_ref")==b["protocol_ref"],
  "Exact genuine independent whole-wrapper pass, never inactive review")
 return deps,b
def metadata_scope(b,caps,planned):
 """Execute only six reviewed metadata predicates, never import their module."""
 def virtual_raw(record):
  if isinstance(record,dict)and record.get("path")in planned:
   data=planned[record["path"]]
   require(set(record)=={"path","sha256"}and hashlib.sha256(data).hexdigest()==record["sha256"],"Exact planned metadata SHA")
   return data
  return raw(record)
 tree=ast.parse(raw_path(WRAPPER+"/root_contract_v1.py").decode())
 names=("reviewed","scientific_review","validate_config","registration","context","selected")
 functions=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in names]
 require({x.name for x in functions}==set(names),"Exact six sealed metadata predicates")
 deps=template("dependencies.v1.json")
 scope=dict(read=lambda r:json.loads(virtual_raw(r)),raw=virtual_raw,ref=reference,require=require,
  binding=lambda:b,HERE=WRAPPER,DRIVER=WRAPPER+"/root_driver_v1.py",MANIFEST=WRAPPER+"/source_manifest.v1.json",
  HANDOFF=WRAPPER+"/source_handoff.v1.json",CAPS=caps,ACCEPTANCE=deps["ACCEPTANCE"],copy=copy,datetime=datetime,
  TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20)))
 exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),"sealed_H060_metadata_predicates_only","exec"),scope)
 return scope
def plan_stage(qin,instruction_ref):
 """Pure planning plus explicit bounded metadata reads; no mutation/observations."""
 expected={"schema","root_authorizer","staging_authorized","issue_full_TRAIN_request","current_CPU_capacity_confirmed",
  "current_runtime_settings_confirmed","never_used_paths_confirmed","cpu_ids","runtime_flags","assembler_source_manifest_ref",
  "preparation_directory","packet_directory","output_directory","created_at_utc","expires_at_utc","registration_ref",
  "registration_decisions_source_ref","registration_ledger_source_ref","storage_policy_selector_ref"}
 require(set(qin)==expected and qin.get("schema")=="H060_isolated_CPU_ROOT_metadata_instruction_v1"
  and qin.get("root_authorizer")=="root_agent"and all(qin.get(k)is True for k in("staging_authorized",
   "issue_full_TRAIN_request","current_CPU_capacity_confirmed","current_runtime_settings_confirmed","never_used_paths_confirmed")),
  "Explicit actual SHA-bound ROOT staging/full-budget/current assertions required")
 cpus=qin["cpu_ids"]
 require(isinstance(cpus,list)and len(cpus)==len(set(cpus))==32 and all(type(i)is int and i>=0 for i in cpus)
  and encoded(qin["runtime_flags"])==encoded(FLAGS),"Exact ROOT-supplied32CPU IDs and accepted CPU32/interop1/FP32/noCUDA/package declarations")
 deps,b=source_packet(qin);caps=template("caps.inactive.v1.json")
 cfg=template("scientific_config.INACTIVE.template.v1.json");cfg["execution_enabled"]=True
 require(cfg["parents_per_epoch"]==20 and cfg["epochs"]==30 and cfg["additional_updates_per_arm"]==600
  and cfg["arms"]==["stable_leaf_memory","no_added_memory"]and cfg["checkpoint_epochs"]==[5,10,15,20,25,30]
  and cfg["model_seed"]==70001 and cfg["parent_order_seed"]==70002 and cfg["training_length_bp"]==2000000
  and caps["CPU_threads"]==32 and caps["dataloader_workers"]==0 and caps["rss_bytes"]==128*2**30
  and cfg["max_cached_parent_bytes"]==caps["max_cached_parent_bytes"]==96*2**30
  and caps["reservation_bytes"]==32*2**20,"Original full1200budget/config/caps, no probe/reduction/new qualification")
 created=datetime.fromisoformat(qin["created_at_utc"]);expires=datetime.fromisoformat(qin["expires_at_utc"]);now=datetime.now(timezone.utc)
 require(created.tzinfo and expires.tzinfo and created<=now and expires-created==timedelta(hours=25)
  and(expires-now).total_seconds()>caps["wall_seconds"]+600,"Fixed ROOT25hour finite dates and >87300s remaining; no clock reset")
 prep=local(qin["preparation_directory"]);packet=local(qin["packet_directory"]);output=local(qin["output_directory"])
 require(prep.parent==packet.parent==ROOT/"research/reviews"and output.parent==ROOT/"experiments/results"
  and prep.name.startswith("H060_isolated_CPU_ROOT_preparation_")
  and packet.name.startswith("H060_isolated_CPU_ROOT_actual_")and output.name.startswith("H060_isolated_CPU_")
  and prep!=packet and absent((prep,packet,output)),"Never-used fresh exclusive preparation/packet/output; leave charged paths absent")
 reserved=tuple(prep/(packet.name+suffix)for suffix in EXCLUSIVE_SUFFIXES)
 require(absent(reserved),"All four driver-exclusive basenames must remain absent")
 require(qin["registration_decisions_source_ref"]["path"]=="research/experiment_decisions.json"
  and qin["registration_ledger_source_ref"]["path"]=="research/experiment_ledger.md","Snapshot genuine current repository decision/ledger bytes")
 decisions_bytes=raw(qin["registration_decisions_source_ref"]);ledger_bytes=raw(qin["registration_ledger_source_ref"])
 decisions=json.loads(decisions_bytes)
 require(len([r for r in decisions["decisions"]if r.get("id")=="H060"])==1 and"H060"in ledger_bytes.decode(),
  "Unique genuine H060 registration decision/text; no relabelling")
 planned={}
 def add(name,value):
  require(Path(name).name==name,"Exclusive flat metadata name")
  data=value if isinstance(value,bytes)else encoded(value)
  require(len(data)<=(32*2**20 if isinstance(value,bytes)else 2**20),"Bounded fully serialized snapshot/JSON before first write")
  path=str((prep/name).relative_to(ROOT));require(path not in planned,"One planned immutable metadata file")
  planned[path]=data;return dict(path=path,sha256=hashlib.sha256(data).hexdigest())
 dr=add("registration_decisions.snapshot.json",decisions_bytes);lr=add("registration_ledger.snapshot.md",ledger_bytes)
 runtime=template("runtime.INACTIVE.template.v1.json");runtime.update(actual_runtime_qualified=True,cpu_ids=copy.deepcopy(cpus))
 rr=add("runtime.root_v1.json",runtime)
 original=read(b["original_training_request_ref"])["bindings"]
 context=template("training_context.INACTIVE.template.v1.json")
 context.update(root_authorizer="root_agent",parent_plan_refs=copy.deepcopy(original["parent_plan_refs"]),
  actual_train_bindings=copy.deepcopy(original["actual_train_bindings"]))
 cr=add("training_context.root_v1.json",context)
 qualification=template("runtime_qualification.INACTIVE.template.v1.json")
 qualification.update(root_authorizer="root_agent",status="qualified",runtime_ref=rr,
  source_manifest_ref=reference(WRAPPER+"/source_manifest.v1.json"),runner_source_ref=b["runner_source_ref"],
  unresolved_items=[],standalone_token_update_probe=False,full_training_fit_measured=False,scientific_probe_updates=0)
 qr=add("runtime_qualification.root_v1.json",qualification)
 request=template("ROOT_request.INACTIVE.template.v1.json")
 request.update(active=True,execution_authorized=True,root_authorizer="root_agent",cpu_ids=copy.deepcopy(cpus),
  preparation_directory=qin["preparation_directory"],packet_directory=qin["packet_directory"],output_directory=qin["output_directory"],
  created_at_utc=qin["created_at_utc"],expires_at_utc=qin["expires_at_utc"],source_manifest_ref=reference(WRAPPER+"/source_manifest.v1.json"),
  source_review_ref=deps["wrapper_review_ref"],scientific_review_ref=b["scientific_review_ref"],
  registration_ref=qin["registration_ref"],registration_decisions_ref=dr,registration_ledger_ref=lr,
  training_context_ref=cr,runtime_ref=rr,runtime_qualification_ref=qr,scientific_config=cfg,
  storage_policy_selector_ref=qin["storage_policy_selector_ref"])
 directive=template("ROOT_directive.INACTIVE.template.v1.json")
 directive.update({k:copy.deepcopy(request[k])for k in DIRECTIVE_FIELDS});directive.update(active=True,root_authorizer="root_agent")
 request["root_directive_ref"]=add("ROOT_directive.root_v1.json",directive)
 request_ref=add("ROOT_request.root_v1.json",request)
 # Exact wrapper predicates over virtual planned files, before prep.mkdir.
 predicates=metadata_scope(b,caps,planned)
 predicates["reviewed"](request["source_review_ref"]);predicates["scientific_review"](request["scientific_review_ref"])
 predicates["validate_config"](cfg);predicates["registration"](request);predicates["context"](cr)
 predicates["selected"](qin["storage_policy_selector_ref"])
 require(runtime["package_report_ref"]==template("dependencies.v1.json")["BASE_RUNTIME"]
  and runtime["device"]=="cpu"and runtime["dtype"]=="float32"and runtime["CPU_threads"]==32
  and runtime["prefix"]==str(ROOT/".runtime/env"),"Exact existing accepted shared runtime metadata")
 raw(runtime["package_report_ref"])
 require(request["TRAIN_truth_only"]is True and request["VAL_truth_authorized"]is False and request["TEST_authorized"]is False
  and request["automatic_retry_resume_or_fallback"]is False and request["standalone_token_update_qualification"]is False
  and request["GPU_authorized"]is False and request["CUDA_authorized"]is False and request["physical_gpu"]is None
  and directive["first_scheduled_update_timed_inside_full_budget"]is True
  and all(directive[k]==request[k]for k in DIRECTIVE_FIELDS),"Exact existing finite TRAIN authority; no extra qualification")
 argv=[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",WRAPPER.replace("/",".")+".root_driver_v1","--admit-ref",json.dumps(request_ref)]
 receipt=dict(schema="H060_isolated_CPU_ROOT_metadata_staging_receipt_v1",instruction_ref=instruction_ref,
  assembler_ref=reference(SOURCE),assembler_source_manifest_ref=qin["assembler_source_manifest_ref"],request_ref=request_ref,
  wrapper_review_ref=deps["wrapper_review_ref"],scientific_review_ref=b["scientific_review_ref"],
  storage_policy_selector_ref=qin["storage_policy_selector_ref"],created_at_utc=qin["created_at_utc"],expires_at_utc=qin["expires_at_utc"],
  source_only_staging=True,guard_admission_or_execution_performed=False,current_resources_observed_by_assembler=False,
  full_training_fit_measured=False,scientific_probe_updates=0,old_failures_charges_clocks_unchanged=True,
  driver_exclusive_paths_left_absent=[str(p.relative_to(ROOT))for p in reserved],next_admission_argv=argv)
 receipt_ref=add("assembly_receipt.root_v1.json",receipt)
 expected_files={"registration_decisions.snapshot.json","registration_ledger.snapshot.md","runtime.root_v1.json",
  "training_context.root_v1.json","runtime_qualification.root_v1.json","ROOT_directive.root_v1.json",
  "ROOT_request.root_v1.json","assembly_receipt.root_v1.json"}
 require(len(planned)==len(expected_files)and{Path(name).name for name in planned}==expected_files
  and all(local(name).parent==prep for name in planned),"Complete serialized eight-file plan stays within preparation before first write")
 require((expires-datetime.now(timezone.utc)).total_seconds()>caps["wall_seconds"]+600 and absent((prep,packet,output)+reserved),
  "All dates/absence revalidated before first write; preserve original fixed timestamps")
 return prep,packet,output,reserved,planned,request_ref,receipt_ref,argv,expires,caps
def stage(instruction_ref):
 require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"ROOT repository -B -s interpreter")
 qin=read(instruction_ref)
 prep,packet,output,reserved,planned,request_ref,receipt_ref,argv,expires,caps=plan_stage(qin,instruction_ref)
 # First filesystem mutation. No rollback/replay: retain any partial fresh prep.
 require(absent((prep,packet,output)+reserved)and(expires-datetime.now(timezone.utc)).total_seconds()>caps["wall_seconds"]+600,
  "Fresh paths/fixed expiry immediately before first write")
 prep.mkdir(mode=0o700)
 for name,data in planned.items():
  path=local(name);require(path.parent==prep,"Only the exclusive preparation may receive metadata")
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
  with os.fdopen(fd,"wb")as f:f.write(data);f.flush();os.fsync(f.fileno())
  require(reference(path)==dict(path=name,sha256=hashlib.sha256(data).hexdigest()),"Actual immutable staged bytes match validated plan")
 require(absent((packet,output)+reserved),"Packet/output and all driver-exclusive files remain absent")
 return dict(request_ref=request_ref,receipt_ref=receipt_ref,metadata_files=len(planned),
  admission_or_execution_performed=False,next_admission_argv=argv)
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--stage-root-instruction-ref",help="Exact ROOT-authored JSON path/SHA; stage metadata only")
 args=parser.parse_args(argv)
 if args.stage_root_instruction_ref is None:
  print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",project_reads_or_writes=False,guard_admission_execution=False)));return 0
 result=stage(json.loads(args.stage_root_instruction_ref));print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
