"""Explicit ROOT metadata staging only. Default invocation reads/writes no project files.

Standard library only. No driver/guard import, admission, execution, model,
resource inspection, shared-ledger overwrite, cleanup or Git operation.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,copy,hashlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/ordered_marker_ROOT_staging_helper_source_20261017_v1"
INTEGRATION="research/reviews/ordered_marker_full_learning_integration_source_20261017_v1"
SOURCE=HERE+"/stage_root_metadata_v1.py"
HYPOTHESIS="ordered_marker_placement_v2"
CPUS=list(range(128,160))
PREPARATION="research/reviews/ordered_marker_CPU_ROOT_preparation_root01_oct17_v1"
PACKET="research/reviews/ordered_marker_CPU_ROOT_root01_oct17_v1"
OUTPUT="experiments/results/ordered_marker_full_parent_CPU_root01_oct17_v1"
FLAGS=dict(backend="cpu",intraop_threads=32,interop_threads=1,default_dtype="torch.float32",CUDA_initialized=False,
 torch_version="2.6.0+cu124",numpy_version="2.4.6",PyG_version="2.7.0")
DIRECTIVE_FIELDS=("preparation_directory","source_manifest_ref","source_review_ref","scientific_review_ref",
 "registration_ref","registration_decisions_ref","registration_ledger_ref","training_context_ref",
 "runtime_ref","runtime_qualification_ref","scientific_config","caps","cpu_ids","created_at_utc","expires_at_utc")
def require(ok,message):
 if not ok:raise ValueError(message)
def local(name):
 p=Path(name);p=p if p.is_absolute()else ROOT/p
 require(p!=ROOT and ROOT in p.parents and ".."not in p.parts and p.resolve()==p,"Canonical contained metadata path")
 return p
def raw_path(name):
 p=local(name);before=p.lstat()
 require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=32*2**20
  and p.suffix in(".json",".py",".md"),"Bounded regular source/metadata only")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as stream:
  opened=os.fstat(stream.fileno());value=stream.read(32*2**20+1)
 after=p.lstat();identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 require(identity(before)==identity(opened)==identity(after)and len(value)==before.st_size,"Stable exact source/metadata read")
 return value
def reference(name):
 p=local(name);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw_path(p)).hexdigest())
def raw(record):
 require(type(record)is dict and set(record)=={"path","sha256"},"Exact source/metadata ref")
 value=raw_path(record["path"]);require(hashlib.sha256(value).hexdigest()==record["sha256"],"Exact source/metadata SHA")
 return value
def read(record):return json.loads(raw(record))
def template(name):
 return json.loads(raw_path(INTEGRATION+"/"+name))
def declaration(name):
 value=template(name);value.pop("SOURCE_ONLY_INACTIVE_TEMPLATE_NOT_AUTHORITY",None);return value

def put(directory,name,value):
 require(Path(name).name==name and Path(name).suffix in(".json",".md")and directory==local(PREPARATION),
  "Only flat fresh preparation metadata")
 data=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 require(len(data)<=32*2**20,"Bounded metadata/snapshot")
 p=directory/name;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,"wb")as stream:
  stream.write(data);stream.flush();os.fsync(stream.fileno())
 return reference(p)
def canonical_hash(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def source_packet(instruction):
 manifest=read(instruction["helper_source_manifest_ref"])
 require(instruction["helper_source_manifest_ref"]==reference(HERE+"/source_manifest.v1.json"),"Exact reviewed staging-helper source packet")
 for path,sha in manifest["files_sha256"].items():raw(dict(path=path,sha256=sha))
 deps=json.loads(raw_path(HERE+"/dependencies.v1.json"))
 for path,sha in deps["pins"].items():raw(dict(path=path,sha256=sha))
 b=template("scientific_binding.v1.json");review=read(deps["integration_review_ref"]);science=read(b["scientific_review_ref"])
 require(review.get("independent")is True and review.get("source_only")is True and review.get("verdict")=="SOURCE_PASS"
  and review.get("blockers")==[]and review.get("source_manifest_ref")==deps["integration_manifest_ref"]
  and review.get("source_handoff_ref")==deps["integration_handoff_ref"]
  and review.get("runner_source_ref")==b["runner_source_ref"]and review.get("scientific_review_ref")==b["scientific_review_ref"]
  and review.get("parameters_per_arm")==33966 and review.get("additional_numerical_equivalence_campaign_required")is False,
  "Exact accepted full integration review and scientific caller")
 require(science.get("verdict")=="SOURCE_PASS"and science.get("source_ref")==b["runner_source_ref"]
  and science.get("blockers")==[],"Exact existing scientific v2 review")
 historical=read(deps["historical_H058_request_pattern_ref"])
 require(historical.get("schema")=="H058_full_parent_CPU_ROOT_request_v1"and historical.get("active")is True
  and historical.get("caps")==template("caps.inactive.v1.json")
  and historical.get("scientific_config",{}).get("execution_enabled")is True,
  "Original actual H058 contextfix declaration pattern; no old authority reused")
 return deps,b
def stage(instruction_ref):
 require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"ROOT repository -B -s interpreter")
 instruction=read(instruction_ref)
 expected={"schema","root_authorizer","staging_authorized","issue_full_TRAIN_declarations",
  "current_CPU_capacity_confirmed","current_runtime_settings_confirmed","CPU_affinity_conflicts_resolved",
  "runtime_identity_verification_ref","cpu_ids","runtime_flags","helper_source_manifest_ref",
  "preparation_directory","packet_directory","output_directory","registration_decisions_source_ref","registration_ledger_source_ref"}
 require(set(instruction)==expected and instruction.get("schema")=="ROOT_ordered_marker_metadata_staging_instruction_v1"
  and instruction.get("root_authorizer")=="root_agent"
  and all(instruction.get(k)is True for k in("staging_authorized","issue_full_TRAIN_declarations",
   "current_CPU_capacity_confirmed","current_runtime_settings_confirmed","CPU_affinity_conflicts_resolved"))
  and instruction.get("cpu_ids")==CPUS and instruction.get("runtime_flags")==FLAGS,
  "Explicit current ROOT declarations for CPU128-159 and exact qualified shared runtime")
 # ROOT creates this compact verification declaration independently; helper never probes runtime/resources.
 verification=read(instruction["runtime_identity_verification_ref"])
 require(verification.get("root_authorizer")=="root_agent"and verification.get("verified")is True
  and verification.get("cpu_ids")==CPUS and verification.get("flags")==FLAGS
  and verification.get("runtime_identity_and_affinity_verified_by_ROOT")is True,
  "Actual source-bound ROOT runtime/affinity verification declaration")
 deps,b=source_packet(instruction)
 require(instruction["preparation_directory"]==PREPARATION and instruction["packet_directory"]==PACKET
  and instruction["output_directory"]==OUTPUT,"Exact one-shot proposed ROOT namespaces")
 prep,packet,output=map(local,(PREPARATION,PACKET,OUTPUT))
 require(not any(os.path.lexists(p)for p in(prep,packet,output)),"Refuse every existing preparation/packet/output name")
 require(instruction["registration_decisions_source_ref"]["path"]=="research/experiment_decisions.json"
  and instruction["registration_ledger_source_ref"]["path"]=="research/experiment_ledger.md","Exact genuine current decision/ledger source refs")
 decision_bytes=raw(instruction["registration_decisions_source_ref"]);ledger_bytes=raw(instruction["registration_ledger_source_ref"])
 decisions=json.loads(decision_bytes);ledger=ledger_bytes.decode("utf-8")
 require(type(decisions)is dict and type(decisions.get("decisions"))is list
  and all(type(row)is dict for row in decisions["decisions"]),"Original complete decision mapping")
 require(not any(row.get("id")==HYPOTHESIS for row in decisions["decisions"])
  and HYPOTHESIS not in ledger,"Refuse duplicate candidate registration; never overwrite existing decision")
 capture=dict(schema="ROOT_ordered_marker_preexisting_registration_capture_v1",
  decisions_source_ref=instruction["registration_decisions_source_ref"],ledger_source_ref=instruction["registration_ledger_source_ref"],
  top_level_decision_keys=sorted(decisions),top_level_value_sha256={key:canonical_hash(value)for key,value in decisions.items()},
  existing_decision_records=[dict(index=i,id=row.get("id"),keys=sorted(row),canonical_sha256=canonical_hash(row))
   for i,row in enumerate(decisions["decisions"])],existing_decision_count=len(decisions["decisions"]),
  shared_sources_modified=False)
 cfg=copy.deepcopy(b["config_defaults"]);caps=template("caps.inactive.v1.json");protocol=cfg["scientific_protocol"]
 require(protocol["hypothesis_id"]==HYPOTHESIS and protocol["TRAIN_parent_ids"]==["ap_expanded_train_%03d"%i for i in range(32)]
  and protocol["passes"]==30and protocol["updates_per_arm"]==960and protocol["updates_all_new_arms"]==1920
  and protocol["VAL_prediction_products_required_before_truth"]==96and set(protocol["parameters_per_arm"].values())=={33966}
  and cfg["CPU_positive_plan_cap_bytes"]==4*2**30and cfg["CPU_prepared_cache_cap_bytes"]==32*2**30
  and cfg["max_cpu_rss_bytes"]==caps["rss_bytes"]==128*2**30and caps["reservation_bytes"]==32*2**20
  and caps["scientific_seconds"]==86400and caps["wall_seconds"]==86700
  and b["operational_ref_keys"]==["runtime_equivalence_ref","full_training_directive_ref","layout_resource_proof_ref","experiment_registration_ref"],
  "Exact complete reviewed science/budgets and all96 pretruth gate")
 selected=read(b["selected_child_selector_ref"])
 require(selected.get("active")is True and selected.get("root_authorizer")=="root_agent"
  and selected.get("source_ref")==b["selected_child_source_ref"]and selected.get("selector_source_ref")==b["selected_child_source_ref"]
  and selected.get("accounting_mode")=="Relate_failed64GiB_retention_v1","Exact bound current selector metadata; no accounting/state query")
 # Detect concurrent shared metadata change before creating the only fresh directory.
 require(raw(instruction["registration_decisions_source_ref"])==decision_bytes
  and raw(instruction["registration_ledger_source_ref"])==ledger_bytes,"Shared metadata changed before staging; refuse stale snapshot")
 created=datetime.now(timezone.utc);expires=created+timedelta(seconds=259200)
 prep.mkdir(mode=0o700) # First mutation: only the exact absent fresh preparation directory.
 capture_ref=put(prep,"preexisting_registration_capture.root_v1.json",capture)
 registration=declaration("registration.INACTIVE.template.v1.json")
 registration.update(active=True,root_authorizer="root_agent",registered_utc=created.isoformat())
 registration_ref=put(prep,"registration.root_active_v1.json",registration)
 decision=declaration("decision.INACTIVE.template.v1.json")
 decision.update(registration_ref=registration_ref,registered_utc=created.isoformat())
 frozen=copy.deepcopy(decisions);frozen["decisions"].append(decision)
 require(frozen["decisions"][:-1]==decisions["decisions"]
  and set(frozen)==set(decisions)and all(frozen[k]==decisions[k]for k in decisions if k!="decisions"),
  "Every existing decision object and unrelated top-level key preserved exactly")
 dr=put(prep,"registration_decisions.snapshot.json",frozen)
 appended=("\n\n## "+HYPOTHESIS+": Matched ordered-marker evidence placement\n\n"
  "Registered "+created.isoformat()+". Separate from H058 aligned-versus-null. Two33966parameter arms receive identical ordered evidence before versus after the second complete tree sweep. "
  "Original32TRAINparents x30passes=960updates per arm/1920total, first scheduled update retained. "
  "All96newVALpredictions precede either arm VALtruth. No numerical result or TEST permission.\n\n"
  "Registration: "+registration_ref["path"]+" SHA256 "+registration_ref["sha256"]+"\n")
 lr=put(prep,"registration_ledger.snapshot.md",ledger_bytes+appended.encode())
 runtime=declaration("runtime.INACTIVE.template.v1.json");runtime.update(actual_runtime_qualified=True,cpu_ids=CPUS)
 rr=put(prep,"runtime.root_v1.json",runtime)
 equivalence=declaration("runtime_equivalence.INACTIVE.template.v1.json")
 equivalence.update(root_authorizer="root_agent",flags=copy.deepcopy(FLAGS),CPU_entry_ref=b["runner_source_ref"])
 er=put(prep,"runtime_equivalence.root_v1.json",equivalence)
 full=declaration("full_training_directive.INACTIVE.template.v1.json")
 full.update(root_authorizer="root_agent",full30pass_training_authorized=True)
 fr=put(prep,"full_training_directive.root_v1.json",full)
 layout=declaration("layout_resource_proof.INACTIVE.template.v1.json")
 layout.update(root_authorizer="root_agent",status="admitted_first_scheduled_full_parent_step")
 pr=put(prep,"layout_resource_proof.root_v1.json",layout)
 context=declaration("training_context.INACTIVE.template.v1.json");context["root_authorizer"]="root_agent"
 cr=put(prep,"training_context.root_v1.json",context)
 qualification=declaration("runtime_qualification.INACTIVE.template.v1.json")
 qualification.update(root_authorizer="root_agent",status="qualified",runtime_ref=rr,
  source_manifest_ref=deps["integration_manifest_ref"],runner_source_ref=b["runner_source_ref"])
 qr=put(prep,"runtime_qualification.root_v1.json",qualification)
 cfg.update(execution_enabled=True,experiment_registration_ref=registration_ref,runtime_equivalence_ref=er,
  full_training_directive_ref=fr,layout_resource_proof_ref=pr)
 request=declaration("ROOT_request.INACTIVE.template.v1.json")
 request.update(active=True,execution_authorized=True,root_authorizer="root_agent",cpu_ids=CPUS,
  preparation_directory=PREPARATION,packet_directory=PACKET,output_directory=OUTPUT,
  created_at_utc=created.isoformat(),expires_at_utc=expires.isoformat(),
  source_manifest_ref=deps["integration_manifest_ref"],source_review_ref=deps["integration_review_ref"],
  scientific_review_ref=b["scientific_review_ref"],registration_ref=registration_ref,
  registration_decisions_ref=dr,registration_ledger_ref=lr,training_context_ref=cr,runtime_ref=rr,
  runtime_qualification_ref=qr,scientific_config=cfg,storage_policy_selector_ref=b["selected_child_selector_ref"])
 directive=declaration("ROOT_directive.INACTIVE.template.v1.json")
 directive.update({key:copy.deepcopy(request[key])for key in DIRECTIVE_FIELDS})
 directive.update(active=True,execution_authorized=True,root_authorizer="root_agent")
 request["root_directive_ref"]=put(prep,"ROOT_directive.root_active_v1.json",directive)
 require((expires-datetime.now(timezone.utc)).total_seconds()>caps["wall_seconds"]+600,"Full original admission margin remains")
 request_ref=put(prep,"ROOT_request.root_active_v1.json",request)
 require(not os.path.lexists(packet)and not os.path.lexists(output),"Entire charged packet/output remain absent")
 for suffix in(".scientific_config.json",".config.json",".source_and_input_closure.json",".reserve_authority.json"):
  require(not os.path.lexists(prep/(packet.name+suffix)),"Exclusive driver staging filename left absent")
 receipt=dict(schema="ROOT_ordered_marker_metadata_staging_receipt_v1",instruction_ref=instruction_ref,helper_ref=reference(SOURCE),
  helper_source_manifest_ref=instruction["helper_source_manifest_ref"],runtime_identity_verification_ref=instruction["runtime_identity_verification_ref"],
  request_ref=request_ref,registration_ref=registration_ref,decision_snapshot_ref=dr,ledger_snapshot_ref=lr,
  preexisting_registration_capture_ref=capture_ref,created_at_utc=created.isoformat(),expires_at_utc=expires.isoformat(),
  exact_authority_seconds=259200,cpu_ids=CPUS,shared_sources_modified=False,shared_metadata_publication_pending=True,
  packet_and_output_created=False,admission_execution_resource_inspection_or_accounting_performed=False,
  next_admission_argv=[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",INTEGRATION.replace("/",".")+".root_driver_v1","--admit-ref",json.dumps(request_ref)])
 receipt_ref=put(prep,"staging_receipt.root_v1.json",receipt)
 return dict(request_ref=request_ref,receipt_ref=receipt_ref,shared_metadata_publication_pending=True,
  admission_or_execution_performed=False,next_admission_argv=receipt["next_admission_argv"])
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--stage-root-instruction-ref",help="Exact ROOT-authored path/SHA JSON; metadata staging only")
 args=parser.parse_args(argv)
 if args.stage_root_instruction_ref is None:
  print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",project_reads_or_writes=False,admission_execution_or_resources=False)));return 0
 result=stage(json.loads(args.stage_root_instruction_ref));print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
