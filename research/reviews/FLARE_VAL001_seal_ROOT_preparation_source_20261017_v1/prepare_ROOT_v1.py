"""Inactive ROOT-only FLARE VAL001 seal metadata preparation. Never admits or launches.
Import is inert. Actual preparation requires a separate ROOT instruction after review.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL001_seal_ROOT_preparation_source_20261017_v1"
SOURCE=HERE+"/prepare_ROOT_v1.py"
PACKAGE="research.reviews.FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2"
PARENT="american_c3_val_001"
PINS={'driver': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2/root_driver_v1.py', 'sha256': 'dbfa0d3f2682088a5aa5536aa7a2d072671134759c22cc571895ccd5efba3710'}, 'manifest': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2/source_manifest.v2.json', 'sha256': '1c3a70a51b0d40cb960e5ffa34d71929e54b2615271c5842fe80bb5323939dfd'}, 'handoff': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2/source_handoff.v2.json', 'sha256': '9458348bbd4bdfc6ceb6adeea53dedc18a54340a8c640e0ad9ed1e511ff151dc'}, 'review': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_independent_20261017_v2/review.v2.operational_update.v1.json', 'sha256': '521fe0e98b5c287c80b49f4b520f5d0561f5ac4daa1bfb455e9406fc7867a870'}, 'context': {'path': 'research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_3h_setupv4_20261017_v1/american_c3_val_001.FLARE_context.json', 'sha256': 'f38075c72f5054dedc3502aa27b2ab7c4d9432e7893160dd73cf573a73acdb3f'}, 'prior_failure': {'path': 'research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_3h_setupv4_20261017_v1/prior_failure_acceptance.root_active_v1.json', 'sha256': '22ce5682a735386b73a5379a1af317345898763084a2376fbbc5e7db29158816'}, 'predecessor': {'path': 'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/conversion_predecessor_stage_evidence.root_active_v1.json', 'sha256': '74c2d2c7d2597356ec764fa2196ba3c3a5597e12478a60b0368b55ad9cb2ae7e'}, 'selector': {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json', 'sha256': 'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}, 'final_certificate': {'path': 'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/final_original_journal_canonical_certificate.root_v1.json', 'sha256': '46bf49a388992ff0e4ed42cf3679f703cbc72d5bb08825fecfc458b98ebec7ae'}, 'native_completed': {'path': 'data/FLARE_original_remaining_VAL001_native_bridgev2_root01_oct17/completed.json', 'sha256': '977e3ea270a88e1c03052edb6cc00e56b3441cf8204babb4374ea662342ca213'}, 'native_wait': {'path': 'research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17/worker_aux/actual_wait.json', 'sha256': '604eb79528b8b5bccdc0767129d19e1557161173fd30883a64c16314679896a3'}, 'native_input': {'path': 'research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17/input.json', 'sha256': 'a88d93a5abd7ce7e84850fbfa7b48a3a1eb82d75ca9d9518a8f29e786f44b970'}, 'native_outer': {'path': 'research/reviews/research_execution_ROOT_20261017_v26/FLARE_VAL001_native_actual_outer_consumed.root_v1.json', 'sha256': '7efeeb510bfe4def75e00a526165be116d5f90c940910497a145524fc125e3ca'}}
NAMES=dict(preparation_directory="research/reviews/FLARE_remaining_ROOT_preparation_VAL001_seal_bridgev2_root01_oct17",
    packet_directory="research/reviews/FLARE_remaining_ROOT_actual_VAL001_seal_bridgev2_root01_oct17",
    output_directory="data/FLARE_original_remaining_VAL001_seal_bridgev2_root01_oct17")
PHASE_BUDGET=dict(setup_seconds=7200,scientific_seconds=3600,wall_seconds=4200,combined_seconds=11400,
    evidence_reserve_seconds=300,authority_margin_seconds=600)

def require(ok,message):
    if not ok:raise ValueError(message)

def local(name):
    require(type(name)is str and not Path(name).is_absolute() and ".."not in Path(name).parts,"Contained relative ROOT path")
    p=ROOT/name;require(p.resolve()==p and ROOT in p.parents,"Canonical contained ROOT path")
    return p

def bounded_bytes(record):
    require(type(record)is dict and set(record)=={"path","sha256"},"Exact bounded metadata/source ref")
    p=local(record["path"]);require(p.suffix in(".json",".py"),"Only JSON metadata or source; no payloads")
    before=p.lstat();require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**20,"Singly linked bounded metadata/source")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb")as f:
        opened=os.fstat(f.fileno());data=f.read(2**20+1);after=os.fstat(f.fileno())
    named=p.lstat();identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(opened)==identity(after)==identity(named) and len(data)==before.st_size
        and hashlib.sha256(data).hexdigest()==record["sha256"],"Exact stable metadata/source bytes")
    return data

def reference(name):
    p=local(name);require(p.suffix in(".json",".py")and p.is_file()and not p.is_symlink()and p.stat().st_size<=2**20,"Only bounded ROOT metadata/source")
    record=dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest());bounded_bytes(record);return record

def read(record):
    require(Path(record["path"]).suffix==".json","Only bounded JSON metadata")
    return json.loads(bounded_bytes(record))

def put(path,value):
    name=str(path.relative_to(ROOT));p=local(name)
    require(p.parent==local(NAMES["preparation_directory"]),"All writes confined to new preparation only")
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=2**20,"Compact ROOT metadata")
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb")as f:f.write(data);f.flush();os.fsync(f.fileno())
    return reference(name)


NATIVE_KEY="H041/FLARE-original-native-FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17"

def validate_native_close_v1(instruction,c):
    """Read exact immutable receipts; never observe, complete or close a live operation."""
    a=c.read(instruction["native_close_acceptance_ref"])
    expected=dict(schema="ROOT_FLARE_VAL001_native_close_acceptance_v1",root_authorizer="root_agent",accepted=True,
        native_input_ref=PINS["native_input"],actual_wait_ref=PINS["native_wait"],completed_ref=PINS["native_completed"],
        native_outer_terminal_ref=PINS["native_outer"],reservation_key=NATIVE_KEY,storage_policy_selector_ref=PINS["selector"],
        original_files_scopes_charges_clocks_and_failure_history_preserved=True,
        all_native_future_write_resume_relaunch_capabilities_revoked=True,seal_execution_authorized=False)
    require(all(type(a.get(k))is type(v)and a.get(k)==v for k,v in expected.items()),
        "Later genuine exact native-close ROOT acceptance required; template/planning/old close rejected")
    outer=c.read(PINS["native_outer"]);wait=c.read(PINS["native_wait"]);inp=c.read(PINS["native_input"])
    require(outer.get("tool_session_id")==69102 and outer.get("outer_tool_terminal_consumed")is True
        and type(outer.get("actual_outer_exit_code"))is int and outer["actual_outer_exit_code"]==0
        and outer.get("actual_direct_worker_returncode")==0 and outer.get("all_owned_processes_and_groups_drained")is True
        and outer.get("completed_ref")==PINS["native_completed"]and outer.get("actual_wait_ref")==PINS["native_wait"]
        and outer.get("no_truth_read")is True and wait.get("actual_direct_worker_wait")is True
        and type(wait.get("returncode"))is int and wait["returncode"]==0
        and wait.get("all_owned_processes_and_groups_drained")is True and wait.get("operational_failure")is None
        and wait.get("input_ref")==PINS["native_input"]and wait.get("completed_ref")==PINS["native_completed"],
        "Exact genuine native wait0/drain and consumed outer69102; no invented completion")
    require(inp.get("root_driver_ref")==PINS["driver"]and inp.get("main_source_ref")==dict(
        path=PACKAGE.replace(".","/")+"/worker_entry_v1.py",sha256="13f5d7a124fd188e6c48a17bab4c0c6389f98abed9ec0fcfff6caa1d8228e9b6")
        and inp.get("source_sha256")==c.read(PINS["manifest"])["source_sha256"],"Exact authentic admitted bridgev2 source closure")
    closed=c.read(a["ordinary_close_receipt_ref"]);reserve=inp["storage"]["reservation"]
    require(closed.get("kind")=="concurrent_completed_scope_close_receipt_v1"and closed.get("key")==NATIVE_KEY==reserve["key"]
        and type(closed.get("artifacts_deleted"))is int and closed["artifacts_deleted"]==0
        and closed.get("value",{}).get("future_writes_and_relaunch_revoked")is True
        and closed["value"].get("request",{}).get("completed_ref")==PINS["native_completed"]
        and closed["value"]["request"].get("direct_wait_ref")==PINS["native_wait"]
        and closed["value"].get("reservation_sha256")==hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
        "Later genuine ordinary native close receipt for exact reservation; no credit/failure relabel")
    terminal=c.read(a["ordinary_close_outer_terminal_ref"])
    require(terminal.get("tool_session_id")==11560 and terminal.get("actual_terminal_consumed")is True
        and any(type(terminal.get(k))is int and terminal[k]==0 for k in("actual_exit_code","actual_tool_exit_code","actual_outer_exit_code"))
        and (terminal.get("ordinary_close_receipt_ref")==a["ordinary_close_receipt_ref"]or terminal.get("receipt_ref")==a["ordinary_close_receipt_ref"]),
        "Later genuine consumed ordinary-close11560 outer0 bound to this receipt required")
    return a

def construct_metadata_v1(instruction,instruction_ref,*,c,put_metadata,own_ref,now):
    """Exact original schemas; writer and metadata contract are injectable for RAM checks."""
    require(instruction.get("schema")=="ROOT_FLARE_VAL001_seal_metadata_preparation_v1"
        and instruction.get("root_authorizer")=="root_agent"and instruction.get("active")is True
        and instruction.get("metadata_preparation_only")is True
        and instruction.get("admission_authorized")is False and instruction.get("launch_authorized")is False
        and instruction.get("seal_request_metadata_authorized")is True
        and instruction.get("ROOT_reviewed_preparer_source")is True
        and instruction.get("preparation_source_ref")==own_ref
        and instruction.get("stage")=="seal"and instruction.get("parent_id")==PARENT
        and all(instruction.get(k)==v for k,v in NAMES.items())
        and instruction.get("storage_policy_selector_ref")==PINS["selector"],
        "Separate explicit ROOT instruction for this reviewed metadata-only seal preparation")
    native_chain=validate_native_close_v1(instruction,c)
    created=datetime.fromisoformat(instruction["created_at_utc"]);expires=datetime.fromisoformat(instruction["expires_at_utc"])
    require(created.tzinfo is not None and expires.tzinfo is not None and created<=now<expires
        and (expires-created).total_seconds()==4*3600
        and (expires-now).total_seconds()>PHASE_BUDGET["combined_seconds"]+600,
        "Fresh finite4h ROOT authority with full original combined work and margin")
    ids=instruction["cpu_ids"]
    require(type(ids)is list and len(ids)==len(set(ids))==4 and all(type(i)is int and i>=0 for i in ids),"ROOT supplies four explicit actual CPU identities")
    review=c.read(PINS["review"])
    require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"
        and review.get("source_ref")==PINS["driver"]and review.get("source_manifest_ref")==PINS["manifest"]
        and review.get("source_handoff_ref")==PINS["handoff"]and review.get("blockers")==[],"Exact independently passed FLAREv2 source")
    ctx=c.completion()._context(PINS["context"],parent_id=PARENT);row=ctx["targets"][PARENT]
    converted=c.read(c.read(PINS["predecessor"])["completion_ref"])
    require(row["haplotypes"]==1600 and row["input_sites"]==227316
        and converted["input_sites"]==227316 and converted["native_expected_markers"]==26609
        and converted["reference_haplotypes"]==600 and converted["query_haplotypes"]==1000
        and converted["common_focal_records"]==10000,"Exact full observed support and genuine native marker count")
    final=c.read(PINS["final_certificate"])
    require(final.get("final_original_journal_accepted")is True and final.get("root_authorizer")=="root_agent"
        and final.get("storage_policy_selector_ref")==PINS["selector"],"Genuine25243 final journal certificate; no historical relabel")
    c.stage_evidence(PINS["predecessor"],stage="convert",parent=PARENT,completion_ref=c.read(PINS["predecessor"])["completion_ref"])
    native=c.read(PINS["native_completed"])
    require(native.get("status")=="completed_C3_native_FLARE"and native.get("parent_id")==PARENT
        and native.get("conversion_ref")==c.read(PINS["predecessor"])["completion_ref"]
        and native.get("focal_registry_ref")==row["focal_registry_ref"]and native.get("native_options")==converted["native_options"]
        and native.get("query_ancestry_truth_opened")is False and native.get("native_outputs_modified")is False,
        "Genuine native run retains original full support/options and no truth")
    prep=ROOT/NAMES["preparation_directory"]
    canonical_outer=dict(schema="ROOT_FLARE_native_canonical_outer_terminal_v1",root_authorizer="root_agent",
        tool_session_id=69102,actual_terminal_consumed=True,actual_exit_code=0,completed_ref=PINS["native_completed"],
        actual_direct_wait_ref=PINS["native_wait"],original_consumed_outer_ref=PINS["native_outer"],
        canonical_alias_only=True,original_record_unchanged=True,no_truth_read=True)
    outer_ref=put_metadata(prep/"native_outer_canonical_alias.root_v1.json",canonical_outer)
    evidence=dict(schema="FLARE_remaining_stage_ROOT_terminal_evidence_v1",root_authorizer="root_agent",stage="native",
        parent_id=PARENT,completion_ref=PINS["native_completed"],actual_wait_ref=PINS["native_wait"],
        input_ref=PINS["native_input"],outer_terminal_ref=outer_ref,
        ordinary_close_receipt_ref=native_chain["ordinary_close_receipt_ref"],
        ordinary_close_terminal_acceptance_ref=instruction["native_close_acceptance_ref"])
    native_evidence_ref=put_metadata(prep/"native_predecessor_stage_evidence.root_v1.json",evidence)
    registration=dict(schema="FLARE_root_target_registration_v1",root_authorizer="root_agent",parent_id=PARENT,
        observation_ref=row["observation_ref"],focal_registry_ref=row["focal_registry_ref"],count_manifest_ref=row["count_manifest_ref"],
        scientific_role="registered_heldout_full30Mb_comparison",query_ancestry_truth_opened=False,score_authorized=False,
        hypothesis_id="H041",missing_original_cell=True,VAL000_rerun=False)
    registration_ref=put_metadata(prep/"american_c3_val_001.FLARE_registration.root_v1.json",registration)
    runtime=dict(prefix=str(ROOT/".runtime/env"),CPU_threads=4,cpu_ids=copy.deepcopy(ids),CUDA_VISIBLE_DEVICES="",package_report_ref=c.BASE_RUNTIME)
    runtime_ref=put_metadata(prep/"runtime.root_active_v1.json",runtime)
    q=dict(schema="FLARE_remaining_CPU_ROOT_request_v1",active=True,root_authorizer="root_agent",execution_authorized=True,
        stage="seal",phase="seal_"+PARENT,parent_id=PARENT,GPU_authorized=False,CUDA_authorized=False,physical_gpu=None,
        remaining_VAL_truth_authorized=False,TEST_authorized=False,automatic_retry_resume_or_fallback=False,VAL000_rerun=False,
        source_manifest_ref=PINS["manifest"],source_review_ref=PINS["review"],storage_policy_selector_ref=PINS["selector"],
        cpu_ids=copy.deepcopy(ids),context_ref=PINS["context"],conversion_ref=c.read(PINS["predecessor"])["completion_ref"],
        native_run_ref=PINS["native_completed"],previous_stage_evidence_ref=native_evidence_ref,prior_failure_acceptance_ref=PINS["prior_failure"],
        registration_ref=registration_ref,runtime_ref=runtime_ref,created_at_utc=instruction["created_at_utc"],expires_at_utc=instruction["expires_at_utc"],
        phase_budget=copy.deepcopy(PHASE_BUDGET),ROOT_metadata_preparation_instruction_ref=instruction_ref,**NAMES)
    cfg=c.make_config(q);caps=c.stage_caps("seal",cfg)
    require(cfg["mode"]=="seal_native"and cfg["native_run_ref"]==PINS["native_completed"]
        and cfg["max_sites_bound"]==227316 and cfg["max_markers"]==26609 and cfg["max_focal_records"]==10000
        and cfg["prediction_ref"]is None and cfg["write_derived_marker_tracts"]is False and cfg["max_tract_records"]==0
        and cfg["external_truth_ref"]is None and cfg["output_cap_bytes"]==32000000
        and caps["reservation_bytes"]==40388608 and caps["cpu_threads"]==4 and caps["rss_bytes"]==4*2**30
        and caps["scientific_seconds"]==3600,"Unchanged original author seal with full retained raw posteriors/focals; no scoring")
    expected_config_ref=put_metadata(prep/"expected_seal_config.review_only.root_v1.json",cfg)
    q["caps"]=caps
    directive=dict(q,schema="FLARE_remaining_CPU_ROOT_directive_v1",
        directive="Execute one exact missing original FLARE stage after genuine fresh admission0")
    q["root_directive_ref"]=put_metadata(prep/"ROOT_directive.root_active_v1.json",directive)
    request_ref=put_metadata(prep/"ROOT_request.root_active_v1.json",q)
    return dict(status="PREPARED_METADATA_ONLY_NOT_ADMITTED",request_ref=request_ref,expected_config_review_only_ref=expected_config_ref,
        registration_ref=registration_ref,runtime_ref=runtime_ref,directive_ref=q["root_directive_ref"],
        original_context_ref=PINS["context"],conversion_predecessor_evidence_ref=PINS["predecessor"],native_predecessor_evidence_ref=native_evidence_ref,
        native_close_acceptance_ref=instruction["native_close_acceptance_ref"],
        seal_output_bytes=32000000,whole_scope_reservation_bytes=40388608,preparation_source_ref=own_ref,
        admission_or_launch_performed=False,capacity_or_current_guard_state_proven=False,no_new_child_activation=True)

def prepare_v1(instruction_ref):
    require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,
        "ROOT exact repository -B -s optimize0 interpreter")
    for record in PINS.values():bounded_bytes(record)
    instruction=read(instruction_ref);own_ref=reference(SOURCE)
    # Only immutable source/metadata contract construction. No activate/guard/admit/execute call.
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    d=importlib.import_module(PACKAGE+".root_driver_v1")
    require(Path(d.__file__)==local(PINS["driver"]["path"]),"Exact reviewed native/seal wrapper origin")
    c=d.engine().contract()
    prep=local(NAMES["preparation_directory"])
    require(not prep.exists() and not local(NAMES["packet_directory"]).exists()
        and not local(NAMES["output_directory"]).exists(),"New preparation and never-reserved seal targets; no replay")
    # Validate pure metadata construction against a RAM writer before the first preparation write.
    ram={}
    def ram_put(path,value):
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();r=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())
        ram[r["path"]]=(r,value);return r
    construct_metadata_v1(instruction,instruction_ref,c=c,put_metadata=ram_put,own_ref=own_ref,now=datetime.now(timezone.utc))
    prep.mkdir(mode=0o700)
    result=construct_metadata_v1(instruction,instruction_ref,c=c,put_metadata=put,own_ref=own_ref,now=datetime.now(timezone.utc))
    # Complete original metadata predicate path; it performs no storage/resource operation.
    c.validate_request(result["request_ref"])
    result["metadata_preparation_receipt_ref"]=put(prep/"prepared_metadata_only.root_v1.json",result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--ROOT-instruction-ref");a=p.parse_args(argv)
    if not a.ROOT_instruction_ref:
        print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",preparation=False,admission=False,launch=False)));return 0
    result=prepare_v1(json.loads(a.ROOT_instruction_ref));print(json.dumps(result,sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
