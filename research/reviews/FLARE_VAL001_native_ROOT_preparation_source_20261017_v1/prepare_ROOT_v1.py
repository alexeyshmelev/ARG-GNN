"""Inactive ROOT-only FLARE VAL001 native metadata preparation. Never admits or launches.
Import is inert. Actual preparation requires a separate ROOT instruction after review.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL001_native_ROOT_preparation_source_20261017_v1"
SOURCE=HERE+"/prepare_ROOT_v1.py"
PACKAGE="research.reviews.FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2"
PARENT="american_c3_val_001"
PINS={'driver': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2/root_driver_v1.py', 'sha256': 'dbfa0d3f2682088a5aa5536aa7a2d072671134759c22cc571895ccd5efba3710'}, 'manifest': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2/source_manifest.v2.json', 'sha256': '1c3a70a51b0d40cb960e5ffa34d71929e54b2615271c5842fe80bb5323939dfd'}, 'handoff': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2/source_handoff.v2.json', 'sha256': '9458348bbd4bdfc6ceb6adeea53dedc18a54340a8c640e0ad9ed1e511ff151dc'}, 'review': {'path': 'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_independent_20261017_v2/review.v2.operational_update.v1.json', 'sha256': '521fe0e98b5c287c80b49f4b520f5d0561f5ac4daa1bfb455e9406fc7867a870'}, 'context': {'path': 'research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_3h_setupv4_20261017_v1/american_c3_val_001.FLARE_context.json', 'sha256': 'f38075c72f5054dedc3502aa27b2ab7c4d9432e7893160dd73cf573a73acdb3f'}, 'prior_failure': {'path': 'research/reviews/FLARE_remaining_ROOT_preparation_VAL001_convert_3h_setupv4_20261017_v1/prior_failure_acceptance.root_active_v1.json', 'sha256': '22ce5682a735386b73a5379a1af317345898763084a2376fbbc5e7db29158816'}, 'predecessor': {'path': 'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/conversion_predecessor_stage_evidence.root_active_v1.json', 'sha256': '74c2d2c7d2597356ec764fa2196ba3c3a5597e12478a60b0368b55ad9cb2ae7e'}, 'selector': {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json', 'sha256': 'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}, 'final_certificate': {'path': 'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/final_original_journal_canonical_certificate.root_v1.json', 'sha256': '46bf49a388992ff0e4ed42cf3679f703cbc72d5bb08825fecfc458b98ebec7ae'}}
NAMES=dict(preparation_directory="research/reviews/FLARE_remaining_ROOT_preparation_VAL001_native_bridgev2_root01_oct17",
    packet_directory="research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17",
    output_directory="data/FLARE_original_remaining_VAL001_native_bridgev2_root01_oct17")
PHASE_BUDGET=dict(setup_seconds=7200,scientific_seconds=14400,wall_seconds=15000,combined_seconds=22200,
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

def construct_metadata_v1(instruction,instruction_ref,*,c,put_metadata,own_ref,now):
    """Exact original schemas; writer and metadata contract are injectable for RAM checks."""
    require(instruction.get("schema")=="ROOT_FLARE_VAL001_native_metadata_preparation_v1"
        and instruction.get("root_authorizer")=="root_agent"and instruction.get("active")is True
        and instruction.get("metadata_preparation_only")is True
        and instruction.get("admission_authorized")is False and instruction.get("launch_authorized")is False
        and instruction.get("native_request_metadata_authorized")is True
        and instruction.get("ROOT_reviewed_preparer_source")is True
        and instruction.get("preparation_source_ref")==own_ref
        and instruction.get("stage")=="native"and instruction.get("parent_id")==PARENT
        and all(instruction.get(k)==v for k,v in NAMES.items())
        and instruction.get("storage_policy_selector_ref")==PINS["selector"],
        "Separate explicit ROOT instruction for this reviewed metadata-only native preparation")
    terminal=c.read(instruction["H058_admission72068_actual_terminal_ref"])
    require(terminal.get("tool_session_id")==72068 and terminal.get("actual_terminal_consumed")is True
        and type(terminal.get("actual_exit_code"))is int,"Wait for genuine consumed H058 admission72068 terminal; no overlapping mutations")
    created=datetime.fromisoformat(instruction["created_at_utc"]);expires=datetime.fromisoformat(instruction["expires_at_utc"])
    require(created.tzinfo is not None and expires.tzinfo is not None and created<=now<expires
        and (expires-created).total_seconds()==8*3600
        and (expires-now).total_seconds()>PHASE_BUDGET["combined_seconds"]+600,
        "Fresh finite8h ROOT authority with full original combined work and margin")
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
    prep=ROOT/NAMES["preparation_directory"]
    registration=dict(schema="FLARE_root_target_registration_v1",root_authorizer="root_agent",parent_id=PARENT,
        observation_ref=row["observation_ref"],focal_registry_ref=row["focal_registry_ref"],count_manifest_ref=row["count_manifest_ref"],
        scientific_role="registered_heldout_full30Mb_comparison",query_ancestry_truth_opened=False,score_authorized=False,
        hypothesis_id="H041",missing_original_cell=True,VAL000_rerun=False)
    registration_ref=put_metadata(prep/"american_c3_val_001.FLARE_registration.root_v1.json",registration)
    runtime=dict(prefix=str(ROOT/".runtime/env"),CPU_threads=4,cpu_ids=copy.deepcopy(ids),CUDA_VISIBLE_DEVICES="",package_report_ref=c.BASE_RUNTIME)
    runtime_ref=put_metadata(prep/"runtime.root_active_v1.json",runtime)
    q=dict(schema="FLARE_remaining_CPU_ROOT_request_v1",active=True,root_authorizer="root_agent",execution_authorized=True,
        stage="native",phase="native_"+PARENT,parent_id=PARENT,GPU_authorized=False,CUDA_authorized=False,physical_gpu=None,
        remaining_VAL_truth_authorized=False,TEST_authorized=False,automatic_retry_resume_or_fallback=False,VAL000_rerun=False,
        source_manifest_ref=PINS["manifest"],source_review_ref=PINS["review"],storage_policy_selector_ref=PINS["selector"],
        cpu_ids=copy.deepcopy(ids),context_ref=PINS["context"],conversion_ref=c.read(PINS["predecessor"])["completion_ref"],
        native_run_ref=None,previous_stage_evidence_ref=PINS["predecessor"],prior_failure_acceptance_ref=PINS["prior_failure"],
        registration_ref=registration_ref,runtime_ref=runtime_ref,created_at_utc=instruction["created_at_utc"],expires_at_utc=instruction["expires_at_utc"],
        phase_budget=copy.deepcopy(PHASE_BUDGET),ROOT_metadata_preparation_instruction_ref=instruction_ref,**NAMES)
    cfg=c.make_config(q);caps=c.stage_caps("native",cfg)
    require(cfg["max_native_vcf_bytes"]==871184687 and cfg["output_cap_bytes"]==1997355311
        and cfg["heap_gib"]==16 and cfg["timeout_seconds"]==14400 and cfg["native_options"]==converted["native_options"]
        and cfg["external_truth_ref"]is None and caps["reservation_bytes"]==2005743919
        and caps["cpu_threads"]==4 and caps["rss_bytes"]==24*2**30,"Unchanged author native configuration and exact count-derived failstops")
    expected_config_ref=put_metadata(prep/"expected_native_config.review_only.root_v1.json",cfg)
    q["caps"]=caps
    directive=dict(q,schema="FLARE_remaining_CPU_ROOT_directive_v1",
        directive="Execute one exact missing original FLARE stage after genuine fresh admission0")
    q["root_directive_ref"]=put_metadata(prep/"ROOT_directive.root_active_v1.json",directive)
    request_ref=put_metadata(prep/"ROOT_request.root_active_v1.json",q)
    return dict(status="PREPARED_METADATA_ONLY_NOT_ADMITTED",request_ref=request_ref,expected_config_review_only_ref=expected_config_ref,
        registration_ref=registration_ref,runtime_ref=runtime_ref,directive_ref=q["root_directive_ref"],
        original_context_ref=PINS["context"],conversion_predecessor_evidence_ref=PINS["predecessor"],
        native_output_bytes=1997355311,whole_scope_reservation_bytes=2005743919,preparation_source_ref=own_ref,
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
        and not local(NAMES["output_directory"]).exists(),"New preparation and never-reserved native targets; no replay")
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
