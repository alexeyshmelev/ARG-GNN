"""Pure fixedchr20 native-v2 metadata constructor. Import is inert.
No filesystem/network, project imports, guards, admissions, payloads or execution.
Root supplies actual JSON metadata bytes and owns materialization and authority.
"""
from pathlib import PurePosixPath
from datetime import datetime
import hashlib,json,re

ROOT="/workspace-SR003.nfs2/ARG-GNN"
HERE="research/reviews/H056_native_binding_constructor_source_20261016_v1"
META="research/reviews/H056_chr20_native_actual_root_20261016_v2"
JOB="H056-native-chr20-recovered-pre-size-v2"
KEY="H056/native-chr20-recovered-pre-size-v2"
OUTPUT="experiments/results/H056_chr20_H36_Relate_pre_size_CPU_v2"
AUXILIARY="experiments/results/H056_chr20_H36_Relate_pre_size_CPU_v2_aux"
PROTOCOL_REF={'path': 'research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json', 'sha256': '9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e'}
PROFILE={'effective_haploid_N': 30000, 'mutation_rate': '1.25e-8', 'years_per_generation': 28, 'population_size_iterations_executed': 0, 'pre_size_only': True, 'seed': 414855340, 'native_memory_GB': '2'}
FIXED_REFS={'guard': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, 'policy': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'renewal': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'renewal_certificate': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'repair_authority': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'installation_metadata': {'path': 'research/reviews/H056_native_runtime_integration_source_20261016_v1/pinned_original_installation.metadata.v1.json', 'sha256': '578e5c04ce297c80c742376297f51412270421958085b2bb28ca86bb9d7e47c8'}, 'map_source_metadata': {'path': 'research/reviews/H056_map_source_metadata_20261016_v1/source_metadata.root_v1.json', 'sha256': '0745c2a210d855607abb710da840b1b513d53c96adfde7a67d6f4c75700a767e'}, 'memo': {'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}, 'v4': {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}}
SOURCE_REFS={'driver': {'path': 'research/reviews/H056_native_runtime_integration_source_20261016_v1/native_h056_v2.py', 'sha256': 'db34a73c1436e151e7e7c300e215611cf0001595b3279fa6d4c0b8bd3acaff4e'}, 'owned_helpers': {'path': 'research/reviews/H056_native_runtime_integration_source_20261016_v1/native_owned_helpers_v1.py', 'sha256': '92e7f13bab3687c1ec17cd9cebcd24eda18548c220207c845dd0811f37a681ea'}, 'native_core': {'path': 'research/reviews/H056_H36_implementation_20261016_v1/native_h36_core_v1.py', 'sha256': '55d63a9717dce70414fd62614090ec53279a9e2070e0447ea0a8af03575a9c5c'}, 'contract': {'path': 'research/reviews/H056_H36_implementation_20261016_v1/contract_v1.py', 'sha256': '4c7727e27be2dddeb698dc78ff6d7ca143d85b47187a1fdd23dc7c17217d8662'}, 'serialized_readback': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/serialized_readback_v1.py', 'sha256': '0deb54f25d1fb1e9bd85b2c93e642d2f12044313519ce03b5e4dfd11dec666e4'}, 'repaired_bridge': {'path': 'research/reviews/H056_acquisition_runtime_source_20261016_v1/bridge_all_candidates_v1.py', 'sha256': 'dbb13e97dc3fb9051abd1ec0289f0f383d85eadb36df0a893703f81460cdb53a'}}
EXPECTED_RUNTIME_IDENTITY={'python_binary_ref': {'path': '.runtime/env/bin/python3.11', 'sha256': '463a65c4055c18d88f6ed863d227383593395c2f7f70976e38df24f17a836289'}, 'python_version': '3.11.17 (main, Oct  7 2026, 00:02:24) [GCC 15.3.0]', 'native_binary_refs': {'Relate': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/bin/Relate', 'sha256': '71ce82f469dd99749a3eb1847e0704be8785a26be94eae2cb8e1ed80881c7ce8'}, 'RelateFileFormats': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/bin/RelateFileFormats', 'sha256': 'cb19cb6e277c27704de506f0e8155f8f53d9c3dc8d0f52615a059caa253332ee'}}, 'license_ref': {'path': '.runtime/tools/relate/relate_v1.2.4_x86_64_static/20180411_15670_RELATE_Academic_Usage_Licence.pdf', 'sha256': 'c55aec267c4f8987f1d67c91cc2fb44a695c3f06244e3bf7be91230e61f2dbe2'}}
MAP_ASSET={'chromosome': '20', 'source_path': 'maps/genetic_maps.b37/chr20.b37.gmap.gz', 'git_blob_sha1': '2d28bfb7f3825930f77802d9513e96cd5b5ba8c4', 'bytes': 583035, 'URL': 'https://raw.githubusercontent.com/odelaneau/GLIMPSE/103ca071ea9e4e648192916142e5ed353ac94e27/maps/genetic_maps.b37/chr20.b37.gmap.gz'}
SUBJECTS=("Bar8","Bon002","KK1","SATP","Loschbour","Bichon","Sidelkino","NEO88",
 "NEO866","NEO945","NEO28","NEO942","NEO790","NEO739","Gjerrild5","NEO735","NEO92","NEO870")
AGES=(8071,10078,9720,13255,8050,13665,11258.5,7858,5457,5444,5461,5495,5668,3971,4110,3995,4218,4255)
CLASSES=("ANA","CHG","WHG","EHG")
DATA_NAMES={"baseline.haps","baseline.sample","sample_ages.generations","qualified.regional.map",
 "frozen_site_identities.json","polarization_equivalence.json","bridge_preparation.json"}
BUNDLE_ROLES={"completed","readback","actual_wait","counts","map_qualification","roster"}
DIRECT_SUFFIXES={".py",".sh",".json"}

def require(ok,message):
    if not ok:
        raise ValueError(message)

def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()

def relative_path(value):
    require(isinstance(value,str) and value and "\\" not in value,"Project POSIX relative path")
    p=PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and "." not in p.parts
            and p.as_posix()==value and value!=".","Normalized contained relative path")
    return value

def exact_ref(value,*,bytes_allowed=False):
    keys=({"path","sha256"},{"path","sha256","bytes"}) if bytes_allowed else ({"path","sha256"},)
    require(isinstance(value,dict) and set(value) in keys
            and isinstance(value["sha256"],str) and re.fullmatch("[0-9a-f]{64}",value["sha256"]),
            "Exact path/SHA reference")
    relative_path(value["path"])
    if "bytes" in value:
        require(type(value["bytes"])is int and value["bytes"]>0,"Positive actual artifact byte annotation")
    return dict(value)

def plain_ref(value):
    exact_ref(value,bytes_allowed=True)
    return {k:value[k] for k in ("path","sha256")}

def metadata_record(record):
    """Authenticate supplied JSON metadata bytes only. Never opens a path."""
    require(isinstance(record,dict) and set(record)=={"ref","raw"},"Explicit metadata ref and actual raw bytes")
    r=exact_ref(record["ref"])
    raw=record["raw"]
    require(PurePosixPath(r["path"]).suffix==".json" and isinstance(raw,bytes)
            and 0<len(raw)<=16*2**20 and hashlib.sha256(raw).hexdigest()==r["sha256"],
            "Bounded exact actual JSON metadata bytes")
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,"Unique JSON metadata keys")
            out[k]=v
        return out
    def invalid(v):
        raise ValueError("No nonfinite JSON metadata constant "+v)
    return r,json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def unique(refs):
    out={}
    for r in refs:
        exact_ref(r)
        require(PurePosixPath(r["path"]).suffix in DIRECT_SUFFIXES,"Original direct PY/SH/JSON suffixes")
        require(r["path"] not in out or out[r["path"]]==r["sha256"],"No conflicting source/metadata refs")
        out[r["path"]]=r["sha256"]
    return [dict(path=p,sha256=h) for p,h in sorted(out.items())]

def ceil_mib(n):return ((n+2**20-1)//2**20)*2**20

def derive_caps_v1(counts,max_chunks):
    """Finite abort ceilings, not predictions or established memory/usage fits."""
    require(set(counts)=={"candidate_sites","sites","constants"} and all(type(v)is int for v in counts.values())
      and counts["candidate_sites"]>=counts["sites"]>=1000 and 0<=counts["constants"]<=counts["sites"],"Actual C/S/K")
    S=counts["sites"]
    require(type(max_chunks)is int and 1<=max_chunks<=S,"Root finite chunk ceiling")
    children=5*max_chunks+3
    workspace=ceil_mib(8*2**20+8192*S+(20+8*max_chunks)+max_chunks*(16+4*S))
    facts=ceil_mib(2**20+3072*S)
    packet=ceil_mib(2**20+32768*children)
    output=workspace+facts+packet+2*2**20
    logs=min(16_000_000,1_000_000*children)
    total=ceil_mib(output+logs+2**20)
    return dict(native_workspace_abort_cap_bytes=workspace,native_facts_cap_bytes=facts,
      packet_cap_bytes=packet,native_output_cap_bytes=output,combined_log_cap_bytes=logs,
      output_cap_bytes=total,native_file_cap_bytes=max(workspace,facts,packet),
      native_output_file_count_cap=1024+32*children,input_bundle_cap_bytes=16*2**20,
      max_chunks=max_chunks,maximum_native_children=children)

def validate_bundle_v1(bundle):
    require(isinstance(bundle,dict) and set(bundle)==BUNDLE_ROLES,"Exactly six actual acquisition JSON records")
    decoded={k:metadata_record(v) for k,v in bundle.items()}
    refs={k:v[0] for k,v in decoded.items()}
    docs={k:v[1] for k,v in decoded.items()}
    done,rb,wait,cd,q,roster=[docs[k] for k in ("completed","readback","actual_wait","counts","map_qualification","roster")]
    require(cd["schema"]=="H056_actual_acquisition_counts_v1" and cd["protocol_ref"]==PROTOCOL_REF
            and cd["chromosome"]=="20","Actual samechr20 acquisition count seal")
    counts=cd["counts"]
    # The unchanged formula validates actual integerC/S/K. J1 here is only a structural validation.
    derive_caps_v1(counts,1)
    require(done["schema"]=="H056_acquisition_runtime_completed_v1"
            and done["status"]=="completed_acquisition_bridge_and_serialized_readback_root_wait_required"
            and done["chromosome"]=="20" and done["protocol_ref"]==PROTOCOL_REF
            and done["counts"]==counts and done["no_native_or_model_execution"] is True
            and plain_ref(done["artifacts"]["counts.json"])==refs["counts"],"Actual completed acquisition/count join")
    require(rb["schema"]=="H056_acquisition_serialized_readback_root_sealed_v1"
            and rb["status"]=="root_actual_wait0_drain_and_serialized_readback_sealed"
            and rb["protocol_ref"]==PROTOCOL_REF and rb["chromosome"]=="20" and rb["counts"]==counts
            and rb["root_preparation_wait_verified"] is True and rb["root_owned_group_absent"] is True
            and rb["worker_completion_ref"]==refs["completed"]
            and rb["worker_readback_ref"]==plain_ref(done["worker_readback_ref"])
            and rb["preparation_actual_wait_ref"]==refs["actual_wait"] and rb["retained_sites"]==counts["sites"],
            "Actual root-sealed readback/completion/count/wait join")
    for key in ("actual_serialized_site_nucleotide_phase_order_verified",
                "actual_serialized_sample_haplotype_date_join_verified",
                "actual_qualified_map_bytes_and_local_site_join_verified"):
        require(rb[key] is True,"Actual independently serialized proof: "+key)
    require(wait["schema"]=="H056_root_actual_preparation_wait_v1" and wait["root_authorizer"]=="root_agent"
            and wait["actual_tool_call_id"] and wait["actual_tool_terminal_consumed"] is True
            and wait["actual_tool_exit_code"]==0
            and type(wait["worker_pid"])is int and wait["worker_pid"]>1
            and wait["process_group"]==wait["worker_pid"]
            and wait["actual_worker_PID_absent"] is True and wait["actual_owned_group_absent"] is True
            and wait["worker_readback_ref"]==plain_ref(done["worker_readback_ref"])
            and wait["worker_completion_ref"]==refs["completed"],"Supplied genuine root externalwait0 and own drain evidence")
    data={name:exact_ref(done["artifacts"][name],bytes_allowed=True) for name in DATA_NAMES}
    require(all(set(r)=={"path","sha256","bytes"} for r in data.values()),"Actual rich artifact refs retained")
    bridge=PurePosixPath(data["baseline.haps"]["path"]).parent.as_posix()
    require(all(r["path"]==bridge+"/"+name and rb["actual_output_sha256"][name]==r["sha256"]
                for name,r in data.items()),"All seven actual same-scope serialized bridge/map hashes")
    require(sum(r["bytes"] for r in data.values())<=16*2**20,"Native unchanged16MiB complete input bundle bound")
    require(q["schema"]=="H056_map_qualification_v1" and done["map_qualification_ref"]==refs["map_qualification"]
            and q["root_authorizer"]=="root_agent" and q["chromosome"]=="20"
            and q["map_sha256"]==data["qualified.regional.map"]["sha256"]
            and q["byte_identity_verified"] is True and q["source_semantics_qualified"] is True
            and q["source_map_brackets_full_requested_interval"] is True
            and q["uniform_map_substitution"] is False and q["source_positions_or_distances_rescaled"] is False
            and q["assembly"]=="GRCh37" and q["physical_unit"]=="bp" and q["genetic_unit"]=="cM"
            and q["source_boundary_bp1"]==[20000001,30000001] and q["local_boundary_bp1"]==[1,10000001],
            "Exact actual original-family qualified samechr20 map/support")
    require(q["source_repository"]=="https://github.com/odelaneau/GLIMPSE"
            and q["source_commit"]=="103ca071ea9e4e648192916142e5ed353ac94e27"
            and q["source_path"]==MAP_ASSET["source_path"] and q["git_blob_sha1"]==MAP_ASSET["git_blob_sha1"]
            and done["original_map_source_refs"]==dict(raw_map=q["raw_map_ref"],map=q["regional_map_ref"],
                 map_semantics=q["source_semantics_ref"],map_qualification=refs["map_qualification"])
            and q["regional_map_ref"]["sha256"]==data["qualified.regional.map"]["sha256"],
            "Original source commit/path/blob and complete actual source-reference join")
    require(isinstance(roster,list) and len(roster)==36,"Actual36-row roster JSON")
    for i,row in enumerate(roster):
        person=i//2;label=person//2 if i<16 else None
        require(type(row["sample_index"])is int and row["sample_index"]==i
                and row["subject_id"]==SUBJECTS[person] and type(row["haplotype"])is int and row["haplotype"]==i%2
                and row["role"]==("reference" if i<16 else "query")
                and row["observed_reference_class"]==label
                and row["observed_cohort"]==(CLASSES[label] if label is not None else None)
                and row["query_ancestry_truth"] is None and row["native_tree_node_id"] is None
                and row["age_mean_cal_BP"]==AGES[person] and row["age_generations"]==AGES[person]/28.
                and row["date_zero"]=="calBP1950" and row["years_per_generation"]==28,
                "Exact frozen H36/R16/Q20 source subject/phase/class/date metadata")
    require(refs["roster"]["sha256"]=="23e298c586bfcd8b668429552bbeb49bffd148a407d5c53d0000f6ece6b92fc1",
            "Exact byte-frozen planned_H36 roster")
    return refs,docs,dict(counts),data,bridge

def construct_native_chr20_v2_v1(bundle,actual_runtime_identity,original_bindings,*,
                                 max_chunks,created_at_utc,expires_at_utc):
    """Return inactive exact dictionaries/bytes. Root alone writes, admits and authorizes."""
    refs,docs,counts,data,bridge=validate_bundle_v1(bundle)
    require(actual_runtime_identity==EXPECTED_RUNTIME_IDENTITY,"Root actual identity must match pinned original native runtime")
    require(isinstance(original_bindings,dict) and original_bindings
            and all(isinstance(k,str) and k for k in original_bindings),"Complete actual original guard binding map")
    for r in original_bindings.values():
        exact_ref(r)
    issued=datetime.fromisoformat(created_at_utc.replace("Z","+00:00"))
    end=datetime.fromisoformat(expires_at_utc.replace("Z","+00:00"))
    require(issued.tzinfo is not None and end.tzinfo is not None
            and 0<(end-issued).total_seconds()<=2400,"Original at-most2400second native authority interval")
    caps=derive_caps_v1(counts,max_chunks)
    scopes=[PurePosixPath(relative_path(p)) for p in (OUTPUT,AUXILIARY,bridge,META)]
    require(all(a!=b and a not in b.parents and b not in a.parents
                for i,a in enumerate(scopes) for b in scopes[i+1:]),"Fresh disjoint intended output/aux/bridge/root metadata scopes")
    cfg=dict(schema="H056_native_Relate_config_v1",protocol_ref=dict(PROTOCOL_REF),chromosome="20",
        scientific_options=dict(PROFILE),caps=caps,device="cpu",physical_gpu=None,cuda_visible_devices="",
        no_replay_or_phase_resume=True,model_execution_authorized=False,
        fixed_refs=json.loads(json.dumps(FIXED_REFS)),job_id=JOB,output_directory=OUTPUT,
        auxiliary_directory=AUXILIARY,bridge_directory=bridge,packet_directory=OUTPUT+"/runtime",
        counts=counts,counts_ref=refs["counts"],bridge_completed_ref=refs["completed"],
        serialized_readback_ref=refs["readback"],bridge_actual_wait_ref=refs["actual_wait"],
        map_qualification_ref=refs["map_qualification"],roster_ref=refs["roster"])
    runtime=dict(schema="H056_native_Relate_runtime_v1",root_authorizer="root_agent",job_id=JOB,
        output_directory=OUTPUT,auxiliary_directory=AUXILIARY,created_at_utc=created_at_utc,
        expires_at_utc=expires_at_utc,python_binary_ref=dict(actual_runtime_identity["python_binary_ref"]),
        python_version=actual_runtime_identity["python_version"],device="cpu",physical_gpu=None,CPU_threads=4,
        global_wall_seconds=2400,native_binary_refs=json.loads(json.dumps(actual_runtime_identity["native_binary_refs"])),
        license_ref=dict(actual_runtime_identity["license_ref"]))
    metadata_refs=dict(counts_ref=refs["counts"],bridge_completed_ref=refs["completed"],
        serialized_readback_ref=refs["readback"],bridge_actual_wait_ref=refs["actual_wait"],
        map_qualification_ref=refs["map_qualification"],roster_ref=refs["roster"],protocol_ref=dict(PROTOCOL_REF))
    manifest=dict(schema="H056_native_Relate_source_manifest_v1",protocol_ref=dict(PROTOCOL_REF),
        source_refs=json.loads(json.dumps(SOURCE_REFS)),metadata_refs=metadata_refs,data_refs=data,
        original_bindings=json.loads(json.dumps(original_bindings)))
    documents={"config.json":cfg,"runtime.json":runtime,"source_manifest.json":manifest}
    docbytes={name:encoded(value) for name,value in documents.items()}
    docrefs={name:dict(path=META+"/"+name,sha256=hashlib.sha256(raw).hexdigest())
             for name,raw in docbytes.items()}
    direct=unique([*original_bindings.values(),*SOURCE_REFS.values(),*metadata_refs.values(),
        *(r for r in FIXED_REFS.values() if PurePosixPath(r["path"]).suffix in DIRECT_SUFFIXES),
        *docrefs.values()])
    require(len(direct)<=64,"Unchanged actual complete direct metadata/source union at most64")
    inactive=dict(schema="H056_native_Relate_root_input_v1",status="inactive",root_authorizer="root_agent",
        hypothesis_id="H056",registered=True,execution_authorized=False,driver_ref=dict(SOURCE_REFS["driver"]),
        config_ref=docrefs["config.json"],runtime_ref=docrefs["runtime.json"],
        source_manifest_ref=docrefs["source_manifest.json"],storage_ref=None)
    documents["input.inactive.json"]=inactive
    docbytes["input.inactive.json"]=encoded(inactive)
    docrefs["input.inactive.json"]=dict(path=META+"/input.inactive.json",
        sha256=hashlib.sha256(docbytes["input.inactive.json"]).hexdigest())
    return dict(schema="H056_native_chr20_v2_pure_binding_construction_v1",
        execution_authorized=False,documents=documents,encoded_documents=docbytes,document_refs=docrefs,
        consumer_bindings=direct,consumer_binding_count=len(direct),actual_counts=counts,caps=caps,
        reservation_expectations=dict(key=KEY,output=ROOT+"/"+OUTPUT,auxiliary_scopes=[ROOT+"/"+AUXILIARY],
            reserved_bytes=caps["output_cap_bytes"],consumer_bindings=direct),
        storage_record_expectations=dict(job_id=JOB,output_directory=OUTPUT,expires_at_utc=expires_at_utc,
            reserved_new_bytes=caps["output_cap_bytes"],completed_overlay_policy_ref=dict(FIXED_REFS["policy"])),
        root_pending=["Independently review constructor and actual metadata bytes/refs",
            "Write exact config/runtime/source_manifest bytes through existing root-owned metadata guard",
            "Use actual original prospective980GB reserve authority and original bindings, key/output/aux and complete direct union",
            "After genuine admission, create fresh input.json from input.inactive.json dictionary with real storage_ref, statusactive and execution_authorizedtrue",
            "Compute exact actual active inputSHA and invoke the already reviewed nativev2; this helper grants no authority",
            "At launch verify current expiry/renewal, exact installed runtime/source hashes and fresh nonexistent output/aux",
            "Require genuine native child+outerwait0/drain/finalscope before bank use"],
        future_active_input_path=META+"/input.json",
        native_argv_template=[".runtime/env/bin/python","-B",SOURCE_REFS["driver"]["path"],"--execute",
            "--root-input",META+"/input.json","--root-input-sha256","<actual-active-input-SHA256>"],
        no_filesystem_or_network_or_project_imports_or_payloads_or_admissions=True)
