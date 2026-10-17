"""Concrete downstream storage/phase contract. No import-time execution.
The64GiB master reservation binds this source/config/runtime. A separate exact
ROOT stage authority binds later genuine prerequisites, never fabricated success.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,stat,sys,time
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import common_v1 as original
ROOT=original.ROOT
HERE="research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2"
DRIVER=HERE+"/execution_v1.py";DIRECT=HERE+"/direct_gpu0_v1.py";CONTEXT=HERE+"/worker_context_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
GUARD={'path': 'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py', 'sha256': '72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb'}
POLICY=None;SELECTOR_REF=None;_SELECTED_GUARD=None;_SELECTOR_VALUE=None;_CONSUMER_MEMO=None
GPU0_UUID=original.GPU0_UUID;BASE_POOL=original.BASE_POOL;GPU0_GUARD=original.GPU0_GUARD
PIDFD=original.PIDFD;OWNED=original.OWNED;NATIVE_PYTHON_REF=original.NATIVE_PYTHON_REF
AUTHOR_ROOT=original.AUTHOR_ROOT;AUTHOR_MANIFEST_REF=original.AUTHOR_MANIFEST_REF
PREP_REF=original.PREP_REF;PREP_CONFIG_REF=original.PREP_CONFIG_REF;COUNTS_REF=original.COUNTS_REF
PREP_OUTPUT=original.PREP_OUTPUT;PARENT=original.PARENT
OWNED_ROSTER_CAP=4096;MINIMAL_WAIT_CAP_BYTES=65536;TERMINAL_RECEIPT_CAP_BYTES=8*2**20
OWNED_ROSTER_POLICY_REF={"path":"research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4/owned_roster_policy.INACTIVE.v4.json","sha256":"ea1ddb59ab4248b8ba95dbbe77d25db30636cc318a4fdaebde1e1ae6e5a7344f"}
SCIENTIFIC_STAGES_REF={"path":"research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/scientific_stages_v1.py","sha256":"455704b88e9247021c09566609801838f88a118420cde856e487aaef4cee30da"}
IO_SOURCE_REF={"path":"research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/efficient_downstream_v1.py","sha256":"bab7057a958e9cfd51f503b5680034c0067226a661b820eb42bb16c7133a94b9"}
IO_POLICY_REF={"path":"research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/io_policy.INACTIVE.v1.json","sha256":"76011c718b4c41f79643b81faf55cacf2d65f549744afb0fc6f3331c3a851b2d"}
IO_REVIEW_REF={"path":"research/reviews/native_VAL000_downstream_IO_efficiency_independent_review_20261016_v1/review.v1.json","sha256":"ea66a23c0d182e63c11cdfc2de8911d25d3637c2077b0b64977b27d905b27c4e"}
GPU_PHASES=("contrastive","embedding_contrastive_train","embedding_contrastive_val","embedding_query_paths_train","embedding_query_paths_val","RAM_retrieve_and_seal")
PHASES=GPU_PHASES+("finalize",)
BASE_CAPS=dict(reservation_bytes=64*2**30,log_bytes=4*2**20,max_seconds=172800,
 max_cpu_rss_bytes=384*2**30,headroom_bytes=4*2**30,cleanup_seconds=30,work_cutoff_margin_seconds=300,
 terminal_evidence_margin_seconds=150,max_checkpoint_bytes=3*2**30,max_checkpoint_files=2)
EXPECTED_CONFIG=dict(schema="native_VAL000_downstream_full_scientific_config_v1",output_cap_bytes=64*2**30,
 IO_policy_ref=IO_POLICY_REF,allowed_phases=list(PHASES),
 scientific_protocol=dict(contrastive=dict(epochs=1,updates=2135,batch=4096,microbatch=4096,precision="fp32",seed=36,temperature=.3,
 save_interval="20ba",save_num_checkpoints_to_keep=1,uniform_class_weight_fallback=True),
 embeddings=dict(batch=256,pooling="head_norm",layer_index=None,max_samples=0,separate_train_val=True),
 retrieval=dict(topk=1,q_batch=1024,e_chunk=100000,reference_rows=9721236,query_rows=10000)),
 author_algorithms_and_scientific_sources_modified=False,private_truth_access=False,
 installed_Composer_nested_Timestamp_metadata_reader_bridge=True,original_failed_MLM_wait_can_be_relabelled_success=False,
 group_scope_caps=dict(contrastive=8*2**30,embedding=56*2**30,RAM_retrieve_and_seal=2**30),
 original_MLM7GiB_scope_extended=False,actual_GPU_fit_qualified=False)
CAPS=dict(BASE_CAPS,max_cuda_reserved_bytes=0)
ROOT_INPUT_REF=None;STATE=None
require=original.require;local=original.local;read=original.read;raw=original.raw;ref=original.ref;load=original.load;save=original.save

def active(value):
    now=datetime.now(timezone.utc)
    require(value.get("root_authorizer")=="root_agent" and value.get("status")=="active" and value.get("template_only") is not True
        and value.get("execution_authorized") is True
        and datetime.fromisoformat(value["created_at_utc"])<=now<datetime.fromisoformat(value["expires_at_utc"]),"Exact unexpired ROOT execution authority")

def current_amendment_ref_v1():
    require(ROOT_INPUT_REF is not None,"Actual ROOT stage input installed")
    return ROOT_INPUT_REF

def set_amendment_v1(record):
    global ROOT_INPUT_REF
    require(ROOT_INPUT_REF is None or ROOT_INPUT_REF==record,"One exact stage input per interpreter")
    ROOT_INPUT_REF=record

def active_amendment_v1():
    require(ROOT_INPUT_REF is not None,"Exact actual ROOT stage input")
    s=read(ROOT_INPUT_REF);q=read(s["root_stage_request_ref"]);master=read(q["master_input_ref"])
    require(q["storage_policy_selector_ref"]==s["storage_policy_selector_ref"]==master["storage_policy_selector_ref"],"Exact master/phase selector, no live route substitution")
    configure_selector_v1(master["storage_policy_selector_ref"])
    active(s);active(q);active(master)
    require(q["source_manifest_ref"]==ref(MANIFEST) and master["source_manifest_ref"]==ref(MANIFEST)
        and q["source_review_passed"] is True and master["source_review_passed"] is True,
        "Exact independently reviewed actual execution source closure")
    for r in (q["source_review_ref"],master["source_review_ref"]):
        review=read(r)
        candidate=review.get("candidate_manifest_ref",review.get("candidate_source_manifest_ref",review.get("source_manifest_ref")))
        require(candidate==ref(MANIFEST) and review.get("independent")is True and review.get("verdict")in("SOURCE_PASS","PASS_SOURCE_ONLY","PASS") and review.get("blockers",review.get("blocking_findings"))==[],
            "Actual independent source review binds this exact manifest")
    end=min(datetime.fromisoformat(q["created_at_utc"]).timestamp()+q["caps"]["max_seconds"],
        datetime.fromisoformat(q["expires_at_utc"]).timestamp(),datetime.fromisoformat(master["expires_at_utc"]).timestamp())
    require(time.time()<end,"Actual phase wall window and master expiry, no automatic reset")
    return q,s,master

def work_deadline_monotonic_v1():
    q,s,master=active_amendment_v1()
    end=min(datetime.fromisoformat(q["created_at_utc"]).timestamp()+CAPS["max_seconds"],
        datetime.fromisoformat(q["expires_at_utc"]).timestamp(),datetime.fromisoformat(master["expires_at_utc"]).timestamp())
    return time.monotonic()+end-time.time()-CAPS["work_cutoff_margin_seconds"]

def frozen_science_metadata_v1():
    prep=read(PREP_REF);pcfg=read(PREP_CONFIG_REF);counts=read(COUNTS_REF)
    require(prep["status"]=="completed_native_C3_path_preparation" and prep["parent_id"]==counts["parent_id"]==PARENT
        and prep["class_order"]==counts["class_order"]==["EUR","EAS","AFR"]
        and prep["query_ancestry_truth_opened"] is False and counts["query_ancestry_truth_opened"] is False
        and prep["observation_ref"]==pcfg["observation_ref"]==original.OBS_REF
        and counts["preparation_ref"]==PREP_REF and counts["preparation_config_ref"]==PREP_CONFIG_REF
        and prep["native_split_seed"]==42 and prep["native_train_fraction"]==.9
        and pcfg["path_sampling_seed"]==93001 and prep["query_focal_count"]==10000
        and prep["vocabulary_nodes"]==counts["vocabulary_nodes"]==107140
        and counts["tokenizer_vocab_size"]==107144 and counts["MLM_optimizer_updates"]==227990
        and counts["contrastive_optimizer_updates"]==2135 and counts["source_corpus_capped_or_sampled"] is False,
        "Exact complete original held-outVAL000 source/seed/observation/row budget")
    expected={"mlm/train":23346222,"mlm/val":2594025,"contrastive/train":8748479,"contrastive/val":972757,"query_paths/train":9003,"query_paths/val":997}
    require(set(prep["native_dataset_refs"])==set(prep["full_path_id_order_refs"])==set(counts["datasets"])==set(expected),"Exact six full native views")
    for key,item in counts["datasets"].items():
        require(item["rows"]==expected[key] and item["dataset_ref"]==prep["native_dataset_refs"][key]
            and item["row_order_ref"]==prep["full_path_id_order_refs"][key],"Exact actual full-path data/order refs")
        view,split=key.split("/")
        require(local(item["dataset_ref"]["path"])==local(PREP_OUTPUT)/view/split/("raw_"+split+"_sequences_and_vocab.pkl")
            and not (local(PREP_OUTPUT)/view/split/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists(),"No raw corpus or tokenized-cache substitution")
    for w,r in ((original.PREP_WAIT_REF,PREP_REF),(original.COUNTS_WAIT_REF,COUNTS_REF)):
        wait=read(w)
        require(wait["returncode"]==0 and wait["completed_ref"]==r
            and (wait.get("owned_worker_group_ESRCH") is True or wait.get("all_owned_processes_and_groups_drained") is True),"Genuine completed preparation/counts and owned drain")
    close=read(original.PREP_CLOSE_REF)
    require(close["value"]["future_writes_and_relaunch_revoked"] is True
        and close["value"]["request"]["completed_ref"]==PREP_REF,"Read-only genuinely closed preparation")
    manifest=read(AUTHOR_MANIFEST_REF)
    require(manifest["author_root"]==AUTHOR_ROOT and manifest["author_commit"]=="f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596","Exact author source release")
    for r in manifest["refs"]:raw(r)
    return prep,pcfg,counts

def validate_input_v1(record):
    global STATE,CAPS
    require(record==ROOT_INPUT_REF,"One exact current phase input")
    q,s,master=active_amendment_v1()
    require(q["schema"]=="native_VAL000_downstream_ROOT_phase_authority_v1" and s["schema"]=="native_VAL000_downstream_ROOT_phase_input_v1"
        and master["schema"]=="native_VAL000_downstream_ROOT_master_input_v1"
        and s["phase"]==q["phase"] and s["phase"] in PHASES
        and s["job_id"]==q["job_id"]==master["job_id"]+"/"+s["phase"]
        and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"],"Exact fresh separate downstream phase identity")
    manifest=read(ref(MANIFEST))
    for r in manifest["executed_source_refs"]+manifest["source_authentication_refs"]:raw(r)
    require(master["capacity_guard_ref"]==GUARD and master["capacity_policy_ref"]==POLICY
        and master["original_MLM7GiB_scope_extended"] is False and master["future_storage_reserved_bytes"]==64*2**30
        and master["saved_MLM_route"]is True and master["saved_MLM_scientific_integration_ref"]==SAVED_INTEGRATION_REF
        and master["saved_MLM_scientific_binding_bridge_ref"]==ref(SCIENCE_BINDING)
        and master["operational_IO_source_ref"]==IO_SOURCE_REF and master["operational_IO_policy_ref"]==IO_POLICY_REF
        and master["owned_roster_policy_ref"]==OWNED_ROSTER_POLICY_REF,
        "Fresh actual64GiB reservation with reviewed I/O and finite4096 policy, not original7GiB")
    raw(IO_REVIEW_REF);require(read(IO_REVIEW_REF)["verdict"]=="PASS_SOURCE_OPTION_ONLY_EXECUTION_NOT_AUTHORIZED","Exact source-option review, not launch authority")
    config=read(master["config_ref"]);storage=read(master["storage_ref"])
    require(config==EXPECTED_CONFIG,"Exact frozen downstream scientific/cap configuration, no hidden fallback")
    require(storage["job_id"]==master["job_id"] and storage["output_directory"]==master["output_directory"]
        and storage["reserved_new_bytes"]==64*2**30 and storage["reservation"]["reserved_bytes"]==64*2**30
        and storage["accounting_mode"]==CENSUS_MODE and storage["storage_policy_selector_ref"]==SELECTOR_REF and storage["completed_overlay_policy_ref"]==POLICY
        and storage["budget_components"]["consumer_adapter"]==GUARD
        and storage["reservation"]["auxiliary_scopes"]==[str(local(master["worker_auxiliary_scope"]))],"Actual fresh current Census-fit2 child64GiB charge")
    bindings=storage["reservation"]["consumer_bindings"]
    require(len(bindings)<=64 and all(r in bindings for r in (ref(MANIFEST),ref(DRIVER),master["config_ref"],master["runtime_ref"])),
        "Original64 direct binding ceiling, master source/config/runtime actually admitted")
    require(master["runtime_ref"]==original.RUNTIME_REF,"Exact existing native inventory is read-only runtime evidence")
    require(s["refs"]["config"]==master["config_ref"] and s["refs"]["runtime"]==master["runtime_ref"]
        and s["refs"]["storage"]==master["storage_ref"] and q["runtime_ref"]==master["runtime_ref"]
        and q["config_ref"]==master["config_ref"] and q["capacity_policy_ref"]==POLICY and q["capacity_guard_ref"]==GUARD,
        "Actual phase uses admitted master storage/source/config/runtime")
    require(datetime.fromisoformat(master["created_at_utc"])<=datetime.fromisoformat(q["created_at_utc"])
        and datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(master["expires_at_utc"]),
        "Every fresh phase remains within actual master authority, no expiry extension")
    gpu_cap=q["caps"]["max_cuda_reserved_bytes"]
    require(type(gpu_cap) is int and ((s["phase"]=="finalize" and gpu_cap==0) or (s["phase"] in GPU_PHASES and 0<gpu_cap<=74*2**30))
        and q["caps"]==dict(BASE_CAPS,max_cuda_reserved_bytes=gpu_cap),"Finite separately ROOT-admitted GPU0 allocator/RSS/wall caps")
    CAPS.clear();CAPS.update(q["caps"])
    if s["phase"].startswith("embedding_"):
        require(q.get("view") in ("contrastive","query_paths") and q.get("split") in ("train","val")
            and s["phase"]=="embedding_"+q["view"]+"_"+q["split"],"Exact four fixed embedding phase/view/split mapping")
    phase=s["phase"];science=master["output_directory"]+"/"+phase;aux=master["worker_auxiliary_scope"]+"/"+phase
    require(s["output_directory"]==science and s["worker_auxiliary_scope"]==aux
        and not any(local(science).is_relative_to(local(r)) or local(aux).is_relative_to(local(r)) for r in
            ("experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2","research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux")),
        "Exact fresh phase descendants outside original MLM7GiB roots")
    require(q["automatic_restart_authorized"] is False and q["partial_scientific_success_authorized"] is False
        and q["operational_IO_source_ref"]==IO_SOURCE_REF and q["operational_IO_policy_ref"]==IO_POLICY_REF
        and q["ROOT_context_source_ref"]==ref(CONTEXT) and q["keep1_rotation_of_new_scope_authorized"] is True
        and q["original_author_recipe_claimed_exact"] is False
        and q["prerequisites"]["MLM"]==master["MLM_prerequisites"]
        and q["ROOT_saved_MLM_artifact_acceptance_ref"]==master["ROOT_saved_MLM_artifact_acceptance_ref"]
        and q["saved_MLM_scientific_integration_ref"]==master["saved_MLM_scientific_integration_ref"]
        and master["MLM_prerequisites"]==dict(saved_artifact_acceptance_ref=master["ROOT_saved_MLM_artifact_acceptance_ref"],historical_MLM_outer_exit_code=1),"Exact science/I/O/context authority and frozen real MLM prerequisite")
    frozen_science_metadata_v1()
    STATE=(copy.deepcopy(master),copy.deepcopy(storage),copy.deepcopy(q),copy.deepcopy(s))
    return s,q,config,storage,config

class MasterStorageGuard:
    def __init__(self,real):self.real=real
    def validate_storage_v9(self,s,cfg,storage,*,source_ref,config_ref,runtime_ref):
        active_amendment_v1();master,stored,q,actual=STATE
        require(storage==stored and source_ref==ref(DRIVER) and config_ref==master["config_ref"] and runtime_ref==master["runtime_ref"]
            and s["job_id"]==actual["job_id"],"Original64GiB master binding plus exact ROOT late-prerequisite phase authority")
        value=self.real.validate_storage_v9(master,dict(output_cap_bytes=64*2**30),stored,
            source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)
        active_amendment_v1();return value
    def budget_check_v9(self,storage,**kwargs):
        active_amendment_v1();require(storage==STATE[1],"Exact unchanged current64GiB storage")
        value=self.real.budget_check_v9(storage,**kwargs);active_amendment_v1();return value

def guard_v1(q,*,consumer=True):
 require(consumer is True and STATE is not None,"Read/check consumer only, no admission/close mutations")
 configure_selector_v1(q["storage_policy_selector_ref"])
 require(q["capacity_guard_ref"]==GUARD and q["capacity_policy_ref"]==POLICY,"Same actual selected child for phase and storage")
 return MasterStorageGuard(selected_guard_v1(consumer=True))

def charged_census_v1(spec,*,required_write_bytes=0):
    require(STATE is not None,"Actual separately charged master roots")
    master=STATE[0];output=original._sizes(local(master["output_directory"]),allow_checkpoint_alias=True)
    auxiliary=original._sizes(local(master["worker_auxiliary_scope"]))
    group_caps={"contrastive":8*2**30,"embedding":56*2**30,"RAM_retrieve_and_seal":2**30}
    totals=dict.fromkeys(group_caps,0)
    for parts,size,kind in output+auxiliary:
        if not parts:continue
        first=parts[0];group="embedding" if first.startswith("embedding_") else ("RAM_retrieve_and_seal" if first in ("finalize","completed.json") else first)
        require(group in group_caps,"Only exact independently admitted downstream phase scopes")
        totals[group]+=size
    current="embedding" if spec["phase"].startswith("embedding_") else ("RAM_retrieve_and_seal" if spec["phase"]=="finalize" else spec["phase"])
    require(all(v+(required_write_bytes if k==current else 0)<=group_caps[k] for k,v in totals.items())
        and sum(totals.values())+required_write_bytes<=64*2**30,"Whole output+TMP/AUX/cache/log64GiB and per-group failstop")
    checkpoints=[(parts,n,kind) for parts,n,kind in output if parts and parts[0]=="contrastive" and "checkpoints" in parts]
    regular=[n-4096 for parts,n,kind in checkpoints if kind=="regular" and parts[-1].endswith(".pt")]
    require(len(regular)<=2 and max(regular,default=0)<=3*2**30,"Actual peak2regular full contrastive checkpoints,3GiB/file")
    return {master["output_directory"]:sum(n for _,n,_ in output),master["worker_auxiliary_scope"]:sum(n for _,n,_ in auxiliary)}

def prepare_master_parents_v1(spec,g,storage):
    for p in (local(STATE[0]["output_directory"]),local(STATE[0]["worker_auxiliary_scope"])):
        if not p.exists():
            g.budget_check_v9(storage,boundary=True,required_write_bytes=8192);active_amendment_v1();p.mkdir()
        require(p.is_dir() and not p.is_symlink(),"Canonical actual master scope")

def compact_v1(spec,name,value,g,storage,*,cap=2**20):
    active_amendment_v1();directory=local(spec["worker_auxiliary_scope"])
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=cap,"Bounded exact operational receipt")
    charged_census_v1(spec,required_write_bytes=len(data)+8192);g.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
    active_amendment_v1();return save(directory,name,value)

def worker_env_paths_v1(spec):return original.worker_env_paths_v1(spec)
def require_work_window_v2(spec):
    require(time.monotonic()<work_deadline_monotonic_v1(),"No new work inside phase cleanup/evidence reserve")


def validate_child_completion_v1(value,s,q):
    require(value.get("parent_id")==PARENT and value.get("query_ancestry_truth_opened") is False
        and value.get("actual_execution_source_manifest_ref")==ref(MANIFEST)
        and value.get("actual_ROOT_phase_input_ref")==ROOT_INPUT_REF,"Exact actual scientific child/input/source provenance")
    if s["phase"]=="contrastive":
        require(value.get("status")=="completed_C3_native_child" and value.get("stage")=="contrastive"
            and value.get("completed_epochs")==1 and value.get("actual_optimizer_updates")==2135
            and value["checkpoint_optimizer_counters"]["unique_step_values"]==[2135]
            and value.get("all_author_checkpoint_save_events_preserved") is True
            and value.get("actual_checkpoint_retention")==1
            and value.get("operational_IO_policy_ref")==IO_POLICY_REF,"Actual full2135contrastive optimizer, unchanged events, disclosed retention")
    elif s["phase"].startswith("embedding_"):
        require(value.get("status")=="completed_C3_native_child" and value.get("stage")=="embedding"
            and value.get("view")==q["view"] and value.get("split")==q["split"]
            and value.get("pooling_strategy")=="head_norm" and value.get("max_samples")==0
            and value.get("raw_dataset_ref")==read(PREP_REF)["native_dataset_refs"][q["view"]+"/"+q["split"]],"Actual complete single-split original native head_norm output")
    elif s["phase"]=="RAM_retrieve_and_seal":
        require(value.get("status")=="completed_native_RAM_retrieval_and_truth_free_seal"
            and value.get("full_reference_rows")==9721236 and value.get("full_query_rows")==10000
            and value.get("native_top1_arithmetic_or_ties_modified") is False
            and value.get("combined_payloads_written_to_disk") is False,
            "Actual full RAM-native-top1 and truth-free scientific seal, not ancestry scoring")

    else:
        require(value.get("status")=="completed_native_VAL000_pipeline_finalizer_child"
            and value.get("all_six_scientific_stages_actual_wait0_and_consumed_outer0") is True
            and value.get("master_completion_ref") is not None,
            "Actual CPU aggregation of genuine complete scientific pipeline, not a fabricated worker wait")

CENSUS_MODE='Census_observation_two_scope_retention_v1'
CENSUS_MANIFEST_REF={'path': 'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json', 'sha256': 'e0d5e04703410f9e2f89336a9059fe73dc2fe948240ea9425bc4e928a5793405'}
CENSUS_MEASUREMENT_REF={'path': 'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/single_root_retained_nodes_v1.py', 'sha256': 'd23f6f24517090367f5fa8437ce86c6139bcff1b03ede44ffd08df14e0bab926'}
SAVED_INTEGRATION_REF={'path': 'research/reviews/native_VAL000_contrastive_saved_MLM_integration_source_20261017_v2/saved_mlm_integration_v1.py', 'sha256': '651e18d11d99083a60face804a3d50b9f1d48bcc83d3b6e0c2dbf827ae12b324'}
SAVED_SCIENTIFIC_ENTRY_REF={'path': 'research/reviews/native_VAL000_contrastive_saved_MLM_integration_source_20261017_v2/scientific_entry_v1.py', 'sha256': 'e1986b32edd4732b56599ea71bc8a5cddec397d0bf4df622f09af7b5a94999a1'}
SAVED_SCIENCE_REVIEW_REF={'path': 'research/reviews/native_VAL000_contrastive_saved_MLM_integration_independent_20261017_v2/review.v2.json', 'sha256': '250b54d82a72602094d1e855e0fac37034029a87fcd29fe814cb37c022c52a63'}

SCIENCE_BINDING=HERE+"/science_binding_v1.py"
BUILDER=HERE+"/root_builder_v1.py"
def configure_selector_v1(selector_ref):
 global POLICY,SELECTOR_REF,_SELECTED_GUARD,_SELECTOR_VALUE
 require(type(selector_ref)is dict and set(selector_ref)=={"path","sha256"},"Exact actual future Census selector path/SHA supplied by ROOT")
 value=read(selector_ref)
 require(value.get("active")is True and value.get("root_authorizer")=="root_agent"
  and value.get("source_ref")==value.get("selector_source_ref")==GUARD
  and value.get("source_manifest_ref")==CENSUS_MANIFEST_REF
  and value.get("measurement_source_ref")==CENSUS_MEASUREMENT_REF and value.get("accounting_mode")==CENSUS_MODE,
  "Exact reviewed planned Census source/manifest/helper, never unrelated old route")
 for r in(selector_ref,GUARD,CENSUS_MANIFEST_REF,CENSUS_MEASUREMENT_REF,value["policy_ref"],value["source_review_ref"]):raw(r)
 if SELECTOR_REF is not None:
  require(SELECTOR_REF==selector_ref and _SELECTOR_VALUE==value and POLICY==value["policy_ref"],
   "Immutable actual selector/policy/source for this interpreter")
  require(_SELECTED_GUARD is not None and _SELECTED_GUARD._SELECTOR_REF==SELECTOR_REF,"Same configured guard singleton")
  return _SELECTED_GUARD
 policy=read(value["policy_ref"])
 require(policy.get("active")is True and policy.get("source_ref")==GUARD and policy.get("journal")==value["journal"]
  and policy.get("source_manifest_ref")==CENSUS_MANIFEST_REF and policy.get("measurement_source_ref")==CENSUS_MEASUREMENT_REF,
  "Actual independently activated exact child policy")
 g=load(GUARD,"_saved_MLM_actual_Census64_guard")
 selected=g.load_selected_guard_v1(selector_ref)
 require(selected is g,"Actual combined child selector returns its own same guard singleton")
 SELECTOR_REF=copy.deepcopy(selector_ref);POLICY=copy.deepcopy(value["policy_ref"]);_SELECTOR_VALUE=copy.deepcopy(value);_SELECTED_GUARD=g
 return g
def selected_guard_v1(*,consumer):
 global _CONSUMER_MEMO
 require(type(consumer)is bool and _SELECTED_GUARD is not None,"Explicit existing selected guard role")
 g=_SELECTED_GUARD
 if not consumer:
  require(_CONSUMER_MEMO is None and all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
   "Fresh original unwrapped ROOT mutation interpreter; no consumer memo")
  return g
 if _CONSUMER_MEMO is None:
  namespace=load(original.NAMESPACE_MEMO,"_saved_MLM_exact_consumer_namespace_memo")
  _CONSUMER_MEMO=namespace.install_consumer_v1(g.budget.v4,g.budget,consumer_role="read_only_consumer",
   expected_source_path=ROOT/original.V4["path"],expected_source_sha256=original.V4["sha256"])
 return g
