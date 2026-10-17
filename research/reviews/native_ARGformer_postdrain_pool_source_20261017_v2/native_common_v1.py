"""Native32 fixed-corpus ROOT metadata and original H020v3 guard binding. Inert import."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib,importlib.util,io,json,os,stat,sys,time,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_ARGformer_postdrain_pool_source_20261017_v2"
SOURCE=HERE+"/native_common_v1.py"
DRIVER=HERE+"/native_root_driver_v1.py"
CONTEXT=HERE+"/native_child_v1.py"
DIRECT=HERE+"/native_direct_gpu0_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
from . import common_v1 as pool_h
from . import pool_contract_v1 as pool_c
require=pool_h.require
CAPS=dict(reservation_bytes=7*2**30,scientific_output_bytes=6*2**30+128*2**20+65536,auxiliary_bytes=768*2**20,directory_and_accounting_allowance_bytes=128*2**20-65536,log_bytes=4*2**20,max_seconds=172800,max_cpu_rss_bytes=384*2**30,max_cuda_reserved_bytes=32*2**30,headroom_bytes=4*2**30,cleanup_seconds=30,work_cutoff_margin_seconds=300,terminal_evidence_margin_seconds=150,max_checkpoint_bytes=3*2**30,max_checkpoint_files=2)
GUARD=pool_h.GUARD;POLICY=pool_h.POLICY;SELECTOR_REF=pool_h.SELECTOR_REF
GPU0_UUID=pool_h.GPU0_UUID;BASE_POOL=pool_h.BASE_POOL;GPU0_GUARD=pool_h.GPU0_GUARD;PIDFD=pool_h.PIDFD;OWNED=pool_h.OWNED;CERT=pool_h.CERT
OWNED_ROSTER_CAP=4096
MAX_NODES=16384;MAX_DEPTH=24
BASE_CONFIG_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/config.json",sha256="7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20")
NATIVE_CPU_QUAL_REF=dict(path="research/reviews/C3_bert_base_configuration_root_v1/native_CPU_qualification.json",sha256="56ed247a09484753a64e9ed3bee6c80eea4e5bfb156ab5acdb09c24ce828e515")
NATIVE_INVENTORY_REF=dict(path="experiments/results/native_argformer_tsinfer_fit_v7_root/environment_inventory.json",sha256="077d3e950c3bb3efdecc7292e06c8157506663293b4efe4723ccf7b877973461")
LEGACY_RUNTIME_REF=dict(path="src/arggnn/native_argformer_tsinfer_fit_v7.py",sha256="b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712")
AUTHOR_ROOT=".runtime/author_sources/argformer_f7545be7bca0"
AUTHOR_MANIFEST_REF={"path":"research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json","sha256":"bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a"}
NATIVE_PYTHON_REF={"path":".runtime/conda/envs/argformer_native_v1/bin/python3.11","sha256":"d87f38a37e44f4e650b03ce940846c435f6c76c1fc0e6614b09865e1b7b44b0b"}
RUNTIME_REF={"path":"research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/runtime_mlm.root_active_v1.json","sha256":"58aa23a7fcadb1b1cddc169cf7f0e1c63a17877d85bdef6405c34a48a5064e61"}

def local(value):
    p=Path(value)
    p=p if p.is_absolute() else ROOT/p
    require(p!=ROOT and '..' not in p.parts and p.resolve()==p and p.is_relative_to(ROOT),'Canonical contained project path')
    return p

def ref(path):
    p=local(path)
    path=str(p.relative_to(ROOT))
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=32*2**20,'Bounded source/metadata')
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r,cap=32*2**20):
    require(isinstance(r,dict) and set(r)=={'path','sha256'} and ref(r['path'])==r,'Exact actual source/JSON ref')
    require(local(r["path"]).stat().st_size<=cap,"Bounded actual JSON")
    return json.loads(local(r["path"]).read_bytes())

def load(r,name):
    require(ref(r['path'])==r,'Exact source before private import')
    sp=importlib.util.spec_from_file_location(name,local(r['path']))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def active(q):
    now=datetime.now(timezone.utc);a=datetime.fromisoformat(q['created_at_utc']);b=datetime.fromisoformat(q['expires_at_utc'])
    require(a.tzinfo and b.tzinfo and a<=now<b,'Current actual ROOT authority')

def unique(rows):
    d={}
    for r in rows:
        require(ref(r['path'])==r and (r['path'] not in d or d[r['path']]==r),'Complete exact unique bindings')
        d[r['path']]=r
    require(len(d)<=64,'Original64 direct-consumer ceiling')
    return list(d.values())

def raw(record):
    require(ref(record["path"])==record,"Exact source/metadata bytes")
    return local(record["path"]).read_bytes()

def save(directory,name,value):
    require(Path(name).name==name,"Flat admission metadata name")
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=8*2**20,"Compact root admission metadata")
    target=directory/name
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    return ref(str(target.relative_to(ROOT)))

def worker_env_paths_v1(spec):
    base=str(local(spec["worker_auxiliary_scope"]))
    return dict(TMPDIR=base+"/tmp",TMP=base+"/tmp",TEMP=base+"/tmp",
        XDG_CACHE_HOME=base+"/cache",XDG_CONFIG_HOME=base+"/config",
        XDG_DATA_HOME=base+"/data",XDG_STATE_HOME=base+"/state",TORCH_HOME=base+"/cache/torch",
        TORCH_EXTENSIONS_DIR=base+"/cache/torch_extensions",PYTORCH_KERNEL_CACHE_PATH=base+"/cache/torch_kernels",
        TRITON_CACHE_DIR=base+"/cache/triton",NUMBA_CACHE_DIR=base+"/cache/numba",
        MPLCONFIGDIR=base+"/cache/matplotlib",PYTHONPYCACHEPREFIX=base+"/cache/pycache",
        CUDA_CACHE_PATH=base+"/cache/cuda",PIP_CACHE_DIR=base+"/cache/pip",
        HF_HOME=base+"/cache/huggingface",HF_HUB_CACHE=base+"/cache/huggingface/hub")

def _sizes(root,*,allow_checkpoint_alias=False):
    """Descriptor-relative no-follow metadata traversal; no file is opened."""
    root=Path(root)
    require(root.is_relative_to(ROOT) and root.resolve()==root and ".." not in root.parts,"Own canonical project scope")
    records=[];count=0
    try:fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    except FileNotFoundError:return records
    def visit(directory,parts):
        nonlocal count
        require(len(parts)<=MAX_DEPTH,"Bounded owned scope depth")
        with os.scandir(directory) as entries:
            names=[e.name for e in entries]
        require(len(names)<=MAX_NODES,"Bounded owned directory entries")
        for name in names:
            count+=1;require(count<=MAX_NODES,"Bounded own metadata census")
            try:s=os.stat(name,dir_fd=directory,follow_symlinks=False)
            except FileNotFoundError:continue  # Normal Composer retention race.
            rel=parts+(name,)
            if stat.S_ISDIR(s.st_mode):
                records.append((rel,4096,"directory"))
                try:child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
                except FileNotFoundError:continue
                try:visit(child,rel)
                finally:os.close(child)
            elif stat.S_ISREG(s.st_mode):
                require(s.st_nlink==1,"Owned scope hardlink forbidden")
                records.append((rel,max(s.st_size,s.st_blocks*512)+4096,"regular"))
            elif stat.S_ISLNK(s.st_mode):
                require(allow_checkpoint_alias and "checkpoints" in parts and name=="latest-rank0.pt"
                    and s.st_size<=256,"Only original own Composer alias is allowed; never followed")
                records.append((rel,s.st_size+4096,"alias"))
            else:raise ValueError("Unexpected nonregular owned scope entry")
    try:visit(fd,())
    finally:os.close(fd)
    return records

def charged_census_v1(spec,*,required_write_bytes=0):
    output=_sizes(local(spec["output_directory"]),allow_checkpoint_alias=True)
    packet_directory=local(spec["worker_auxiliary_scope"]).parent
    packet=_sizes(packet_directory);auxiliary=_sizes(local(spec["worker_auxiliary_scope"]))
    checkpoints=[n-4096 for parts,n,kind in output if "checkpoints"in parts and kind=="regular"and parts[-1].endswith(".pt")]
    require(len(checkpoints)<=CAPS["max_checkpoint_files"] and max(checkpoints,default=0)<=CAPS["max_checkpoint_bytes"],"Original transient full-checkpoint cap")
    outbytes=sum(n for _,n,_ in output);packetbytes=sum(n for _,n,_ in packet);auxbytes=sum(n for _,n,_ in auxiliary)
    require(outbytes+required_write_bytes<=CAPS["scientific_output_bytes"] and auxbytes+required_write_bytes<=CAPS["auxiliary_bytes"]
        and packetbytes-auxbytes+required_write_bytes<=CAPS["directory_and_accounting_allowance_bytes"]
        and outbytes+packetbytes+required_write_bytes<=CAPS["reservation_bytes"],"Whole7GiB including complete packet/AUX, original source caps failstop")
    return {spec["output_directory"]:outbytes,str(packet_directory.relative_to(ROOT)):packetbytes}

def compact_v1(spec,name,value,g,storage,*,cap=2**20):
    directory=local(spec["worker_auxiliary_scope"])
    raw_bytes=(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(raw_bytes)<=cap,"Fixed compact operational receipt cap")
    charged_census_v1(spec,required_write_bytes=len(raw_bytes)+8192)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw_bytes)+8192)
    active(spec)  # A contended original guard may have consumed the time margin.
    return save(directory,name,value)

def require_work_window_v2(spec):
    require(time.time()+CAPS["work_cutoff_margin_seconds"]<
        datetime.fromisoformat(spec["expires_at_utc"]).timestamp(),
        "Scientific work stops before reserved cleanup and terminal-evidence margin")

PARENT_REGISTRY=HERE+"/parent_registry.v1.json"
AUTHOR_LOADER_REF={"path":".runtime/author_sources/argformer_f7545be7bca0/src/arg_data.py","sha256":"2e7df2c955ece89ced3f05c609a4450f997250df5a7cb06ff4a2ca957e49ffdf"}
AUTHOR_MLM_YAML_REF={"path":".runtime/author_sources/argformer_f7545be7bca0/yamls/mlm.yaml","sha256":"df1cbb747a1bb6740fffd1e6eee5eb0ac3a1fa43d8fbbb15ce70d6b9cc7799b6"}

def guard_v1(q,*,consumer=True):
    require(q["capacity_guard_ref"]==GUARD and q["capacity_policy_ref"]==POLICY
        and q["storage_policy_selector_ref"]==SELECTOR_REF,"Exact current explicit Relate guard/selector, no fallback")
    return pool_h.guard_v1(consumer=consumer)

def source_refs_v1():return pool_h.source_refs_v1()

def cardinality_v1(n,drop_last):
    require(type(n) is int and n>0 and type(drop_last) is bool,"Original actual dataset cardinality and drop_last")
    batches=n//512 if drop_last else (n+511)//512
    require(batches>0,"Nonzero complete author loader")
    return dict(dataset_cardinality=n,drop_last=drop_last,world_size=1,batch_size=512,
        batches_per_epoch=batches,epochs=5,expected_optimizer_updates=5*batches)

def science_v1(q):
    registry=read(ref(PARENT_REGISTRY));parent=q["parent_id"]
    require(parent in registry["remaining_fresh_fit_parents"],"Only unfinished original VAL001..004, never replay current VAL000")
    row=registry["parents"][parent];prep=read(q["preparation_ref"]);counts=read(q["counts_ref"]);pcfg=read(q["preparation_config_ref"])
    require(q["preparation_ref"]==row["preparation_ref"] and q["preparation_directory"]==row["preparation_directory"]
        and prep["parent_id"]==counts["parent_id"]==parent and prep["status"]=="completed_native_C3_path_preparation"
        and prep["observation_ref"]==pcfg["observation_ref"]==row["observation_ref"]
        and prep["class_order"]==counts["class_order"]==["EUR","EAS","AFR"]
        and prep["native_split_seed"]==42 and prep["native_train_fraction"]==.9
        and pcfg["path_sampling_seed"]==row["original_seed_row"]["path_sampling_seed"]
        and prep["query_focal_count"]==counts["query_focal_count"]==10000
        and prep["query_ancestry_truth_opened"] is False and counts["query_ancestry_truth_opened"] is False
        and prep["contrastive_labels_reference_only"] is True,"Exact original full30Mb parent preparation, original seeds and private-truth barrier")
    require(counts["status"]=="completed_C3_native_observation_only_counts"
        and counts["preparation_ref"]==q["preparation_ref"] and counts["preparation_config_ref"]==q["preparation_config_ref"]
        and counts["source_corpus_capped_or_sampled"] is False and counts["drop_last_release_batches_preserved"] is True
        and counts["native_training_eligible"] is True,"Genuine complete original per-parent counts, no subsampling")
    keys={v+"/"+s for v in ("mlm","contrastive","query_paths")for s in ("train","val")}
    require(set(counts["datasets"])==set(prep["native_dataset_refs"])==keys,"Complete all-six original native view cardinality provenance")
    for key,item in counts["datasets"].items():
        require(item["dataset_ref"]==prep["native_dataset_refs"][key]
            and item["row_order_ref"]==prep["full_path_id_order_refs"][key]
            and type(item["rows"]) is int and item["rows"]>0
            and type(item["raw_pickle_bytes"]) is int and item["raw_pickle_bytes"]>0,"Exact counted original dataset/order ref")
        view,split=key.split("/");expected=local(q["preparation_directory"])/view/split/("raw_"+split+"_sequences_and_vocab.pkl")
        require(local(item["dataset_ref"]["path"])==expected,"Original same-parent complete corpus path")
        require(not (expected.parent/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists(),"No shadow tokenized corpus")
    require(counts["vocabulary_nodes"]==prep["vocabulary_nodes"]==row["original_vocab_nodes"]
        and counts["tokenizer_vocab_size"]==prep["vocabulary_nodes"]+4,"Original actual parent vocabulary plus author special tokens")
    for wait_ref,completed_ref in ((q["preparation_wait_ref"],q["preparation_ref"]),(q["counts_wait_ref"],q["counts_ref"])):
        wait=read(wait_ref)
        require(wait["returncode"]==0 and wait["completed_ref"]==completed_ref
            and (wait.get("owned_worker_group_ESRCH") is True or wait.get("all_owned_processes_and_groups_drained") is True),"Genuine original preparation/count direct0 and owned drain")
    frozen=read(q["preparation_readonly_ref"])
    require(frozen["root_authorizer"]=="root_agent" and frozen["preparation_ref"]==q["preparation_ref"]
        and frozen["original_preparation_future_writes_and_relaunch_revoked"] is True
        and frozen["files_and_original_charges_preserved"] is True,"Original complete corpus frozen read-only, no storage credit implied")
    cardinality=cardinality_v1(counts["datasets"]["mlm/train"]["rows"],True)
    require(counts["MLM_optimizer_updates"]==cardinality["expected_optimizer_updates"],"Count receipt independently agrees with exact original loader derivation")
    author=read(AUTHOR_MANIFEST_REF)
    require(author["author_root"]==AUTHOR_ROOT and author["author_commit"]=="f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596"
        and AUTHOR_LOADER_REF in author["refs"] and AUTHOR_MLM_YAML_REF in author["refs"],"Pinned full original author algorithm/loader/config")
    for r in author["refs"]:raw(r)
    runtime=read(q["runtime_ref"])
    require(runtime==read(RUNTIME_REF) and runtime["native_python_ref"]==NATIVE_PYTHON_REF
        and runtime["native_inventory_ref"]==NATIVE_INVENTORY_REF and runtime["target_physical_gpu"]==0
        and runtime["target_GPU_UUID"]==GPU0_UUID,"Exact existing native runtime reused read-only, no old execution authority")
    protocol=dict(parent_id=parent,split="val",phase="mlm",author_commit=author["author_commit"],
        epochs=5,optimizer_updates=cardinality["expected_optimizer_updates"],batch=512,microbatch=512,
        precision="amp_bf16",seed=17,vocabulary=counts["tokenizer_vocab_size"],max_seq_len=128,layers=12,
        heads=12,hidden_size=768,intermediate_size=3072,num_workers=8,drop_last=True,world_size=1,
        save_interval="500ba",save_num_checkpoints_to_keep=1,full30Mb_corpus=True,
        author_model_objective_heads_unchanged=True,fresh_optimizer=True,initializer_checkpoint_ref=None,
        query_ancestry_truth_opened=False,cardinality=cardinality)
    cfg=read(q["config_ref"])
    require(cfg==dict(schema="native_original_parent_postdrain_science_v1",scientific_protocol=protocol,
        max_checkpoint_files=CAPS["max_checkpoint_files"],max_checkpoint_bytes=CAPS["max_checkpoint_bytes"]),"Exact derived whole-parent author science; no resume/transfer or budget fallback")
    return cfg,prep,counts,runtime

def validate_request_v1(record):
    q=read(record);active(q);source_refs_v1()
    require(q.get("schema")=="native_original_parent_postdrain_ROOT_request_v1" and q.get("status")=="active"
        and q.get("root_authorizer")=="root_agent" and q.get("execution_authorized") is True
        and q.get("template_only") is False and q.get("phase")=="mlm"
        and q.get("caps")==CAPS and q.get("adapter_manifest_ref")==ref(MANIFEST)
        and q.get("resume_checkpoint_ref") is None and q.get("initializer_checkpoint_ref") is None
        and q.get("automatic_restart_authorized") is False and q.get("partial_scientific_success_authorized") is False
        and isinstance(q.get("IPC_namespace"),str) and 1<=len(q["IPC_namespace"])<=14
        and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789_"for ch in q["IPC_namespace"]),
        "Fresh finite explicit original-parent ROOT admission, never old job/clock reset")
    pool_c.reviewed_v1(q["source_review_ref"])
    cap=pool_c.validate_cap_v1(q["postdrain_pool_capability_ref"])
    job=cap["jobs"].get(q["job_id"],{})
    require(job.get("worker_kind")=="native" and job.get("parent_id")==q["parent_id"]
        and job.get("execution_enabled") is True and job.get("native_phase","mlm")=="mlm","One declared native-parent logical slot in new postdrain owner")
    require(q["packet_directory"].startswith("research/reviews/native_postdrain_ROOT_actual_")
        and q["output_directory"].startswith("experiments/results/native_postdrain_full30Mb_")
        and q["preparation_directory"] not in (q["packet_directory"],q["output_directory"]),"Fresh separate fully charged scope, no corpus reuse as output")
    end=datetime.fromisoformat(q["expires_at_utc"]).timestamp();renewal=read(CERT)
    require(time.time()+CAPS["max_seconds"]+600<end<=cap["expires_unix"]
        and end<datetime.fromisoformat(renewal["renewed_expires_at_utc"]).timestamp(),"Full48h bounded attempt plus drain/evidence inside fresh pool/renewal expiry")
    repeat=read(q["unfinished_cell_registration_ref"])
    require(repeat["root_authorizer"]=="root_agent" and repeat["hypothesis_id"]=="H041"
        and repeat["parent_id"]==q["parent_id"] and repeat["phase"]=="mlm"
        and repeat["condition"]=="original_unfinished_parent_native_fit"
        and repeat["all_old_scopes_charges_clocks_failures_unknowns_preserved"] is True
        and repeat["same_parent_completed_full_native_fit_exists"] is False,"Original registered unfinished cell, not an answered repeat")
    guard_v1(q)
    cfg,prep,counts,runtime=science_v1(q)
    return q,cfg,None,cfg["scientific_protocol"],read(ref(MANIFEST))

def validate_input_v1(record):
    s=read(record);active(s);q,cfg,_,protocol,manifest=validate_request_v1(s["root_stage_request_ref"])
    require(record["path"]==q["packet_directory"]+"/input.json" and s["schema"]=="native_original_parent_postdrain_input_v1"
        and s["status"]=="active" and s["root_authorizer"]=="root_agent" and s["execution_authorized"] is True
        and s["job_id"]==q["job_id"] and s["parent_id"]==q["parent_id"] and s["phase"]==q["phase"]
        and s["output_directory"]==q["output_directory"] and s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux"
        and s["refs"]["config"]==q["config_ref"] and s["refs"]["runtime"]==q["runtime_ref"]
        and s["created_at_utc"]==q["created_at_utc"] and s["expires_at_utc"]==q["expires_at_utc"],"Exact whole-parent admitted input")
    storage=read(s["refs"]["storage"]);closure=read(s["consumer_manifest_ref"])
    require(storage["job_id"]==s["job_id"] and storage["output_directory"]==s["output_directory"]
        and storage["reserved_new_bytes"]==CAPS["reservation_bytes"]
        and storage["reservation"]["auxiliary_scopes"]==[str(local(q["packet_directory"]))]
        and storage["completed_overlay_policy_ref"]==POLICY and storage["budget_components"]["consumer_adapter"]==GUARD
        and closure["root_request_ref"]==s["root_stage_request_ref"] and closure["adapter_manifest_ref"]==ref(MANIFEST)
        and closure["parent_id"]==q["parent_id"] and closure["preparation_ref"]==q["preparation_ref"]
        and closure["counts_ref"]==q["counts_ref"] and closure["scientific_protocol"]==protocol
        and s["source_sha256"]=={r["path"]:r["sha256"] for r in source_refs_v1()},"Complete current guard/source/parent/count provenance, fresh7GiB scope")
    needed=[ref(MANIFEST),ref(DRIVER),ref(DIRECT),s["root_stage_request_ref"],q["config_ref"],q["runtime_ref"],s["consumer_manifest_ref"],q["source_review_ref"],q["direct_attachment_authority_ref"],SELECTOR_REF]
    require(len(storage["reservation"]["consumer_bindings"])<=64 and all(r in storage["reservation"]["consumer_bindings"] for r in needed),"Original direct-role64 ceiling, unchanged")
    return s,q,cfg,storage,protocol
