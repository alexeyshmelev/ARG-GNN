"""Additive H050-only metadata closure fix. Default inert; root executes."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,ast,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H050_shared_admission_source_metadata_closure_fix_20261016_v1"
SOURCE=HERE+"/root_h050_shared_admission_fix_v1.py"
CONTEXT_REF={'path': 'research/reviews/H050_shared_admission_source_metadata_closure_fix_20261016_v1/context_metadata_fix_v1.py', 'sha256': '6c704a0b47d5b6f136479879ad9c11036f6d80760a93933ae3d0f222db3ce67d'}
FROZEN_ROUTE_DRIVER=dict(path="research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2/root_shared_admission_v2.py",
 sha256="b22a18f7e4126b6435fdb335bfa2b972cb4e6e29d611ed4548196dc35eb32662")
FROZEN_H050_DRIVER=dict(path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/root_driver_v1.py",
 sha256="f8399361305ecb3b27172695d6e6e5c1827da330d4fa3d4565bda2df1ed6d6ac")
MODULE="arggnn.H050_direct_readout_cut_memory_exec_v1"
JOB="H050-direct-readout-shared-TRAIN-v2"


def require(ok,message):
    if not ok:raise ValueError(message)


def ref(name):
    p=ROOT/name;require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),"Canonical source/metadata ref")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(record):
    require(ref(record["path"])==record,"Exact immutable ref");return json.loads((ROOT/record["path"]).read_text())


def save(folder,name,value):
    p=folder/name;require(not p.exists(),"Fresh no-overwrite metadata")
    with p.open("x") as h:json.dump(value,h,indent=2,sort_keys=True);h.write("\n")
    return ref(str(p.relative_to(ROOT)))


def load(record,label):
    require(ref(record["path"])==record,"Exact frozen/private source")
    sp=importlib.util.spec_from_file_location(label,ROOT/record["path"]);m=importlib.util.module_from_spec(sp)
    sys.modules[label]=m;sp.loader.exec_module(m);return m


def adapter():return load(CONTEXT_REF,"_H050_source_metadata_adapter_fix_v1")


def original_route_driver(label="_H050_frozen_route_driver_for_fix_v1"):
    return load(FROZEN_ROUTE_DRIVER,label)


def bind(record):
    base=original_route_driver();request,r,p=base.prepare_admission(record,MODULE)
    original_request_ref=request["worker_request_ref"];original_request=read(original_request_ref)
    original_cfg_ref=original_request["config_ref"];cfg=read(original_cfg_ref)
    require(original_request["phase"]=="train" and original_request["physical_gpu"]==4,"Original pending H050 TRAIN request")
    a=adapter();require(cfg["warmstart_checkpoint_ref"]==a.WARM_REF,"Exact original warm-start ref")
    folder=ROOT/HERE/"metadata_v1";require(not folder.exists(),"Fresh fix metadata; failed packet preserved");folder.mkdir()
    proxy_ref=save(folder,"scientific_input_closure.json",dict(schema="H050_scientific_input_transitive_closure_v1",
        hypothesis_id="H050",scientific_input_refs=[a.WARM_REF],metadata_adapter_ref=CONTEXT_REF,
        admission_successor_ref=ref(SOURCE),original_common_ref=a.COMMON_REF,original_context_ref=a.CONTEXT_REF,
        original_config_ref=original_cfg_ref,original_source_manifest_ref=cfg["source_manifest_ref"],
        scientific_payload_hash_checks_unchanged=True,scientific_execution_started=False))
    manifest=read(cfg["source_manifest_ref"]);sources=dict(manifest["source_sha256"])
    for x in [CONTEXT_REF,ref(SOURCE)]:sources[x["path"]]=x["sha256"]
    metadata=dict(manifest.get("metadata_refs",{}));metadata["scientific_input_proxy"]=proxy_ref
    manifest_ref=save(folder,"source_manifest.shared.metadata_fix.json",dict(manifest,source_sha256=sources,
        metadata_refs=metadata,original_shared_manifest_ref=cfg["source_manifest_ref"]))
    cfg=copy.deepcopy(cfg);cfg.update(root_context_module_ref=CONTEXT_REF,source_manifest_ref=manifest_ref,
        scientific_input_closure_ref=proxy_ref,metadata_binding_admission_successor_ref=ref(SOURCE))
    config_ref=save(folder,"config_train.shared.metadata_fix.json",cfg)
    c=a.private_common_v1();reader=c.MetadataReaderV1();c.validate_config_v1(cfg,reader);c.exact_sources_v1(reader,cfg)
    fixed_request=dict(original_request,config_ref=config_ref)
    request_ref=save(folder,"admission_request.train.shared.metadata_fix.json",fixed_request)
    result=dict(schema="H050_shared_source_metadata_fix_binding_v1",original_worker_admission_request_ref=record,
        worker_request_ref=request_ref,config_ref=config_ref,scientific_input_closure_ref=proxy_ref,
        shared_route_ref=request["shared_route_ref"],new_job_id=JOB,old_failed_packet_preserved=True,admitted=False,launched=False)
    result_ref=save(folder,"binding.json",result);print(json.dumps(dict(binding_ref=result_ref,**result)),flush=True);return result


def admit(record):
    binding=read(record);require(binding.get("schema")=="H050_shared_source_metadata_fix_binding_v1","Exact new bound fix")
    base=original_route_driver();r,p=base.route(binding["shared_route_ref"]);a=adapter();c=a.private_common_v1()
    worker_request_ref=binding["worker_request_ref"];req=read(worker_request_ref);cfg=read(req["config_ref"])
    require(req["phase"]=="train" and req["physical_gpu"]==4 and cfg["root_context_module_ref"]==CONTEXT_REF
        and cfg["scientific_input_closure_ref"]==binding["scientific_input_closure_ref"],"Exact fresh H050 context/proxy request")
    source=(ROOT/FROZEN_H050_DRIVER["path"]).read_text();require(hashlib.sha256(source.encode()).hexdigest()==FROZEN_H050_DRIVER["sha256"],"Frozen H050 admission source")
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="admit")
    text=ast.get_source_segment(source,node)
    changes={
        'packet=ROOT/HERE/("admitted_"+phase)':'packet=ROOT/FIX_HERE/"admitted_H050_shared_train_v2"',
        'output="experiments/results/H050_GPU%d_direct_readout_cut_memory_%s_seed101_v1"%(gpu,phase)':'output="experiments/results/H050_shared_direct_readout_train_seed101_v2"',
        'job="H050-direct-readout-%s-seed101-v1-GPU%d"%(phase,gpu)':'job=FIX_JOB',
        'expiry=(now+timedelta(hours=24)).isoformat()':'expiry=min(now+timedelta(hours=24),datetime.fromtimestamp(FIX_EXPIRES,timezone.utc)).isoformat()',
        'validate_spec_v1(reader,spec,cfg)':'spec.update(FIX_OPERATIONAL)\n    validate_spec_v1(reader,spec,cfg)',
        'direct_required_refs_v1(cfg,spec)+list(budget_bindings.values())':'direct_required_refs_v1(cfg,spec)+FIX_REQUIRED+list(budget_bindings.values())',
        'admitted_ref=save(packet,"admitted.json",admitted)':'admitted.update(admission_successor_ref=FIX_ADMISSION_REF,scientific_input_closure_ref=FIX_PROXY_REF,shared_route_ref=FIX_ROUTE_REF)\n    admitted_ref=save(packet,"admitted.json",admitted)'}
    for old,new in changes.items():require(text.count(old)==1,"Exact single frozen adaptation boundary");text=text.replace(old,new)
    operational=base.operational_spec(binding["shared_route_ref"],MODULE)
    operational.update(operational_H050_admission_successor_ref=ref(SOURCE),scientific_input_closure_ref=binding["scientific_input_closure_ref"])
    namespace={k:getattr(c,k) for k in dir(c) if not k.startswith("__")}
    namespace.update(datetime=datetime,timedelta=timedelta,timezone=timezone,json=json,
        FIX_HERE=HERE,FIX_JOB=JOB,FIX_EXPIRES=r["expires_unix"],FIX_OPERATIONAL=operational,
        FIX_REQUIRED=base.operational_refs()+base.renewal_refs(r["marker_repair_renewal_ref"])+[
            ref("src/arggnn/H050_direct_readout_cut_memory_exec_v1.py"),ref(SOURCE),CONTEXT_REF,binding["scientific_input_closure_ref"],record,binding["shared_route_ref"]],
        FIX_ADMISSION_REF=ref(SOURCE),FIX_PROXY_REF=binding["scientific_input_closure_ref"],FIX_ROUTE_REF=binding["shared_route_ref"])
    exec(compile(text,"<frozen_H050_new_context_metadata_admission>","exec"),namespace)
    return namespace["admit"](worker_request_ref)


def assemble(record):
    # Keep C3's old admission exactly. Only fresh controller bindings gain the
    # actual H050 successor source/context/proxy, never retrofitted into C3.
    req=read(record);h=read(req["H050_admitted_ref"]);h_spec=read(h["input_ref"])
    require(h_spec["job_id"]==JOB and h_spec["root_context_module_ref"]==CONTEXT_REF
        and h_spec.get("operational_H050_admission_successor_ref")==ref(SOURCE),"Exact actually admitted H050 successor")
    extra=[ref(SOURCE),CONTEXT_REF,h_spec["scientific_input_closure_ref"]]
    storage=read(h_spec["refs"]["storage"])
    require(all(r in storage["reservation"]["consumer_bindings"] for r in extra),"Actual H050 successor sources/proxy reserved")
    base=original_route_driver("_H050_fix_assemble_original_v1")
    source=(ROOT/FROZEN_ROUTE_DRIVER["path"]).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="assemble")
    text=ast.get_source_segment(source,node)
    changes={
        'consumers=c.unique(list(bindings.values())+operational_refs()+renewal_refs(':
            'consumers=c.unique(list(bindings.values())+operational_refs()+EXTRA_METADATA_REFS+renewal_refs(',
        'refs={x["path"]:x for x in operational_refs()+renewal_refs(':
            'refs={x["path"]:x for x in operational_refs()+EXTRA_METADATA_REFS+renewal_refs(',
        'controller_job="GPU4-C3cache-H050-shared-controller-v1"':
            'controller_job="GPU4-C3cache-H050-shared-controller-v2"'}
    for old,new in changes.items():
        require(text.count(old)==1,"Exact narrow controller-only metadata adaptation");text=text.replace(old,new)
    namespace=dict(base.__dict__);jobs=dict(base.JOBS);jobs[MODULE]=h_spec["job_id"]
    namespace.update(HERE=HERE,JOBS=jobs,EXTRA_METADATA_REFS=extra)
    exec(compile(text,"<unchanged_C3_admission_fresh_H050_controller_assembly>","exec"),namespace)
    return namespace["assemble"](record)



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("describe","bind","admit","assemble"),nargs="?",default="describe")
    p.add_argument("--execute",action="store_true");p.add_argument("--record");p.add_argument("--sha256");a=p.parse_args(argv)
    if a.action=="describe":
        print(json.dumps(dict(status="source_only_H050_metadata_closure_fix",C3_files_modified=False,guards_or_jobs_called=False)));return 0
    require(a.execute and a.record and a.sha256,"Explicit future root action/ref")
    {"bind":bind,"admit":admit,"assemble":assemble}[a.action](dict(path=a.record,sha256=a.sha256));return 0

if __name__=="__main__":raise SystemExit(main())
