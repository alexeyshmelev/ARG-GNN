"""Additive H050 canonical launch binding; frozen admission/context/science reused."""
import argparse,importlib.util,json
from types import SimpleNamespace
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bridge_common_v1 as c
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bind_root_packet_v1 as binder
HERE=c.HERE+"/canonical_activation_v2"
SOURCE=HERE+"/root_canonical_driver_v2.py"
ALIAS=HERE+"/launch_canonical_exclusive_v2.py"
CANONICAL="src/arggnn/H050_direct_readout_cut_memory_exec_v1.py"
BASE_DRIVER_REF=dict(path=c.DRIVER,sha256="f8399361305ecb3b27172695d6e6e5c1827da330d4fa3d4565bda2df1ed6d6ac")
BASE_BINDER_REF=dict(path=c.BINDER,sha256="4a9d3a39afbac81183d649ba99efbdb5ef9b669c3b036c9fd864a01755e86731")
MODE="original_exclusive_GPU45_canonical_alias_v2"


def operational_refs_v2():return [c.ref(ALIAS),c.ref(SOURCE),c.ref(CANONICAL)]


def validate_operational_config_v2(cfg,reader):
    c.validate_config_v1(cfg,reader);c.exact_sources_v1(reader,cfg)
    c.require(cfg.get("operational_launcher_ref")==c.ref(ALIAS)
        and cfg.get("operational_root_entry_ref")==c.ref(SOURCE)
        and cfg.get("operational_canonical_module_ref")==c.ref(CANONICAL)
        and cfg.get("operational_launch_mode")==MODE
        and cfg.get("underlying_exclusive_launcher_ref")==c.LAUNCHER_REF,"Exact separate actual canonical/alias/underlying exclusive launcher chain")
    sources=reader.decode_json(cfg["source_manifest_ref"])["source_sha256"]
    c.require(all(sources.get(r["path"])==r["sha256"] for r in operational_refs_v2()),"All additive operational source bytes in exact closure")
    return cfg


def isolated_admission_driver_v2():
    c.require(c.ref(BASE_DRIVER_REF["path"])==BASE_DRIVER_REF,"Exact frozen H050 driver")
    sp=importlib.util.spec_from_file_location("_H050_canonical_binding_private_driver_v2",c.ROOT/BASE_DRIVER_REF["path"])
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    original_required=module.direct_required_refs_v1
    def required(cfg,spec):return c.unique(original_required(cfg,spec)+operational_refs_v2())
    module.direct_required_refs_v1=required
    original_factory=module.isolated_original_supervisor_v1
    def canonical_supervisor():
        supervisor=original_factory()
        # The original supervisor launches this bounded alias, which delegates
        # to its unchanged scheduler after authenticating the canonical shim.
        supervisor.context=lambda:SimpleNamespace(POLICY_REF=c.POLICY_REF,LAUNCHER_REF=c.ref(ALIAS))
        return supervisor
    module.isolated_original_supervisor_v1=canonical_supervisor
    return module


def bind_v2(request_ref):
    request=c.MetadataReaderV1().decode_json(request_ref)
    c.require(request.get("canonical_operational_refs")==operational_refs_v2(),"Root pins exact additive canonical/alias source bytes")
    c.require(c.ref(BASE_BINDER_REF["path"])==BASE_BINDER_REF,"Exact frozen metadata binder")
    result=binder.bind_root_packet_v1(request_ref)
    reader=c.MetadataReaderV1();cfg=reader.decode_json(result["config_ref"])
    base_manifest=reader.decode_json(cfg["source_manifest_ref"]);sources=dict(base_manifest["source_sha256"])
    for r in operational_refs_v2():
        c.require(r["path"] not in sources or sources[r["path"]]==r["sha256"],"No conflicting canonical operational sources")
        sources[r["path"]]=r["sha256"]
    folder=c.ROOT/binder.BOUND
    manifest=c.save(folder,"source_manifest.canonical.root_bound.json",dict(base_manifest,source_sha256=sources,
        original_root_bound_manifest_ref=cfg["source_manifest_ref"],operational_launch_mode=MODE,
        actual_operational_launcher_ref=c.ref(ALIAS),actual_canonical_module_ref=c.ref(CANONICAL),
        underlying_exclusive_launcher_ref=c.LAUNCHER_REF))
    cfg.update(source_manifest_ref=manifest,operational_launcher_ref=c.ref(ALIAS),operational_root_entry_ref=c.ref(SOURCE),
        operational_canonical_module_ref=c.ref(CANONICAL),operational_launch_mode=MODE,underlying_exclusive_launcher_ref=c.LAUNCHER_REF)
    config_ref=c.save(folder,"config_train_evaluate.canonical.root_active.json",cfg)
    validate_operational_config_v2(cfg,reader)
    request=reader.decode_json(result["train_request_ref"]);request["config_ref"]=config_ref
    train=c.save(folder,"admission_request.train.canonical.root_active.json",request)
    pending=reader.decode_json(result["pending_evaluate_request_ref"]);pending["config_ref"]=config_ref
    pending_ref=c.save(folder,"admission_request.evaluate.canonical.pending_train.json",pending)
    receipt=c.save(folder,"canonical_metadata_binding.json",dict(schema="H050_canonical_operational_metadata_binding_v2",request_ref=request_ref,
        config_ref=config_ref,train_request_ref=train,pending_evaluate_request_ref=pending_ref,
        original_metadata_binding_ref=result["metadata_binding_ref"],source_manifest_ref=manifest,
        operational_refs=operational_refs_v2(),underlying_exclusive_launcher_ref=c.LAUNCHER_REF,
        admitted=False,launched=False,exclusive_lease_required=True,live_C3_cannot_share=True))
    answer=dict(canonical_metadata_binding_ref=receipt,config_ref=config_ref,train_request_ref=train,pending_evaluate_request_ref=pending_ref)
    print(json.dumps(answer),flush=True);return answer


def bind_evaluate_v2(request_ref):
    reader=c.MetadataReaderV1();r=reader.decode_json(request_ref)
    c.require(r.get("schema")=="H050_root_evaluation_binding_request_v1" and r.get("status")=="active"
        and r.get("root_authorizer")=="root_agent","Explicit root EVALUATE binding")
    c.require(r["config_ref"]["path"]==binder.BOUND+"/config_train_evaluate.canonical.root_active.json","Same canonical immutable TRAIN/EVAL config")
    cfg=reader.decode_json(r["config_ref"]);validate_operational_config_v2(cfg,reader)
    wait=reader.decode_json(r["train_actual_wait_ref"])
    spec=dict(phase="evaluate",train_actual_wait_ref=r["train_actual_wait_ref"],train_launch_ref=wait.get("launch_ref"),train_completion_ref=wait.get("completed_ref"))
    c.validate_train_lineage_v1(reader,spec,cfg)
    c.require(type(r.get("physical_gpu")) is int and r["physical_gpu"] in (4,5)
        and type(r.get("reserved_bytes")) is int and r["reserved_bytes"]>0,"Root GPU4/5 and complete EVALUATE reservation")
    plan=c.count_plan_v1(reader.decode_json(cfg["count_manifest_ref"]),"evaluate",reader.decode_json(cfg["preparation_completion_ref"]),cfg)
    c.validate_plan_caps_v1(plan,cfg,r["reserved_bytes"])
    record=c.save(c.ROOT/binder.BOUND,"admission_request.evaluate.canonical.root_active.json",dict(schema="H050_root_admission_request_v1",status="active",
        root_authorizer="root_agent",config_ref=r["config_ref"],physical_gpu=r["physical_gpu"],reserved_bytes=r["reserved_bytes"],**spec))
    print(json.dumps(dict(evaluate_request_ref=record)),flush=True);return record


def admit_v2(request_ref):
    reader=c.MetadataReaderV1();r=reader.decode_json(request_ref)
    validate_operational_config_v2(reader.decode_json(r["config_ref"]),reader)
    return isolated_admission_driver_v2().admit(request_ref)


def launch_v2(admitted_ref):
    reader=c.MetadataReaderV1();admitted=reader.decode_json(admitted_ref);spec=reader.decode_json(admitted["input_ref"])
    cfg=reader.decode_json(spec["refs"]["config"]);validate_operational_config_v2(cfg,reader)
    storage=reader.decode_json(spec["refs"]["storage"])
    c.require(all(r in storage["reservation"]["consumer_bindings"] for r in operational_refs_v2()),"Actual canonical/alias bindings in original full reservation")
    return isolated_admission_driver_v2().launch(admitted_ref)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("describe","bind","bind-evaluate","admit","launch"),nargs="?",default="describe")
    p.add_argument("--execute",action="store_true");p.add_argument("--record");p.add_argument("--sha256");a=p.parse_args(argv)
    if a.action=="describe":
        print(json.dumps(dict(status="inactive_source_canonical_binding_only",hypothesis_id="H050",exclusive_lease=True,
            launch_blocked_while_C3_GPU4_and_native_GPU5_hold_original_leases=True,canonical_module="arggnn.H050_direct_readout_cut_memory_exec_v1")));return 0
    c.require(a.execute and a.record and a.sha256,"Explicit root action and exact metadata ref")
    record=dict(path=a.record,sha256=a.sha256)
    actions={"bind":bind_v2,"bind-evaluate":bind_evaluate_v2,"admit":admit_v2,"launch":launch_v2}
    return actions[a.action](record) if a.action in ("launch",) else (actions[a.action](record) and 0)

if __name__=="__main__":raise SystemExit(main())
