"""Pinned H050 canonical alias into the unchanged exclusive current980 launcher."""
import argparse,importlib.util,json,sys
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bridge_common_v1 as c
HERE=c.HERE+"/canonical_activation_v2"
SOURCE=HERE+"/launch_canonical_exclusive_v2.py"
DRIVER=HERE+"/root_canonical_driver_v2.py"
CANONICAL="src/arggnn/H050_direct_readout_cut_memory_exec_v1.py"
MODULE="arggnn.H050_direct_readout_cut_memory_exec_v1"
SCIENTIFIC_MODULE=c.worker.SOURCE[:-3].replace("/",".")


def parse_and_authenticate_v2(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",type=int,choices=(4,5),required=True);p.add_argument("--job",required=True)
    for name in ("storage-record","expected-storage-sha256","expected-budget-source-sha256","expected-launcher-sha256"):p.add_argument("--"+name,required=True)
    p.add_argument("module");p.add_argument("module_args",nargs=argparse.REMAINDER)
    a=p.parse_args(argv);c.require(a.module==SCIENTIFIC_MODULE,"Only the exact frozen H050 scientific entry may use this alias")
    c.require(len(a.module_args)==5 and a.module_args[0:2]==["--execute","--input"]
        and a.module_args[3]=="--expected-input-sha256","Exact original H050 root capability CLI")
    record=dict(path=a.module_args[2],sha256=a.module_args[4]);reader=c.MetadataReaderV1()
    spec=reader.decode_json(record);cfg=reader.decode_json(spec["refs"]["config"])
    c.require(spec.get("hypothesis_id")=="H050" and spec.get("arm")=="joint_four_arm"
        and spec.get("status")=="active" and spec.get("phase") in ("train","evaluate")
        and spec["root_driver_ref"]==c.ref(c.DRIVER) and spec["root_context_module_ref"]==c.ref(c.CONTEXT)
        and spec["producer_ref"]==spec["canonical_dispatcher_ref"]==c.ref(c.worker.SOURCE)
        and spec["job_id"]==a.job and spec["physical_gpu"]==a.gpu,"Original H050 scientific admission and exact job/GPU")
    required=[c.ref(SOURCE),c.ref(DRIVER),c.ref(CANONICAL)]
    c.require(cfg.get("operational_launcher_ref")==required[0] and cfg.get("operational_root_entry_ref")==required[1]
        and cfg.get("operational_canonical_module_ref")==required[2]
        and cfg.get("operational_launch_mode")=="original_exclusive_GPU45_canonical_alias_v2"
        and cfg.get("underlying_exclusive_launcher_ref")==c.LAUNCHER_REF,"Actual separate alias/canonical/delegated-launcher lineage")
    c.require(spec["refs"]["storage"]==dict(path=a.storage_record,sha256=a.expected_storage_sha256)
        and a.expected_budget_source_sha256==c.CORE and a.expected_launcher_sha256==c.BASE,"Original exact storage/core/launcher pins")
    storage=reader.decode_json(spec["refs"]["storage"])
    c.require(all(r in storage["reservation"]["consumer_bindings"] for r in required),"Actual canonical shim and both operational entries directly reserved")
    for r in required:
        c.require(spec["source_sha256"].get(r["path"])==r["sha256"],"Actual canonical/alias sources in admitted closure")
        reader.read(r)
    return a,record


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv);a,record=parse_and_authenticate_v2(argv)
    c.require(c.ref(c.LAUNCHER_REF["path"])==c.LAUNCHER_REF,"Exact original authorized exclusive scheduler")
    sp=importlib.util.spec_from_file_location("_H050_original_exclusive_GPU45_launcher",c.ROOT/c.LAUNCHER_REF["path"])
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    # One declared alias only. The original scheduler receives an arggnn module,
    # checks that exact source binding, takes its original flock and records it.
    indexes=[i for i,x in enumerate(argv) if x==SCIENTIFIC_MODULE]
    c.require(len(indexes)==1,"Single exact scientific module argument")
    argv[indexes[0]]=MODULE
    print(json.dumps(dict(event="H050_canonical_alias",operational_launcher_ref=c.ref(SOURCE),
        delegated_exclusive_launcher_ref=c.LAUNCHER_REF,actual_module_ref=c.ref(CANONICAL),scientific_producer_ref=c.ref(c.worker.SOURCE),input_ref=record)),flush=True)
    previous=sys.argv;sys.argv=[str(c.ROOT/c.LAUNCHER_REF["path"]),*argv]
    try:return module.main()
    finally:sys.argv=previous

if __name__=="__main__":raise SystemExit(main())
