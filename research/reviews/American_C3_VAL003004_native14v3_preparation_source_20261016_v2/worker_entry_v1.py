"""Authorized CPU worker: unchanged original execute functions and existing plan worker."""
from pathlib import Path
import argparse,fcntl,importlib,json,os,sys
from research.reviews.American_C3_VAL003004_native14v3_preparation_source_20261016_v2 import root_contract_v1 as c
def worker(record,gate_fd):
    c.require(os.read(gate_fd,1)==b"G","No scientific execution before owned launch gate");os.close(gate_fd)
    s,q,cfg=c.input_spec(record)
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and os.getpid()==os.getpgrp()==os.getsid(0),"Owned CPU-only session")
    fd=int(os.environ["ARGGNN_C3_PREPARATION_SLOT_FD"]);identity=os.fstat(fd)
    candidates=[c.local(c.SLOT_REL+"/cpu_slot_"+str(i)+".lock") for i in range(2)]
    paths=[p for p in candidates if (p.lstat().st_dev,p.lstat().st_ino)==(identity.st_dev,identity.st_ino)]
    c.require(len(paths)==1 and identity.st_size==0,"Inherited exact two-producer slot")
    # Probe a distinct OFD; the inherited supervisor/worker OFD must still hold it.
    with paths[0].open("r+") as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(probe,fcntl.LOCK_UN);raise ValueError("CPU slot lock not actually held")
    if q["stage"]=="plan":
        from research.reviews.American_C3_VAL003004_native14v3_preparation_source_20261016_v2 import plan_worker_v1
        return plan_worker_v1.execute(record)
    g,_=c.consumer_guard()
    basis=c.read(c.ref(c.BASIS));stage=q["stage"]
    producer_ref=basis["inference_producer_ref" if stage=="infer" else "native_producer_ref"]
    common_ref=basis["common_producer_ref"]
    closure=[common_ref,producer_ref,basis["simulation_sanitize_helper_ref"] if stage=="infer" else basis["author_path_helper_ref"]]
    for r in closure:c.raw(r)
    c.require(all(r in s["storage"]["reservation"]["consumer_bindings"] for r in [*closure,s["refs"]["config"],s["runtime_ref"],c.GUARD_REF]),
        "Original common/producer/helper/config/runtime/current guard all reserved")
    common=importlib.import_module("arggnn.american_c3_common_v1")
    c.require(Path(common.__file__).resolve()==c.local(common_ref["path"]),"Original common writer origin")
    runtime=c.read(s["runtime_ref"])
    c.require(runtime.get("prefix")==str(Path(sys.prefix).resolve()) and runtime.get("CUDA_VISIBLE_DEVICES")=="",
        "Exact original qualified runtime/interpreter")
    if "packages" in runtime:
        actual=common.runtime_report()["packages"]
        c.require(all(actual.get(k)==v for k,v in runtime["packages"].items()),"Original runtime package metadata")
    g.validate_storage_v9(s,cfg,s["storage"],source_ref=producer_ref,config_ref=s["refs"]["config"],runtime_ref=s["runtime_ref"])
    producer=importlib.import_module("arggnn."+Path(producer_ref["path"]).stem)
    c.require(Path(producer.__file__).resolve()==c.local(producer_ref["path"]),"Exact original scientific function origin")
    # Reuse existing scientific execute and strict Writer; no algorithm clone.
    return producer.execute(cfg,common.Writer(s,g))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker-ref");p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
    if a.worker_ref is None:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",execution=False)));return 0
    c.require(a.gate_fd is not None,"Actual inherited gate");result=worker(json.loads(a.worker_ref),a.gate_fd)
    print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
