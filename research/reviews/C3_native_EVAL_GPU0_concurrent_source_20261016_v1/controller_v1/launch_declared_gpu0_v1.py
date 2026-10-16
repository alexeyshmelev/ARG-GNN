"""Additive finite CLI and exact renewal bootstrap in one actual source role."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,runpy,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
REL="research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py"
MODULE="research.reviews.C3_native_EVAL_GPU0_concurrent_source_20261016_v1.controller_v1.gpu0_declared_pool_v1"
def bootstrap_main_v1(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--renewal-certificate");parser.add_argument("--renewal-sha256")
    parser.add_argument("--renewal-source-sha256");parser.add_argument("--module",choices=(MODULE,))
    parser.add_argument("args",nargs=argparse.REMAINDER)
    a=parser.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="source_only_no_import_hook_installed")));return 0
    if not all((a.renewal_certificate,a.renewal_sha256,a.renewal_source_sha256,a.module)):
        raise ValueError("Exact explicit renewal certificate/source and canonical pool module")
    path=ROOT/REL
    if hashlib.sha256(path.read_bytes()).hexdigest()!=a.renewal_source_sha256:
        raise ValueError("Renewal bootstrap source bytes changed")
    sp=importlib.util.spec_from_file_location("_H039_fresh_pool_renewal_v1",path)
    renewal=importlib.util.module_from_spec(sp);sp.loader.exec_module(renewal)
    renewal.install_renewal_v1(dict(path=a.renewal_certificate,sha256=a.renewal_sha256))
    args=a.args[1:] if a.args[:1]==["--"] else a.args
    sys.argv=[a.module,*args];runpy.run_module(a.module,run_name="__main__",alter_sys=True)
    return 0
def main(argv=None):
    argv=sys.argv[1:] if argv is None else list(argv)
    if any(x=="--renewal-certificate" or x.startswith("--renewal-certificate=") for x in argv):
        return bootstrap_main_v1(argv)
    from research.reviews.C3_native_EVAL_GPU0_concurrent_source_20261016_v1.controller_v1.gpu0_declared_pool_v1 import main as pool_main
    return pool_main(argv)

if __name__=="__main__":raise SystemExit(main())
