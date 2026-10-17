"""Inactive native/seal-only wrapper. Exact original v2 lifecycle, one v4 predecessor."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,stat,sys
from . import predecessor_bridge_v1 as bridge
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v1"
SOURCE=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
ENGINE_REF=dict(path="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/root_driver_v1.py",sha256="c1d6a5bb835800f30d0c1e487c885ede13db253c18286b20bc115d8d70a7447f")
AUTH_REF=dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py",sha256="2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98")
_ENGINE=None

def _load_exact_source(record,name):
    p=ROOT/record["path"]
    bridge.require(p.resolve()==p and ROOT in p.parents,"Exact canonical existing source")
    before=p.lstat()
    bridge.require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**20,"Bounded singly linked source")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb")as f:
        opened=os.fstat(f.fileno());raw=f.read(2**20+1);after=os.fstat(f.fileno())
    named=p.lstat();identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    bridge.require(identity(before)==identity(opened)==identity(after)==identity(named)and len(raw)==before.st_size
        and hashlib.sha256(raw).hexdigest()==record["sha256"],"Exact stable inherited source SHA")
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    exec(compile(raw,str(p),"exec",dont_inherit=True,optimize=0),module.__dict__)
    return module

def engine():
    global _ENGINE
    if _ENGINE is None:
        bridge.require(sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,"Original fresh -B -s optimize0")
        d=_load_exact_source(ENGINE_REF,"_FLARE_exact_native_seal_v2_lifecycle")
        for name,value in dict(HERE=HERE,SOURCE=SOURCE,ENTRY=ENTRY,MANIFEST=MANIFEST,HANDOFF=HANDOFF).items():setattr(d,name,value)
        inherited_contract=d.contract
        def contract():
            c=inherited_contract()
            c.STAGES=("native","seal");c.VAL=(bridge.PARENT,)
            auth=_load_exact_source(AUTH_REF,"_FLARE_exact_existing_source_authenticator")
            bridge.install_stage_bridge_v1(c,auth)
            inherited_validate=c.validate_request
            def validate_request(record):
                q=c.read(record)
                c.require(q.get("stage")in("native","seal")and q.get("parent_id")==bridge.PARENT
                    and q.get("conversion_ref")==bridge.COMPLETED_REF,
                    "Only new native/seal stages for this exact completed conversion; conversion/replay prohibited")
                c.require(q.get("storage_policy_selector_ref")==bridge.SELECTOR_REF,"Exact current Relate64 child")
                return inherited_validate(record)
            c.validate_request=validate_request
            return c
        d.contract=contract;_ENGINE=d
    return _ENGINE

def activate():return engine().activate()
def __getattr__(name):
    if name.startswith("__"):raise AttributeError(name)
    return getattr(engine(),name)
def main(argv=None):
    parser=argparse.ArgumentParser();actions=parser.add_mutually_exclusive_group()
    actions.add_argument("--admit-ref");actions.add_argument("--execute-ref")
    args=parser.parse_args(argv)
    if not(args.admit_ref or args.execute_ref):
        print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY_NEEDS_INDEPENDENT_REVIEW",guard_imported=False,operations=False)));return 0
    return engine().main(argv)
if __name__=="__main__":raise SystemExit(main())
