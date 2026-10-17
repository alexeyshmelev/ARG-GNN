"""Explicit static/RAM-only source checks. No target imports or real resources."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,json
HERE=Path(__file__).parent
ROOT=HERE.parents[2]
def run_v1():
    parsed={p.name:ast.parse(p.read_text(),filename=str(p))for p in HERE.glob("*.py")}
    old=ROOT/"research/reviews/H058_GPU0_full_training_successor_source_20261017_v2"
    for name in("producer_v1.py","observations_gpu_v1.py","context_factory_v1.py"):
        text=(old/name).read_text().replace(str(old.relative_to(ROOT)),str(HERE.relative_to(ROOT))).replace(str(old.relative_to(ROOT)).replace("/","."),str(HERE.relative_to(ROOT)).replace("/","."))
        assert ast.dump(ast.parse(text),include_attributes=False)==ast.dump(parsed[name],include_attributes=False),name
    class FakeShared:
        def _memory(self,incoming=0):return dict(blocked_by=self.fake_reasons)
        def _fail_owned_supervision(self,message):self.failure=message;self.jobs["H058"]["state"]="draining"
    legacy=SimpleNamespace(ACTIVE={"launching","running","draining"})
    base=SimpleNamespace()
    reviewed=SimpleNamespace(legacy=legacy,base=base,Shared0Controller=FakeShared)
    sources={"down":(ROOT/"research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py").read_bytes(),"old":(ROOT/"src/arggnn/gpu45_c3cache_h050_pool_v1.py").read_bytes()}
    fake_h=SimpleNamespace(source_refs_v1=lambda:None,load=lambda *a:reviewed,raw=lambda r:sources[r],POLICY={})
    fake_c=SimpleNamespace(SHARED="shared",DOWN="down",OLD="old",guard_v1=lambda:None,guarded_check_v1=lambda *a:None,validate_cap_v1=lambda *a,**k:None)
    fn=next(n for n in parsed["pool_v1.py"].body if isinstance(n,ast.FunctionDef)and n.name=="build_controller_class_v1")
    def need(ok,message):
        if not ok:raise ValueError(message)
    ns=dict(h=fake_h,c=fake_c,ROOT=ROOT,MODULE="exact.future.module",require=need,ast=ast,os=None,time=None,json=json,types=None,_verify_new_exclusive_v1=lambda *a:None,_open_old_inode_v1=lambda *a:None)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"<RAM_only_factory>","exec"),ns)
    cls,_=ns["build_controller_class_v1"]()
    app=object.__new__(cls);app.jobs={"H058":dict(state="running")};app.fake_reasons=["actual_free_memory_or_reserved_headroom"]
    assert app._memory(0)["blocked_by"]
    assert app.failure.startswith("Periodic active pool capacity failed:") and app.jobs["H058"]["state"]=="draining"
    assert app._memory(30720)["blocked_by"]
    app.fake_reasons=[];assert app._memory(0)==dict(blocked_by=[])
    # Independent arithmetic fixture: all remaining allocations plus headroom,
    # external/driver occupancy, exact80GiB boundary and next-MiB rejection.
    def memory(external,owned):
        peaks=(30720,34816);total=81920;used=sum(owned)+external
        return sum(peaks)+external+4096<=total and total-used>=sum(max(0,p-u)for p,u in zip(peaks,owned))+4096
    assert memory(12288,(10240,20480))and not memory(12289,(10240,20480))
    hook=(HERE/"native_attachment_seam_v1.py").read_text();assert "227990"not in hook
    return dict(schema="GPU0_postdrain_static_RAM_checks_v1",status="PASS_STATIC_RAM_ONLY",Python_AST_files=len(parsed),H058_producer_observation_context_AST_preserved=True,real_source_dynamic_start_queue_factory_compiles_under_RAM_fakes=True,periodic_capacity_blocker_requests_failstop_without_skipping_drain=True,incoming_blocker_remains_rejected_by_admission=True,nominal80GiB_boundary_and_next_MiB_rejection=True,native_hook_has_no_VAL000_update_constant=True,target_module_imports=0,scientific_or_numerical_fixtures=0,real_guard_lock_process_device_accounting_operations=0,full_join_independent_review_required=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",action="store_true");a=p.parse_args()
    print(json.dumps(run_v1()if a.run else dict(status="SOURCE_ONLY_INACTIVE",checks_run=False),sort_keys=True))
if __name__=="__main__":main()
