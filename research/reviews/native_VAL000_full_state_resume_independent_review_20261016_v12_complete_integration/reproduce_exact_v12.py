from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/native_VAL000_full_state_resume_source_20261016_v12'
REVIEW='research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v12_complete_integration'
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def save(name,value):
    with (ROOT/REVIEW/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def check_seals():
    expected={HERE+'/source_handoff.v12.json':'bd22f9ced9e81da91c4a63850606facc3a8350e690826ee6ca9b671ca4a07ca7',HERE+'/source_manifest.v1.json':'734c8f55fc6a06c20fba9b63b9d72bf733ee9e6b49261f8da26c8bb6b27ae664',HERE+'/prepare_continuation_root_v12.py':'04d8c8a6b1e9480ad660bae64e49213b92300bdf4dfa60d889457cca1caf0bcf'}
    refs={}
    def walk(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x and isinstance(x['path'],str):refs[(x['path'],x['sha256'])]=dict(path=x['path'],sha256=x['sha256'])
            for value in x.values():walk(value)
        elif isinstance(x,list):
            for value in x:walk(value)
    for n in ('source_handoff.v12.json','source_manifest.v1.json'):walk(json.loads((ROOT/HERE/n).read_text()))
    checked=[];excluded=[]
    for record in refs.values():
        path=record['path'];p=ROOT/path
        if p.suffix=='.pt':excluded.append(dict(ref=record,reason='Payload/checkpoint excluded; metadata ref only'));continue
        assert p.suffix in ('.py','.json','.md','.patch','.yaml'),path
        assert not Path(path).is_absolute() and '..' not in p.parts and p.resolve()==p and not p.is_symlink() and p.stat().st_size<=32*2**20,path
        assert ref(path)==record,(path,'Hash mismatch')
        checked.append(record)
    for path,digest in expected.items():assert ref(path)['sha256']==digest
    save('hash_verification.v12.json',dict(status='PASS',checked_ref_count=len(checked),checked_refs=checked,payloads_excluded=excluded))
    return len(checked)
def run():
    n=check_seals()
    f=module(HERE+'/dual_guard_memo_inert_v12.py','_independent_v12_real_factory_fixture')
    f.HERE=REVIEW
    a=f.run();save('author_real_factory_reproduction.v12.json',a)
    v=module(HERE+'/fresh_scope_integration_inert_v12.py','_independent_v12_VFS_ROOT_fixture')
    b=v.run();save('author_VFS_ROOT_reproduction.v12.json',b)
    save('reproduction_summary.v12.json',dict(status='PASS',sealed_ref_count=n,real_factory_check_count=a['check_count'],VFS_ROOT_check_count=b['check_count'],roles=b['prospective_role_union'],fixture_source_refs=[ref(HERE+'/dual_guard_memo_inert_v12.py'),ref(HERE+'/fresh_scope_integration_inert_v12.py')],actual_guards_probes_payload_accounting_admission_activation_jobs_or_Git=False,all_writes_owned_review_directory_only=True,finished_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':
    try:run()
    except BaseException as error:
        save('reproduction.FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
