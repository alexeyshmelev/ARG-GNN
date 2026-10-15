"""Focused operational-v2 source and synthetic-only checks; no driver/runtime import."""
from pathlib import Path, PurePosixPath
import ast,base64,csv,hashlib,json,posixpath,types,argparse,io,contextlib
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H018_native_Relate_qualification_source_v1'
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def require(ok,message):
    if not ok:raise ValueError(message)
def reject(name,call):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append(name);return
    raise AssertionError(name+' accepted')
def node(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
def pure(source,name,ns):
    exec(compile(ast.Module(body=[node(source,name)],type_ignores=[]),'<synthetic-only-'+name+'>','exec'),ns)
    return ns[name]
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
new=(HERE/'root_qualifier_v2.py').read_text();old=(HERE/'root_qualifier.py').read_text()
for name in ('require','utc','load','common','syncdir','save','identity','bounded_file','error_record','direct_authority_refs','write_bytes','group_absent','direct_wait_owned','limit_process','native_limits','file_inventory','owned_rss','supervise','verify_request','admitted_inputs','deadline','launch'):
    check('unchanged operational/scientific function '+name,ast.dump(node(new,name),include_attributes=False)==ast.dump(node(old,name),include_attributes=False))
for name in ('root_qualifier.py','root_qualifier_v2.py'):
    compile((HERE/name).read_text(),name,'exec');checks.append('AST compile '+name)
cfg=json.loads((HERE/'config_v2.json').read_bytes());prior=json.loads((HERE/'config.json').read_bytes())
check('only four operational config namespace values changed',{k:v for k,v in cfg.items() if k not in ('job_id','output_directory','packet_directory','auxiliary_directory')}=={k:v for k,v in prior.items() if k not in ('job_id','output_directory','packet_directory','auxiliary_directory')})
check('exact50MB cap preserved',cfg['reserved_bytes']==50000000 and cfg['output_cap_bytes']==32000000)
check('fresh v2 job/output/packet/aux',cfg['job_id'].endswith('-v2') and cfg['output_directory'].endswith('_v2') and cfg['packet_directory'].endswith('/admission_v2') and cfg['auxiliary_directory'].endswith('/worker_aux_v2'))
settings=next(n.value for n in ast.parse(new).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets));check('literal settings equal actual config_v2',ast.literal_eval(settings)==cfg)
request=json.loads((HERE/'root_request_inactive_v2.json').read_bytes());manifest=json.loads((HERE/'source_manifest_v2.json').read_bytes())
check('inactive request no authority',request['status']=='inactive_source_plan' and request['root_authorizer'] is None and request['native_execution_authorized'] is False and request['registered_before_execution'] is False and request['registration_ref'] is None)
check('exact current driver/config/manifest refs',request['driver_ref']==manifest['source_refs']['driver']==ref(HERE/'root_qualifier_v2.py') and request['config_ref']==manifest['source_refs']['config']==ref(HERE/'config_v2.json') and request['source_manifest_ref']==ref(HERE/'source_manifest_v2.json'))
check('RECORD refs unchanged',cfg['runtime_package_RECORD_refs']==prior['runtime_package_RECORD_refs'])
check('binary/licence/fixture/helper refs unchanged',all(cfg[k]==prior[k] for k in ('native_binary_refs','license_ref','fixture_profile_ref','fixture_expected_ref','helper_ref')))
profile=json.loads((HERE/'fixture_profile.json').read_bytes());check('exact512site40haplotype fixture',profile['sites']==512 and profile['haplotypes']==40)
ns=dict(Path=Path,ROOT=ROOT);recognize=pure(new,'generated_project_pycache_path_v2',ns)
for path in [ROOT/'.runtime/cache/pycache/numpy/core.cpython-311.pyc',ROOT/'.runtime/cache/pycache/a.pyc']:
    check('accept canonical generated pyc '+path.name,recognize(path))
for label,path in [('outside_project',Path('/tmp/.runtime/cache/pycache/a.pyc')),('prefix_collision',ROOT/'.runtime/cache/pycache-other/a.pyc'),('wrong_suffix',ROOT/'.runtime/cache/pycache/a.py'),('suffix_trick',ROOT/'.runtime/cache/pycache/a.pyc.so'),('arbitrary_dunder_pycache',ROOT/'.runtime/env/lib/__pycache__/a.pyc'),('dotdot',ROOT/'.runtime/cache/pycache/../a.pyc'),('relative',Path('.runtime/cache/pycache/a.pyc'))]:
    check('reject generated-path '+label,not recognize(path))
# The actual verifier receives fake canonical paths/files and in-memory RECORD CSV.
class FakePath(PurePosixPath):
    def resolve(self):return type(self)(posixpath.normpath(str(self)))
    def is_file(self):return True
    def is_symlink(self):return False
fake_root=FakePath(str(ROOT));record=dict(path='synthetic/RECORD',sha256='0'*64)
payload=b'abcde';hashed_pyc=b'1234567';hash1=hashlib.sha256(payload).hexdigest();hash2=hashlib.sha256(hashed_pyc).hexdigest()
def token(hexdigest):return base64.urlsafe_b64encode(bytes.fromhex(hexdigest)).decode().rstrip('=')
rows=[['numpy/core.py','sha256='+token(hash1),'5'],['../../../../cache/pycache/numpy/generated.pyc','',''],['numpy.fake.dist-info/RECORD','',''],['../../../../cache/pycache/numpy/hashed.pyc','sha256='+token(hash2),'7']]
state={'rows':rows};hashed_calls=[]
expected={'numpy/core.py':(hash1,5),'hashed.pyc':(hash2,7)}
def fake_bounded(b,rec,cap,retain=True):
    if rec==record:
        out=io.StringIO();csv.writer(out).writerows(state['rows']);return out.getvalue().encode()
    hashed_calls.append(rec)
    sha,size=expected['hashed.pyc'] if rec['path'].endswith('hashed.pyc') else expected['numpy/core.py']
    require(rec['sha256']==sha,'mock installed file SHA mismatch')
    return dict(bytes=size)
verify_ns=dict(Path=FakePath,ROOT=fake_root,SETTINGS={'runtime_package_RECORD_refs':[record]},csv=csv,base64=base64,require=require,bounded_file=fake_bounded)
pure(new,'generated_project_pycache_path_v2',verify_ns);verify=pure(new,'verify_runtime_packages',verify_ns)
result=verify(object());check('all hashed entries including hashedpyc authenticated',result['hashed_file_entries']==2 and len(hashed_calls)==2)
check('generatedpyc excluded honestly',result['generated_pyc_entries_without_RECORD_hash']==1 and result['generated_pyc_contents_authenticated'] is False and result['wheel_all_files_authenticated_claimed'] is False)
check('blank RECORD row retained policy',result['blank_RECORD_entries']==1)
for label,row in [('outside_containment',['/tmp/fake.pyc','','']),('non_pyc_under_cache',['../../../../cache/pycache/numpy/not-generated.so','','']),('blank_real_source',['numpy/core.py','','']),('arbitrary___pycache__',['numpy/__pycache__/fake.pyc','','']),('wrong_hash',['numpy/core.py','sha256='+token('1'*64),'5']),('wrong_size',['numpy/core.py','sha256='+token(hash1),'6']),('wrong_algorithm',['numpy/core.py','md5='+token(hash1),'5'])]:
    state['rows']=[row];reject('verifier rejects '+label,lambda:verify(object()))
check('canonical resolution and containment precede any blank skip',new.index('path=(site/name).resolve()')<new.index('Project-contained installed wheel file')<new.index('if not digest:'))
admit=ast.get_source_segment(new,node(new,'admit'));check('real package verification gates ROOT before accounting',admit.index('verify_runtime_packages(b)')<admit.index('packet.mkdir()')<admit.index('m=b.guard()'))
check('public preflight no guard action','verify_runtime_packages(common())' in ast.get_source_segment(new,node(new,'runtime_preflight_v2')))
main_ns=dict(argparse=argparse,json=json,require=require);main=pure(new,'main',main_ns)
with contextlib.redirect_stdout(io.StringIO()):status=main([])
check('noargs default inert integer0',type(status) is int and status==0)
for name,sha in json.loads((HERE/'preserved_v1_pins_v2.json').read_bytes())['files_sha256'].items():check('preserved v1 '+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha)
result=dict(schema='H018_Relate_operational_v2_source_pure_checks',status='passed_source_and_synthetic_cases_only',checks=len(checks),check_names=checks,driver_imported=False,guard_native_or_package_imported=False,actual_runtime_preflight_executed=False,actual_process_or_admission_executed=False,fixture_science_or_date_policy_changed=False)
print(json.dumps(result,sort_keys=True,indent=2))
