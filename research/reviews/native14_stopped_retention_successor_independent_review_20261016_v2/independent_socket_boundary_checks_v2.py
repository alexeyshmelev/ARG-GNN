import ast,copy,json,os,posixpath,stat,types
from pathlib import Path
C=Path('research/reviews/native14_stopped_retention_successor_source_20261016_v2')
R=Path(__file__).parent
source=(C/'native14_stopped_retention_v1.py').read_text()
names={'require','_parse_kernel_endpoints_v2','_native_kernel_path_absent_v2','_native_socket_absent','_native_targets','_native_measurement','consumer_bindings_v1','_bindings'}
nodes=[copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes}==names
report=[]
def reject(label,fn):
    try:fn()
    except (ValueError,TypeError,KeyError,OSError):report.append(dict(case=label,rejected=True));return
    raise AssertionError('Incorrect acceptance: '+label)
def integer(x,label,minimum=0):
    if type(x) is not int or x<minimum:raise ValueError(label)
    return x
def absent_group(*a):raise ProcessLookupError()
env=dict(os=types.SimpleNamespace(fsencode=os.fsencode,killpg=absent_group),posixpath=posixpath,stat=stat,copy=copy)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<independent-isolated-source-only>','exec'),env)
target='/SYNTHETIC_ONLY/scoped/socket.sock'
for label,address in [('canonical',b'/SYNTHETIC_ONLY/scoped/socket.sock'),('relative',b'socket.sock'),('redundant_slash',b'/SYNTHETIC_ONLY//scoped/socket.sock'),('dot',b'/SYNTHETIC_ONLY/scoped/./socket.sock'),('alias_parent',b'/alias/socket.sock'),('renamed_basename',b'/external/old-name.sock'),('unrelated',b'@unrelated')]:
    env['_kernel_endpoints']=lambda address=address:[address]
    reject('production_callback_'+label,lambda:env['_native_socket_absent'](target))
header=b'Num RefCount Protocol Flags Type St Inode Path\n'
row=b'00000000: 00000002 00000000 00010000 0001 01 12345 /external/unrelated.sock\n'
parse=env['_parse_kernel_endpoints_v2']
assert parse(header+row)==[b'/external/unrelated.sock']
for label,data in [('empty',b''),('header_only',header),('partial',(header+row)[:-1]),('oversize',b'x'*(16*2**20+1)),('unknown_header',b'Unknown\n'+row),('unknown_row',header+b'unknown\n')]:reject('catalog_'+label,lambda data=data:parse(data))
m=json.loads((C/'source_manifest.v1.json').read_text())
env.update(ROOT=Path('/SYNTHETIC_ONLY'),_kernel_endpoints=lambda:[b'@unrelated'],_utc=lambda:'FAKE',_absent=lambda pid:dict(checked_utc='FAKE'))
def missing(*a):raise FileNotFoundError()
helper=types.SimpleNamespace(lstat_no_follow_v1=missing)
env['_native_helper']=lambda:helper
facts=env['_native_targets'](m)
assert len(facts['PID_observations'])==150 and len(facts['group_observations'])==11 and len(facts['endpoint_observations'])==24
assert all(x[1]=='absent_path' for x in facts['endpoint_observations'])
for i,entry in enumerate(m['recorded_filesystem_endpoints']):
    retained=Path('/SYNTHETIC_ONLY')/entry['path']
    metadata=types.SimpleNamespace(**dict(zip(('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns'),entry['original_lstat'])))
    def observe(root,path,retained=retained,metadata=metadata):
        if path==retained:return metadata
        raise FileNotFoundError()
    helper.lstat_no_follow_v1=observe
    reject('known_endpoint_present_at_index_'+str(i),lambda:env['_native_targets'](m))
helper.lstat_no_follow_v1=missing
env['_kernel_endpoints']=lambda:[b'@arggnn-c3mlm-v12-any-fake-suffix']
reject('source_bound_abstract_namespace',lambda:env['_native_targets'](m))
roots=['/SYNTHETIC_ONLY/output','/SYNTHETIC_ONLY/aux','/SYNTHETIC_ONLY/tmp']
budget=types.SimpleNamespace(v4=types.SimpleNamespace(integer=integer))
env.update(_native_manifest=lambda:m,budget=budget,NATIVE_RESERVED=15032385536,NATIVE_ROOTS=roots,_paths=lambda own:map(Path,roots))
certificate=dict(schema='native14_exact_stopped_retained_node_measurement_v1',conservative_scope_bytes=12288,entries=3,scope_paths=roots,stat_only=True,
    per_scope=[dict(path=p,entries=1,conservative_scope_bytes=4096,node_types=dict(directory=1,regular=0,symlink=0,socket=0)) for p in roots],inventory_sha256='0'*64,
    two_complete_passes_and_final_no_follow_revalidation=True,per_node_additional_metadata_charge_bytes=4096,regular_payloads_opened=False,
    symlink_targets_followed=False,unsafe_nodes_rejected=True,node_metadata_preserved=True,namespace_normalization_performed=False,cooperative_snapshot_not_atomic_quota=True)
own=dict(reserved_bytes=15032385536)
assert env['_native_measurement'](certificate,own)==12288
for i in range(3):
    bad=copy.deepcopy(certificate);bad['entries']+=1;bad['conservative_scope_bytes']+=4096
    bad['per_scope'][i]['entries']+=1;bad['per_scope'][i]['conservative_scope_bytes']+=4096;bad['per_scope'][i]['node_types']['socket']+=1
    reject('nonzero_socket_certificate_root_'+str(i),lambda bad=bad:env['_native_measurement'](bad,own))
plan=json.loads((C/'direct_role_plan.v1.json').read_text())
refs={x['path']:x for x in plan['authored_direct_refs']}
def fake(name):return dict(path='SYNTHETIC_ONLY/'+name+'.json',sha256='f'*64)
env['_raw']=lambda ref:None
env.update(SOURCE=str(C/'native14_stopped_retention_v1.py'),MANIFEST=str(C/'source_manifest.v1.json'),MEASURE_SOURCE=str(C/'retained_nodes_v1.py'),_ref=lambda p:refs[p],_POLICY_REF=fake('policy'))
policy=dict(root_activation_ref=fake('activation'),permanent_revocation_ref=fake('revocation'),source_review_ref=fake('review'))
base={str(i):x for i,x in enumerate(plan['current_Census23_refs'])}
state=env['_bindings'](base,policy)
state_refs=env['consumer_bindings_v1'](list(state.values()))
assert len(state_refs)==62 and all(x in state_refs for x in plan['current_Census23_refs']+plan['historical45_refs'])
assert len(env['consumer_bindings_v1'](state_refs,[fake('observation'),fake('operation')]))==64
reject('additional_role65',lambda:env['consumer_bindings_v1'](state_refs,[fake('observation'),fake('operation'),fake('extra')]))
result=dict(schema='native14_independent_v2_fake_delta_boundaries',status='PASS_SOFTWARE_ONLY',negative_cases=len(report),negative_results=report,
    zero_socket_certificate_positive=True,all24_known_absent_paths_positive=True,all24_present_known_endpoint_positions_rejected=True,
    production_callback_blocks_nonmatching_renamed_socket=True,strict_parser_unknown_and_empty_rejected=True,complete_state62_and_invocation64_preserve_current23_and_historical45=True,
    actual_original_filesystem_process_kernel_endpoint_device_census_accounting_authority_access=False)
(R/'independent_socket_boundary_checks.v2.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],independent_negative_cases=len(report),state_refs=62,invocation_refs=64)))
