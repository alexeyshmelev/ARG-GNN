"""Source-only exact bank21 close helper. Root explicit --execute only. No execution during authorship.
Adapts the unchanged bank20 helper, preserves all outputs, and uses original980 guard.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,traceback,os,json,hashlib,importlib.util
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H056_banks21_22_completed_close_source_20261016_v1/chr21'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
def ref(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(n,v):
 p=HERE/n
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return ref(p)
def load(r,name):
 p=ROOT/r['path'];assert ref(p)==r
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
assert sys.argv[1:]==['--execute']
PINNED_EVIDENCE=[{'path': 'experiments/results/H056_chr21_H36_original_GT_bank_CPU_v1/completed.json', 'sha256': 'e5891fba79634f4c74b58700e8fe1c6af07d8f0082da1dc34802c70d50ec2702'}, {'path': 'experiments/results/H056_chr21_H36_original_GT_bank_CPU_v1_aux/actual_wait_minimal.json', 'sha256': 'e612a1fd44b97921af1351bdabd78390bac756c296912fe1ec8eea57784142ca'}, {'path': 'experiments/results/H056_chr21_H36_original_GT_bank_CPU_v1_aux/outer_wait.json', 'sha256': 'c669e58174e00c90dff414d2247aece69e7516292757c8a3f89ac418e14d09a7'}, {'path': 'experiments/results/H056_chr22_H36_original_GT_bank_CPU_v1/completed.json', 'sha256': '3dc014c110665f7dc837bd4ab53e9b2ad00f217b2248b4bf4055ac4a52c4b457'}, {'path': 'experiments/results/H056_chr22_H36_original_GT_bank_CPU_v1_aux/actual_wait_minimal.json', 'sha256': '0d06f7ea6dc42bf8234be8f2a910fdaa14475526d9acc7d0cac256a6271ff901'}, {'path': 'experiments/results/H056_chr22_H36_original_GT_bank_CPU_v1_aux/outer_wait.json', 'sha256': 'ddf75c2cd185730bd0b9402d0beccd821ba131be978fe7bf3b2f187903efddad'}, {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, {'path': 'research/reviews/H056_chr20_bank_completed_close_root_20261016_v1/close.root_v1.py', 'sha256': '3e8008d0c92e17e47bd2dc03c0fd6ad8da9d037adc784782e218730907a530e6'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/actual_bank_tool_wait.root_v1.json', 'sha256': '942e5cd7665e6f93db8e06f84d69a5cf93f1a0b1d94d884377e0f205f6c091c8'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/actual_launch.root_v1.json', 'sha256': '0486c30caae3b2e701da69f427a53ef380d6cf266a18fd266806f150bc93c8d0'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/actual_tool_terminal.root_v1.json', 'sha256': 'da404049e046b48a9cbefc91bc1bfe9e49d484c0327d535440d27dd50ae3b9a7'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/actual_wait.root_v1.json', 'sha256': '091ab6dcf8064ae5dd98df918f1847cba2a730859b8d8c1ac35189e124710ba6'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/input.json', 'sha256': '0438786cb95f3821a27140fab982eec142dac12ed4bbe9506f681fe35d28b582'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/preparation_wait.root_sealed_v1.json', 'sha256': '92b7a0e521033e6bd0025078f2a7752bba9c1325f544721b459d39eab0ff9289'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/root_supervisor_identity.root_v1.json', 'sha256': 'a7b2372e74b4487d80cea62321cc274ccd3b0e0ebff7950fe93d1e07b891b9af'}, {'path': 'research/reviews/H056_chr21_bank_actual_root_20261016_v1/storage.json', 'sha256': '7ab6b7ca5820aac98692be824dccd774295994fdbe48c68fa100c86d24d42c6a'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/actual_bank_tool_wait.root_v1.json', 'sha256': 'c46da1806badccfb21ff9885d9f2967fc2cca176dea03f9b5a82fcd9cfe9ee60'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/actual_launch.root_v1.json', 'sha256': 'ee003d3ebd49885e7851e8ccaa6d48d6273a065cc2aac43b4b1f6f86cad9ec31'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/actual_tool_terminal.root_v1.json', 'sha256': 'b2dbeeee430d65c04cd9930acf2dd0bb20c0eb048f0e93e0c6186b8a2fb18672'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/actual_wait.root_v1.json', 'sha256': '5f360891d89a914b65f4db1256d41eef8e59e2ed51194d62af5ae2b3832191cc'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/input.json', 'sha256': '1516b37ac940ac661429945b85c00854b343e8de4c7d167cc0bbbc6dcead4913'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/preparation_wait.root_sealed_v1.json', 'sha256': '3d6b71c3900095964b19400dd9f92667b75b1514043bfaa32a1d4416ed2b4969'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/root_supervisor_identity.root_v1.json', 'sha256': 'a7b2372e74b4487d80cea62321cc274ccd3b0e0ebff7950fe93d1e07b891b9af'}, {'path': 'research/reviews/H056_chr22_bank_actual_root_20261016_v1/storage.json', 'sha256': '7b54b634b5b7e314d4ebed41852e39eecd1faaa3c484724cef615cc759558f98'}, {'path': 'research/reviews/H056_maps_actual_root_20261016_v2/config.json', 'sha256': '86daf84e78643225d889226fbdcbd4c33213419f96933a5bbc0a7b86dc815ebd'}, {'path': 'research/reviews/H056_maps_actual_root_20261016_v2/source_closure.json', 'sha256': 'a8c47f888ce5b3fe550b2026726639b94b28cd61ec35e86185275641bb5a3a1f'}, {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, {'path': 'research/reviews/H056_banks21_22_completed_close_source_20261016_v1/chr21/close_request.inactive.v1.json', 'sha256': 'f7ee78f4bd38af15a2fbd1a7d7c6e45aa107e9f239f4e2f9d6eed9ba9b391128'}]
for evidence_ref in PINNED_EVIDENCE:
 assert ref(ROOT/evidence_ref['path'])==evidence_ref
origin=ROOT/'research/reviews/H056_chr21_bank_actual_root_20261016_v1'
out=ROOT/'experiments/results/H056_chr21_H36_original_GT_bank_CPU_v1'
launch=json.loads((origin/'actual_launch.root_v1.json').read_text())
wait=json.loads((origin/'actual_wait.root_v1.json').read_text())
storage=json.loads((origin/'storage.json').read_text())
done=json.loads((out/'completed.json').read_text())
tool=json.loads((origin/'actual_tool_terminal.root_v1.json').read_text())
assert tool['actual_tool_terminal_consumed'] and tool['actual_tool_exit_code']==0
inner=json.loads((ROOT/'experiments/results/H056_chr21_H36_original_GT_bank_CPU_v1_aux/actual_wait_minimal.json').read_text())
assert inner['actual_direct_wait'] and inner['returncode']==0 and inner['pid']==3248721

assert launch['chromosome']=='21' and done['chromosome']=='21'
assert launch['root_supervisor_identity']['pid']==3248512 and launch['root_supervisor_identity']['ppid']==3248511
assert launch['launcher_identity']['pid']==3248513 and wait['launcher_pid']==3248513
assert launch['launcher_identity']['start_ticks']==wait['launcher_start_ticks']
assert wait['actual_direct_wait_observed'] is True and wait['returncode']==0
assert done['input_ref']==launch['input_ref']==ref(origin/'input.json')
assert tool['actual_tool_call_id']==10155 and tool['tool_chunk_id']=='2af26e'
assert tool['actual_direct_wait_ref']==ref(origin/'actual_wait.root_v1.json')
assert tool['bank_completion_ref']==ref(out/'completed.json')
prep=json.loads((origin/'preparation_wait.root_sealed_v1.json').read_text())
tool_wait=json.loads((origin/'actual_bank_tool_wait.root_v1.json').read_text())
aux=ROOT/'experiments/results/H056_chr21_H36_original_GT_bank_CPU_v1_aux'
outer=json.loads((aux/'outer_wait.json').read_text())
assert prep['actual_direct_wait_observed'] is True and prep['actual_exit_code']==0 and prep['process_group_drained'] is True
assert prep['input_ref']==launch['input_ref'] and prep['completed_ref']==ref(out/'completed.json')
assert prep['actual_root_tool_wait_ref']==ref(origin/'actual_bank_tool_wait.root_v1.json')
assert prep['original_worker_minimal_wait_ref']==ref(aux/'actual_wait_minimal.json')
assert prep['original_worker_outer_wait_ref']==ref(aux/'outer_wait.json')
assert tool_wait['actual_tool_terminal_ref']==ref(origin/'actual_tool_terminal.root_v1.json')
assert tool_wait['actual_tool_terminal_consumed'] is True and tool_wait['actual_tool_exit_code']==0
assert tool_wait['bank_completion_ref']==ref(out/'completed.json')
assert outer['actual_wait_ref']==ref(aux/'actual_wait_minimal.json')
assert outer['returncode']==0 and outer['observed_owned_groups_ESRCH'] is True
assert inner['input_ref']==launch['input_ref'] and inner['worker_pgid']==3248721
assert outer['observed_owned_groups']==[3248721]

request=dict(key=storage['reservation']['key'],reservation_sha256=None,storage_ref=ref(origin/'storage.json'),storage_locator=[],
 launch_ref=ref(origin/'actual_launch.root_v1.json'),launch_pid_locator=['launcher_identity','pid'],
 launch_identity={k:launch[k] for k in ['schema','root_supervisor_identity','launcher_identity','input_ref']},
 direct_wait_ref=ref(origin/'actual_wait.root_v1.json'),wait_pid_locator=['launcher_pid'],wait_exit_locator=['returncode'],
 wait_identity={k:wait[k] for k in ['schema','launcher_pid','launcher_start_ticks','actual_direct_wait_observed','returncode']},
 completed_ref=ref(out/'completed.json'),
 completed_identity={k:done[k] for k in ['schema','status','chromosome','counts','input_ref','bank_ref']})
def original_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
request['reservation_sha256']=original_digest(storage['reservation'])
request['observation_ref']=None
assert request==json.loads((HERE/'close_request.inactive.v1.json').read_text())
pids=[3248511, 3248512, 3248513, 3248514, 3248721, 3248683]
obs=[]
for pid in pids:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Original owned PID still present')
 obs.append(dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=datetime.now(timezone.utc).isoformat()))
try:os.killpg(3248721,0)
except ProcessLookupError:pass
else:raise RuntimeError('Original owned worker group still present')
for owned_group in [3248721,3248683]:
 try:os.killpg(owned_group,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Original shared bank worker group still present')
for k in ['storage_ref','launch_ref','direct_wait_ref','completed_ref']:
 assert ref(ROOT/request[k]['path'])==request[k]
observation=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=request['key'],reservation_sha256=request['reservation_sha256'],storage_ref=request['storage_ref'],launch_ref=request['launch_ref'],direct_wait_ref=request['direct_wait_ref'],completed_ref=request['completed_ref'],launcher_pid=3248513,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=obs,known_worker_PGID_ESRCH=3248721,original_separate_outer_tool_terminal_not_found=False)
observation.update(shared_parallel_banks_full_original_roster=True,known_shared_worker_PGIDs_ESRCH=[3248721,3248683],actual_tool_terminal_ref=ref(origin/'actual_tool_terminal.root_v1.json'),original_preparation_wait_ref=ref(origin/'preparation_wait.root_sealed_v1.json'),original_inner_minimal_wait_ref=ref(aux/'actual_wait_minimal.json'),original_inner_outer_wait_ref=ref(aux/'outer_wait.json'))
request['observation_ref']=write('targeted_root_observation.root_active_v1.json',observation)
cfg=json.loads((ROOT/'research/reviews/H056_maps_actual_root_20261016_v2/config.json').read_text());fixed=cfg['fixed_refs']
renewal=load(fixed['renewal'],'_H056_bank21_close_renewal');renewal.install_renewal_v1(fixed['renewal_certificate'])
guard=load(fixed['guard'],'_H056_bank21_close_original980');guard.configure_v2(fixed['repair_authority'],policy_ref=fixed['policy'])
b=json.loads((ROOT/'research/reviews/H056_maps_actual_root_20261016_v2/source_closure.json').read_text())['original_bindings']
kw=dict(expected_source_sha256=b['implementation']['sha256'],expected_launcher_sha256=b['launcher']['sha256'])
with guard.budget.v4.mutex(ROOT):
 state=guard._state_locked(kw['expected_source_sha256'],kw['expected_launcher_sha256'],fresh=True)
 guard.assert_open(request['key'],state[0],state[1]);assert state[3]==b
 assert guard.budget.digest(state[0][request['key']])==request['reservation_sha256']
 view=state[2]
auth=guard.operation_draft_v1('close_completed',request,expected_before=guard._pins(view),bindings=b)
auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat())
ar=write('close_authority.root_active_v1.json',auth)
receipt=guard.apply_close_completed_v1(ar,**kw)
write('close_receipt.actual_v1.json',receipt)
print(json.dumps({k:receipt[k] for k in ['key','retained_bytes','credit_bytes','projected_bytes','artifacts_deleted']}),flush=True)
