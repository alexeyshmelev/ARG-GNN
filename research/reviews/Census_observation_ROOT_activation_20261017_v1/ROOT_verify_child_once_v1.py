TERMINAL_REF={'path': 'research/reviews/research_execution_ROOT_20261017_v27/Census_observation_activation_actual_outer_consumed.root_v1.json', 'sha256': 'fd6516a57a5f92ddd716eaf639bdeaa713ad8a8e62f7c728ae1b3b042df7da38'}
SELECTOR_REF={'path': 'research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json', 'sha256': 'fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4'}
from pathlib import Path
import json,hashlib,importlib,datetime,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
P=ROOT/'research/reviews/Census_observation_ROOT_activation_20261017_v1'
def ref(n):
 return dict(path=n,sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest())
def read(r):
 assert ref(r['path'])==r
 return json.loads((ROOT/r['path']).read_text())
try:
 assert Path.cwd()==ROOT and sys.dont_write_bytecode
 terminal=read(TERMINAL_REF)
 assert terminal['actual_terminal_consumed'] is True and terminal['actual_outer_exit_code']==0 and terminal['tool_session_id']==88986
 pub=read(terminal['activation_publication_ref'])
 assert pub['actual_credit_bytes']==0
 selref=pub['selector_ref'];assert selref==SELECTOR_REF
 cap=json.loads((P/'actual_parent_capture.root_v1.json').read_text())
 g=importlib.import_module('research.reviews.Census_observation_two_scope_retention_source_20261017_v1.census_observation_retention_v1')
 assert g.load_selected_guard_v1(selref) is g
 pol=read(pub['policy_ref'])
 with g.budget.v4.mutex(ROOT):
  state=g._state_locked(pol['implementation_ref']['sha256'],pol['launcher_ref']['sha256'],fresh=True)
  assert state[2]['ancestor_projected_bytes']==cap['actual_parent_projected_bytes']
  assert state[2]['projected_bytes']==cap['actual_parent_projected_bytes']+g.BOOTSTRAP_CHARGE
  assert state[2]['base_pins']==cap['actual_parent_pins']
  assert state[2]['completed_overlay_prefix']==dict(events=0,tail_sha256=None)
  assert state[2]['completed_credit_bytes']==0 and state[2]['new_reserved_bytes']==0
  assert all(state[3].get(k)==v for k,v in cap['all_actual_parent_bindings'].items())
  assert len(state[3])==len(cap['all_actual_parent_bindings'])+1==47
  for k in g.TARGET_KEYS:
   assert k in state[0] and k not in state[1] and state[0][k]['reserved_bytes']==20_000_000_000
  value=dict(schema='ROOT_Census_observation_fresh_child_verification_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selector_ref=selref,policy_ref=pub['policy_ref'],activation_terminal_ref=TERMINAL_REF,all_parent_pins_and_46roles_preserved=True,actual_direct_guard_roles=len(state[3]),projected_commitment_bytes=state[2]['projected_bytes'],headroom_to_980GB_bytes=980_000_000_000-state[2]['projected_bytes'],physical_disk_usage_measured=False,activation_bootstrap_bytes=g.BOOTSTRAP_CHARGE,actual_credit_bytes=0,both_Census_original_20GB_keys_still_open=True,existing_jobs_and_original_guards_unchanged=True)
  p=P/'fresh_child_verification.root_v1.json'
  with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(value),flush=True)
except Exception:
 print(traceback.format_exc(),flush=True);raise SystemExit(1)
