"""ROOT verifies the genuine close in the original selected journal, metadata only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
P=ROOT/'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2'
def ref(p):
 raw=(ROOT/p).read_bytes()
 return {'path':p,'sha256':hashlib.sha256(raw).hexdigest()}
def load(r):
 raw=(ROOT/r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];return json.loads(raw)
def save(name,v):
 p=P/name
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 return ref(str(p.relative_to(ROOT)))
sr={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
sel=load(sr);assert ref(sel['source_ref']['path'])==sel['source_ref']
sp=importlib.util.spec_from_file_location('_ROOT_FLARE_close_journal_acceptance',ROOT/sel['source_ref']['path']);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
g.configure_with_renewal_v1(sel['repair_authority_ref'],policy_ref=sel['policy_ref'],selector_ref=sr)
cr={'path':'research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/ordinary_close_receipt.root_v1.json','sha256':'3da38af704512caa5368d013cd4b4febaf135b593c8de06834515552954c2a7b'}
receipt=load(cr)
with g.budget.v4.mutex(ROOT):
 state=g._state_locked('483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e','a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3',fresh=True)
 assert receipt['key'] in state[1] and state[1][receipt['key']]==receipt['value']
 g.check_prefix(receipt['completed_overlay_prefix'],state[5])
 inventory=[{'key':k,'reserved_bytes':v['reserved_bytes'],'output':v['output'],'auxiliary_scopes':v.get('auxiliary_scopes',[]),'status':'open_accounting_reservation_not_process_liveness'} for k,v in state[0].items() if k not in state[1] and v['reserved_bytes']>=5000000000]
 inventory.sort(key=lambda x:x['reserved_bytes'],reverse=True)
 result={'schema':'ROOT_FLARE_VAL001_actual_close_final_journal_acceptance_v1','utc':datetime.now(timezone.utc).isoformat(),'close_tool_session_id':96129,'actual_close_outer_exit_code':0,'actual_close_terminal_consumed':True,'receipt_ref':cr,'original_journal_value_exact':True,'original_close_prefix_is_accepted_ancestor':True,'current_selected_prefix':g._pins(state[2]),'current_view':state[2],'direct_guard_roles':len(state[3]),'files_deleted':0,'new_scientific_score':None}
 ar=save('final_original_journal_acceptance.root_v1.json',result)
 ir=save('large_open_reservation_metadata.root_v1.json',{'schema':'ROOT_large_open_reservation_metadata_v1','observed_utc':result['utc'],'source_acceptance_ref':ar,'scopes':inventory,'no_file_measurement_or_liveness_claim':True,'not_credit_authority':True})
 print(json.dumps({'final_acceptance_ref':ar,'large_open_reservation_inventory_ref':ir,'large_open_scopes':len(inventory),'original_journal_close_accepted':True}),flush=True)
