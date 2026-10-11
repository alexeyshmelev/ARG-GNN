"""Handmade metadata-only restoration fixtures, no tensors or actual payloads."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import ast,io,json
source=Path('src/arggnn/train_reference_logcavity_resume_v1.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='restore_states_v1')
namespace={'fit':SimpleNamespace(torch=SimpleNamespace(load=lambda stream,**kwargs:json.loads(stream.read()))),'io':io}
exec(compile(ast.Module(body=[node],type_ignores=[]),'restore_fixture','exec'),namespace)
restore=namespace['restore_states_v1']
def ref(name):return {'path':'handmade/'+name,'sha256':'0'*64}
def row(epoch):
 return {'arm':'static','epoch':epoch,'optimizer_steps':16*epoch,'checkpoint':ref(f'cp{epoch}'),
  'selected_epoch':1,'cells':[{'prediction_seal':ref(f'p{epoch}a')},{'prediction_seal':ref(f'p{epoch}b')}],
  'equal_parent_metrics':{'primary_present_day_64':{'NLL_is_infinite':False,'nll':.4}}}
rows=[row(1),row(2)];historyrefs=[ref('metrics1'),ref('metrics2')]
anchor={'last_complete_epoch':2,'restored_optimizer_steps':32,'history_metrics_refs':historyrefs,'latest_metrics_ref':historyrefs[-1],
 'latest_checkpoint_ref':rows[-1]['checkpoint'],'latest_probability_refs':[c['prediction_seal'] for c in rows[-1]['cells']],
 'prior_selected_epoch':1,'prior_selected_metrics_ref':historyrefs[0],'prior_selected_probability_refs':[c['prediction_seal'] for c in rows[0]['cells']]}
manifest={'arms':{'static':anchor}}
cfg={'resume_state_manifest_ref':ref('manifest'),'original_input_ref':ref('original-input'),'original_frozen_baseline_checkpoint':ref('teacher')}
saved={'epoch':2,'arm':'static','seed':101,'input_ref':cfg['original_input_ref'],'frozen_baseline_checkpoint':cfg['original_frozen_baseline_checkpoint'],
 'model_state':{'handmade_marker':1},'optimizer_state':{'state':{'handmade_head':{'step':32}},'param_groups':[{'lr':.001}]}}
class Reader:
 def decode_json(self,item):
  if item==cfg['resume_state_manifest_ref']:return deepcopy(manifest)
  return deepcopy(rows[historyrefs.index(item)])
 def read(self,item):return json.dumps(saved).encode() if item==rows[-1]['checkpoint'] else b'handmade-reference-only'
class StateHolder:
 def load_state_dict(self,value,**kwargs):self.value=deepcopy(value)
def state():return {'model':StateHolder(),'optimizer':StateHolder()}
states={'static':state()};restore(states,cfg,Reader())
assert states['static']['steps']==32 and states['static']['best_epoch']==1
assert states['static']['model'].value==saved['model_state']
assert states['static']['optimizer'].value==saved['optimizer_state']
assert states['static']['retained_best']==anchor['prior_selected_probability_refs']+[rows[0]['checkpoint']]
checks={'restores_model_and_AdamW_state_without_optimizer_update':True,'preserves_earliest_minimum_NLL_tie':True,'preserves_prior_selected_external_refs':True}
for field,bad in [('latest_probability_refs',[ref('wrong')]),('restored_optimizer_steps',31)]:
 old=deepcopy(anchor[field]);anchor[field]=bad
 try:restore({'static':state()},cfg,Reader());raise AssertionError('Invalid '+field+' accepted')
 except ValueError:pass
 anchor[field]=old
checks['rejects_invalid_latest_pair_and_optimizer_step_metadata']=True
real=json.loads(Path('research/reviews/reference_logcavity_resume_v1_author/resume_state_manifest.json').read_text())
epochs={arm:value['last_complete_epoch'] for arm,value in real['arms'].items()}
new={arm:0 for arm in epochs};schedule={}
for epoch in range(min(epochs.values())+1,11):
 active=[arm for arm in epochs if epochs[arm]<epoch];schedule[str(epoch)]=active
 for arm in active:epochs[arm]=epoch;new[arm]+=16
assert len(schedule['4'])==3 and len(schedule['5'])==5
assert new=={'carrier_gru':112,'carrier_matrix':112,'logcavity_only':96,'sham_ancestor':112,'static':96}
assert sum(new.values())==528
checks.update(per_arm_new_steps=new,total_new_steps=528,final_steps_per_arm={a:real['arms'][a]['restored_optimizer_steps']+new[a] for a in new},
 epoch4_completed_kernel_static_not_replayed=True,actual_data_or_checkpoint_payloads_read=False,GPU_used=False,optimizer_updates=0)
Path('research/reviews/reference_logcavity_resume_v1_author/metadata_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
