from pathlib import Path
import os,json,re,datetime,traceback
out=Path('research/reviews/user_progress_20261017_v5')
try:
 import torch
 torch.set_num_threads(1)
 base=Path('experiments/results/native_VAL000_verified22000_continuation_v12_root01_oct16/checkpoints')
 options=[p for p in base.iterdir() if re.fullmatch(r'ep\d+-ba\d+-rank0\.pt',p.name)]
 p=max(options,key=lambda p:int(re.search(r'-ba(\d+)-',p.name).group(1)))
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  before=os.fstat(f.fileno()); obj=torch.load(f,map_location='cpu',weights_only=False);after=os.fstat(f.fileno())
  assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  state=obj['state'];timestamp=state['timestamp']
  def scalar(v):
   if hasattr(v,'value'):v=v.value
   if isinstance(v,dict) and 'value' in v:v=v['value']
   return int(v)
  def field(k):return scalar(timestamp[k] if isinstance(timestamp,dict) else getattr(timestamp,k))
  vals=[];names=[]
  for name,opt in state['optimizers'].items():
   names.append(name)
   for v in opt['state'].values():
    if 'step' in v:vals.append(int(v['step']))
  step=field('batch');assert len(vals)==77 and set(vals)=={step}
  rec={'schema':'ROOT_native_v12_actual_saved_optimizer_progress_v5','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_path':str(p),'stable_open_readonly_FD_content_identity':[before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns],'metadata_before':{'nlink':before.st_nlink,'ctime_ns':before.st_ctime_ns},'metadata_after':{'nlink':after.st_nlink,'ctime_ns':after.st_ctime_ns},'saved_epoch':field('epoch'),'saved_batch':step,'saved_batch_in_epoch':field('batch_in_epoch'),'optimizer_parameter_state_count':len(vals),'optimizer_unique_step_values':sorted(set(vals)),'optimizer_names':names,'previous_verified_batch':91000,'new_saved_optimizer_updates':step-91000,'new_updates_since22000_resume':step-22000,'checkpoint_full_SHA_repeated':False,'full_scientific_training_complete':step>=227990,'ancestry_accuracy_available':False,'checkpoint_or_training_modified':False,'content_observed_through_one_O_RDONLY_O_NOFOLLOW_FD':True,'observation_uses_existing_native_interpreter':True}
  with (out/'saved_native_optimizer_progress.root_v1.json').open('x') as h:json.dump(rec,h,indent=2)
  print(json.dumps(rec),flush=True)
except BaseException as e:
 rec={'schema':'ROOT_native_progress_observation_error_v2','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'no_scientific_or_checkpoint_changes':True}
 with (out/'native_progress_observer_native_runtime_error.root_v1.json').open('x') as h:json.dump(rec,h,indent=2)
 print(json.dumps(rec),flush=True)
 raise
