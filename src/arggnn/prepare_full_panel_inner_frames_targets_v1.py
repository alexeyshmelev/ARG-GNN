from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, traceback
from . import prepare_full_panel_native_supervised_v1 as bridge
from .project_storage_full_reservation_v1 import validate_storage_v9, budget_check_v9
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='src/arggnn/prepare_full_panel_inner_frames_targets_v1.py'
PARENTS=('ap_full_train_000','ap_full_train_001','ap_full_train_002')
def decode(ref):
 return json.loads(bridge.checked(ref).read_text())
def run(record):
 spec=decode(record)
 assert spec['schema']=='full_panel_inner_frames_targets_input_v1'
 assert spec['root_authorizer']=='root_agent' and spec['execution_authorized'] is True and spec['status']=='active'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert set(spec['parents'])==set(PARENTS)
 cfg=decode(spec['refs']['config'])
 assert cfg['output_cap_bytes']==100000000 and cfg['stages']==['observed_frames','native_argmix_inner_targets']
 assert cfg['model_fitting'] is False and cfg['outer_access'] is False and cfg['GPU_used'] is False
 for ref in spec['source_refs']:bridge.checked(ref,observed=True)
 rt=decode(spec['refs']['runtime'])
 assert rt['job_id']==spec['job_id'] and rt['output_directory']==spec['output_directory']
 storage=decode(spec['refs']['storage'])
 validate_storage_v9(spec,cfg,storage,source_ref=bridge.record(ROOT/SOURCE),config_ref=spec['refs']['config'],runtime_ref=spec['refs']['runtime'])
 def guard(n):return budget_check_v9(storage,boundary=True,required_write_bytes=n)
 def save(name,obj):
  return bridge.write_json(out/name,obj,guard)
 out=bridge.new_directory(spec['output_directory'],guard)
 started=time.monotonic()
 save('started.json',dict(schema=spec['schema'],pid=os.getpid(),GPU_used=False,
  job_id=spec['job_id'],input_ref=record,config_ref=spec['refs']['config'],runtime_ref=spec['refs']['runtime']))
 frames={};targets={};observation_seals={}
 try:
  observations={}
  # All inner observations bind before any inner truth opens.
  for pid in PARENTS:
   item=spec['parents'][pid]
   obs=bridge.read_completed_parent(pid,item['inference'],item['completion'],require_argmix_native_ids=True)
   observations[pid]=obs
   frames[pid]=bridge.prepare_observed_frame(obs,out/'observed'/pid,guard=guard)
   observation_seals[pid]=frames[pid]['manifest']
  seal=save('all_inner_observed_frames_sealed.json',dict(frames=observation_seals,
   truth_arrays_opened=False,outer_access=False,queries_per_parent=686,pure_source_proxies_per_parent=266))
  for pid in PARENTS:
   targets[pid]=bridge.prepare_argmix_targets(observations[pid],spec['parents'][pid]['truth'],out/'targets'/pid,guard=guard)
  result=dict(schema='full_panel_inner_frames_targets_receipt_v1',
   status='completed_native_inner_frames_and_ARGMix_targets',input_ref=record,
   config_ref=spec['refs']['config'],hypothesis_id='H034',job_id=spec['job_id'],pid=os.getpid(),
   frames=frames,targets=targets,observation_seal=seal,source_refs=spec['source_refs'],
   actual_observations_read=True,actual_inner_truth_read=True,outer_truth_read=False,
   GPU_used=False,native_Rust_run=False,models_fitted=False,
   elapsed_seconds=time.monotonic()-started,scope_budget=guard(0))
  save('completed.json',result)
  print(json.dumps(dict(status=result['status'],pid=os.getpid(),elapsed_seconds=result['elapsed_seconds'],GPU_used=False)),flush=True)
 except Exception as error:
  save('failed.json',dict(status='failed_partial_preserved',error_type=type(error).__name__,error=str(error),
   elapsed_seconds=time.monotonic()-started))
  raise
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--execute',action='store_true')
 p.add_argument('--input')
 p.add_argument('--expected-input-sha256')
 args=p.parse_args()
 if not args.execute:
  print(json.dumps(dict(status='source_only_not_executed',source=bridge.record(ROOT/SOURCE),GPU_used=False)))
  return
 assert args.input and args.expected_input_sha256
 run(dict(path=args.input,sha256=args.expected_input_sha256))
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc(file=sys.stdout)
  raise SystemExit(1)
