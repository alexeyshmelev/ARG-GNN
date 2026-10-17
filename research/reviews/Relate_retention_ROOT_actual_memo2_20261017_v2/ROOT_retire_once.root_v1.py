from pathlib import Path
import json,hashlib,datetime,importlib,os,traceback,time,fcntl,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'research/reviews/Relate_retention_ROOT_actual_memo2_20261017_v2'
def ref(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def put(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
phase='construct'
try:
 assert not (OUT/'actual_retirement_receipt.root_v1.json').exists()
 wait=json.loads((OUT/'actual_preoperation_subprocess_wait.root_v1.json').read_text());assert wait['actual_returncode']==0
 before=json.loads((OUT/'actual_preoperation_state.root_v1.json').read_text())
 assert ref(OUT/'actual_preoperation_state.root_v1.json')['sha256']=='4d09cabbeb644c8bbe2b45d3af6c3abc214a2b6828ebbda5dba6b9899bf477a4'
 instruction=json.loads((OUT/'ROOT_instruction.root_active_v1.json').read_text())
 m=importlib.import_module('research.reviews.Relate_failed64GiB_retention_successor_source_20261017_v1.relate_failed64_retention_v1')
 g=m.load_selected_guard_v1(before['selector_ref'])
 h=importlib.import_module('research.reviews.Relate_retirement_transaction_memo_source_20261017_v1.retire_transaction_memo_v1')
 helper=h.RetireStoppedMemo(g,integration_manifest_ref=instruction['source_manifest_ref'],helper_ref=instruction['helper_ref'],review_ref=instruction['independent_review_ref'],memo_review_ref=instruction['memo_review_ref'],root_instruction_ref=ref(OUT/'ROOT_instruction.root_active_v1.json'),cache_refs=instruction['cache_refs'],uncached_current_refs=[])
 lock=g.budget.v4.canonical(ROOT,g.budget.v4.LOCK);a=lock.lstat();assert stat.S_ISREG(a.st_mode) and a.st_nlink==1
 fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
 try:
  b=os.fstat(fd);assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  c=lock.lstat();assert (b.st_dev,b.st_ino)==(c.st_dev,c.st_ino)
  probe={'path':str(lock.relative_to(ROOT)),'dev':b.st_dev,'inode':b.st_ino,'exclusive_nonblocking_probe_passed':True,'observed_utc':g._utc(),'retirement_does_not_claim_inherited_lock':True}
  fcntl.flock(fd,fcntl.LOCK_UN)
 finally:os.close(fd)
 put(OUT/'actual_queue_free_probe.root_v1.json',probe)
 key=instruction['key'];phase='fresh_evidence'
 rev=g.stopped_revocation_draft_v1(key);rev.update(active=True,revoked_at_utc=g._utc())
 put(OUT/'permanent_failed_key_revocation.root_active_v1.json',rev);revref=ref(OUT/'permanent_failed_key_revocation.root_active_v1.json')
 facts=g._observe_targets(g._target(key));obs=g.stopped_observation_draft_v1(key,revocation_ref=revref)
 obs.update(active=True,observed_at_utc=g._utc(),**facts)
 put(OUT/'fresh_known_targets_observation.root_active_v1.json',obs)
 request={'key':key,'reservation_sha256':before['actual_original_reservation_sha256'],'revocation_ref':revref,'observation_ref':ref(OUT/'fresh_known_targets_observation.root_active_v1.json')}
 auth=g.operation_draft_v1(g.RETIRE_OPERATION,request,expected_before=before['expected_before'],bindings=before['actual_bindings']);auth.update(active=True,expires_at_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=15)).isoformat())
 put(OUT/'retirement_authority.root_active_v1.json',auth);authority=ref(OUT/'retirement_authority.root_active_v1.json')
 policy=json.loads((ROOT/'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json').read_text())
 phase='actual_single_retirement';t0=time.monotonic();put(OUT/'ROOT_single_retirement_invocation.root_v1.json',{'schema':'ROOT_Relate64_single_retirement_invocation_v1','pid':os.getpid(),'time_utc':g._utc(),'authority_ref':authority,'original_deadline_seconds':300,'automatic_retry':False,'ROOT_authorizer':'root_agent'})
 print(json.dumps({'phase':phase,'pid':os.getpid(),'authority_ref':authority,'time_utc':g._utc()}),flush=True)
 receipt=helper.apply_retire_stopped_v1(authority,expected_source_sha256=policy['implementation_ref']['sha256'],expected_launcher_sha256=policy['launcher_ref']['sha256'])
 elapsed=time.monotonic()-t0
 assert receipt['kind']=='Relate64_exact_failed_retention_receipt_v1' and receipt['key']==key and receipt['artifacts_deleted']==0 and receipt['scientific_success_asserted'] is False
 put(OUT/'actual_retirement_receipt.root_v1.json',receipt)
 summary={'schema':'ROOT_Relate64_actual_operation_result_v1','returncode':0,'actual_operation_elapsed_seconds':elapsed,'receipt_ref':ref(OUT/'actual_retirement_receipt.root_v1.json'),'retained_bytes':receipt['retained_bytes'],'credit_bytes':receipt['credit_bytes'],'projected_bytes':receipt['projected_bytes'],'memo_counts':helper.counts,'ROOT_outer_terminal_and_final_state_acceptance_pending':True,'no_file_deleted_or_scientific_success_asserted':True}
 put(OUT/'actual_retirement_result.root_v1.json',summary);print(json.dumps(summary),flush=True)
except BaseException:
 text=traceback.format_exc();put(OUT/'actual_retirement_failure.root_v1.json',{'schema':'ROOT_Relate64_actual_retirement_failure_v1','phase':phase,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'traceback':text,'automatic_retry':False,'credit_not_accepted':True})
 print(text,flush=True);raise
