"""ROOT stops only two authenticated obsolete H060 supervisors through pidfds."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,signal
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
P=ROOT/'research/reviews/H060_obsolete_preparation_stop_ROOT_20261017_v2'
backend_path=ROOT/'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py'
assert hashlib.sha256(backend_path.read_bytes()).hexdigest()=='61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'
sp=importlib.util.spec_from_file_location('_ROOT_existing_pidfd_backend',backend_path)
backend=importlib.util.module_from_spec(sp);sp.loader.exec_module(backend)
P.mkdir(exist_ok=False)
def now():return datetime.now(timezone.utc).isoformat()
def ref(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest()}
def obs(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),ppid=int(v[1]),uid=p.stat().st_uid,state=v[0],argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
targets=[
dict(tool=36502,packet='research/reviews/H060_isolated_CPU_ROOT_actual_tx_20261017_v1',worker=3889463,worker_start=487457986,supervisor=3888134,supervisor_start=487439694),
dict(tool=97638,packet='research/reviews/H060_isolated_CPU_ROOT_actual_fast_observed_root01_oct17_v1',worker=3965209,worker_start=488505215,supervisor=3964320,supervisor_start=488492595)]
directive=dict(schema='ROOT_H060_obsolete_preparation_supervisor_stop_v1',root_authorizer='root_agent',utc=now(),reason='Keep the current full legacy-fast implementation and stop obsolete duplicate preparation implementations competing on the same32CPUaffinity. No model budget is shortened in the retained run.',targets=targets,retained_worker=4012891,retained_tool=1723,retained_full_additional_updates=1200,signals='One SIGTERM through authenticated supervisor pidfd only. Original supervisor retains owned cleanup.',all_files_charges_clocks_and_failures_preserved=True,original_final_optimizer_totals_unknown=True,no_truth_or_score=True,no_accounting_credit=True,old_future_launch_resume_routes_revoked=True)
(P/'ROOT_directive.root_active_v1.json').write_text(json.dumps(directive,indent=2,sort_keys=True)+'\n')
events=[]
for t in targets:
 packet=ROOT/t['packet'];inp_ref=ref(packet/'input.json');inp=json.loads((packet/'input.json').read_text());launch=json.loads((packet/'worker_aux/launch.json').read_text())
 assert launch['input_ref']==inp_ref and launch['pid']==t['worker']
 w=obs(t['worker']);s=obs(t['supervisor'])
 assert w['start_ticks']==t['worker_start'] and w['ppid']==t['supervisor'] and w['uid']==os.getuid()
 assert s['start_ticks']==t['supervisor_start'] and s['uid']==os.getuid() and s['state'] not in ('Z','X')
 argv=s['argv'];assert '--execute-ref' in argv and '-m' in argv
 assert argv[argv.index('-m')+1]==inp['root_driver_ref']['path'][:-3].replace('/','.')
 assert json.loads(argv[argv.index('--execute-ref')+1])==inp_ref
 fd=backend.pidfd_open_v1(t['supervisor'],0)
 try:
  again=obs(t['supervisor']);assert again['start_ticks']==s['start_ticks'] and again['uid']==s['uid'] and again['argv']==s['argv']
  ev=dict(utc=now(),input_ref=inp_ref,launch_ref=ref(packet/'worker_aux/launch.json'),worker_before=w,supervisor_before=s,signal='SIGTERM',signal_method='backend.pidfd_send_signal_v1')
  with (P/(str(t['tool'])+'.pre_signal.root_v1.json')).open('x') as f:json.dump(ev,f,indent=2,sort_keys=True);f.write('\n')
  backend.pidfd_send_signal_v1(fd,signal.SIGTERM,None,0)
  ev.update(signal_sent=True,tool_terminal_and_owned_drain_pending=True)
  with (P/(str(t['tool'])+'.signal_sent.root_v1.json')).open('x') as f:json.dump(ev,f,indent=2,sort_keys=True);f.write('\n')
  events.append(dict(tool=t['tool'],supervisor=t['supervisor'],signal_sent=True))
 finally:os.close(fd)
print(json.dumps(dict(events=events,retained_tool1723_untouched=True,old_actual_terminals_still_required=True)),flush=True)
