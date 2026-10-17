"""RAM-execute the complete actual parent function with explicit OS/process boundaries."""
from pathlib import Path
import ast,contextlib,copy,io,json,os as real_os,types
D=Path('research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3')
source=(D/'owner_parent_v1.py').read_text();tree=ast.parse(source)
checks=[];cases=[]
def need(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def execute_case(name,*,owner_rc=0,refresh_failure=False,reused_helper=False,gate_cancel=False,persistent_absence_failure=False,helper_completed_normally=False):
    state=dict(clock=0.,spawned=False,root_reaped=False,helper_live=False,helper_zombie=False,helper_reaped=helper_completed_normally,signals=[],subreaper=False,gate=False,adopted=False,refresh_failed=False,waitid_calls=0)
    class Clock:
        @staticmethod
        def time():return 1000.+state['clock']
        @staticmethod
        def monotonic():return state['clock']
        @staticmethod
        def sleep(seconds):state['clock']+=max(seconds,.001)
    leader=dict(pid=6000,start=101,ppid=5000,group=6000,session=6000,state='S')
    helper=dict(pid=6001,start=102,ppid=6000,group=6000,session=6000,state='S')
    class FakeTree:
        def __init__(self,pid):
            need(pid==6000 and state['subreaper'],'Original subreaper precedes exact-owned tree '+name)
            self.known={6000:copy.deepcopy(leader)}
        def refresh(self):
            if state['spawned'] and not state['helper_reaped']:
                self.known[6001]=copy.deepcopy(helper)
            if state['root_reaped']and persistent_absence_failure:raise OSError('RAM persistent tree refresh failure')
            if refresh_failure and state['root_reaped']and not state['refresh_failed']:
                state['refresh_failed']=True
                raise OSError('RAM helper metadata read failure')
            return self
        def identities_absent(self):
            if persistent_absence_failure:raise OSError('RAM persistent final absence failure')
            return state['root_reaped'] and (state['helper_reaped']or reused_helper)
        def evidence(self):return dict(owned_identities=[dict(pid=r['pid'],start=r['start'])for r in self.known.values()],signals_use_pidfd_only=True)
    def children(pid):
        if not state['spawned']:return []
        if not state['root_reaped']:return [6000]
        return [6001] if (state['helper_live']or state['helper_zombie'])and not reused_helper else []
    def signal_identity(row,signum):
        if row['pid']==6000 and state['root_reaped']:return False
        if row['pid']==6001:
            if reused_helper:
                state['signals'].append(dict(pid=6001,start=row['start'],sent=False,reason='reused excluded'));state['helper_live']=False
                return False
            state['signals'].append(dict(pid=6001,start=row['start'],sent=True,after_root_wait=state['root_reaped']))
            state['helper_live']=False;state['helper_zombie']=True;return True
        state['signals'].append(dict(pid=row['pid'],sent=True));return True
    owned=types.SimpleNamespace(_children=children,_stat=lambda pid:copy.deepcopy(leader),OwnedTree=FakeTree,signal_identity=signal_identity)
    backend=types.SimpleNamespace(self_probe_v1=lambda:dict(RAM_ONLY=True))
    def subreaper():state['subreaper']=True
    registry={}
    def load(record,label):
        if label=='pidfd_backend_v1':registry['pidfd_backend_v1']=backend;return backend
        if label=='_empty_owner_original_owned_lifecycle':return owned
        if label=='_empty_owner_original_subreaper':return types.SimpleNamespace(enable_child_subreaper=subreaper)
        raise AssertionError(label)
    cap=dict(drain_expires_unix=2000.,base_pool_ref=dict(path='RAM_BASE',sha256='x'),ROOT_record_directory='RAM_RECORDS')
    C=types.SimpleNamespace(HERE=str(D),ROOT=Path('/workspace-SR003.nfs2/ARG-GNN'),need=need,load=load,validate_cap_v1=lambda record:(cap,{},{},{}),guard_v1=lambda cap,storage:types.SimpleNamespace())
    class Pipe:
        def fileno(self):return 88
        def close(self):pass
    class Proc:
        pid=6000;stdout=Pipe();returncode=None
    def popen(command,**kwargs):
        need(state['subreaper'],'Subreaper enabled before actual Popen '+name)
        need(kwargs['start_new_session']is True and kwargs['pass_fds']==(10,),'Exact new session and closed gate '+name)
        state['spawned']=True;state['helper_live']=not reused_helper and not helper_completed_normally;return Proc()
    def waitid(*args):
        state['waitid_calls']+=1
        return None if state['waitid_calls']==1 else types.SimpleNamespace(si_pid=6000)
    def waitpid(pid,flags):
        if pid==6000:
            if state['root_reaped']:raise ChildProcessError()
            state['root_reaped']=True
            return 6000,owner_rc<<8
        need(pid==-1 and state['root_reaped'],'Adopted waits occur only after genuine root wait '+name)
        if state['helper_zombie']and not state['helper_reaped']:
            state['helper_reaped']=True;state['helper_zombie']=False;state['adopted']=True
            return 6001,15
        raise ChildProcessError()
    fakeos=types.SimpleNamespace(environ={},getpid=lambda:5000,pipe=lambda:(10,11),close=lambda fd:None,set_blocking=lambda fd,v:None,
        waitid=waitid,waitpid=waitpid,waitstatus_to_exitcode=real_os.waitstatus_to_exitcode,P_PID=1,WEXITED=4,WNOHANG=1,WNOWAIT=16777216,
        read=lambda fd,n:b'',write=lambda fd,b:state.update(gate=True)or 1)
    fakesignal=types.SimpleNamespace(SIGTERM=15,SIGINT=2,SIGKILL=9,signal=lambda s,h:None)
    fakeselect=types.SimpleNamespace(select=lambda r,w,x,t:(state.update(clock=state['clock']+.1)or r,w,x))
    fakesys=types.SimpleNamespace(dont_write_bytecode=True,flags=types.SimpleNamespace(no_user_site=1,optimize=0),modules=registry)
    env={'__name__':'RAM_PARENT','c':C}
    nodes=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.If))]
    env.update(dict(Path=Path,argparse=None,hashlib=None,json=json,os=fakeos,select=fakeselect,signal=fakesignal,subprocess=types.SimpleNamespace(Popen=popen,PIPE=-1,STDOUT=-2),sys=fakesys,time=Clock))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_COMPLETE_ACTUAL_PARENT','exec',dont_inherit=True,optimize=0),env)
    output={}
    def put(cap,name,value):
        output[name]=copy.deepcopy(value)
        return dict(path='RAM_ONLY/'+name,sha256='RAM_ONLY')
    env['put']=put
    with contextlib.redirect_stdout(io.StringIO()):rc=env['execute_v1'](dict(path='RAM_CAP',sha256='RAM_ONLY'))
    outcome=output['actual_wait.json']
    if 'actual_launch.json'not in output:raise AssertionError(outcome)
    need(output['actual_launch.json']['controller_start_ticks']==101,'Original start mapped to controller_start_ticks '+name)
    need(outcome['controller_start_ticks']==101,'Actual outcome start mapping '+name)
    need(state['gate'],'Launch gate released only after identity record '+name)
    need(outcome['all_owned_processes_drained']is (not persistent_absence_failure),'Truthful current owned drain or uncertainty '+name)
    need(outcome['actual_raw_wait']==owner_rc<<8 and outcome['actual_direct_returncode']==owner_rc,'Genuine owner wait preserved '+name)
    need(outcome['actual_tool_outer_terminal_consumed']is False,'External tool wait not invented '+name)
    if helper_completed_normally:
        need(outcome['adopted_wait_count']==0,'Normally reaped inventory helpers need no manufactured adopted wait '+name)
    elif reused_helper:
        need(not any(x['pid']==6001 and x['sent']for x in state['signals']),'Reused helper never signaled '+name)
    else:
        need(any(x['pid']==6001 and x['sent']and x.get('after_root_wait')for x in state['signals']),'Helper pidfd cleanup continues after direct owner wait '+name)
        need(state['adopted']and outcome['adopted_wait_count']==1 and outcome['adopted_waits'][0]['returncode']==-15,'Actual adopted helper wait captured '+name)
    if persistent_absence_failure:
        need(outcome['failure']is not None and rc==1 and outcome['actual_raw_wait']==owner_rc<<8,'Persistent absence failure retains genuine direct wait '+name)
    elif refresh_failure:
        need(state['refresh_failed']and len(outcome['cleanup_errors'])==1 and outcome['failure']is not None and rc==1,'Refresh uncertainty remains failure while helper drains '+name)
    else:need(rc==(0 if owner_rc==0 and(reused_helper or helper_completed_normally)else 1),'Truthful outer return including adopted-helper failure '+name)
    cases.append(dict(name=name,passed=True,returncode=rc,helper_signal_after_root_exit=any(x.get('after_root_wait')and x.get('sent')for x in state['signals']),adopted_wait_count=outcome['adopted_wait_count'],owner_direct_returncode=owner_rc))
execute_case('owner0_inventory_helpers_completed',helper_completed_normally=True)
execute_case('owner0_helper_still_live')
execute_case('owner1_helper_still_live',owner_rc=1)
execute_case('owner0_refresh_error_retained_helper_cleanup',refresh_failure=True)
execute_case('owner0_reused_helper_identity_excluded',reused_helper=True)
execute_case('owner0_persistent_absence_failure_retains_wait',persistent_absence_failure=True)
result=dict(schema='ROOT_empty_owner_v2_complete_parent_RAM_lifecycle_checks_v1',passed=True,assertions=len(checks),cases=cases,checks=checks,actual_processes_or_pidfd_or_signals_or_guards=0,independent_review_claimed=False,all_OS_process_boundaries_are_RAM_fixtures=True)
print(json.dumps(result,sort_keys=True,indent=2))
