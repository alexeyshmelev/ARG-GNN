"""Inactive one-shot source. Only explicit CLI refs in reviewed ROOT staging can run it."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,stat,sys,threading,traceback,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
DIRECTORY='research/reviews/Census_retention_inline_ROOT_actual_numerical_postfreeze_source_20261017_v1'
KEY='census_tsinfer_observation_v1/census-tsinfer-numerical-v1'
CALLBACK=DIRECTORY+'/ROOT_authorize_inline_retirement.root_v1.py'
SOURCE=DIRECTORY+'/ROOT_retire_once.root_v1.py'
INSTRUCTION=DIRECTORY+'/ROOT_instruction.root_active_v1.json'
INTENT=DIRECTORY+'/ROOT_actual_operation_intent.root_active_v1.json'
MANIFEST=DIRECTORY+'/integration_manifest.root_v1.json'
REVIEW=DIRECTORY+'/whole_join_review.root_v1.json'
NAMES={'invocation':DIRECTORY+'/ROOT_actual_retirement_invocation.root_v1.json',
    'receipt':DIRECTORY+'/actual_retirement_receipt.root_v1.json',
    'result':DIRECTORY+'/actual_retirement_result.root_v1.json',
    'failure':DIRECTORY+'/actual_retirement_failure.root_v1.json'}
FRESH={'attempt':DIRECTORY+'/inline_attempt.root_v1.json',
    'observation':DIRECTORY+'/known_target_observation.root_active_v1.json',
    'authority':DIRECTORY+'/retirement_authority.root_active_v1.json'}
GUARD={'path':'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py','sha256':'72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb'}
HELPER={'path':'research/reviews/Census_retention_inline_source_20261017_v1/inline_ROOT_retire_v1.py','sha256':'bc4235d3dcedb0b3ed9058cd70826c47eb3f243b132dac26dcb232d84f488d21'}
SELECTOR={'path':'research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4'}
POLICY={'path':'research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json','sha256':'5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5'}
REVOCATION={'path':'research/reviews/Census_numerical_permanent_revocation_ROOT_20261017_v1/permanent_writer_revocation.root_active_v1.json','sha256':'c7a4411960259dc536e25bbc2b0f19f4540301d9186413b0eb91b5a9e5cb019e'}
IMPLEMENTATION_SHA='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
LAUNCHER_SHA='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
def need(ok,message):
    if not ok:
        raise ValueError(message)
def stamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    need(result.tzinfo is not None,'Timezone-aware ROOT finite instruction')
    return result
def raw(record,cap=1048576):
    need(isinstance(record,dict) and set(record)=={'path','sha256'},'Exact explicit immutable ref')
    relative=Path(record['path'])
    need(not relative.is_absolute() and str(relative)==record['path']
        and '..' not in relative.parts,'Repository-relative exact path')
    path=ROOT/relative
    need(path.resolve()==path,'No control/source symlinks')
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap,'Bounded regular immutable ref')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        opened=os.fstat(stream.fileno())
        data=stream.read(cap+1)
        after=os.fstat(stream.fileno())
    identity=lambda v:(v.st_dev,v.st_ino,v.st_mode,v.st_nlink,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    need(identity(before)==identity(opened)==identity(after)==identity(path.lstat())
        and len(data)==before.st_size and hashlib.sha256(data).hexdigest()==record['sha256'],
        'Stable exact authenticated ref bytes')
    return data
def document(record):
    return json.loads(raw(record))
def put(kind,value):
    path=ROOT/NAMES[kind]
    need(path.parent.resolve()==path.parent and path.parent.is_dir(),'Exact existing ROOT staging directory')
    data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode('utf-8')
    need(len(data)<=65536,'Bounded compact operation metadata')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        need(stream.write(data)==len(data),'Full metadata write')
        stream.flush()
        os.fsync(stream.fileno())
        opened=os.fstat(stream.fileno())
    named=path.lstat()
    need(stat.S_ISREG(opened.st_mode) and stat.S_ISREG(named.st_mode)
        and opened.st_nlink==named.st_nlink==1 and opened.st_size==named.st_size==len(data)
        and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino),'Immutable operation metadata name')
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path':NAMES[kind],'sha256':hashlib.sha256(data).hexdigest()}
def load(record,name):
    need(name not in sys.modules,'Fresh source module only')
    code=raw(record)
    module=types.ModuleType(name)
    module.__file__=str(ROOT/record['path'])
    sys.modules[name]=module
    exec(compile(code,module.__file__,'exec',dont_inherit=True,optimize=0),module.__dict__)
    return module
def controls(instruction_ref,intent_ref):
    need(instruction_ref['path']==INSTRUCTION and intent_ref['path']==INTENT,'Exact separately explicit ROOT controls')
    instruction=document(instruction_ref)
    intent=document(intent_ref)
    need(instruction['schema']=='ROOT_exact_Census_inline_evidence_instruction_v1'
        and instruction['active'] is True and instruction['root_authorizer']=='root_agent'
        and instruction['key']==KEY and instruction['metadata_directory']==DIRECTORY
        and instruction['guard_ref']==GUARD and instruction['helper_ref']==HELPER
        and instruction['policy_ref']==POLICY and instruction['selector_ref']==SELECTOR
        and instruction['permanent_revocation_ref']==REVOCATION
        and instruction['fresh_metadata_paths']==FRESH
        and instruction['callback_source_ref']['path']==CALLBACK
        and instruction['source_manifest_ref']['path']==MANIFEST
        and instruction['whole_join_review_ref']['path']==REVIEW
        and instruction['operation_wall_seconds']==300 and instruction['fresh_target_seconds']==120
        and isinstance(instruction['expected_before'],dict)
        and isinstance(instruction['expected_guard_roles'],dict)
        and len(instruction['expected_guard_roles'])==47,'Exact numerical integration and explicit ROOT prefix/all47roles')
    now=datetime.now(timezone.utc)
    need(stamp(instruction['created_at_utc'])<=now<stamp(instruction['expires_at_utc']),'Finite unchanged ROOT deadline')
    manifest=document(instruction['source_manifest_ref'])
    source_ref={'path':SOURCE,'sha256':manifest['source_sha256'][SOURCE]}
    expected=dict(schema='ROOT_one_Census_numerical_inline_actual_operation_intent_v1',
        active=True,root_authorizer='root_agent',key=KEY,metadata_directory=DIRECTORY,
        instruction_path=INSTRUCTION,source_manifest_ref=instruction['source_manifest_ref'],
        whole_join_review_ref=instruction['whole_join_review_ref'],orchestration_source_ref=source_ref,
        callback_source_ref=instruction['callback_source_ref'],fresh_metadata_paths=FRESH,
        outcome_metadata_paths=NAMES,permanent_revocation_ref=REVOCATION,
        expected_before=instruction['expected_before'],expected_guard_roles=instruction['expected_guard_roles'],
        created_at_utc=instruction['created_at_utc'],expires_at_utc=instruction['expires_at_utc'],
        ROOT_authorizes_one_actual_original_Census_retirement=True,
        automatic_retry=False,prefix_discovery=False,credit_accepted=False,
        failed29922_and_all_old_files_charges_clocks_and_unknown_history_preserved=True)
    need(intent==expected and set(intent)==set(expected),'Genuine explicit one-shot ROOT intent with no fallback')
    need(intent_ref in instruction['uncached_current_refs'],'Actual intent remains uncached in original helper transaction')
    review=document(instruction['whole_join_review_ref'])
    need(review.get('independent') is True and review.get('verdict')=='SOURCE_PASS'
        and review.get('blockers')==[] and review.get('actual_callback_complete_source_review') is True
        and review.get('orchestration_complete_source_review') is True
        and review.get('orchestration_source_ref')==source_ref
        and review.get('outcome_metadata_paths')==NAMES
        and review.get('actual_intent_path')==INTENT,'Independent whole review includes actual orchestration and immutable names')
    need(manifest['source_sha256'].get(CALLBACK)==instruction['callback_source_ref']['sha256']
        and manifest['source_sha256'].get(GUARD['path'])==GUARD['sha256']
        and manifest['source_sha256'].get(HELPER['path'])==HELPER['sha256'],'Exact actual code closure')
    raw(source_ref)
    raw(instruction['callback_source_ref'])
    raw(GUARD)
    raw(HELPER)
    raw(REVOCATION)
    selector=document(SELECTOR)
    policy=document(POLICY)
    need(selector['source_ref']==GUARD and selector['policy_ref']==POLICY
        and policy['source_ref']==GUARD and policy['active'] is True
        and policy['implementation_ref']['sha256']==IMPLEMENTATION_SHA
        and policy['launcher_ref']['sha256']==LAUNCHER_SHA,'Exact current Census child; no old guard fallback')
    return instruction
def run_once(*,instruction_ref,intent_ref):
    need(sys.flags.optimize==0 and sys.flags.no_user_site==1 and sys.dont_write_bytecode
        and threading.active_count()==1,'Fresh -B -s nonoptimized single-thread ROOT interpreter')
    need(Path(__file__).resolve()==ROOT/SOURCE and Path.cwd()==ROOT,'Only exact reviewed ROOT-staged source and repository')
    need('scripts.project_budget_v9' not in sys.modules,'Select before any original guard or memo construction')
    instruction=controls(instruction_ref,intent_ref)
    # Exclusive invocation consumption precedes imports/configuration; any uncertainty stops this attempt.
    put('invocation',dict(schema='Census_numerical_inline_actual_invocation_v1',
        instruction_ref=instruction_ref,intent_ref=intent_ref,key=KEY,
        pid=os.getpid(),created_at_utc=datetime.now(timezone.utc).isoformat(),
        automatic_retry=False,credit_accepted=False))
    phase='configure_exact_current_Census'
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        guard=load(GUARD,'_ROOT_actual_Census_numerical_guard_v1')
        guard=guard.load_selected_guard_v1(SELECTOR)
        phase='construct_exact_reviewed_inline_helper'
        helper_module=load(HELPER,'_ROOT_actual_Census_numerical_inline_helper_v1')
        helper=helper_module.InlineRootRetireStoppedMemo(guard,root_instruction_ref=instruction_ref)
        phase='one_actual_original_mutex_retirement'
        receipt=helper.apply_retire_stopped_v1(expected_source_sha256=IMPLEMENTATION_SHA,
            expected_launcher_sha256=LAUNCHER_SHA)
        phase='persist_actual_receipt_pending_ROOT_final_acceptance'
        receipt_ref=put('receipt',receipt)
        result=dict(schema='Census_numerical_inline_actual_outcome_v1',key=KEY,
            instruction_ref=instruction_ref,intent_ref=intent_ref,receipt_ref=receipt_ref,
            receipt_kind=receipt.get('kind'),retained_bytes=receipt.get('retained_bytes'),
            receipt_credit_bytes=receipt.get('credit_bytes'),projected_bytes=receipt.get('projected_bytes'),
            memo_counts=helper.counts,automatic_retry=False,credit_accepted=False,
            ROOT_outer_wait_consumption_and_final_original_journal_acceptance_pending=True,
            scientific_success_relabelled=False,files_deleted=False)
        put('result',result)
        return result
    except BaseException:
        failure=dict(schema='Census_numerical_inline_actual_failure_v1',phase=phase,
            instruction_ref=instruction_ref,intent_ref=intent_ref,key=KEY,
            created_at_utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()[-16384:],
            automatic_retry=False,credit_accepted=False,
            append_or_late_failure_requires_original_journal_audit=True)
        try:
            put('failure',failure)
        except BaseException:
            print(json.dumps({'failure_metadata_persistence_uncertain':True,
                'automatic_retry':False,'credit_accepted':False}),flush=True)
        raise
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-root-reviewed-once',action='store_true')
    parser.add_argument('--instruction-path')
    parser.add_argument('--instruction-sha256')
    parser.add_argument('--intent-path')
    parser.add_argument('--intent-sha256')
    args=parser.parse_args(argv)
    if not args.execute_root_reviewed_once:
        print(json.dumps({'status':'INACTIVE_SOURCE_ONLY','operations':0}))
        return 0
    need(all((args.instruction_path,args.instruction_sha256,args.intent_path,args.intent_sha256)),
        'Separate explicit ROOT instruction and intent refs required')
    result=run_once(instruction_ref={'path':args.instruction_path,'sha256':args.instruction_sha256},
        intent_ref={'path':args.intent_path,'sha256':args.intent_sha256})
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
