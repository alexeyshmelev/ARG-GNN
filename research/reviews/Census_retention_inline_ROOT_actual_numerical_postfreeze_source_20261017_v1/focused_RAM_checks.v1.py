"""Source-only author RAM checks. Never loads operational modules or controls."""
from pathlib import Path
from datetime import datetime as RealDatetime, timezone
import argparse, ast, contextlib, copy, hashlib, io, json

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
DIRECTORY='research/reviews/Census_retention_inline_ROOT_actual_numerical_postfreeze_source_20261017_v1'
OLD='research/reviews/Census_retention_inline_ROOT_actual_numerical_root01_oct17'
D=ROOT/DIRECTORY
results=[]
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    results.append({'name':name,'passed':True})
def read(relative):
    return (ROOT/relative).read_bytes()
def sha(data):
    return hashlib.sha256(data).hexdigest()
manifest=json.loads((D/'integration_manifest.root_v1.json').read_bytes())
for name in ('ROOT_authorize_inline_retirement.root_v1.py','ROOT_retire_once.root_v1.py'):
    old=read(OLD+'/'+name)
    new=read(DIRECTORY+'/'+name)
    normalized=new.replace(DIRECTORY.encode(),OLD.encode())
    check(name+': exact sole namespace replacement',new==old.replace(OLD.encode(),DIRECTORY.encode()) and normalized==old)
    check(name+': normalized AST identity',ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False))
    compile(new,str(D/name),'exec',dont_inherit=True,optimize=0)
    check(name+': source compiles',True)
check('complete closure contains exactly226 entries',len(manifest['source_sha256'])==226)
check('closure excludes runtime journal paths',not any(p.startswith('.runtime/') for p in manifest['source_sha256']))
mismatches=[]
for relative,expected in manifest['source_sha256'].items():
    actual=sha(read(relative))
    if actual!=expected:
        mismatches.append({'path':relative,'expected':expected,'actual':actual})
check('all226 pinned source and metadata hashes authenticate',mismatches==[])
oldmanifest=json.loads(read(OLD+'/integration_manifest.root_v1.json'))
check('all original dependency pins retained exactly',all(manifest['source_sha256'].get(p)==s for p,s in oldmanifest['source_sha256'].items()))
source=read(DIRECTORY+'/ROOT_retire_once.root_v1.py')
tree=ast.parse(source)
allowed_names={'need','stamp','controls','main'}
nodes=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.Assign)) or isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in allowed_names]
check('RAM extraction excludes operational entry and all file/process helpers',set(node.name for node in nodes if isinstance(node,ast.FunctionDef))==allowed_names)
env={'__name__':'_RAM_ONLY'}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_AST_EXTRACTED_RUNNER','exec',dont_inherit=True,optimize=0),env)
operation_calls=[]
def forbidden(*args,**kwargs):
    operation_calls.append(True)
    raise AssertionError('Operational call forbidden in source-only RAM harness')
env['run_once']=forbidden
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    status=env['main']([])
check('default main returns inactive JSON without operational entry',status==0 and json.loads(buf.getvalue())=={'status':'INACTIVE_SOURCE_ONLY','operations':0} and operation_calls==[])
class FixedDatetime:
    @staticmethod
    def now(tz):
        return RealDatetime(2026,10,17,12,0,0,tzinfo=timezone.utc)
    @staticmethod
    def fromisoformat(value):
        return RealDatetime.fromisoformat(value)
env['datetime']=FixedDatetime
def ref(path,label):
    return {'path':path,'sha256':sha(('RAM_ONLY:'+label).encode())}
instruction_ref=ref(env['INSTRUCTION'],'instruction')
intent_ref=ref(env['INTENT'],'intent')
manifest_ref={'path':env['MANIFEST'],'sha256':sha((D/'integration_manifest.root_v1.json').read_bytes())}
review_ref=ref(env['REVIEW'],'review')
source_ref={'path':env['SOURCE'],'sha256':manifest['source_sha256'][env['SOURCE']]}
callback_ref={'path':env['CALLBACK'],'sha256':manifest['source_sha256'][env['CALLBACK']]}
instruction={
    'schema':'ROOT_exact_Census_inline_evidence_instruction_v1','active':True,'root_authorizer':'root_agent',
    'key':env['KEY'],'metadata_directory':DIRECTORY,'guard_ref':env['GUARD'],'helper_ref':env['HELPER'],
    'policy_ref':env['POLICY'],'selector_ref':env['SELECTOR'],'permanent_revocation_ref':env['REVOCATION'],
    'fresh_metadata_paths':env['FRESH'],'callback_source_ref':callback_ref,'source_manifest_ref':manifest_ref,
    'whole_join_review_ref':review_ref,'operation_wall_seconds':300,'fresh_target_seconds':120,
    'expected_before':{'RAM_ONLY':True,'prefix':'SYNTHETIC_NOT_ACTUAL'},
    'expected_guard_roles':{'RAM_ROLE_%02d'%i:{'RAM_ONLY':True} for i in range(47)},
    'created_at_utc':'2026-10-17T11:59:00+00:00','expires_at_utc':'2026-10-17T12:01:00+00:00',
    'uncached_current_refs':[intent_ref]}
intent=dict(schema='ROOT_one_Census_numerical_inline_actual_operation_intent_v1',
    active=True,root_authorizer='root_agent',key=env['KEY'],metadata_directory=DIRECTORY,
    instruction_path=env['INSTRUCTION'],source_manifest_ref=manifest_ref,whole_join_review_ref=review_ref,
    orchestration_source_ref=source_ref,callback_source_ref=callback_ref,fresh_metadata_paths=env['FRESH'],
    outcome_metadata_paths=env['NAMES'],permanent_revocation_ref=env['REVOCATION'],
    expected_before=instruction['expected_before'],expected_guard_roles=instruction['expected_guard_roles'],
    created_at_utc=instruction['created_at_utc'],expires_at_utc=instruction['expires_at_utc'],
    ROOT_authorizes_one_actual_original_Census_retirement=True,automatic_retry=False,
    prefix_discovery=False,credit_accepted=False,
    failed29922_and_all_old_files_charges_clocks_and_unknown_history_preserved=True)
review={'independent':True,'verdict':'SOURCE_PASS','blockers':[],
    'actual_callback_complete_source_review':True,'orchestration_complete_source_review':True,
    'orchestration_source_ref':source_ref,'outcome_metadata_paths':env['NAMES'],'actual_intent_path':env['INTENT']}
selector={'source_ref':env['GUARD'],'policy_ref':env['POLICY']}
policy={'source_ref':env['GUARD'],'active':True,'implementation_ref':{'sha256':env['IMPLEMENTATION_SHA']},
    'launcher_ref':{'sha256':env['LAUNCHER_SHA']}}
base={'instruction':instruction,'intent':intent,'manifest':manifest,'review':review,'selector':selector,'policy':policy,
    'instruction_ref':instruction_ref,'intent_ref':intent_ref}
case_raw_calls=[]
case_document_calls=[]
def run_case(data):
    records={
        (instruction_ref['path'],instruction_ref['sha256']):data['instruction'],
        (intent_ref['path'],intent_ref['sha256']):data['intent'],
        (manifest_ref['path'],manifest_ref['sha256']):data['manifest'],
        (review_ref['path'],review_ref['sha256']):data['review'],
        (env['SELECTOR']['path'],env['SELECTOR']['sha256']):data['selector'],
        (env['POLICY']['path'],env['POLICY']['sha256']):data['policy']}
    allowed_raw={(x['path'],x['sha256']) for x in (source_ref,callback_ref,env['GUARD'],env['HELPER'],env['REVOCATION'])}
    def ram_document(record):
        key=(record['path'],record['sha256'])
        case_document_calls.append(key)
        if key not in records:
            raise ValueError('Unknown RAM document ref')
        return copy.deepcopy(records[key])
    def ram_raw(record,cap=1048576):
        key=(record['path'],record['sha256'])
        case_raw_calls.append(key)
        if key not in allowed_raw:
            raise ValueError('Unknown RAM raw ref')
        return b'RAM_ONLY_BOUNDARY_NO_FILESYSTEM'
    env['document']=ram_document
    env['raw']=ram_raw
    return env['controls'](data['instruction_ref'],data['intent_ref'])
positive=copy.deepcopy(base)
check('positive exact fresh namespace join passes in RAM',run_case(positive)==positive['instruction'])
check('positive authenticates all five expected source boundary refs',set(case_raw_calls)=={(x['path'],x['sha256']) for x in (source_ref,callback_ref,env['GUARD'],env['HELPER'],env['REVOCATION'])})
negative=[]
def reject(name, mutate):
    data=copy.deepcopy(base)
    mutate(data)
    try:
        run_case(data)
    except (ValueError,KeyError,TypeError) as error:
        negative.append({'name':name,'rejected':True,'error_type':type(error).__name__})
    else:
        raise AssertionError('Failed rejection: '+name)
reject('old instruction namespace',lambda d:d['instruction_ref'].update(path=OLD+'/ROOT_instruction.root_active_v1.json'))
reject('old intent namespace',lambda d:d['intent_ref'].update(path=OLD+'/ROOT_actual_operation_intent.root_active_v1.json'))
reject('inactive instruction',lambda d:d['instruction'].update(active=False))
reject('inactive intent',lambda d:d['intent'].update(active=False))
reject('mismatched expected prefix',lambda d:d['intent'].update(expected_before={'RAM_ONLY':'MISMATCH'}))
reject('intent missing from uncached refs',lambda d:d['instruction'].update(uncached_current_refs=[]))
reject('incomplete46role set',lambda d:d['instruction']['expected_guard_roles'].pop('RAM_ROLE_46'))
reject('extended48role set',lambda d:d['instruction']['expected_guard_roles'].update(RAM_EXTRA={}))
reject('review fail verdict',lambda d:d['review'].update(verdict='SOURCE_FAIL'))
reject('review not independent',lambda d:d['review'].update(independent=False))
reject('review callback incomplete',lambda d:d['review'].update(actual_callback_complete_source_review=False))
reject('review orchestration incomplete',lambda d:d['review'].update(orchestration_complete_source_review=False))
reject('review blocker present',lambda d:d['review'].update(blockers=['RAM_ONLY_BLOCKER']))
reject('review old intent path',lambda d:d['review'].update(actual_intent_path=OLD+'/ROOT_actual_operation_intent.root_active_v1.json'))
reject('manifest changed callback sha',lambda d:d['manifest']['source_sha256'].update({env['CALLBACK']:'0'*64}))
reject('expired finite clock',lambda d:d['instruction'].update(expires_at_utc='2026-10-17T11:59:59+00:00'))
reject('future finite clock',lambda d:d['instruction'].update(created_at_utc='2026-10-17T12:00:01+00:00'))
reject('operation cap changed',lambda d:d['instruction'].update(operation_wall_seconds=301))
reject('target freshness changed',lambda d:d['instruction'].update(fresh_target_seconds=121))
reject('changed child selector source',lambda d:d['selector'].update(source_ref={'path':'RAM_ONLY_WRONG','sha256':'0'*64}))
reject('inactive selected policy',lambda d:d['policy'].update(active=False))
reject('wrong original launcher',lambda d:d['policy']['launcher_ref'].update(sha256='0'*64))
reject('closed intent extra authority field',lambda d:d['intent'].update(RAM_ONLY_EXTRA=True))
reject('closed intent retry enabled',lambda d:d['intent'].update(automatic_retry=True))
result={'schema':'Census_postfreeze_author_focused_RAM_checks_v1','source_only':True,
    'independent_review_claimed':False,'operational_functions_executed':0,
    'guard_or_helper_imports':0,'active_controls_written':0,'actual_state_or_process_probes':0,
    'actual_accounting_operations':0,'RAM_controls_are_synthetic_and_never_written_as_authority':True,
    'source_namespace_delta_only':True,'closure_entries_authenticated':226,
    'checks':results,'negative_checks':negative,'passed':True,
    'limitations':['Author source/RAM checks are not an independent review.',
        'RAM boundaries do not exercise filesystem authentication, target observations, accounting or runtime.',
        'Historical evidence pins do not add a new runtime prerequisite validator.',
        'ROOT must resolve actual97786 and review fresh state-bound controls before any operation.']}
print(json.dumps(result,sort_keys=True,indent=2))
