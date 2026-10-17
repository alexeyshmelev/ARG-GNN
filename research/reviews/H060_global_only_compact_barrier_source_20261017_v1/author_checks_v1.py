"""Author source-only full runner/writer control flow. Numerical, truth, guard, lock and file providers are RAM."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import ast,copy,hashlib,io,json,re,stat,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/H060_global_only_compact_barrier_source_20261017_v1'
checks=[];negative=[]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def check(msg,ok=True):need(ok,msg);checks.append(msg)
def load_defs(path,names,env,constants=False):
 raw=path.read_bytes();tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names or constants and isinstance(n,ast.Assign)]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec',dont_inherit=True,optimize=0),env);return raw
helper={'__name__':'_RAM_compact_helper'};exec(compile((D/'compact_barrier_v1.py').read_bytes(),str(D/'compact_barrier_v1.py'),'exec'),helper)
proto=dict(Path=Path,copy=copy,hashlib=hashlib,json=json)
protocol_path=ROOT/'research/reviews/H060_global_observation_seal_source_20261017_v2/global_observation_protocol_v1.py'
proto_names=('need','json_sha','short_ref','validate_plan_metadata_v1','_seal_shape_v1','make_global_seal_v1','make_all20_barrier_v1','validate_barrier_v1','global_observation_gate_v1','validate_target_protocol_binding_v1')
protocol_raw=load_defs(protocol_path,proto_names,proto,True);TRAIN=proto['TRAIN'];ARMS=('stable_leaf_memory','no_added_memory')
def source_ref(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
sources=sorted([*proto['STATIC_SOURCE_REFS'],source_ref(protocol_path),source_ref(D/'compact_barrier_v1.py'),source_ref(D/'warmstart_train_v3.py')],key=lambda r:r['path'])
plans={};seals={};plan_refs={};trees=2000
for number,parent in enumerate(TRAIN):
 ref=dict(path='research/reviews/RAM_plan_%03d.json'%number,sha256='%064x'%number);plan_refs[parent]=ref
 observation_ref=dict(path='research/reviews/RAM_observed_%03d.json'%number,sha256='a'*64);focal_ref=dict(path='research/reviews/RAM_focals_%03d.json'%number,sha256='b'*64);native_ref=dict(path='research/reviews/RAM_native_%03d.json'%number,sha256='c'*64)
 blocks=[dict(start_tree=i,stop_tree=i+500,nodes=500*1600,sites=0)for i in range(0,trees,500)]
 plan=dict(schema='american_c3_whole_tree_block_index_v1',parent_id=parent,sequence_length_bp=2000000,haplotypes=1600,class_order=list(proto['CLASSES']),original_reference_counts=[200]*3,effective_reference_counts=[200]*3,focal_records=10000,all_trees_and_samples_retained=True,targets_opened=False,trees=trees,sites=0,blocks=blocks,plan_ref=ref,observation_ref=observation_ref,focal_registry_ref=focal_ref,config=dict(class_order=list(proto['CLASSES']),expected_haplotypes=1600,input_reference_class_mask=None,input_reference_sample_indices=None,allow_missing=False,donor_age_split=350.,max_trees_per_block=500,max_nodes_per_block=1000000,max_sites_per_block=1000000,native_preparation_ref=native_ref))
 plans[parent]=plan;identity=dict(tensor_sha256='d'*64,tensor_bytes=100000,focal_identity_sha256='e'*64,trees=trees,sites=0,all_global_structural_recipes_before_truth=True)
 metadata=dict(schema='American_TreeLoom_full_observation_preparation_v1',parent_id=parent,split='train',observation_only=True,observation_ref=observation_ref,focal_registry_ref=focal_ref,native_preparation_ref=native_ref,focal_records=10000,focal_identity_sha256='e'*64,all_native_trees_nodes_sites_samples_retained=True,structural_recipes_global_before_partition=True,artificial_block_boundaries=False,private_truth_opened=False,model_executed=False,
  observation_metadata=dict(isolated_zero_site_holds=list(range(trees))),episode_metadata=dict(proper_clades_per_tree=list(range(trees)),native_nodes_per_tree=[1600]*trees,unary_equivalences_per_tree=[0]*trees),cut_metadata=dict(native_hold_cut_group_counts=[1]*trees,episode_starts=list(range(trees)),conditional_information_nats_per_leaf=[0.]*trees),isolated_zero_site_holds=list(range(trees)))
 support=dict(schema='H060_observed_plan_support_without_legacy_features_v1',trees=trees,sites=0,nodes=trees*1600,blocks=len(blocks),plan_blocks_sha256=proto['json_sha'](blocks),focal_records=10000,every_focal_row_covered_exactly_once=True,original_query_sample_columns=True,all_original_block_counts_verified=True,legacy_neural_features_constructed=False)
 seals[parent]=proto['make_global_seal_v1'](plan,plan_ref=ref,preparation_metadata=metadata,global_identity=identity,plan_support=support,source_refs=sources)
identities={p:seals[p]['global_identity']for p in TRAIN};full=proto['make_all20_barrier_v1'](plan_refs,seals,validated_cache_identities=identities);original_sha=proto['json_sha'](full)
commit=helper['make_durable_barrier_v1'](full,full_barrier_sha256=original_sha,validate_full_barrier=proto['validate_barrier_v1']);large_bytes=len((json.dumps(full,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());compact_bytes=len((json.dumps(commit,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
check('streamed_canonical_digest_exactly_matches_original_full_barrier',commit['full_original_barrier']['sha256']==original_sha and helper['canonical_commitment_v1'](full)['canonical_JSON_bytes']==len(json.dumps(full,sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
check('compact_distinct_schema_retains_all20_per_parent_source_identity_and_support',commit['schema']!=proto['BARRIER_SCHEMA']and commit['parent_ids']==list(TRAIN)and commit['source_refs']==sources and all(commit['parents'][p]['original_full_seal']['sha256']==proto['json_sha'](seals[p])and commit['parents'][p]['plan_support']==seals[p]['plan_support']for p in TRAIN))
check('per_hold_values_remain_in_original_memory_and_are_committed_not_truncated',len(full['parents'][TRAIN[0]]['preparation_metadata']['episode_metadata']['proper_clades_per_tree'])==trees and commit['parents'][TRAIN[0]]['structural_list_commitments']['episode_metadata/proper_clades_per_tree']['length']==trees)
check('large_metadata_fixture_exceeds_original3MiB_but_compact_within256KiB',large_bytes>3145728 and compact_bytes<=262144)
for label,bad,expected in [('wrong_full_SHA',full,'f'*64),('missing_parent',dict(full,parent_ids=list(TRAIN[:-1])),original_sha)]:
 try:helper['make_durable_barrier_v1'](bad,full_barrier_sha256=expected,validate_full_barrier=proto['validate_barrier_v1'])
 except ValueError:negative.append(label)
 else:raise AssertionError('Accepted '+label)
bad=copy.deepcopy(full);bad['parents'][TRAIN[0]]['preparation_metadata']['episode_metadata']['proper_clades_per_tree'][0]+=1
try:helper['make_durable_barrier_v1'](bad,full_barrier_sha256=original_sha,validate_full_barrier=proto['validate_barrier_v1'])
except ValueError:negative.append('changed_structural_array_breaks_full_commitment')
else:raise AssertionError('Array change accepted')
# Exact original save body with RAM OS/path and no actual guard, process, filesystem or clocks.
SINK={};FD={};serial=[0]
class RAMPath(PurePosixPath):
 def lstat(self):return metadata(str(self))
def metadata(path):return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=1,st_ino=abs(hash(path)),st_size=len(SINK.get(path,b'')))
class RAMFile(io.BytesIO):
 def __init__(self,fd):super().__init__();self.fd=fd
 def fileno(self):return self.fd
 def write(self,data):
  count=super().write(data);SINK[FD[self.fd]]=self.getvalue();return count
class RAMOS:
 O_WRONLY=1;O_CREAT=2;O_EXCL=4;O_NOFOLLOW=8
 def open(self,path,flags,mode):
  path=str(path);need(path not in SINK,'RAM exclusive output');serial[0]+=1;FD[serial[0]]=path;SINK[path]=b'';return serial[0]
 def fdopen(self,fd,mode):return RAMFile(fd)
 def fsync(self,fd):pass
 def fstat(self,fd):return metadata(FD[fd])
 def statvfs(self,path):return SimpleNamespace(f_bavail=1000000,f_frsize=4096)
CAPS=json.loads((ROOT/'research/reviews/H060_isolated_CPU_runtime_source_20261017_v1/caps.inactive.v1.json').read_text())
class FakeTorch:
 float32='RAM_FLOAT32'
 def save(self,value,buffer):buffer.write(b'RAM_ONLY_NOT_A_SCIENTIFIC_CHECKPOINT')
 def device(self,name):return SimpleNamespace(type=name,index=None)
 def get_default_dtype(self):return self.float32
 def is_autocast_enabled(self):return False
torch=FakeTorch();writerenv=dict(Path=RAMPath,ROOT=ROOT,io=io,json=json,re=re,stat=stat,hashlib=hashlib,os=RAMOS(),c=SimpleNamespace(CAPS=CAPS,require=need));worker_path=ROOT/'research/reviews/H060_global_only_full_training_source_20261017_v2/worker_entry_v1.py';worker_tree=ast.parse(worker_path.read_bytes());classes=[n for n in worker_tree.body if isinstance(n,ast.ClassDef)and n.name in ('CappedBuffer','BoundedWriter')];exec(compile(ast.Module(body=classes,type_ignores=[]),str(worker_path),'exec'),writerenv)
def writer(name):
 obj=writerenv['BoundedWriter'].__new__(writerenv['BoundedWriter']);obj.path=RAMPath(str(ROOT/'research/reviews/RAM_ONLY'/name));obj.s=dict(storage='RAM');obj.used=8192;obj.json_used=0;obj.torch_files=0;obj.names=set();obj.fast=lambda event:None;obj.torch=torch;obj.jsonable=lambda value:value;obj.g=SimpleNamespace(budget_check_v9=lambda *a,**k:None);return obj
blocked=writer('full')
try:blocked.save('TRAIN_observation_barrier.json',full)
except ValueError:negative.append('original_writer_rejects_large_full_fixture_without_partial_write')
else:raise AssertionError('Original cap not reached')
check('original_failed_full_writer_has_no_partial_RAM_product',not any('/full/'in p for p in SINK))
# Full 20-parent runner/write/control route; scientific and external providers are explicit RAM doubles.
events=[];update_counts={a:0 for a in ARMS};truth=[];barriers=[];old_binding=dict(registry_ref=proto['REGISTRY_REF'],target_equations_unchanged=True)
class Bundle:
 def to(self,device):return self
bundle=Bundle()
class Protocol:
 STATIC_SOURCE_REFS=proto['STATIC_SOURCE_REFS'];SOURCE=proto['SOURCE'];SEAL_SCHEMA=proto['SEAL_SCHEMA'];BARRIER_SCHEMA=proto['BARRIER_SCHEMA'];TARGET_SCHEMA=proto['TARGET_SCHEMA']
 def prepare_and_seal_global_parent_v1(self,plan_ref,cfg,guard,source_refs):
  parent=next(p for p in TRAIN if plan_refs[p]==plan_ref);events.append(('prepare',parent));return bundle,seals[parent]
 def make_all20_barrier_v1(self,*a,**k):
  result=proto['make_all20_barrier_v1'](*a,**k);barriers.append(result);events.append(('all20_barrier',len(result['parents'])));return result
 validate_barrier_v1=staticmethod(proto['validate_barrier_v1'])
 validate_target_protocol_binding_v1=staticmethod(proto['validate_target_protocol_binding_v1'])
 def make_actual_global_target_joiner_factory_v1(self,bindings,*,barrier,source_refs,runtime_guard):
  proto['validate_barrier_v1'](barrier);need(barrier['schema']==proto['BARRIER_SCHEMA'],'Full in-memory barrier only');events.append(('target_factory',len(barrier['parents'])));frozen=barrier
  class Joiner:
   execution_binding=dict(schema=proto['TARGET_SCHEMA'],registry_ref=proto['REGISTRY_REF'],original_actual_target_binding=old_binding,component_source_ref=source_ref(protocol_path),source_refs=sources,global_observation_barrier_sha256=proto['json_sha'](barrier),global_seal_schema=proto['SEAL_SCHEMA'],target_equations_unchanged=True,actual_target_paths_unchanged=True,legacy_block_seal_fabricated=False,original_observation_protocol_or_checkpoint_identity_claimed=False)
   def __init__(self,resolve_train_truth,**kwargs):self.resolve=resolve_train_truth
   def join(self,parent,*,split,plan,observation_seal):
    proto['global_observation_gate_v1'](parent,split=split,plan=plan,seal=observation_seal,barrier=frozen);self.resolve(parent);events.append(('target_join',parent));return SimpleNamespace(receipt=dict(parent_id=parent,full_observation_sha256=original_sha))
  return Joiner
protocol=Protocol();w=writer('future');w.save('started.json',dict(status='RAM_SOURCE_FIXTURE_ONLY'))
env=dict(copy=copy,json_sha=proto['json_sha'],global_protocol=protocol,compact_barrier=SimpleNamespace(**helper),TRAIN=TRAIN,ARMS=ARMS,FAMILY='American_TreeLoom_saved600_warmstart_matched_v1',CONDITION='accepted_C3_saved600_backbone_plus600_new_native_focal_updates_per_arm',PARAMETERS={a:0 for a in ARMS},REGISTRY_REF=proto['REGISTRY_REF'],SCHEMA='RAM_SOURCE_CONTROL_FLOW',CHECKPOINT_EPOCHS=(5,10,15,20,25,30),torch=torch,time=SimpleNamespace(monotonic=lambda:0.),require=need)
env.update(validate_config_v1=lambda cfg:None,authoritative_registry_v1=lambda:None,load_accepted_backbone_v1=lambda *a,**k:(None,{},plan_refs,old_binding),_source_snapshot_v2=lambda:('RAM_source_frames',),_source_check_v2=lambda token:None,_global_preparation_refs_v1=lambda:sources,_validated_global_cache_identities_v1=lambda cache,s,p,g:identities,_signature_v1=lambda *a:({},'f'*64),_seed_fresh=lambda device:None,make_matched_family_v1=lambda *a,**k:({arm:None for arm in ARMS}, {arm:None for arm in ARMS}, {}),epoch_order_v1=lambda epoch:list(TRAIN),read=lambda ref:next(plans[p]for p in TRAIN if plan_refs[p]==ref),_cache_get_v2=lambda cache,p:cache[p]['bundle'])
def admit(cache,parent,b,seal,*,current_bytes,maximum_bytes,runtime_guard):cache[parent]=dict(bundle=b,charge=1);return current_bytes+1
def step(arm,*args,**kwargs):update_counts[arm]+=1;return dict(first_real_update_checks={'RAM_ONLY':True},complete_parent_step_wall_seconds_including_checks=0.)
env['_admit_global_parent_v1']=admit;env['train_complete_parent_step_v1']=step;env['_snapshot_v1']=lambda *a,step,**k:dict(trainer_state=dict(additional_updates_per_arm=step))
runner_raw=load_defs(D/'warmstart_train_v3.py',('run_train_v1',),env)
cfg=dict(max_cached_parent_bytes=103079215104,max_truth_uncompressed_bytes=32000000)
result=env['run_train_v1'](cfg,parent_plan_refs=plan_refs,artifact_acceptance_ref='RAM_ONLY',writer=w,device='cpu',worker_tmp_directory='RAM_ONLY',resolve_train_truth=lambda p:truth.append(p),runtime_guard=lambda event:None,actual_train_bindings='RAM_ONLY',resume_ref=None)
completed=json.loads(SINK[str(w.path/'completed_training.json')]);durable=json.loads(SINK[str(w.path/'TRAIN_observation_barrier.json')])
check('full20_preparations_then_durable_barrier_before_any_target_join',len([e for e in events if e[0]=='prepare'])==20 and events.index(('all20_barrier',20))<events.index(('target_factory',20))and len(truth)==20)
check('target_factory_and_original_gates_receive_full_not_compact_barrier',barriers[0]['schema']==proto['BARRIER_SCHEMA']and durable['schema']==helper['SCHEMA']and len([e for e in events if e[0]=='target_join'])==20)
check('all600_pairs_1200_arm_update_calls_and6_checkpoints_preserved',update_counts=={a:600 for a in ARMS}and w.torch_files==6 and completed['new_saved_logical_optimizer_updates_total']==1200 and all(len(completed['history'][a])==600 for a in ARMS))
check('all60_prediction_seals_and_original_all30_barrier_flags_preserved',completed['total_new_prediction_seals_required']==60 and completed['all60_matched_prediction_seals_required_before_any_VAL_truth']is True and completed['existing_C3_all30_barrier_unchanged']is True)
check('checkpoint_completion_full_observation_digest_unchanged',completed['observation_seal_sha256']==original_sha==durable['full_original_barrier']['sha256']and completed['durable_observation_commitment_schema']==helper['SCHEMA'])
check('exact_original_writer_caps_and32MiB_reservation_not_raised',CAPS['reservation_bytes']==33554432 and CAPS['total_JSON_bytes']==3145728 and CAPS['output_bytes']==26214400 and w.json_used<=CAPS['total_JSON_bytes']and w.used<=CAPS['output_bytes'])
oldraw=(ROOT/'research/reviews/H060_global_only_full_training_source_20261017_v2/warmstart_train_v2.py').read_bytes();oldnodes={n.name:n for n in ast.parse(oldraw).body if isinstance(n,ast.FunctionDef)};newnodes={n.name:n for n in ast.parse(runner_raw).body if isinstance(n,ast.FunctionDef)};changed={k for k in oldnodes if ast.dump(oldnodes[k],include_attributes=False)!=ast.dump(newnodes[k],include_attributes=False)}
check('only_source_ref_protocol_disclosure_and_durable_write_functions_differ',changed=={'_global_preparation_refs_v1','_signature_v1','run_train_v1'})
need(hashlib.sha256(oldraw).hexdigest()=='a94aee031661432eb27665a7d2ba3d9f92f4d0efaf26ffb106e2e9e453ddd2b5','Current global runner immutable')
result=dict(schema='H060_compact_barrier_author_source_RAM_route_checks_v1',status='AUTHOR_SOURCE_RAM_CHECKS_PASS',checks=checks,positive_count=len(checks),negative_cases=negative,negative_count=len(negative),full_fixture_pretty_JSON_bytes=large_bytes,compact_fixture_pretty_JSON_bytes=compact_bytes,RAM_full_route_JSON_bytes=w.json_used,RAM_full_route_accounted_output_bytes=w.used,complete_parent_preparations=20,full_native_support_metadata_preserved=True,RAM_update_calls=update_counts,paired_checkpoints=6,all60_barrier_flags_preserved=True,original_canonical_full_barrier_SHA_unchanged=True,actual_numerical_training_or_payload_truth_or_guard_resource_lock_clock_process_operations=False,actual_disk_products=False,actual_live_adoption=False,whole_ROOT_control_admission_join_qualified=False,runtime_fit_or_speed_claimed=False,changed_original_functions=sorted(changed))
with(D/'author_checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:result[k]for k in ('status','positive_count','negative_count','full_fixture_pretty_JSON_bytes','compact_fixture_pretty_JSON_bytes','RAM_update_calls','actual_live_adoption')},sort_keys=True))
