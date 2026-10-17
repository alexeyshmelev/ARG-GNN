"""Inactive H060 global observation/target protocol component; stdlib-only inert import.

No legacy block seal is emitted. A future independently reviewed fresh runner
must own actual preparation/cache, all20 barrier, targets, models and resources.
"""
from pathlib import Path
from types import CodeType
import ast,copy,hashlib,json,os,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/H060_global_observation_seal_source_20261017_v1'
SOURCE=HERE+'/global_observation_protocol_v1.py'
TRAIN=tuple('american_c3_short_train_%03d'%i for i in range(20))
CLASSES=('EUR','EAS','AFR')
REGISTRY_REF=dict(path='research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json',sha256='c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37')
SEAL_SCHEMA='H060_global_complete_parent_observation_seal_v1'
BARRIER_SCHEMA='H060_all20_global_observation_cache_barrier_v1'
TARGET_SCHEMA='H060_global_observation_actual_TRAIN_target_protocol_v1'
STATIC_SOURCE_REFS=({'path': 'src/arggnn/american_c3_gnn_targets_v1.py', 'sha256': '065160f5d1e05ed00a663c1a374116fbc4f46bfc9446869185e891630689ce46'}, {'path': 'src/arggnn/american_c3_gnn_dataset_isolated_v2.py', 'sha256': '6e4f32fcec9e6af0c6aa2c81cdbcc4ef1bf7ff83da04b233f0064b6a65c33cb2'}, {'path': 'src/arggnn/american_c3_gnn_stream_train_v1.py', 'sha256': '4a403ba9f5205d8ff38a3509d767daee3d71bd58651c1840fa636b994b37fe9b'}, {'path': 'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py', 'sha256': 'b7418267d329b0d83df45c5f5d053ffd034db0e88c794d5a59c7b59ca52900d1'}, {'path': 'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/metadata_contract_v1.py', 'sha256': '9ffb99b62db75cee95b5721ee0d7002ff0b79bca2503b8619067d5a9900a3bd4'}, {'path': 'research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/full_preparation_fast_v1.py', 'sha256': 'f6aecdc34432c99c0c941963a4ecdfbe68ff970b5abd54b725f75ff105541fe1'}, {'path': 'research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/observations_fast_v1.py', 'sha256': 'ad9dff84272798afd8da89d1b467df6af19dea96e1e3b2f033dad6b56f1d8ceb'}, {'path': 'research/reviews/level_batched_observed_preparation_source_20261017_v1/level_batched_support_v1.py', 'sha256': 'f726a148a96f50c936a2514ea0491e69e338b7450b57d4710eaa8aad0ea09397'}, {'path': 'research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/episode_preparation_v1.py', 'sha256': '3981905c4fc8f959abdc9d4fed3ebfd5432f5f048d68f8c792cce7a7c4503199'}, {'path': 'research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/cut_preparation_v1.py', 'sha256': '185372055111fc849935763513bbcc0dad318db9ddcdef0d35e53e000b587ea0'}, {'path': 'research/reviews/American_TreeLoom_full_mechanism_source_20261017_v2/american_treeloom_full_c3_v2.py', 'sha256': 'f6f4b5ec49231268ea2ae8607739ee52c03953a1b8cc6ac34729d08a3fe24f66'}, {'path': 'research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py', 'sha256': '3b496b163e11147643aa576dfe1f37a75544d466624ef2bd4377fc62d1533265'})

def need(ok,message):
    if not ok:raise ValueError(message)
def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def short_ref(r):
    need(type(r)is dict and set(r)=={'path','sha256'}and type(r['path'])is str and type(r['sha256'])is str
        and not Path(r['path']).is_absolute()and '..'not in Path(r['path']).parts and str(Path(r['path']))==r['path']
        and len(r['sha256'])==64 and all(c in '0123456789abcdef'for c in r['sha256']),'Exact contained immutable ref')
    return r
def source_bytes(r):
    short_ref(r);p=ROOT/r['path'];before=p.lstat()
    need(p.resolve()==p and stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**20,'Bounded canonical source/metadata only')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb')as stream:
        opened=os.fstat(stream.fileno());raw=stream.read(2**20+1);after=os.fstat(stream.fileno())
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(stamp(before)==stamp(opened)==stamp(after)==stamp(p.lstat())and len(raw)==before.st_size
        and hashlib.sha256(raw).hexdigest()==r['sha256'],'Stable exact source SHA')
    return raw

def checked_sources_v1(source_refs):
    need(type(source_refs)in(list,tuple)and len(source_refs)<=256,'Bounded explicit future scientific source closure')
    rows=[copy.deepcopy(short_ref(r))for r in source_refs]
    need(len({r['path']for r in rows})==len(rows)and all(Path(r['path']).suffix=='.py'for r in rows),'Unique Python source refs, never data')
    bypath={r['path']:r for r in rows}
    need(all(bypath.get(r["path"])==r for r in STATIC_SOURCE_REFS),"Exact unchanged original global preparation and target sources")
    need(SOURCE in bypath,'New typed protocol implementation explicitly authenticated')
    for row in rows:source_bytes(row)
    return sorted(rows,key=lambda r:r['path'])

def validate_plan_metadata_v1(plan, *, parent_id, length_bp):
    cfg = plan['config']
    blocks = plan['blocks']
    need(plan['schema'] == 'american_c3_whole_tree_block_index_v1' and plan['parent_id'] == parent_id and (plan['sequence_length_bp'] == length_bp) and (plan['class_order'] == list(CLASSES)) and (plan['haplotypes'] == 1600) and (plan['original_reference_counts'] == [200, 200, 200]) and (plan['effective_reference_counts'] == [200, 200, 200]) and (plan['focal_records'] == 10000) and (plan['all_trees_and_samples_retained'] is True) and (plan['targets_opened'] is False) and blocks and (plan['trees'] > 0) and (cfg['class_order'] == list(CLASSES)) and (cfg['expected_haplotypes'] == 1600) and (cfg['input_reference_class_mask'] in (None, [True, True, True])) and (cfg['input_reference_sample_indices'] is None) and (cfg['allow_missing'] is False) and (cfg['donor_age_split'] == 350.0), 'Exact complete original C3 baseline plan/mask/cardinality')
    cursor = 0
    for b in blocks:
        need(type(b['start_tree']) is int and type(b['stop_tree']) is int and (b['start_tree'] == cursor) and (b['stop_tree'] > cursor) and (b['nodes'] > 0) and (b['sites'] >= 0) and (b['stop_tree'] - b['start_tree'] <= cfg['max_trees_per_block']) and (b['nodes'] <= cfg['max_nodes_per_block']) and (b['sites'] <= cfg['max_sites_per_block']), 'Every original whole tree must occur in one contiguous bounded block')
        cursor = b['stop_tree']
    need(cursor == plan['trees'] and sum((b['sites'] for b in blocks)) == plan['sites'], 'Complete original tree/site plan coverage')

def _seal_shape_v1(seal):
    keys={'schema','parent_id','split','plan_ref','observation_ref','focal_registry_ref','native_preparation_ref',
        'sequence_length_bp','haplotypes','reference_haplotypes','query_haplotypes','class_order','trees','sites','focal_records',
        'focal_identity_sha256','global_identity','preparation_metadata','plan_support','source_refs','observation_only',
        'all_native_trees_nodes_sites_samples_retained','all_global_recipes_sealed_before_truth','private_truth_opened',
        'model_executed','legacy_neural_features_constructed','legacy_block_seal_claimed'}
    need(type(seal)is dict and set(seal)==keys and seal['schema']==SEAL_SCHEMA,'New typed global seal only; legacy schema denied')
    need(seal['parent_id']in TRAIN and seal['split']=='train'and seal['sequence_length_bp']==2000000
        and seal['haplotypes']==1600 and seal['reference_haplotypes']==600 and seal['query_haplotypes']==1000
        and seal['class_order']==list(CLASSES)and seal['focal_records']==10000
        and type(seal['trees'])is int and seal['trees']>0 and type(seal['sites'])is int and seal['sites']>=0,'Exact original TRAIN support')
    for field in ('plan_ref','observation_ref','focal_registry_ref','native_preparation_ref'):short_ref(seal[field])
    need(seal['observation_only']is True and seal['all_native_trees_nodes_sites_samples_retained']is True
        and seal['all_global_recipes_sealed_before_truth']is True and seal['private_truth_opened']is False
        and seal['model_executed']is False and seal['legacy_neural_features_constructed']is False
        and seal['legacy_block_seal_claimed']is False,'Truth-free global preparation; no legacy tensor/seal claim')
    identity=seal['global_identity'];metadata=seal['preparation_metadata']
    need(type(identity)is dict and set(identity)=={'tensor_sha256','tensor_bytes','focal_identity_sha256','trees','sites','all_global_structural_recipes_before_truth'}
        and type(identity['tensor_bytes'])is int and identity['tensor_bytes']>0
        and identity['focal_identity_sha256']==seal['focal_identity_sha256']
        and identity['trees']==seal['trees']and identity['sites']==seal['sites']
        and identity['all_global_structural_recipes_before_truth']is True,'Exact current global tensor/support identity')
    for digest in (identity['tensor_sha256'],seal['focal_identity_sha256']):
        need(type(digest)is str and len(digest)==64 and all(c in '0123456789abcdef'for c in digest),'Actual ordered observation/focal digest')
    need(metadata['schema']=='American_TreeLoom_full_observation_preparation_v1'and metadata['parent_id']==seal['parent_id']
        and metadata['split']=='train'and metadata['observation_only']is True and metadata['focal_records']==10000
        and metadata['focal_identity_sha256']==seal['focal_identity_sha256']
        and metadata['all_native_trees_nodes_sites_samples_retained']is True
        and metadata['structural_recipes_global_before_partition']is True and metadata['artificial_block_boundaries']is False
        and metadata['private_truth_opened']is False and metadata['model_executed']is False
        and all(metadata[k]==seal[k]for k in ('observation_ref','focal_registry_ref','native_preparation_ref')),'Original global preparation metadata, never a legacy certificate')
    need(type(seal['source_refs'])is list and seal['source_refs']and len({r['path']for r in seal['source_refs']})==len(seal['source_refs']),'Explicit source identity in every new global seal')
    for r in seal['source_refs']:short_ref(r)
    need(SOURCE in{r['path']for r in seal['source_refs']},'New protocol source bound in seal')
    support=seal["plan_support"]
    need(type(support)is dict and support["schema"]=="H060_observed_plan_support_without_legacy_features_v1"
        and support["trees"]==seal["trees"]and support["sites"]==seal["sites"]and support["focal_records"]==10000
        and type(support["nodes"])is int and support["nodes"]>0
        and support["legacy_neural_features_constructed"]is False and support["all_original_block_counts_verified"]is True
        and support["every_focal_row_covered_exactly_once"]is True and support["original_query_sample_columns"]is True,
        "Complete observed metadata support check replaces no legacy feature claim")
    need(all({r["path"]:r for r in seal["source_refs"]}.get(r["path"])==r for r in STATIC_SOURCE_REFS),
        "Original global math/target sources remain exact in new seal")
    return seal

def make_global_seal_v1(plan,*,plan_ref,preparation_metadata,global_identity,plan_support,source_refs):
    """Metadata constructor after actual unchanged full global preparation/hash only."""
    validate_plan_metadata_v1(plan,parent_id=plan["parent_id"],length_bp=2000000)
    need(plan['parent_id']in TRAIN and plan.get('plan_ref',plan_ref)==plan_ref
        and plan['sequence_length_bp']==2000000 and plan['haplotypes']==1600 and plan['focal_records']==10000
        and plan['class_order']==list(CLASSES)and plan['targets_opened']is False
        and plan['all_trees_and_samples_retained']is True,'Frozen original complete TRAIN plan')
    seal=dict(schema=SEAL_SCHEMA,parent_id=plan['parent_id'],split='train',plan_ref=copy.deepcopy(plan_ref),
        observation_ref=copy.deepcopy(plan['observation_ref']),focal_registry_ref=copy.deepcopy(plan['focal_registry_ref']),
        native_preparation_ref=copy.deepcopy(plan['config']['native_preparation_ref']),sequence_length_bp=2000000,
        haplotypes=1600,reference_haplotypes=600,query_haplotypes=1000,class_order=list(CLASSES),trees=plan['trees'],sites=plan['sites'],
        focal_records=10000,focal_identity_sha256=global_identity['focal_identity_sha256'],global_identity=copy.deepcopy(global_identity),
        preparation_metadata=copy.deepcopy(preparation_metadata),plan_support=copy.deepcopy(plan_support),source_refs=copy.deepcopy(source_refs),observation_only=True,
        all_native_trees_nodes_sites_samples_retained=True,all_global_recipes_sealed_before_truth=True,
        private_truth_opened=False,model_executed=False,legacy_neural_features_constructed=False,legacy_block_seal_claimed=False)
    return _seal_shape_v1(seal)

def make_all20_barrier_v1(parent_plan_refs,seals,*,validated_cache_identities):
    """Caller supplies all20 actual owned-cache integrity results before any truth."""
    need(type(parent_plan_refs)is dict and type(seals)is dict and type(validated_cache_identities)is dict
        and set(parent_plan_refs)==set(seals)==set(validated_cache_identities)==set(TRAIN),'Complete original20 roster before TRAIN truth')
    first_sources=seals[TRAIN[0]]["source_refs"]
    for parent in TRAIN:
        seal=_seal_shape_v1(seals[parent])
        need(seal["source_refs"]==first_sources,"Same exact source protocol for all20 parents")
        need(seal['parent_id']==parent and seal['plan_ref']==parent_plan_refs[parent]
            and validated_cache_identities[parent]==seal['global_identity'],'Actual current owned cache and seal bind identical full-parent identity')
    return dict(schema=BARRIER_SCHEMA,registry_ref=copy.deepcopy(REGISTRY_REF),parent_ids=list(TRAIN),
        parent_plan_refs=copy.deepcopy(parent_plan_refs),parents=copy.deepcopy(seals),
        validated_cache_identities=copy.deepcopy(validated_cache_identities),all20_ready=True,observation_only=True,
        all20_global_cache_before_TRAIN_truth=True,truth_resolver_calls_before_barrier=0,
        legacy_block_seal_claimed=False,protocol='global_observation_only_target_join_v1')

def validate_barrier_v1(barrier):
    need(type(barrier)is dict and set(barrier)=={'schema','registry_ref','parent_ids','parent_plan_refs','parents','validated_cache_identities',
        'all20_ready','observation_only','all20_global_cache_before_TRAIN_truth','truth_resolver_calls_before_barrier','legacy_block_seal_claimed','protocol'}
        and barrier['schema']==BARRIER_SCHEMA and barrier['registry_ref']==REGISTRY_REF and barrier['parent_ids']==list(TRAIN)
        and barrier['all20_ready']is True and barrier['observation_only']is True
        and barrier['all20_global_cache_before_TRAIN_truth']is True and type(barrier['truth_resolver_calls_before_barrier'])is int
        and barrier['truth_resolver_calls_before_barrier']==0 and barrier['legacy_block_seal_claimed']is False
        and barrier['protocol']=='global_observation_only_target_join_v1','Distinct truthful complete global barrier')
    make_all20_barrier_v1(barrier['parent_plan_refs'],barrier['parents'],validated_cache_identities=barrier['validated_cache_identities'])
    return barrier

def global_observation_gate_v1(parent_id,*,split,plan,seal,barrier):
    """No resolver/payload argument. Every new target join checks the entire20-parent barrier."""
    need(split=='train'and parent_id in TRAIN,'No held-out/old30Mb target resolution')
    validate_barrier_v1(barrier);_seal_shape_v1(seal)
    validate_plan_metadata_v1(plan,parent_id=parent_id,length_bp=2000000)
    need(seal['plan_support']['plan_blocks_sha256']==json_sha(plan['blocks'])
        and seal['plan_support']['blocks']==len(plan['blocks'])
        and seal['plan_support']['nodes']==sum(b['nodes']for b in plan['blocks']),'Same original observed plan coverage witness')
    need(seal==barrier['parents'][parent_id]and seal['parent_id']==parent_id
        and plan['parent_id']==parent_id and plan['sequence_length_bp']==2000000 and plan['haplotypes']==1600
        and plan['focal_records']==10000 and plan['class_order']==list(CLASSES)
        and plan['plan_ref']==seal['plan_ref']and plan['observation_ref']==seal['observation_ref']
        and plan['focal_registry_ref']==seal['focal_registry_ref']and plan['config']['native_preparation_ref']==seal['native_preparation_ref']
        and plan['trees']==seal['trees']and plan['sites']==seal['sites'],'Exact same global observations/native support/plan in this all20 barrier')
    return seal

def derive_base_join_v1(node):
    """Replace only the old legacy observation gate; preserve membership/target equations."""
    expected=ast.parse('''require(plan["parent_id"]==parent_id and plan["sequence_length_bp"]==2000000
        and observation_seal["parent_id"]==parent_id and observation_seal["observation_only"] is True
        and observation_seal["all_blocks_sealed_before_truth"] is True
        and observation_seal["plan_ref"]==plan["plan_ref"],"Complete observation-only plan seal before truth")''').body[0]
    need(node.name=='join'and ast.dump(node.body[2],include_attributes=False)==ast.dump(expected,include_attributes=False),'Exact original single legacy observation gate')
    result=copy.deepcopy(node)
    result.body[2]=ast.parse('_global_observation_gate(parent_id,split=split,plan=plan,seal=observation_seal,barrier=self._global_barrier)').body[0]
    return ast.fix_missing_locations(result)

def derive_actual_join_v1(node):
    result=copy.deepcopy(node);hits=0
    for call in ast.walk(result):
        if isinstance(call,ast.Call)and ast.unparse(call.func)=='super().join':
            hits+=1;call.func=ast.Name(id='_global_base_join',ctx=ast.Load());call.args.insert(0,ast.Name(id='self',ctx=ast.Load()))
    need(hits==1,'Exactly one existing actual-path delegation to the new typed base observation gate')
    return ast.fix_missing_locations(result)

def _exact_method_v1(module,record,fn,path):
    raw=source_bytes(record);need(Path(module.__file__)==ROOT/record['path']and fn.__module__==module.__name__,'Original pinned target module/method')
    code=compile(raw,str(ROOT/record['path']),'exec',dont_inherit=True,optimize=0);tree=ast.parse(raw)
    for name in path:
        codes=[c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name]
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name]
        need(len(codes)==len(nodes)==1,'Exact source-derived nested callable path')
        code=codes[0];tree=nodes[0]
    need(fn.__code__==code and fn.__globals__ is module.__dict__ and not hasattr(fn,'__wrapped__'),'Unchanged original target callable code object')
    return tree

def make_actual_global_target_joiner_factory_v1(bindings,*,barrier,source_refs,runtime_guard):
    """Future admitted caller only. The old factory/target module are never modified."""
    need(callable(runtime_guard),'Future current runtime/source gate')
    runtime_guard('before_new_global_target_protocol_import');checked=checked_sources_v1(source_refs)
    validate_barrier_v1(barrier)
    need(all(barrier["parents"][p]["source_refs"]==checked for p in TRAIN),"All20 seals bind this exact current source closure")
    frozen=copy.deepcopy(barrier)
    import importlib
    targets=importlib.import_module('arggnn.american_c3_gnn_targets_v1')
    hooks=importlib.import_module('research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.execution_hooks_prefetch_v1')
    bypath={r['path']:r for r in checked}
    tr=bypath['src/arggnn/american_c3_gnn_targets_v1.py']
    hr=bypath['research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py']
    base_node=_exact_method_v1(targets,tr,targets.TrainTargetJoinV1.join,('TrainTargetJoinV1','join'))
    scope=dict(targets.__dict__,_global_observation_gate=global_observation_gate_v1)
    exec(compile(ast.Module(body=[derive_base_join_v1(base_node)],type_ignores=[]),SOURCE+'#original-target-body-new-typed-observation-gate','exec'),scope)
    base_join=scope['join']
    actual=hooks.make_actual_target_joiner_factory_v1(bindings)
    actual_node=_exact_method_v1(hooks,hr,actual.join,('make_actual_target_joiner_factory_v1','ActualTrainTargetJoinV1','join'))
    closure={k:c.cell_contents for k,c in zip(actual.join.__code__.co_freevars,actual.join.__closure__or())}
    scope=dict(hooks.__dict__,**closure,_global_base_join=base_join)
    exec(compile(ast.Module(body=[derive_actual_join_v1(actual_node)],type_ignores=[]),SOURCE+'#original-actual-path-body-new-typed-delegation','exec'),scope)
    class ActualGlobalTrainTargetJoinV1(actual):
        def __init__(self,resolve_train_truth,*,max_truth_uncompressed_bytes):
            super().__init__(resolve_train_truth,max_truth_uncompressed_bytes=max_truth_uncompressed_bytes)
            self._global_barrier=copy.deepcopy(frozen)
    ActualGlobalTrainTargetJoinV1.join=scope['join']
    ActualGlobalTrainTargetJoinV1.execution_binding=dict(schema=TARGET_SCHEMA,registry_ref=copy.deepcopy(REGISTRY_REF),
        component_source_ref=copy.deepcopy(bypath[SOURCE]),source_refs=copy.deepcopy(checked),
        original_actual_target_binding=copy.deepcopy(actual.execution_binding),
        global_observation_barrier_sha256=json_sha(frozen),global_seal_schema=SEAL_SCHEMA,
        target_equations_unchanged=True,actual_target_paths_unchanged=True,legacy_block_seal_fabricated=False,
        original_observation_protocol_or_checkpoint_identity_claimed=False)
    runtime_guard('after_new_global_target_protocol_source_join')
    need(checked_sources_v1(source_refs)==checked,'Source identity stable through new target factory construction')
    return ActualGlobalTrainTargetJoinV1

def validate_target_protocol_binding_v1(binding,original_binding,*,barrier):
    validate_barrier_v1(barrier)
    need(binding['schema']==TARGET_SCHEMA and binding['registry_ref']==REGISTRY_REF
        and binding['original_actual_target_binding']==original_binding
        and binding['component_source_ref']in binding['source_refs']and binding['component_source_ref']['path']==SOURCE
        and all(barrier['parents'][p]['source_refs']==binding['source_refs']for p in TRAIN)
        and original_binding['registry_ref']==REGISTRY_REF and original_binding['target_equations_unchanged']is True
        and binding['global_observation_barrier_sha256']==json_sha(barrier)and binding['global_seal_schema']==SEAL_SCHEMA
        and binding['target_equations_unchanged']is True and binding['actual_target_paths_unchanged']is True
        and binding['legacy_block_seal_fabricated']is False
        and binding['original_observation_protocol_or_checkpoint_identity_claimed']is False,'Preserved actual target path/equations in an explicit NEW observation protocol')
    return binding

def global_identity_v1(bundle):
    """Original ordered CPU-observation tensor digest, for these new bytes."""
    from research.reviews.TreeLoom_observed_preparation_fast_source_20261017_v1.full_preparation_fast_v1 import _bank_tensors, _episode_tensors
    require = need
    tensors = _bank_tensors(bundle.bank) + _episode_tensors(bundle.episodes) + bundle.cut.tensors() + (bundle.focal_hold, bundle.focal_sample_column)
    digest = hashlib.sha256()
    total = 0
    for i, tensor in enumerate(tensors):
        require(tensor.device.type == 'cpu' and (not tensor.requires_grad), 'CPU observation-only global feature seal')
        value = tensor.detach().contiguous().numpy()
        digest.update(str(i).encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
        total += value.nbytes
    digest.update(bundle.focal_identity_sha256.encode())
    return dict(tensor_sha256=digest.hexdigest(), tensor_bytes=total, focal_identity_sha256=bundle.focal_identity_sha256, trees=bundle.native_holds, sites=bundle.original_sites, all_global_structural_recipes_before_truth=True)

def verify_observed_plan_support_v1(plan,*,tree_rows,site_positions,focal_holds,focal_sample_columns,registry_sample_columns,resource_check):
    """Observed topology/count/index witness only; never constructs a legacy tensor.

    tree_rows is a streaming (index,left,right,active_node_count) enumeration
    from the same sanitized tree sequence. Native half-open focal joins remain
    checked again by the unchanged full global preparation.
    """
    from bisect import bisect_left
    need(callable(resource_check)and plan['parent_id']in TRAIN and plan['sequence_length_bp']==2000000
        and plan['focal_records']==10000 and plan['haplotypes']==1600,'Original complete TRAIN support only')
    blocks=plan['blocks'];cursor=0;last_right=0.;nodes_total=sites_total=0
    iterator=iter(tree_rows)
    for index,block in enumerate(blocks):
        resource_check('global_observed_plan_metadata_block')
        need(type(block['start_tree'])is int and type(block['stop_tree'])is int
            and block['start_tree']==cursor and block['stop_tree']>cursor,'Original contiguous complete tree partition')
        nodes=sites=0
        for expected in range(block['start_tree'],block['stop_tree']):
            row=next(iterator,None)
            need(row is not None and len(row)==4,'No original tree omitted')
            tree,left,right,count=row
            need(type(tree)is int and tree==expected and left==last_right and right>left
                and type(count)is int and count>0,'Same complete ordered native tree intervals/nodes')
            sites+=bisect_left(site_positions,right)-bisect_left(site_positions,left);nodes+=count;last_right=right
        need(nodes==block['nodes']and sites==block['sites'],'Original per-block observed node/site counts without feature construction')
        nodes_total+=nodes;sites_total+=sites;cursor=block['stop_tree']
    need(cursor==plan['trees']and next(iterator,None)is None and last_right==2000000
        and sites_total==plan['sites']==len(site_positions),'Every original tree/node/site retained in exact original plan support')
    need(len(focal_holds)==len(focal_sample_columns)==len(registry_sample_columns)==10000
        and all(0<=int(hold)<plan['trees']and int(hold)==hold for hold in focal_holds)
        and all(600<=int(column)<1600 and int(column)==column and column==expected
            for column,expected in zip(focal_sample_columns,registry_sample_columns)),
        'All10000 focal rows occur exactly once in complete tree partition with original query columns')
    return dict(schema='H060_observed_plan_support_without_legacy_features_v1',trees=cursor,sites=sites_total,nodes=nodes_total,
        blocks=len(blocks),plan_blocks_sha256=json_sha(blocks),focal_records=10000,
        every_focal_row_covered_exactly_once=True,original_query_sample_columns=True,
        all_original_block_counts_verified=True,legacy_neural_features_constructed=False)

def prepare_and_seal_global_parent_v1(plan_ref,cfg,runtime_guard,*,source_refs):
    """Explicit future admitted observation-only call; no provider/legacy encoder exists here."""
    need(callable(runtime_guard),'Actual future admitted runtime boundary')
    runtime_guard('before_global_only_source_authentication');checked=checked_sources_v1(source_refs)
    from arggnn.american_c3_common_v1 import read
    from arggnn.american_c3_gnn_dataset_isolated_v2 import load_observation
    from arggnn.american_c3_gnn_stream_train_v1 import validate_plan_v1
    from research.reviews.TreeLoom_observed_preparation_fast_source_20261017_v1.full_preparation_fast_v1 import prepare_from_loaded_observation_v1
    original_plan=read(plan_ref);plan=dict(original_plan,plan_ref=copy.deepcopy(plan_ref))
    validate_plan_v1(plan,parent_id=plan['parent_id'],length_bp=2000000)
    need(plan['parent_id']in TRAIN,'Only original20 independent TRAIN parents')
    state=None
    try:
        runtime_guard('before_global_only_observed_load')
        state=load_observation(plan['observation_ref'],plan['focal_registry_ref'],plan['config'])
        tree_rows=((int(tree.index),float(tree.interval.left),float(tree.interval.right),sum(1 for _ in tree.nodes()))for tree in state['ts'].trees())
        support=verify_observed_plan_support_v1(plan,tree_rows=tree_rows,site_positions=state['positions'],
            focal_holds=state['focal_tree'],focal_sample_columns=state['focals']['sample_indices'],
            registry_sample_columns=state['focals']['sample_indices'],resource_check=runtime_guard)
        runtime_guard('before_unchanged_global_FP32_preparation')
        bundle,metadata=prepare_from_loaded_observation_v1(state,split='train',
            observation_ref=plan['observation_ref'],focal_registry_ref=plan['focal_registry_ref'],
            native_preparation_ref=plan['config']['native_preparation_ref'],max_nodes=cfg['max_native_nodes'],
            max_holds=cfg['max_native_holds'],max_episode_payload_bytes=cfg['max_episode_payload_bytes'])
        need(bundle.native_holds==plan['trees']and bundle.original_sites==plan['sites']
            and bundle.bank.parent.numel()==support['nodes'],'Same complete native support in actual global bank')
        runtime_guard('before_actual_global_tensor_seal');identity=global_identity_v1(bundle)
        seal=make_global_seal_v1(plan,plan_ref=plan_ref,preparation_metadata=metadata,global_identity=identity,
            plan_support=support,source_refs=checked)
        need(read(plan_ref)==original_plan and checked_sources_v1(source_refs)==checked,'Original plan/source identity retained through actual hashing')
        runtime_guard('after_actual_global_tensor_seal')
        return bundle,seal
    finally:
        state=None

if __name__=='__main__':
    print(json.dumps(dict(status='INACTIVE_SOURCE_ONLY_COMPONENT',scientific_imports=False,preparation=False,target_resolution=False,execution_authority=False)))
