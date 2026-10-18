"""Concrete shared-American campaign factories. Numerical implementations stay frozen."""
from __future__ import annotations
import copy
import re
from pathlib import Path
from . import contracts as c
from . import american_preparation as generation
from . import american_observation as observation
from . import american_focals as focals

MODULE = 'arggnn.benchmarking.american_workflow'
DEFAULT_PYTHON = '.runtime/env/bin/python'
DESIGN = 'configs/benchmarks/fresh_american_data_design.json'
REGISTRY = 'configs/benchmarks/fresh_american_parent_registry.json'
GIB = 2**30

def descriptor(module,function,spec,python=DEFAULT_PYTHON,source_refs=()):
    return dict(module=module,function=function,spec=spec,python=python,
                source_refs=[c.file_ref(__file__),*source_refs])


def generation_factory(*,task_id,output_directory,limits):
    spec=dict(schema='american_fresh_preparation_v1',task_id=task_id,
        campaign_ref=c.file_ref(DESIGN),comparison_ref=c.file_ref('configs/benchmarks/paper_comparisons/argformer_american.json'),
        recipe_ref=c.file_ref('configs/benchmarks/recipes/argformer_american.json'),
        output_directory=output_directory,declared_conventions=copy.deepcopy(generation.CONVENTIONS),
        max_sites=5000000,max_tables_bytes=4*GIB,max_truth_intervals=2000000,**limits)
    generation.prepare(spec)
    return descriptor('arggnn.benchmarking.american_preparation','run',spec,
                      source_refs=[*generation.SOURCE_REFS,c.file_ref(generation.__file__)])


def inference_factory(*,input_ref,output_directory,limits):
    spec=dict(schema='american_fresh_observation_stage_v1',stage='shared_inference',input_ref=input_ref,
        output_directory=output_directory,declared_inference_conventions=copy.deepcopy(observation.INFERENCE_CONVENTIONS),
        max_sites=5000000,max_tables_bytes=4*GIB,max_node_map_bytes=256*2**20,max_sample_store_bytes=16*GIB,**limits)
    observation.prepare(spec)
    return descriptor('arggnn.benchmarking.american_observation','run_inference',spec,
        python='.runtime/envs/biological_tsinfer_v1/bin/python',
        source_refs=[*observation.SOURCE_REFS,c.file_ref(observation.__file__)])


def focal_factory(*,input_ref,output_directory,limits):
    spec=focals.build_spec(input_ref=input_ref,output_directory=output_directory,
        max_unique_paths=32000000,max_path_tokens=500000000,max_path_occurrences=250000000,
        max_sample_unique_paths=2000000,max_sample_path_tokens=100000000,max_raw_path_nodes=1024,**limits)
    return descriptor('arggnn.benchmarking.american_focals','run',spec,source_refs=focals.requirements()['source_refs'])


def observed_binding(preparation_ref):
    prepared=c.read_ref(preparation_ref)
    return dict(observation_ref=prepared['observation_ref'],focal_registry_ref=prepared['focals_ref'],
                native_preparation_ref=preparation_ref)


def train_binding(parent,preparation_ref,generation_ref):
    # Only private TRAIN metadata is resolved here. Held-out metadata/payloads
    # have no path through fitting or prediction factories.
    registry=c.read_ref(c.file_ref(REGISTRY))
    assert any(p['parent_id']==parent and p['split']=='train' for p in registry['parents'])
    ready=c.read_ref(generation_ref)
    assert ready['parent_id']==parent and ready['split']=='train'
    private_path=c.contained_path(generation_ref['path']).parent/'private/generation_completed.json'
    private=c.read_ref(c.file_ref(private_path))
    assert private['parent_id']==parent and private['observation_ref']==ready['observation_ref']
    return dict(observed_binding(preparation_ref),truth_ref=private['private_truth_ref'])


def fit_factory(*,preparation_refs,generation_refs,output_directory,max_rss_bytes):
    from . import treeloom_prediction as model
    registry=c.file_ref(REGISTRY)
    train={p['parent_id'] for p in c.read_ref(registry)['parents'] if p['split']=='train'}
    if set(preparation_refs)!=train or set(generation_refs)!=train:
        raise ValueError('Every complete TRAIN parent required')
    spec=model.build_fit_spec(parent_registry_ref=registry,output_directory=output_directory,
        train_inputs={p:train_binding(p,preparation_refs[p],generation_refs[p]) for p in sorted(train)},
        max_rss_bytes=max_rss_bytes,preparation_headroom_bytes=128*GIB,
        training_headroom_bytes=128*GIB,observation_cache_bytes=128*GIB)
    return descriptor(MODULE,'run_fit',spec,source_refs=model.requirements()['source_refs'])


def run_fit(spec,*,resource_check):
    from . import treeloom_prediction as model
    result=model.run_fit(spec,resource_check=resource_check)
    completion_ref=c.file_ref(c.contained_path(spec['training_spec']['output_directory'])/'training_completed.json')
    if c.read_ref(completion_ref)!=result:raise ValueError('Actual final training receipt differs')
    return dict(result,completion_ref=completion_ref)


def prediction_factory(*,fit_result,preparation_refs,output_directory,max_rss_bytes):
    from . import treeloom_prediction as model
    return model.prediction_factory(parent_registry_ref=c.file_ref(REGISTRY),
        checkpoint_ref=fit_result['checkpoint_ref'],training_completion_ref=fit_result['completion_ref'],
        heldout_inputs={p:observed_binding(r) for p,r in preparation_refs.items()},output_directory=output_directory,
        max_rss_bytes=max_rss_bytes,preparation_headroom_bytes=128*GIB,
        prediction_headroom_bytes=64*GIB,observation_cache_bytes=0,max_output_bytes=64*2**20)


def dependency(task,*,read=False,path=()):
    return {'$task':task,'read':read,'path':list(path)}


def build_tasks(*,prefix,output_root):
    """All35 shared parents, fixed fit, all45 predictions, one held-out barrier."""
    tasks=[]
    def add(id,stage,resource,deps,arguments,limits,module=MODULE,function=None):
        match=re.search(r'argformer_american_(?:train|validation|test)_[0-9]{3}',id)
        scope=Path(output_root)/(match.group(0) if match else 'shared')/id
        task=dict(id=id,resource=resource,dependencies=list(dict.fromkeys(deps)),
                  factory_module=module,factory_function=function or stage+'_factory',factory_arguments=arguments,
                  output_directory=str(scope),**limits)
        if resource=='gpu0':task.update(minimum_free_gpu_bytes=74*GIB,max_cuda_reserved_bytes=74*GIB)
        tasks.append(task)
        return id
    limits={
        'generation':dict(max_seconds=7200,max_output_bytes=21*GIB,max_rss_bytes=128*GIB),
        'inference':dict(max_seconds=24*3600,max_output_bytes=29*GIB,max_rss_bytes=128*GIB),
        'focal':dict(max_seconds=12*3600,max_output_bytes=32*2**20,max_rss_bytes=64*GIB),
        'prepare':dict(max_seconds=24*3600,max_output_bytes=21*GIB,max_rss_bytes=256*GIB),
        'mlm':dict(max_seconds=7*86400,max_output_bytes=13*GIB,max_rss_bytes=256*GIB),
        'contrastive':dict(max_seconds=2*86400,max_output_bytes=13*GIB,max_rss_bytes=256*GIB),
        'embeddings':dict(max_seconds=2*86400,max_output_bytes=65*GIB,max_rss_bytes=256*GIB),
        'retrieval':dict(max_seconds=86400,max_output_bytes=3*GIB,max_rss_bytes=256*GIB),
        'flare_prepare':dict(max_seconds=12*3600,max_output_bytes=3*GIB,max_rss_bytes=32*GIB),
        'flare_native':dict(max_seconds=86400,max_output_bytes=16*GIB,max_rss_bytes=32*GIB),
    }
    generation_refs={};inference_refs={};focal_refs={};focal_ids={};generation_ids={}
    rows=c.read_ref(c.file_ref(DESIGN))['tasks']
    # One early evaluation parent feeds native fitting while the rest of TRAIN
    # is prepared. These scientific splits and random streams stay unchanged.
    rows=rows[:6]+[rows[20]]+rows[6:20]+rows[21:]
    for row in rows:
        parent=row['shared_parent_id'];short=prefix+'_'+parent
        if row['split']=='train' and int(parent[-3:])<6:
            i=int(parent[-3:]);eid=short+'_external'
            scope='experiments/results/benchmark_campaign_20261018/fresh_american/'+parent
            queue='.runtime/benchmarking/'+('american_cpu_queue.json' if i==0 else 'american_cpu_batch001_005.json')
            tasks.append(dict(id=eid,resource='external',dependencies=[],queue_path=queue,
                job_id='fresh_american_train%03d_generation_inference_20261018'%i,
                actual_wait_path=scope+'/actual_wait.json',result_path=scope+'/pipeline_completed.json'))
            generation_refs[parent]=dependency(eid,read=True,path=['generation_ready_ref'])
            inference_refs[parent]=dependency(eid,read=True,path=['inference_ready_ref'])
            generation_ids[parent]=eid;inferid=eid
        else:
            gid=short+'_generation';iid=short+'_inference'
            for stage,id,deps,args in [('generation',gid,[],dict(task_id=row['id'])),
                                       ('inference',iid,[gid],dict(input_ref=dependency(gid)))]:
                scientific=dict(limits[stage]);scientific['max_output_bytes']-=GIB
                add(id,stage,'cpu',deps,dict(args,limits=scientific),limits[stage])
            generation_refs[parent]=dependency(gid);inference_refs[parent]=dependency(iid)
            generation_ids[parent]=gid;inferid=iid
        fid=short+'_focals';focal_ids[parent]=fid
        scientific=dict(limits['focal']);scientific['max_output_bytes']-=2**20
        add(fid,'focal','cpu',[inferid],dict(input_ref=inference_refs[parent],limits=scientific),limits['focal'])
        focal_refs[parent]=dependency(fid)
    train={r['shared_parent_id'] for r in rows if r['split']=='train'}
    heldout=[r['shared_parent_id'] for r in rows if r['split']!='train']
    fitid=prefix+'_treeloom_fit'
    fitlimit=dict(max_seconds=7*86400,max_output_bytes=GIB,max_rss_bytes=512*GIB)
    add(fitid,'fit','gpu0',[focal_ids[p] for p in sorted(train)],dict(
        preparation_refs={p:focal_refs[p] for p in sorted(train)},generation_refs={p:generation_refs[p] for p in sorted(train)},
        max_rss_bytes=fitlimit['max_rss_bytes']),fitlimit)
    predictions={m:{} for m in ('treeloom','argformer','flare')};prediction_ids=[]
    for parent in heldout:
        pid=prefix+'_'+parent+'_treeloom_prediction'
        limit=dict(max_seconds=2*86400,max_output_bytes=GIB,max_rss_bytes=256*GIB)
        add(pid,'prediction','gpu0',[fitid,focal_ids[parent]],dict(fit_result=dependency(fitid),
            preparation_refs={parent:focal_refs[parent]},max_rss_bytes=limit['max_rss_bytes']),limit)
        predictions['treeloom'][parent]=dependency(pid,path=['prediction_seal_refs',parent]);prediction_ids.append(pid)
    prior_retire=None
    for parent in heldout:
        short=prefix+'_'+parent
        prepare_id=short+'_argformer_prepare';previous=None;native_ids={}
        for stage in ('prepare','mlm','contrastive','embeddings','retrieval'):
            id=short+'_argformer_'+stage;native_ids[stage]=id
            deps=[focal_ids[parent]]+([] if previous is None else [previous])
            if stage=='prepare' and prior_retire:deps.append(prior_retire)
            scientific=dict(limits[stage]);scientific['max_output_bytes']-=GIB
            args=dict(stage=stage,inference_ref=inference_refs[parent],focal_preparation_ref=focal_refs[parent],limits=scientific)
            if stage=='prepare':args['path_limits']=dict(max_unique_paths=32000000,max_path_tokens=500000000,
                max_path_occurrences=250000000,max_vocabulary_nodes=250000,max_raw_path_nodes=1024)
            else:
                args['preparation_ref']=dependency(prepare_id)
                if stage!='mlm':args['producer_ref']=dependency(previous)
            add(id,stage,'cpu' if stage=='prepare' else 'gpu0',deps,args,limits[stage],
                module='arggnn.benchmarking.american_competitors',function='argformer_factory')
            previous=id
        predictions['argformer'][parent]=dependency(previous);prediction_ids.append(previous)
        prior_retire=short+'_retire_native_scratch'
        add(prior_retire,'retire','cpu',[previous],dict(
            preparation_ref=dependency(prepare_id),embedding_ref=dependency(native_ids['embeddings']),
            prediction_ref=dependency(previous),
            owner_scopes=[str(Path(output_root)/parent/native_ids[s]) for s in ('prepare','embeddings','retrieval')]),
            dict(max_seconds=3600,max_output_bytes=32*2**20,max_rss_bytes=GIB),
            module='arggnn.benchmarking.retention',function='factory')
        tasks[-1]['remeasure_succeeded_tasks']=[native_ids['prepare'],native_ids['embeddings']]
        conv=short+'_flare_prepare';native=short+'_flare_native'
        runtime=dict(binary_provenance_ref=c.file_ref('data/native_FLARE_build_existing_JDK_root_v1/receipts/binary_provenance.json'),
            jar_ref={'path':'data/native_FLARE_build_existing_JDK_root_v1/flare.jar','sha256':'89c88219297a4802f70e00bceaef8957420f1e345b059c7f38da12c287746c2e'},
            java_executable='/home/jovyan/miniconda3/envs/busco/lib/jvm/bin/java',
            java_executable_sha256='dd85541dfdb514a1a577e27e6843d67e4e015f25741fdccb2bb320caf17478c5',
            java_version_output_sha256='58f40f8cf431b24fd4ed3429b533a6051b650e20a84f5e9592e76b9f0c89123f')
        for stage,id,deps in [('prepare',conv,[focal_ids[parent]]),('native',native,[conv])]:
            lim=limits['flare_'+stage];scientific=dict(lim);scientific['max_output_bytes']-=GIB
            args=dict(stage=stage,inference_ref=inference_refs[parent],focal_preparation_ref=focal_refs[parent],limits=scientific)
            if stage=='native':args.update(conversion_ref=dependency(conv),native_runtime=runtime)
            add(id,stage,'cpu',deps,args,lim,module='arggnn.benchmarking.american_competitors',function='flare_factory')
        predictions['flare'][parent]=dependency(native);prediction_ids.append(native)
    add(prefix+'_score','score','cpu',prediction_ids,dict(parent_registry_ref=c.file_ref(REGISTRY),
        prediction_refs_by_method=predictions,generation_ready_refs_by_parent={p:generation_refs[p] for p in heldout},
        focal_preparation_refs_by_parent={p:focal_refs[p] for p in heldout}),
        dict(max_seconds=7200,max_output_bytes=128*2**20,max_rss_bytes=32*GIB),
        module='arggnn.benchmarking.american_scoring',function='factory')
    # Native preparation should not wait behind all remaining CPU simulations.
    # The controller still checks every explicit dependency before dispatch.
    priority={'retire':0,'focals':1,'argformer_prepare':2,'inference':2,'generation':3}
    def rank(task):
        if task['resource']!='cpu':return 0
        name=task['id']
        return next((v for k,v in priority.items() if k in name),5)
    tasks.sort(key=rank)
    return tasks
