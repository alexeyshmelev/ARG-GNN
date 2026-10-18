"""Freeze the concrete benchmark DAG, without releasing or starting it."""
from pathlib import Path
import hashlib
import json
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import american_workflow as american
from arggnn.benchmarking import european_workflow as european
from arggnn.benchmarking import european_argmix_workflow as european_c4
from arggnn.benchmarking import workflow

GIB=2**30
OUTPUT='experiments/results/benchmark_campaign_20261018_workflow'
RECORDS='.runtime/benchmarking/workflow_20261018'
PLAN='configs/benchmarks/executable_workflow_20261018.json'

def write(path,value):
    p=c.contained_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return c.file_ref(p)


def interleave_tasks(tasks):
    """Stable round robin across independent benchmark families."""
    from itertools import zip_longest
    families=("AM","AP","C4")
    buckets={name:[] for name in families}
    other=[]
    for task in tasks:
        family=next((name for name in families if task["id"].startswith(name+"_")),None)
        (buckets[family] if family in buckets else other).append(task)
    ordered=[task for row in zip_longest(*(buckets[name] for name in families))
             for task in row if task is not None]
    return ordered+other


def build():
    if c.contained_path(RECORDS+"/state.json").exists():
        raise RuntimeError("A started workflow is immutable. Do not overwrite its plan.")
    native=json.loads(c.contained_path('configs/ancestralpaths_full_panel_discovery_v1.json').read_text())
    native['relate']['timeout_seconds_per_step']=2*86400
    native['native_step_headroom_bytes']=GIB
    native_config_ref=write('configs/benchmarks/ap200mb_native_runtime.json',
        dict(relate=native['relate'],native_step_headroom_bytes=native['native_step_headroom_bytes']))
    campaign=c.read_ref(c.file_ref('configs/benchmarks/campaign.json'))
    parents=[]
    def seed(parent,stage):
        return 1+int.from_bytes(hashlib.sha256((parent+':'+stage+':20261018').encode()).digest()[:8],'big')%(2**31-100)
    for row in campaign['tasks']:
        if row['recipe']=='ancestralpaths_european' and row['stage']=='simulate_once':
            parent=row['shared_parent_id']
            parents.append(dict(parent_id=parent,split=row['split'],simulation_stream=row['simulation_stream'],seeds=dict(
                ancestry=row['seeds']['ancestry'],mutations=row['seeds']['mutations'],
                relate_initial=seed(parent,'relate_initial'),relate_size_base=seed(parent,'relate_size_base'))))
    parent_ref=write('configs/benchmarks/ap200mb_parent_registry.json',dict(parents=parents,
        seed_convention='Original campaign ancestry/mutation streams, independent deterministic SHA256 Relate streams'))
    def lim(seconds,disk,rss):return dict(max_seconds=seconds,max_output_bytes=disk,max_rss_bytes=rss)
    limits=dict(ap_generate=lim(86400,24*GIB,128*GIB),ap_infer=lim(7*86400,96*GIB,128*GIB),
        ap_registry=lim(7200,GIB,32*GIB),ap_features=lim(86400,2*GIB,64*GIB),
        ap_train_targets=lim(7200,GIB,32*GIB),ap_fit=lim(86400,2*GIB,32*GIB),
        ap_predict=lim(7200,GIB,32*GIB),ap_score=lim(7200,GIB,32*GIB),
        treeloom_fit=dict(lim(10*86400,2*GIB,512*GIB),minimum_free_gpu_bytes=74*GIB,max_cuda_reserved_bytes=74*GIB),
        treeloom_predict=dict(lim(5*86400,2*GIB,256*GIB),minimum_free_gpu_bytes=74*GIB,max_cuda_reserved_bytes=74*GIB))
    parameters=dict(write_headroom_bytes=GIB,max_metadata_bytes=128*2**20,max_feature_bytes=512*2**20,
        max_truth_uncompressed_bytes=16*GIB,native_training_seed=20261018,native_tensorflow_threads=8)
    preparation=dict(max_nodes=1000000000,max_holds=500000,max_episode_payload_bytes=64*GIB)
    common=dict(max_sites=5000000,max_metadata_bytes=128*2**20,preparation_limits=preparation,max_output_bytes=64*2**20)
    tree_parameters=dict(
        fit=dict(common,max_rss_bytes=512*GIB,preparation_headroom_bytes=128*GIB,
                 training_headroom_bytes=128*GIB,observation_cache_bytes=128*GIB),
        predict=dict(common,max_rss_bytes=256*GIB,preparation_headroom_bytes=128*GIB,
                     prediction_headroom_bytes=64*GIB,observation_cache_bytes=0))
    tasks=american.build_tasks(prefix='AM',output_root=OUTPUT+'/american')
    ap_tasks=european.build_tasks(prefix='AP',output_root=OUTPUT+'/european',parent_records=parents,
        sampling_plan_ref=c.file_ref('configs/benchmarks/recipes/ap_paper_sampled_plan_v1.json'),native_config_ref=native_config_ref,
        python_data='.runtime/env/bin/python',python_tensorflow='.runtime/conda/envs/ancestralpaths_cpu_v1/bin/python',
        stage_limits=limits,stage_parameters=parameters,treeloom_parameters=tree_parameters)
    # The available-release C4 comparison uses the same fresh observed parents,
    # its own source-verified endpoint and an independent native VAL parent.
    c4_config_ref=c.file_ref('configs/benchmarks/recipes/argmix_release_ap200mb_c4_v2.json')
    build_ref=dict(path='experiments/results/argmix_native_abi210_execution_v1/v6/build/receipt.json',
        sha256='fa1898dc4f55638d4e106a2da9b6fb7df139f2719b50b664b433c7955e0dd0dd')
    cpu_ref=dict(path='experiments/results/argmix_native_cpu_admission_v6/native_receipt.json',
        sha256='592561fe97bd74e8678ba2ebb1bf749c0c09124c13eda03d0ded98d120c99a39')
    for ref in (build_ref,cpu_ref):
        if c.file_ref(ref['path'])!=ref:
            raise RuntimeError('Existing native runtime qualification receipt changed')
    def gpu_lim(seconds,disk,rss):
        return dict(lim(seconds,disk,rss),minimum_free_gpu_bytes=74*GIB,
                    max_cuda_reserved_bytes=74*GIB)
    c4_limits=dict(
        validation_generate=dict(limits['ap_generate']),
        validation_infer=dict(limits['ap_infer']),
        release_frame=lim(86400,GIB,64*GIB),
        release_c4_registry=lim(7200,GIB,64*GIB),
        release_c4_targets=lim(7200,GIB,32*GIB),
        release_native_targets=lim(86400,10*GIB,64*GIB),
        # Bounds are complete-data fail stops. Actual native tree count is
        # checked before Rust allocation, with no chunk or sample reduction.
        release_archive=lim(7*86400,128*GIB,512*GIB),
        release_fit=gpu_lim(10*86400,2*GIB,768*GIB),
        release_infer=gpu_lim(5*86400,64*GIB,256*GIB),
        release_seal=lim(86400,GIB,32*GIB),
        release_c4_score=lim(7200,GIB,32*GIB),
        c4_treeloom_fit=dict(limits['treeloom_fit']),
        c4_treeloom_predict=dict(limits['treeloom_predict']))
    c4_parameters=dict(parameters,
        native_build_ref=build_ref,native_cpu_ref=cpu_ref,
        native_archive_command_limits=dict(max_seconds=7*86400),
        max_native_archive_bytes=128*GIB,
        max_snp_truth_bytes=8*GIB,max_h5_chunk_rows=65536)
    tasks+=european_c4.build_tasks(ap_tasks=ap_tasks,ap_prefix='AP',prefix='C4',
        output_root=OUTPUT+'/european_c4',config_ref=c4_config_ref,
        sampling_plan_ref=c.file_ref('configs/benchmarks/recipes/ap_paper_sampled_plan_v1.json'),
        native_config_ref=native_config_ref,python_data='.runtime/env/bin/python',
        python_native='.runtime/official_argmix_abi210_v1/cu129/env/bin/python',
        stage_limits=c4_limits,stage_parameters=c4_parameters,treeloom_parameters=tree_parameters)
    # A genuine current C4 checkpoint is a predecessor. This is a descriptive
    # application to the existing original observed banks, without accuracy.
    tasks.append(dict(
        id='EMP__current_C4_H056__apply',resource='cpu',
        dependencies=['C4__all_train__c4_treeloom_fit'],
        output_directory=str(c.contained_path(OUTPUT+'/empirical/current_C4_H056')),
        factory_module='arggnn.benchmarking.empirical_treeloom',
        factory_function='factory',
        factory_arguments=dict(
            fit_completion_ref={'$task':'C4__all_train__c4_treeloom_fit','path':['completion_ref']},
            max_rss_bytes=4*GIB,max_bank_read_bytes=18*2**20,
            max_checkpoint_read_bytes=8*2**20,max_episode_payload_bytes=2**20,
            max_nodes=100000,max_holds=1000,max_output_bytes=32*2**20),
        **lim(86400,64*2**20,4*GIB)))
    tasks=interleave_tasks(tasks)
    external_storage=[];external_resources=[]
    for i in range(6):
        parent='argformer_american_train_%03d'%i
        scope='experiments/results/benchmark_campaign_20261018/fresh_american/'+parent
        evidence=dict(queue_path='.runtime/benchmarking/'+('american_cpu_queue.json' if i==0 else 'american_cpu_batch001_005.json'),
            job_id='fresh_american_train%03d_generation_inference_20261018'%i,wait_path=scope+'/actual_wait.json')
        external_storage.append(dict(cap_bytes=56*GIB,scopes=[scope],completion_evidence=[evidence]))
        external_resources.append(dict(evidence,resource='cpu',max_rss_bytes=128*GIB))
    external_storage.append(dict(cap_bytes=92*GIB,scopes=[
        'experiments/results/benchmark_campaign_20261018/native_argformer_val000_contrastive_ordered_state',
        'experiments/results/benchmark_campaign_20261018/native_argformer_predictions'],completion_evidence=[]))
    # Freeze every maintained scientific adapter plus its existing declared pins.
    source_paths={'scripts/build_benchmark_workflow.py'}
    for task in tasks:
        if task['resource']!='external':source_paths.add('src/'+task['factory_module'].replace('.','/')+'.py')
    source_paths.update(str(p.relative_to(c.ROOT)) for p in (c.ROOT/'src/arggnn/benchmarking').glob('*.py'))
    sources=[c.file_ref(p) for p in sorted(source_paths)]
    plan=dict(schema=workflow.SCHEMA,physical_gpu=0,cpu_parallelism=8,tasks=tasks,sources=sources,
        publish_progress=True,task_family_dispatch_order="stable round robin AM, AP, C4",
        records_directory=RECORDS,storage_baseline_ref=c.file_ref(
            'experiments/results/benchmark_campaign_20261018/native_argformer_val000_contrastive_ordered_state/storage_admission.json'),
        storage_ceiling_bytes=980000000000,metadata_margin_bytes=2*GIB,
        max_parallel_rss_bytes=1280*GIB,host_ram_margin_bytes=64*GIB,
        external_storage=external_storage,external_resources=external_resources,external_native_rss_bytes=256*GIB,
        scientific_scope=['Fresh American native three-source comparison','AP200Mb native six-path comparison',
            'Available-release ARGMix and genuinely trained TreeLoom on common AP200Mb C4 support',
            'Current C4 descriptive empirical application, without ancestry accuracy'],
        additional_comparison_integration='ARGMix C4 and current empirical application have concrete descriptors, scientific comparisons remain pending',
        c4_recipe_ref=c4_config_ref,native_build_ref=build_ref,native_cpu_ref=cpu_ref,
        execution_authority='Explicit latest user instruction, exact canonical policy release still required',
        original_campaign_is_not_an_executable_queue=True,ap_parent_registry_ref=parent_ref,
        retained_artifacts='All observations/truth/focals/checkpoints/predictions/results retained; only successful new native scratch retired')
    ref=write(PLAN,plan)
    workflow.load_plan(ref)
    print(json.dumps(dict(plan_ref=ref,tasks=len(tasks),resources={r:sum(t['resource']==r for t in tasks) for r in ('cpu','gpu0','external')},
        mode='validated_plan_only'),indent=2))
    return plan

if __name__=='__main__':build()
