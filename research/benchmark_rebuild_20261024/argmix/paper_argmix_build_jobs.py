"""Write a concrete additive DAG; this script never launches its jobs."""
from pathlib import Path
import json
from arggnn.benchmarking.paper_argmix import paper_argmix_generator as g

config=g.metadata(g.CONFIG)
base=g.confined(config['campaign_root'])
recipe=str(g.CONFIG)
module='arggnn.benchmarking.paper_argmix.paper_argmix_cli'
python_generator=str(g.ROOT/config['runtime']['generator'])
python_model=str(g.ROOT/config['runtime']['model'])
GiB=1024**3
MiB=1024**2
jobs=[]

def add(key,stage,path,dependencies,arguments=(),*,cap=32*MiB,seconds=1800,cpu=1,rss=4*GiB,gpu=False,expected=None):
 argv=[python_model if stage in ('train','predict') else python_generator,'-B','-m',module,stage,'--config',recipe,
       '--output',str(path),'--max-output-bytes',str(cap),'--timeout-seconds',str(seconds)]+list(arguments)
 record=dict(id=key,stage=stage,argv=argv,cwd=str(g.ROOT),dependencies=list(dependencies),output_directory=str(path),
             output_cap_bytes=cap,timeout_seconds=seconds,resources=dict(cpu_threads=cpu,max_rss_bytes=rss,
             gpu='campaign' if gpu else None,physical_gpu_allowlist=[3,4,5] if gpu else [],logical_cuda_device=0 if gpu else None,max_gpu_bytes=None if gpu else 0,gpu_memory_policy='full_device' if gpu else None),
             env=dict(OMP_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),MKL_NUM_THREADS=str(cpu),
                      **({'ARGGNN_GPU_MEMORY_POLICY':'full_device'} if gpu else {'CUDA_VISIBLE_DEVICES':''})),campaign_gpu_binding_required=gpu,expected_output_bytes=expected,
             required_environment_source=str(g.ROOT/'scripts/project_env.sh'),retry=False)
 jobs.append(record)
 return key

previous=[]
fit_dependencies=[]
evaluation=[]
for parent in config['parents']:
 name=parent['parent_id']
 path=base/name
 common=['--parent-id',name]
 generated=add(name+'.generated','adopt-train0' if name=='published_recon_train_000' else 'generate',path/'generated',previous,
               common,cap=8*GiB,seconds=7200,rss=16*GiB,expected=150*MiB if name.endswith('train_000') else 350*MiB)
 prepared=add(name+'.relate_inputs','prepare-relate',path/'relate_work',[generated],
              common+['--parent-output',str(path/'generated')],cap=4*GiB,seconds=3600,rss=2*GiB,expected=2100*MiB)
 native=add(name+'.native_relate','native-relate',path/'native_run',[prepared],
            ['--native-work',str(path/'relate_work')],cap=256*MiB,seconds=86400,cpu=8,rss=32*GiB,expected=32*MiB)
 jobs[-1]['additional_mutable_scopes']=[dict(path=str(path/'relate_work'),max_bytes=24*GiB,meaning='new transient native files and prepared inputs')]
 jobs[-1]['requires_periodic_supervisor_scope_check']=True
 inferred=add(name+'.inferred','finalize-relate',path/'inferred',[native],
              common+['--parent-output',str(path/'generated'),'--native-work',str(path/'relate_work'),
                      '--terminal-receipt',str(path/'native_run/terminal.json')],cap=2*GiB,rss=16*GiB,expected=500*MiB)
 support=add(name+'.support','sample',path/'support',[inferred],common+['--inferred-output',str(path/'inferred')],
             cap=32*MiB,seconds=1800,rss=4*GiB,expected=12*MiB)
 features=add(name+'.features','extract',path/'features',[support],
              common+['--inferred-output',str(path/'inferred'),'--support-output',str(path/'support')],
              cap=256*MiB,seconds=14400,cpu=1,rss=16*GiB,expected=182700384)
 release_dependencies=[features]
 if parent['split'] in ('train','val'):
  labels=add(name+'.labels','labels',path/'labels',[support],
             common+['--parent-output',str(path/'generated'),'--support-output',str(path/'support')],
             cap=32*MiB,seconds=1800,rss=4*GiB,expected=128*1024)
  fit_dependencies.extend([features,labels])
  release_dependencies.append(labels)
 else:
  evaluation.append(parent)
 # Only Root invokes this explicit new-scratch release stage after reviewing it.
 released=add(name+'.release_new_scratch','release-scratch',path/'scratch_release',release_dependencies,
              common+['--parent-output',str(path/'generated'),'--native-work',str(path/'relate_work'),
                      '--inferred-output',str(path/'inferred')],cap=32*MiB,seconds=3600,rss=2*GiB,expected=MiB)
 jobs[-1]['destructive_scope']='Only newly declared input.haps and inferred native scratch; verified original observations, truth, sanitized trees, logs and receipts are retained.'
 previous=[released]
assembled=add('fit_inputs','assemble-fit',base/'fit_inputs',fit_dependencies,['--stage-root',str(base)],cap=MiB,expected=8192)
trained=add('argmix.train','train',base/'argmix_train',[assembled],['--fit-manifest',str(base/'fit_inputs/fit_inputs.json'),
            '--condition','equation_corrected','--device','cuda:0','--cpu-threads','4'],
            cap=768*MiB,seconds=86400,cpu=4,rss=8*GiB,gpu=True,expected=64*MiB)
prediction_jobs=[]
prediction_args=[]
for parent in evaluation:
 name=parent['parent_id'];path=base/name
 prediction_jobs.append(add(name+'.argmix_prediction','predict',path/'argmix_prediction',[trained,name+'.features'],
    ['--selected-model',str(base/'argmix_train/completed.json'),'--feature-manifest',str(path/'features/completed.json'),
     '--device','cuda:0','--cpu-threads','2'],cap=32*MiB,seconds=7200,cpu=2,rss=4*GiB,gpu=True,expected=2*MiB))
 prediction_args+=['--prediction',str(path/'argmix_prediction/completed.json')]
barrier=add('argmix.prediction_barrier','barrier',base/'argmix_prediction_barrier',prediction_jobs,prediction_args,cap=MiB,expected=16384)
label_jobs=[];label_args=[]
for parent in evaluation:
 name=parent['parent_id'];path=base/name
 label_jobs.append(add(name+'.labels','labels',path/'labels',[barrier,name+'.support'],
   ['--parent-id',name,'--parent-output',str(path/'generated'),'--support-output',str(path/'support'),
    '--prediction-barrier',str(base/'argmix_prediction_barrier/barrier.json')],cap=32*MiB,rss=4*GiB,expected=128*1024))
 label_args+=['--label',str(path/'labels/completed.json')]
scored=add('argmix.score','score',base/'argmix_scores',label_jobs+[barrier],
    ['--prediction-barrier',str(base/'argmix_prediction_barrier/barrier.json')]+label_args,cap=8*MiB,expected=256*1024)
# The unchanged TreeLoom architecture consumes every native tree/node/site.
# No whole-tree bank is persisted in the scarce filesystem scope.
tl_observations=[]
for parent in config['parents']:
 name=parent['parent_id'];path=base/name
 observed=add(name+'.treeloom_observation','treeloom-observation',path/'treeloom_observation',[name+'.support'],
   ['--parent-id',name,'--parent-output',str(path/'generated'),'--inferred-output',str(path/'inferred'),
    '--support-output',str(path/'support')],cap=16*MiB,seconds=3600,rss=16*GiB,expected=4*MiB)
 if parent['split']=='train': tl_observations.extend([observed,name+'.labels'])
 next(job for job in jobs if job['id']==name+'.release_new_scratch')['dependencies'].append(observed)
tl_train=add('treeloom.train','treeloom-train',base/'treeloom_train',tl_observations,
   ['--stage-root',str(base),'--device','cuda:0','--cpu-threads','8'],cap=128*MiB,seconds=604800,cpu=8,
   rss=config['treeloom']['max_rss_bytes'],gpu=True,expected=8*MiB)
tl_predictions=[];tl_prediction_args=[]
for parent in evaluation:
 name=parent['parent_id'];path=base/name
 tl_predictions.append(add(name+'.treeloom_prediction','treeloom-predict',path/'treeloom_prediction',
   [tl_train,name+'.treeloom_observation'],['--selected-model',str(base/'treeloom_train/completed.json'),
    '--observation-seal',str(path/'treeloom_observation/completed.json'),'--device','cuda:0','--cpu-threads','8'],
    cap=32*MiB,seconds=86400,cpu=8,rss=config['treeloom']['max_rss_bytes'],gpu=True,expected=2*MiB))
 tl_prediction_args+=['--prediction',str(path/'treeloom_prediction/completed.json')]
tl_barrier=add('treeloom.prediction_barrier','barrier',base/'treeloom_prediction_barrier',tl_predictions,tl_prediction_args,cap=MiB,expected=16384)
joint=add('paired.prediction_barrier','joint-barrier',base/'paired_prediction_barrier',[barrier,tl_barrier],
 ['--prediction-barrier',str(base/'argmix_prediction_barrier/barrier.json'),'--other-barrier',str(base/'treeloom_prediction_barrier/barrier.json')],cap=MiB,expected=4096)
for key in label_jobs:
 job=next(job for job in jobs if job['id']==key)
 job['dependencies'].append(joint)
 argv=job['argv'];argv[argv.index('--prediction-barrier')+1]=str(base/'paired_prediction_barrier/barrier.json')
add('treeloom.score','score',base/'treeloom_scores',label_jobs+[tl_barrier,scored],
 ['--prediction-barrier',str(base/'treeloom_prediction_barrier/barrier.json'),'--comparator',str(base/'argmix_scores/completed.json')]+label_args,
 cap=8*MiB,expected=512*1024)

# Stable topological ordering makes the contract usable by simple DAG workers.
remaining=list(jobs);ordered=[];done=set()
while remaining:
 ready=[job for job in remaining if set(job['dependencies'])<=done]
 if not ready: raise RuntimeError('Cyclic or missing job dependency')
 for job in ready:
  ordered.append(job);done.add(job['id']);remaining.remove(job)
if len(done)!=len(jobs): raise RuntimeError('Duplicate job ID')
jobs=ordered
completion_names={'generate':'generation.json','adopt-train0':'generation.json','prepare-relate':'commands.json','native-relate':'terminal.json',
                  'assemble-fit':'fit_inputs.json','barrier':'barrier.json','joint-barrier':'barrier.json'}
for job in jobs:
 job['completion_file']=str(Path(job['output_directory'])/completion_names.get(job['stage'],'completed.json'))
plan=dict(schema='paper_argmix_execution_dag_v1',config_ref=g.file_ref(g.CONFIG),campaign_root=str(base),jobs=jobs,
    execution_launched=False,source_ref=g.file_ref(__file__),
    whole_new_campaign_failstop_bytes=config['bounds']['whole_campaign_new_bytes'],
    required_root_actions=['Measure actual aggregate project bytes once before dispatch and enforce980GB project ceiling plus40GiB new-scope ceiling.',
      'Source project_env.sh, pin all package sources and pass finite actual RSS/wall/output caps.',
      'Keep native preparation sequential through explicit release_new_scratch jobs; do not modify historical data.',
      'The DAG already includes both-method prediction barrier before all evaluation labels. Do not weaken it.',
      'Select one physicalGPU3/4/5 with an exclusive campaign lease; inject its exact UUID as CUDA_VISIBLE_DEVICES and its index as ARGGNN_PHYSICAL_GPU; expose logicalcuda0 and consume real subprocess waits.'],
    storage_forecast=dict(compact_feature_payload_all14_bytes=14*182700384,
      retained_observations_truth_per_new_parent_estimated_bytes=[300*MiB,500*MiB],
      retained_inferred_tree_per_parent_estimated_bytes=[300*MiB,1024*MiB],
      simultaneous_native_cap_bytes=24*GiB,additional_input_cap_bytes=4*GiB,
      note='Forecasts are unmeasured except fixed feature payload and historical TRAIN0 sizes; first full completion must be measured. Capacity failures preserve partial data without reducing science.'),
    immutable_interfaces=dict(inferred='PARENT/inferred/completed.json: tree_ref, roster_ref, observed_coverage_bp0, local_to_rebased_offset_bp; sanitized inferred genealogy only',
      support='PARENT/support/completed.json and shared_targets.tsv: fixed query/tree pairs, stable identities and local/rebased midpoints',
      labels='PARENT/labels/completed.json and classes.u1: same row order, four founder classes, separate truth namespace',
      features='PARENT/features/completed.json and compact tokens/distances/nodes.npy: uint8/uint16/int32,29 genealogy tokens',
      predictions='PARENT/argmix_prediction/completed.json and probabilities.npy:float32,100000x4,truth_read=false'))
output=g.ROOT/'research/benchmark_rebuild_20261024/argmix/paper_argmix_jobs.json'
output.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(output),jobs=len(jobs),parents=len(config['parents']),launched=False),indent=2))
