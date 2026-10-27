"""Prospective full-parent paired training. Importing this module runs nothing.

Root must admit an explicit GPU job and pass its actual resource callback.
No worker/connector changes or subprocess launchers exist here. Both arms use
all original biological inputs and10000 focal targets per complete parent.
"""
from contextlib import contextmanager
from pathlib import Path
import copy,gc,hashlib,json,os,time

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OWNED=ROOT/'research/efficiency_20261027/architecture'
ARMS=('frozen_current','transport')


def need(condition,message):
    if not condition:raise ValueError(message)


def reference(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())


def read(record):
    need(reference(record['path'])==record,'Immutable source reference changed')
    return json.loads((ROOT/record['path']).read_text())


class StageProfile:
    """Host-side call durations; these are not useful GPU duty measurements."""
    def __init__(self):self.rows={}
    @contextmanager
    def stage(self,name):
        start=time.perf_counter()
        try:yield
        finally:
            row=self.rows.setdefault(name,dict(calls=0,host_seconds=0.))
            row['calls']+=1;row['host_seconds']+=time.perf_counter()-start
    def mark(self,name,**values):pass


def run(config,*,resource_check):
    need(config.get('execution_enabled') is True and callable(resource_check),
         'Root-admitted execution and actual worker resource callback required')
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3','4','5') and
         os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and
         ',' not in os.environ['CUDA_VISIBLE_DEVICES'] and
         os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device',
         'One root-selected visible GPU3/4/5 UUID and existing full-device policy required')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from arggnn.benchmarking import contracts as c
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_evaluate_v1 import registry as read_focals,verify_prediction
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh,training_v2 as original_training
    from research.benchmark_rebuild_20261024.american.treeloom_speed.gru_v2 import install_chunked_gru
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward,geometry_from_spans
    from research.scalability_20261024.integration.executors import build_executor
    from research.scalability_20261024.integration.cache import LayoutCache
    from .treeloom_transport import TreeLoomTransport
    from .bounded_transport import BoundedTransport
    need(torch.cuda.device_count()==1,'Exactly one admitted logical CUDA device required')
    torch.set_num_threads(config['CPU_threads']);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out=(ROOT/config['output_directory']).resolve()
    need(out.is_relative_to(OWNED/'runs') and not out.exists(),'Fresh owned research output directory required')
    out.mkdir(parents=True)
    start=time.monotonic();deadline=start+config['max_seconds']
    def rss():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    def check(stage):
        resource_check(stage)
        need(time.monotonic()<deadline,'Registered wall budget exhausted; do not crop parents or extend silently')
        need(rss()<=config['max_RSS_bytes'],'Declared process RSS budget exceeded')
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved()<=config['max_GPU_reserved_bytes'],'Declared device reserve budget exceeded')
    def write(name,payload):
        check('research_artifact_write');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
        return reference(p)
    def log(payload):
        with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(payload,allow_nan=False)+'\n')
    result=dict(status='running',schema='TreeLoom_Transport_paired_full_parent_training_v1',
                config=config,TEST_opened=False,VAL_truth_opened=False,seeds=[])
    write('config.json',config)
    try:
        for source_ref in config['source_refs']:
            need(reference(source_ref['path'])==source_ref,'Registered source changed before run')
        original=read(config['original_training_job_ref'])['call']['spec']['arguments']
        registry=fresh.FreshParentRegistry.from_ref(original['parent_registry_ref'])
        need(len(registry.ids('train'))==20 and len(registry.ids('validation'))==5,'Original20TRAIN/5VAL cohort required')
        need(set(config['validation_inputs'])==set(registry.ids('validation')),'Every original VAL parent and noTEST required')
        canonical=read(config['original_validation_bindings_source_ref'])
        score_tasks=[task for task in canonical['tasks'] if task['id']=='paper_american_all_methods_score']
        need(len(score_tasks)==1,'One immutable original evaluation bindings task required')
        original_VAL=score_tasks[0]['call']['factory']['kwargs']
        for parent in registry.ids('validation'):
            fpref=original_VAL['focal_preparation_refs_by_parent'][parent];fp=read(fpref)
            expected=dict(native_preparation_ref=fpref,observation_ref=fp['observation_ref'],focal_registry_ref=fp['focals_ref'])
            need(config['validation_inputs'][parent]==expected and
                 config['validation_generation_ready_refs'][parent]==original_VAL['generation_ready_refs_by_parent'][parent],
                 'VAL must equal the exact original canonical inferred observations, focals and generation lineage')
        need(1<=len(config['seeds'])<=3 and len(set(config['seeds']))==len(config['seeds']) and
             20<=config['updates_per_arm']<=600 and config['updates_per_arm']%20==0,
             'Predeclared1..3 paired seeds and complete20-parent epochs required')
        check('before_lineage_checkpoint_metadata')
        checkpoint_ref=config['observation_barrier_checkpoint_ref']
        need(reference(checkpoint_ref['path'])==checkpoint_ref,'Original lineage checkpoint changed')
        saved=torch.load(ROOT/checkpoint_ref['path'],map_location='cpu',weights_only=True,mmap=True)
        barrier_ref=saved['observation_barrier_ref'];del saved
        barrier=c.read_ref(barrier_ref)
        need(barrier['all20_observations_sealed_before_truth'] is True and
             barrier['parent_registry_ref']==registry.reference and
             set(barrier['parents'])==set(registry.ids('train')),'Original complete observation barrier required')
        plans={}
        for parent,item in original['train_inputs'].items():
            seal=barrier['parents'][parent]
            need(all(item[k]==seal[k] for k in fresh.INPUT_REFS),'Both arms must use exact frozen observed TRAIN bindings')
            plans[parent]=dict(c.read_ref(seal['plan_ref']),plan_ref=seal['plan_ref'])
        runtime=fresh.FreshTreeLoom(registry)
        joiner=fresh.FreshTrainTargetJoin(registry,lambda parent:original['train_inputs'][parent]['truth_ref'],
            train_inputs=original['train_inputs'],max_truth_uncompressed_bytes=original['max_truth_uncompressed_bytes'])
        def prepare(parent,split):
            check('before_complete_parent_preparation');item=original['train_inputs'][parent] if split=='train' else config['validation_inputs'][parent]
            need(registry.split_for(parent)==split,'Original split must remain unchanged')
            tick=time.perf_counter()
            state=load_observation(item['observation_ref'],item['focal_registry_ref'],dict(
                native_preparation_ref=item['native_preparation_ref'],class_order=list(fresh.CLASSES),
                expected_haplotypes=1600,reference_counts=[200,200,200],input_reference_class_mask=None,input_reference_sample_indices=None))
            bundle,metadata=runtime.prepare_observation(state,split=split,resource_check=check,
                **{k:item[k] for k in fresh.INPUT_REFS},**original_training.LIMITS)
            del state;gc.collect()
            need(bundle.sequence_length_bp==30000000 and bundle.bank.sample_index.shape[1]==1600 and
                 len(bundle.focal_hold)==10000 and all(getattr(bundle,k)==item[k] for k in fresh.INPUT_REFS),
                 'Complete original30Mb/all1600/all10000 parent support required')
            bundle.validate();check('complete_parent_prepared')
            return bundle,metadata,time.perf_counter()-tick
        def executor(model,bundle,metadata):
            return BoundedTransport(model,bundle.bank,bundle.episodes.flags.node_flags,
                metadata['episode_metadata']['native_nodes_per_tree'],resource_check=check,**config['transport_partitions'])
        for seed in config['seeds']:
            check('paired_seed_begin');models={};optimizers={};arm_seconds={arm:0. for arm in ARMS}
            for arm in ARMS:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model=install_chunked_gru(runtime.frozen.AmericanTreeLoomC3FullV1(),chunk_holds=1024) if arm=='frozen_current' else TreeLoomTransport()
                models[arm]=model.to('cuda:0')
                optimizers[arm]=torch.optim.AdamW(models[arm].parameters(),lr=.001,weight_decay=.0001,
                    betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)
            seed_record=dict(seed=seed,updates=[],common_preparation_seconds=0.,common_target_join_seconds=0.,checkpoints={},predictions={arm:{} for arm in ARMS})
            schedule=[parent for epoch in range(1,config['updates_per_arm']//20+1) for parent in registry.epoch_order(epoch)]
            for step,parent in enumerate(schedule,1):
                bundle,metadata,prep_seconds=prepare(parent,'train');seed_record['common_preparation_seconds']+=prep_seconds
                need(bundle.focal_identity_sha256==barrier['parents'][parent]['focal_identity_sha256'],'Actual parent matches original pretruth seal')
                join_start=time.perf_counter();target=joiner.join(parent,split='train',plan=plans[parent],observation_seal=barrier['parents'][parent])
                seed_record['common_target_join_seconds']+=time.perf_counter()-join_start
                labels=target.labels_cpu
                need(labels.shape==(10000,) and labels.dtype==torch.long,'All original10000 TRAIN targets required')
                order=ARMS if step%2 else tuple(reversed(ARMS))
                for arm in order:
                    check('paired_update_begin');model,optimizer=models[arm],optimizers[arm];model.train();optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();profile=StageProfile()
                    with torch.autocast(device_type='cuda',enabled=False):
                        if arm=='transport':
                            with profile.stage('transport_forward'):
                                execution=executor(model,bundle,metadata);logits=execution.forward_focals(bundle.focal_hold,bundle.focal_sample_column)
                                loss=F.cross_entropy(logits,labels.to('cuda:0'))
                            with profile.stage('transport_backward'):loss.backward()
                            del logits,execution
                        else:
                            cache=LayoutCache(max_bytes=config['baseline_layout_cache_bytes'],admit=lambda amount,stage:need(rss()+amount<=config['max_RSS_bytes'],'Layout allocation exceeds RSS reserve'))
                            execution=build_executor('integrated',model,bundle,metadata,execution='staged',layout_cache=cache,
                                frozen_module=runtime.frozen,resource_check=check,profile=profile,**config['baseline_partitions'])
                            with profile.stage('baseline_context'):context=execution.context_forward()
                            with profile.stage('baseline_decoder'):
                                loss,vjp,decoder=decode_parent_backward(model,context,geometry_from_spans(bundle.bank.spans),
                                    bundle.focal_hold,bundle.focal_sample_column,labels,tile_haplotypes=config['baseline_partitions']['haplotypes_per_gru'],profile=profile,resource_check=check)
                            with profile.stage('baseline_trunk_backward'):context.backward(vjp)
                            execution.release_layout();del context,vjp,execution,cache
                    need(bool(torch.isfinite(loss)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Finite connected objective and every model gradient required')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    torch.cuda.synchronize();elapsed=time.perf_counter()-tick;arm_seconds[arm]+=elapsed
                    need(all(int(v['step'].item())==step for v in optimizer.state.values()),'Identical optimizer update exposure required')
                    row=dict(step=step,parent_id=parent,arm=arm,seed=seed,loss=float(loss),gradient_norm=float(norm),
                        complete_native_holds=bundle.native_holds,native_nodes=len(bundle.bank.parent),sites=bundle.original_sites,
                        supervised_focals=10000,update_seconds=elapsed,cumulative_arm_update_seconds=arm_seconds[arm],
                        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        process_RSS_bytes=rss(),host_stage_timings=profile.rows,useful_GPU_duty='not_measured_by_this_timer')
                    seed_record['updates'].append(row);log(row);check('paired_update_complete')
                    optimizer.zero_grad(set_to_none=True);del loss;gc.collect()
                del target,labels,bundle,metadata;gc.collect();torch.cuda.empty_cache()
            for arm in ARMS:
                path=out/f'seed_{seed}/{arm}_final.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},seed=seed,
                    updates=config['updates_per_arm'],architecture=arm,parent_registry_ref=registry.reference),path)
                seed_record['checkpoints'][arm]=reference(path)
                models[arm].eval()
            # No VAL truth resolver exists in this GPU process.
            for parent in registry.ids('validation'):
                bundle,metadata,prep_seconds=prepare(parent,'validation')
                item=config['validation_inputs'][parent];records=read_focals(item['focal_registry_ref'])
                for arm in ARMS:
                    check('VAL_full_parent_prediction')
                    with torch.no_grad():
                        if arm=='transport':
                            execution=executor(models[arm],bundle,metadata)
                            logits=execution.forward_focals(bundle.focal_hold,bundle.focal_sample_column);del execution
                        else:
                            execution=build_executor('integrated',models[arm],bundle,metadata,execution='bounded',
                                layout_cache=LayoutCache(max_bytes=config['baseline_layout_cache_bytes'],admit=lambda amount,stage:need(rss()+amount<=config['max_RSS_bytes'],'Prediction layout exceeds RSS reserve')),
                                resource_check=check,**config['baseline_partitions'])
                            logits=execution.forward();execution.release_layout();del execution
                        prediction={k:records[k] for k in fresh.IDENTITY};prediction['probabilities']=logits.softmax(-1).cpu().numpy()
                    verify_prediction(prediction,records)
                    path=out/f'seed_{seed}/predictions/{arm}/{parent}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as stream:np.savez(stream,**prediction)
                    seed_record['predictions'][arm][parent]=reference(path)
                    del logits,prediction
                del bundle,metadata,records;gc.collect();torch.cuda.empty_cache()
            need(all(set(v)==set(registry.ids('validation')) for v in seed_record['predictions'].values()),'Both arms/all5VAL must seal before later CPU truth access')
            seed_record['all10_VAL_predictions_sealed_before_truth']=True
            seed_record['arm_update_seconds']=arm_seconds
            result['seeds'].append(seed_record);write(f'seed_{seed}/sealed.json',seed_record)
            del models,optimizers;gc.collect();torch.cuda.empty_cache()
        result.update(status='completed_training_and_all_VAL_prediction_seals',elapsed_seconds=time.monotonic()-start,
                      parent_registry_ref=registry.reference,observation_barrier_ref=barrier_ref,
                      initialization='both fresh; lineage checkpoint model/optimizer state never loaded into either arm',
                      budget_accounting='Common CPU preparation/target-join costs reported once and must be charged to both standalone methods; arm update times measured separately.',
                      accuracy_preservation_established=False,full_parent_resource_qualification_only_if_run_completes=True)
        for source_ref in config['source_refs']:
            need(reference(source_ref['path'])==source_ref,'Registered source changed during run')
        return write('completed.json',result)
    except BaseException as error:
        # Preserve exact incomplete budget/failure evidence. Never change inputs.
        p=out/'failed.json'
        with p.open('x') as stream:json.dump(dict(status='failed_or_budget_exhausted',error_type=type(error).__name__,message=str(error),elapsed_seconds=time.monotonic()-start),stream,indent=2)
        raise
