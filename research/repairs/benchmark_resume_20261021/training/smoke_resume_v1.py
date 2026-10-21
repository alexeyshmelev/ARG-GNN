"""One tiny CPU integration check: stop after update 7, resume exactly to 600."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import copy
import json
import random
import time
import numpy as np
import torch
from arggnn.benchmarking import treeloom_training as training
from arggnn import american_c3_gnn_dataset_v1 as dataset
from research.repairs.benchmark_recovery_20261020.runtime import binding
from research.repairs.benchmark_recovery_20261020.training import step
from research.repairs.benchmark_recovery_20261020.training.candidate_fit_cpu_test import tiny_bundle,CPUModel
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import resident_pool_v2 as pool_module
from research.repairs.benchmark_recovery_20261020.training.candidate_common import ROOT,GIB,need,file_ref,write_json
from . import fit_v1 as fitter,checkpoint_v1 as checkpoint


def build_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70001)
        return CPUModel()


def update(model,optimizer):
    scale=torch.rand(())+float(np.random.random())+random.random()
    optimizer.zero_grad(set_to_none=True)
    loss=sum(parameter.square().mean() for parameter in model.parameters())*scale
    loss.backward();optimizer.step()


def main():
    started=time.monotonic()
    base=ROOT/'experiments/results/benchmark_resume_20261021/synthetic_resume_smoke_v1'
    need(not base.exists(),'Fresh synthetic smoke scope');base.mkdir(parents=True)
    parents=tuple('synthetic_train_%03d'%i for i in range(20))
    class Registry:
        reference={'mock':'synthetic_registry'}
        def ids(self,split):return parents if split == 'train' else ()
        def split_for(self,parent):return 'train' if parent in parents else 'test'
        def epoch_order(self,epoch):
            result=list(parents);random.Random(70002+epoch).shuffle(result);return result
    registry=Registry()
    refs={parent:{key:dict(path='synthetic/'+parent+'/'+key,sha256='a'*64)
        for key in (*training.fresh.INPUT_REFS,'truth_ref')} for parent in parents}
    observed={p:{k:refs[p][k] for k in training.fresh.INPUT_REFS} for p in parents}
    original=dict(train_inputs=refs,max_truth_uncompressed_bytes=2**20,preparation_limits={},
        settings=copy.deepcopy(training.SETTINGS))
    plans={p:dict(native_holds=4,original_sites=3) for p in parents}
    barrier=dict(parents={p:dict(focal_identity_sha256='c'*64,**observed[p]) for p in parents})
    visits=[];pools=[];models=[]
    class Runtime:
        def __init__(self,registry):pass
        def build_model(self):
            model=build_model();models.append(model);return model
        def prepare_observation(self,state,**kwargs):
            return tiny_bundle(state['parent'],observed[state['parent']],registry.reference)
        def train_step(self,model,bundle,target,optimizer,metadata,*,resource_check):
            visits.append(bundle.parent_id);update(model,optimizer)
            return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,native_focals=10000,
                synthetic_objective_only=True)
    class Joiner:
        def __init__(self,*args,**kwargs):pass
        def join(self,parent,**kwargs):
            return SimpleNamespace(parent_id=parent,focal_registry_ref=refs[parent]['focal_registry_ref'],
                focal_identity_sha256='c'*64,receipt=dict(parent_id=parent,split='train',
                parent_registry_ref=registry.reference,truth_ref=refs[parent]['truth_ref'],
                focal_registry_ref=refs[parent]['focal_registry_ref'],focal_identity_sha256='c'*64,
                records=10000,labels_in_observation_bank=False))
    fresh=SimpleNamespace(INPUT_REFS=training.fresh.INPUT_REFS,CLASSES=training.fresh.CLASSES,
        FreshTrainTargetJoin=Joiner)
    def pool_factory(spec,original,registry,fresh,guard,event,tick):
        pool=pool_module.CompactParentPool(parent_bounds={p:80*2**20 for p in parents},input_refs=observed,
            external_arrays={},max_cache_bytes=352*GIB,max_rss_bytes=512*GIB,
            preparation_headroom_bytes=128*GIB,rss_bytes=lambda:0,resource_check=tick,event=event)
        pools.append(pool);return pool
    def spec(name,resume=None):
        return dict(schema='treeloom_upfront_RAM_resume_fit_v1',job_ref=fitter.JOB,
            observation_barrier_ref=fitter.BARRIER,output_directory=str((base/name).relative_to(ROOT)),
            max_rss_bytes=512*GIB,observation_cache_bytes=352*GIB,training_headroom_bytes=128*GIB,
            max_cuda_reserved_bytes=16*GIB,max_output_bytes=64*2**20,source_refs=fitter.source_refs(),
            physical_gpu=5,gpu_uuid=fitter.GPU_UUID,resume_checkpoint_ref=resume)
    def seed():
        random.seed(4201);np.random.seed(4202);torch.manual_seed(4203)
    seed()
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(fitter,'metadata_bindings',return_value=(fresh,original,registry,plans)))
        actual_read=fitter.read_ref
        stack.enter_context(mock.patch.object(fitter,'read_ref',side_effect=lambda ref:barrier if ref == fitter.BARRIER else actual_read(ref)))
        stack.enter_context(mock.patch.object(binding,'ReadableTreeLoom',Runtime))
        stack.enter_context(mock.patch.object(step,'install_execution',side_effect=lambda *args,**kwargs:None))
        stack.enter_context(mock.patch.object(dataset,'load_observation',side_effect=lambda ref,*args:{'parent':ref['path'].split('/')[1]}))
        stack.enter_context(mock.patch.object(fitter,'create_pool',side_effect=pool_factory))
        stack.enter_context(mock.patch.object(fitter,'enable_gpu',side_effect=lambda *args:None))
        stack.enter_context(mock.patch.object(fitter.gc,'collect',return_value=0))
        stack.enter_context(mock.patch.object(torch,'set_num_interop_threads',side_effect=lambda value:None))
        stack.enter_context(mock.patch.object(torch,'set_num_threads',side_effect=lambda value:None))
        stack.enter_context(mock.patch.object(torch.cuda,'synchronize',side_effect=lambda:None))
        stack.enter_context(mock.patch.object(torch.cuda,'empty_cache',side_effect=lambda:None))
        stack.enter_context(mock.patch.object(torch.cuda,'max_memory_reserved',return_value=0))
        def first_guard(label):
            need(not torch.cuda.is_initialized(),'CPU smoke only')
            if label == 'TreeLoom_before_complete_parent' and len(visits) == 7:
                raise RuntimeError('Synthetic stop after persisted update 7')
        try:
            fitter.run_fit(spec('first'),resource_check=first_guard)
        except RuntimeError as error:
            need(str(error) == 'Synthetic stop after persisted update 7','Expected controlled synthetic stop')
        else:
            raise AssertionError('First synthetic process must stop after update 7')
        resume=file_ref(base/'first/checkpoints/latest.pt')
        before=torch.load(ROOT/resume['path'],map_location='cpu',weights_only=True)
        need(before['complete_parent_updates'] == 7,'Actual persisted step 7')
        # Scramble all random sources before the second process simulation.
        random.seed(3);np.random.seed(4);torch.manual_seed(5)
        result=fitter.run_fit(spec('resumed',resume),resource_check=lambda label:need(not torch.cuda.is_initialized(),'CPU only'))
    expected=[p for e in range(1,31) for p in registry.epoch_order(e)]
    need(visits == expected and result['updates_executed_this_process'] == 593
         and result['resumed_from_updates'] == 7,'No repeated or skipped complete update')
    need(file_ref(base/'first/checkpoints/latest.pt') == resume,'Original resume checkpoint preserved')
    saved=torch.load(ROOT/result['checkpoint_ref']['path'],map_location='cpu',weights_only=True)
    seed();gold=build_model()
    optimizer=torch.optim.AdamW(gold.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
        maximize=False,amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)
    for _ in range(600):update(gold,optimizer)
    need(all(torch.equal(value,gold.state_dict()[name]) for name,value in saved['model_state'].items()),
         'Resumed actual model must exactly equal uninterrupted CPU result')
    for key,row in optimizer.state_dict()['state'].items():
        for name,value in row.items():
            need(torch.equal(value,saved['optimizer_state']['state'][key][name]),'Exact resumed AdamW state')
    final_rng=checkpoint.random_state()
    need(final_rng['python'] == saved['random_state']['python']
         and final_rng['numpy'] == saved['random_state']['numpy']
         and torch.equal(final_rng['torch_CPU'],saved['random_state']['torch_CPU']),
         'Python, NumPy and Torch random states exactly preserved')
    receipt=dict(schema='treeloom_exact_resume_CPU_smoke_v1',status='passed',synthetic_only=True,
        CUDA_initialized=False,scientific_data_loaded=False,scientific_truth_loaded=False,
        persisted_initial_updates=7,resumed_updates=593,total_actual_updates=600,
        repeated_updates=0,omitted_updates=0,parameter_tensors=49,parameters=32132,
        actual_compact_pool_codec_used=True,exact_uninterrupted_model_and_AdamW_match=True,
        exact_Python_NumPy_Torch_RNG_match=True,maintained_prediction_checkpoint_validator_passed=True,
        first_checkpoint_preserved=True,elapsed_seconds=time.monotonic()-started,
        source_refs=[file_ref(fitter.__file__),file_ref(checkpoint.__file__),file_ref(__file__)])
    write_json(Path(__file__).parent/'smoke_resume_v1.result.json',receipt)
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    main()
