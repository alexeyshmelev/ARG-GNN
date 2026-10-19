"""Exercise actual fit orchestration and RAM codec with tiny synthetic CPU data.

The 600 synthetic optimizer steps verify bookkeeping, not scientific training or
GPU performance. No original observation, truth or CUDA context is accessed.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import copy
import json
import random
import time
import numpy as np
import torch
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from arggnn.benchmarking import treeloom_training as training, treeloom_prediction as prediction
from arggnn import american_c3_gnn_dataset_v1 as dataset
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_preparation_v1 as original
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import resident_pool_v2 as pool_module
from ..runtime import observations, episodes, cuts, binding
from . import candidate_fit as candidate, step
from .candidate_common import ROOT, GIB, need, write_json, file_ref, read_ref


@dataclass(frozen=True)
class SyntheticBundle:
    bank:object
    episodes:object
    cut:object
    parent_id:str
    split:str
    sequence_length_bp:int
    native_holds:int
    original_sites:int
    focal_hold:object
    focal_sample_column:object
    focal_identity_sha256:str
    observation_ref:dict
    focal_registry_ref:dict
    native_preparation_ref:dict
    bank_stamps:tuple
    episode_stamps:tuple
    focal_stamps:tuple

    def validate(self):
        need(self.bank_stamps == core._stamps(original._bank_tensors(self.bank)), 'Synthetic bank stamps')
        need(self.episode_stamps == core._stamps(original._episode_tensors(self.episodes)), 'Synthetic episode stamps')
        need(self.focal_stamps == core._stamps((self.focal_hold,self.focal_sample_column)), 'Synthetic focal stamps')
        self.cut.validate(self.bank)


def tiny_bundle(parent, refs, registry_ref):
    ages=np.array([0.,400.,0.,400.,0.,100.,349.,351.])
    trees=[]
    for at in range(4):
        if at == 2:
            trees.append(NativeObservedTreeV1(np.full(8,-1,dtype=np.int64),ages.copy(),np.arange(8),np.arange(8)))
            continue
        count=12 if at == 1 else 11
        links=np.full(count,count-1,dtype=np.int64);links[-1]=-1
        links[[4,6] if at == 3 else [4,5]]=8;links[[0,2]]=9
        if at == 1:links[8]=10
        times=np.r_[ages,350. if at == 3 else 200.,250.,*([300.] if at == 1 else []),1000.]
        trees.append(NativeObservedTreeV1(links,times,np.arange(8),np.arange(8)))
    bank,bank_meta=observations.encode_block(trees,np.array([[0.,100.],[100.,200.],[200.,300.],[300.,400.]]),
        np.random.default_rng(9).integers(0,2,size=(3,8),dtype=np.int8),np.full(3,2,dtype=np.int8),
        np.array([50.,150.,350.]),np.array([0,1,2,0,-1,-1,-1,-1]),ages,
        num_classes=3,query_columns=np.arange(4,8))
    isolated=tuple(bank_meta.get('isolated_zero_site_holds',()))
    episode,episode_meta=episodes.build_episodes(bank,max_payload_bytes=2**20,isolated_hold_indices=isolated)
    cut,cut_meta=cuts.prepare_cut_plan_v1(bank,expected_haplotypes=8,max_nodes=1000,max_holds=10,
        isolated_hold_indices=isolated)
    hold,sample=torch.tensor([0,1,3]),torch.tensor([4,5,6])
    bundle=SyntheticBundle(bank,episode,cut,parent,'train',30000000,4,3,hold,sample,'c'*64,**refs,
        bank_stamps=core._stamps(original._bank_tensors(bank)),
        episode_stamps=core._stamps(original._episode_tensors(episode)),
        focal_stamps=core._stamps((hold,sample)))
    metadata=dict(parent_id=parent,declared_split='train',parent_registry_ref=registry_ref,
        all_native_trees_nodes_sites_samples_retained=True,private_truth_opened=False,
        synthetic_fixture=True,observation=bank_meta,episodes=episode_meta,cut=cut_meta)
    return bundle,metadata


class CPUModel(frozen.AmericanTreeLoomC3FullV1):
    def to(self, *, device, dtype):
        need(device == 'cuda:0' and dtype == torch.float32, 'Actual fit device request preserved in test')
        return super().to(device='cpu',dtype=dtype)


def main():
    started=time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base=Path(__file__).parent
    output=base/'candidate_fit_cpu_test.output'
    need(not output.exists(), 'Fresh synthetic smoke output required')
    output.mkdir();(output/'checkpoints').mkdir()
    parents=tuple('synthetic_train_%03d'%i for i in range(20))
    class Registry:
        reference={'mock':'synthetic_registry'}
        def ids(self,split):return parents if split == 'train' else ()
        def split_for(self,parent):return 'train' if parent in parents else 'test'
        def epoch_order(self,epoch):
            values=list(parents);random.Random(70002+epoch).shuffle(values);return values
    registry=Registry()
    refs={parent:{key:dict(path='synthetic/'+parent+'/'+key,sha256='a'*64)
        for key in (*training.fresh.INPUT_REFS,'truth_ref')} for parent in parents}
    observed={parent:{key:refs[parent][key] for key in training.fresh.INPUT_REFS} for parent in parents}
    plans={parent:dict(native_holds=4,original_sites=3) for parent in parents}
    barrier=dict(parent_registry_ref=registry.reference,train_parent_ids=list(parents),
        all20_observations_sealed_before_truth=True,private_truth_opened=False,
        parents={parent:dict(focal_identity_sha256='c'*64,**observed[parent]) for parent in parents})
    barrier_ref=write_json(output/'synthetic_barrier.json',barrier)
    preflight_ref=write_json(output/'synthetic_preflight.json',dict(native_holds=4))
    request=dict(preflight_ref=preflight_ref,observation_barrier_ref=barrier_ref,
        source_refs=[file_ref(base/'candidate_fit.py')],max_rss_bytes=128*GIB)
    request_ref=write_json(output/'synthetic_request.json',request)
    spec=dict(train_inputs=refs,max_truth_uncompressed_bytes=2**20,
        preparation_limits={},settings=copy.deepcopy(training.SETTINGS))
    visits,preparations,gates,pools=[],[],[],[]
    first_bundle=[None]
    model=CPUModel()
    class Runtime:
        def __init__(self,registry):pass
        def build_model(self):return model
        def prepare_observation(self,state,**kwargs):
            parent=state['parent'];preparations.append(parent)
            result=tiny_bundle(parent,observed[parent],registry.reference)
            if len(preparations) == 1:first_bundle[0]=result[0]
            return result
        def train_step(self,model,bundle,target,optimizer,metadata,*,resource_check):
            visits.append(bundle.parent_id)
            optimizer.zero_grad(set_to_none=True)
            loss=sum(parameter.square().mean() for parameter in model.parameters())
            loss.backward();optimizer.step()
            return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,
                native_focals=10000,synthetic_objective_only=True)
    class Joiner:
        def __init__(self,*args,**kwargs):pass
        def join(self,parent,**kwargs):
            return SimpleNamespace(parent_id=parent,focal_registry_ref=refs[parent]['focal_registry_ref'],
                focal_identity_sha256='c'*64,receipt=dict(parent_id=parent,split='train',
                parent_registry_ref=registry.reference,truth_ref=refs[parent]['truth_ref'],
                focal_registry_ref=refs[parent]['focal_registry_ref'],focal_identity_sha256='c'*64,
                records=10000,labels_in_observation_bank=False))
    class Guard:
        def __init__(self):self.request=request
        def check(self,label):need(not torch.cuda.is_initialized(), 'Synthetic smoke remains CPU only')
        def wait_gate(self,name,actual_ref,*,qualification_ref):
            need(actual_ref == request_ref and len(visits) == 1, 'Gate follows one actual synthetic update')
            qualification=read_ref(qualification_ref)
            need(qualification['optimizer_updates'] == 1, 'Qualified update preserved')
            gates.append(name)
            return write_json(output/'synthetic_continuation_gate.json',dict(qualification_ref=qualification_ref))
    guard=Guard()
    def pool_factory(request,spec,registry,fresh,guard,event,resource_check):
        need(len(visits) == 1 and guard.request['max_rss_bytes'] == 512*GIB,
             'RAM pool starts only after first-step continuation allocation')
        compact=pool_module.CompactParentPool(parent_bounds={p:80*2**20 for p in parents},
            input_refs=observed,external_arrays={},max_cache_bytes=352*GIB,max_rss_bytes=512*GIB,
            preparation_headroom_bytes=128*GIB,rss_bytes=lambda:0,resource_check=resource_check,event=event)
        original_pack=compact.pack
        def pack(bundle,metadata,**kwargs):
            if not compact.records:
                need(bundle is first_bundle[0], 'Qualification bundle reused without reprepare')
            record=original_pack(bundle,metadata,**kwargs)
            restored,restored_meta=compact.restore(record,input_refs=kwargs['input_refs'],verify_integrity=True)
            need(pool_module.exact_signature(bundle,metadata) == pool_module.exact_signature(restored,restored_meta),
                 'Actual compact RAM codec preserves all synthetic tensors and metadata')
            return record
        compact.pack=pack;pools.append(compact);return compact
    cuda=SimpleNamespace(is_initialized=lambda:False,synchronize=lambda:None,
        empty_cache=lambda:None,max_memory_reserved=lambda index:0)
    torch_facade=SimpleNamespace(cuda=cuda,optim=torch.optim,float32=torch.float32,save=torch.save,load=torch.load)
    fresh=SimpleNamespace(INPUT_REFS=training.fresh.INPUT_REFS,CLASSES=training.fresh.CLASSES,
        FreshTrainTargetJoin=Joiner)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(candidate,'configure_torch',return_value=torch_facade))
        stack.enter_context(mock.patch.object(candidate,'metadata_bindings',return_value=(fresh,spec,registry,plans)))
        stack.enter_context(mock.patch.object(candidate,'enable_gpu',side_effect=lambda *args:None))
        stack.enter_context(mock.patch.object(candidate,'create_pool',side_effect=pool_factory))
        stack.enter_context(mock.patch.object(candidate,'continuation_allocation',return_value=dict(
            max_rss_bytes=512*GIB,cpu_admission_ref={'synthetic':'no_external_admission'})))
        stack.enter_context(mock.patch.object(binding,'ReadableTreeLoom',Runtime))
        stack.enter_context(mock.patch.object(step,'install_execution',side_effect=lambda *args,**kwargs:None))
        stack.enter_context(mock.patch.object(dataset,'load_observation',side_effect=lambda ref,*args:{'parent':ref['path'].split('/')[1]}))
        # This isolated smoke avoids repeated full-process GC; actual driver retains it.
        stack.enter_context(mock.patch.object(candidate.gc,'collect',return_value=0))
        candidate.fit(request,request_ref,{},barrier,output,guard)
    compact=pools[0]
    expected=[p for epoch in range(1,31) for p in registry.epoch_order(epoch)]
    need(visits == expected and preparations == expected[:20], 'Exact original schedule and once-only preparations')
    need(compact.preparations == {p:1 for p in parents} and compact.restorations == {p:29 for p in parents},
         'Exactly 20 preparations and 580 restores')
    need(gates == ['continue_gate.json'], 'One first-update continuation gate')
    completion=read_ref(file_ref(output/'training_completed.json'))
    saved=torch.load(ROOT/completion['checkpoint_ref']['path'],map_location='cpu',weights_only=True)
    prediction.validate_saved_checkpoint(saved,model,registry,completion)
    prediction._completion_binding(registry,completion['checkpoint_ref'],file_ref(output/'training_completed.json'))
    need(len(list((output/'checkpoints').glob('*.pt'))) == 7, 'First update plus all six scheduled checkpoints retained')
    need(not torch.cuda.is_initialized(), 'No CUDA context in synthetic smoke')
    receipt=dict(schema='treeloom_candidate_fit_CPU_smoke_v1',status='passed',
        synthetic_only=True,scientific_observations_loaded=False,scientific_truth_loaded=False,
        CUDA_initialized=False,actual_synthetic_AdamW_updates=600,actual_parameter_tensors=49,
        actual_parameter_count=sum(p.numel() for p in model.parameters()),
        preparations=20,exact_restorations=580,first_prepared_parent_reused=True,
        all20_roundtrip_tensor_metadata_checks_passed=True,maintained_prediction_contract_passed=True,
        retained_checkpoints=7,elapsed_seconds=time.monotonic()-started,
        fit_source_ref=file_ref(base/'candidate_fit.py'),test_source_ref=file_ref(Path(__file__)),
        compact_pool_receipt=compact.receipt())
    write_json(base/'candidate_fit_cpu_test.result.json',receipt)
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
