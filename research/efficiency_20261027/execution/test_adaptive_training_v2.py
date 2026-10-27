"""CPU-only exact remaining600-entry schedule test of candidate training wiring."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import numpy as np
import torch
from arggnn.benchmarking import contracts as c
from research.efficiency_20261027.integration_candidate import training_adaptive_v2 as training
from research.efficiency_20261027.root.compare_host_parallelism import CHECKPOINT
from .adaptive_prefetch_v2 import AdaptiveCompactParents,AdaptiveMemoryGovernor,EncoderCredits

BASE=Path('research/efficiency_20261027/execution')


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=c.read_json('.runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json')
    original=job['call']['spec']['training_spec'];inputs=original['train_inputs']
    saved=torch.load(c.ROOT/CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    registry=training.fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
    barrier=c.read_ref(saved['observation_barrier_ref'])
    before=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().numpy().tobytes())
    schedule=[(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
    updates=[];joins=[];durable=[];periodic=[];active=[False]
    def prepare(parent,*,resource_check):
        seal=barrier['parents'][parent]
        return SimpleNamespace(parent_id=parent,split='train',sequence_length_bp=30000000,
            focal_identity_sha256=seal['focal_identity_sha256'],validate=lambda:None,
            **{key:inputs[parent][key] for key in training.fresh.INPUT_REFS}),dict(
                parent_id=parent,declared_split='train',parent_registry_ref=registry.reference,
                all_native_trees_nodes_sites_samples_retained=True,private_truth_opened=False)
    def pack(bundle,meta,**kwargs):return dict(bundle=bundle,meta=meta,charged_bytes=10,read_peak_upper_bound_bytes=10)
    def restore(record,**kwargs):return record['bundle'],record['meta']
    gov=AdaptiveMemoryGovernor(maximum_bytes=1000000,rss_bytes=lambda:0,
        resource_check=lambda stage:None,reserve_bytes=1000)
    credits=EncoderCredits(maximum_bytes=1000000,allowance_bytes=10,
        family_memory=lambda:dict(bytes=0,recursive_children=0))
    pool=AdaptiveCompactParents(schedule=[parent for _,parent in schedule],start_cursor=268,
        parent_bounds=dict.fromkeys(registry.ids('train'),10),input_refs=inputs,external_arrays={},
        max_cache_bytes=200,governor=gov,credits=credits,resource_check=lambda stage:None,
        event=lambda *args:None,prepare=prepare,pack=pack,restore=restore)
    optimizer=SimpleNamespace(steps=268)
    def join(parent,bundle,meta):
        assert pool.training and bundle.parent_id==parent
        joins.append(parent)
        receipt=dict(parent_id=parent,split='train',parent_registry_ref=registry.reference,
            truth_ref=inputs[parent]['truth_ref'],focal_registry_ref=bundle.focal_registry_ref,
            focal_identity_sha256=bundle.focal_identity_sha256,records=10000,labels_in_observation_bank=False)
        return SimpleNamespace(parent_id=parent,focal_registry_ref=bundle.focal_registry_ref,
            focal_identity_sha256=bundle.focal_identity_sha256,receipt=receipt)
    def step(model,bundle,target,opt,metadata,**kwargs):
        assert pool.training;updates.append(bundle.parent_id);opt.steps+=1
        return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,native_focals=10000)
    def checkpoint(epoch,steps,model,opt,latest=False):
        if latest:
            durable.append((steps,schedule[steps-1][1]))
            pool.durable_checkpoint(parent=schedule[steps-1][1],completed_updates=steps)
        else:periodic.append((epoch,steps))
        return dict(path='CPU_fake_checkpoint',sha256=str(steps))
    try:
        result=training.train_loop(registry=registry,runtime=SimpleNamespace(train_step=step),
            model=None,optimizer=optimizer,train_inputs=inputs,observation_barrier=barrier,
            load_parent=pool.load,join_targets=join,checkpoint=checkpoint,event=lambda *args:None,
            resource_check=lambda stage:pool.raise_if_failed(),optimizer_counts=lambda opt:[opt.steps]*49,
            start_steps=268,initial_checkpoint_refs={str(epoch):dict(CPU_fake=True) for epoch in (5,10)},
            training_phase=pool.training_phase)
        assert updates==joins==[parent for _,parent in schedule[268:]]
        assert len(durable)==332 and pool.cursor==optimizer.steps==result['complete_parent_updates']==600
        assert pool.preparations==dict.fromkeys(registry.ids('train'),1)
        assert periodic==[(15,300),(20,400),(25,500),(30,600)]
        after=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().numpy().tobytes())
        assert before==after and not torch.cuda.is_initialized()
        source=Path(training.__file__).read_text();ast.parse(source)
        assert 'if parent not in plans:' not in source and 'pool.prefetch_next' not in source
        assert 'pool.durable_checkpoint' in source and 'preprocessing_bindings = install_tree(child_credits)' in source
        receipt=dict(status='passed',start_updates=268,remaining_updates=332,final_updates=600,
            all20_prepared_once=True,exact_original_schedule_identity=training.schedule_identity(schedule),
            fake_optimizer_counters_per_update=49,actual_optimizer_executed=False,
            full_native_observation_or_truth_payload_opened=False,global_RNG_unchanged=True,
            fixed_checkpoint_events_preserved=True,pool=pool.receipt())
    finally:pool.close()
    with (BASE/'ADAPTIVE_TRAINING_CPU_v2.json').open('x') as handle:json.dump(receipt,handle,indent=2)
    print(json.dumps(dict(status='passed',remaining_updates=332,final_updates=600)))


if __name__=='__main__':main()
