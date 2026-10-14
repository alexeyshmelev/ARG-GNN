"""Bounded complete-tree C3 blocks over exact native ARGformer observation/focals.

CPU workers load a parent tree sequence once and memory-map genotype columns.
No truth, full-parent graph bank, subsampled tree or collapsed unary is read.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import json,math,os,tempfile,sys,uuid,multiprocessing.connection as mp_connection
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,get_worker_info
from torch_geometric.data import Data
from .american_c3_common_v1 import ROOT,CLASSES,require,local,read,bound,ref,cli
from .american_c3_evaluate_v1 import registry
from .american_c3_gnn_observations_v1 import native_tree,encode_block
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,_LEVEL_FIELDS
SOURCE='src/arggnn/american_c3_gnn_dataset_v1.py'
GENEALOGY='inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG'
BANK_FIELDS=tuple(f.name for f in fields(MutationRoutedWholeTreeInputsV1))
BLOCK_TENSORS=('genomic_geometry','focal_rows','focal_hold','focal_sample_column')
DEFAULTS=dict(schema='american_c3_whole_tree_block_plan_v1',execution_enabled=False,observation_ref=None,
    focal_registry_ref=None,native_preparation_ref=None,class_order=list(CLASSES),expected_haplotypes=None,reference_counts=None,input_reference_class_mask=None,input_reference_sample_indices=None,
    max_trees_per_block=32,max_nodes_per_block=250000,max_sites_per_block=4096,
    donor_age_split=350.,allow_missing=False,output_cap_bytes=20000000)


def load_observation(observation_ref,focal_registry_ref,config):
    import tskit
    m=read(observation_ref);roster=read(m['roster_ref']);prep=read(config['native_preparation_ref'])
    require(prep['observation_ref']==observation_ref and prep['focals_ref']==focal_registry_ref
        and prep['parent_id']==m['parent_id'] and prep['genealogy_kind']==GENEALOGY
        and prep['class_order']==m['class_order'] and prep['query_ancestry_truth_opened'] is False,
        'Exact native ARGformer preparation tree/observation/focal binding required')
    require(m['schema']=='american_c3_observed_v1' and m['genealogy_kind']==GENEALOGY
        and m['query_ancestry_truth_in_inputs'] is False and m['class_order']==config['class_order']
        and roster['parent_id']==m['parent_id'] and roster['class_order']==m['class_order'],'Same sanitized inferred C3 observation as native required')
    rows=roster['samples'];h=len(rows);classes=len(m['class_order'])
    require(classes>=2 and len(set(m['class_order']))==classes and [x['sample_index'] for x in rows]==list(range(h))
        and (config['expected_haplotypes'] is None or h==config['expected_haplotypes']),'Explicit generic class/sample roster required')
    labels=[];ages=[];keys=[];query_columns=[]
    for row in rows:
        role=row['role'];label=row['observed_reference_class']
        require(role in ('query','pure_source_reference'),'Only actual queries/known source references supported')
        require((label is None if role=='query' else type(label) is int and 0<=label<classes),'No query ancestry in inputs')
        if role=='query':query_columns.append(row['sample_index'])
        labels.append(-1 if label is None else label);ages.append(row['age_generations']);keys.append((row['sample_id'],row['haplotype']))
    require(len(set(keys))==h and all(math.isfinite(a) and a>=0 for a in ages),'Unique observed phased sample/date joins')
    labels=np.asarray(labels,dtype=np.int64);ages=np.asarray(ages,dtype=np.float64)
    counts=[int(np.count_nonzero(labels==c)) for c in range(classes)]
    require(sum(counts)>0 and np.any(labels<0) and (config['reference_counts'] is None or counts==config['reference_counts']),
        'Explicit original observed reference count contract required')
    class_mask=config['input_reference_class_mask']
    class_mask=[True]*classes if class_mask is None else class_mask
    require(len(class_mask)==classes and all(type(x) is bool for x in class_mask),'Explicit per-class observed reference mask')
    original_reference=labels>=0;active=original_reference.copy()
    for c,enabled in enumerate(class_mask):
        if not enabled:active &= labels!=c
    selected=config['input_reference_sample_indices']
    if selected is not None:
        require(isinstance(selected,list) and all(type(x) is int and 0<=x<h and original_reference[x] for x in selected)
            and len(set(selected))==len(selected),'Only unique originally known reference sample columns can be enabled')
        active &= np.isin(np.arange(h),selected)
    labels=np.where(active,labels,-1).astype(np.int64)
    require(active.any(),'At least one observed reference remains; absent class/cohort donors stay masked')
    query_columns=np.asarray(query_columns,dtype=np.int64)
    ts=tskit.load(str(bound(m['tree_ref'])));nodes=np.asarray(ts.samples(),dtype=np.int64)
    require(ts.num_migrations==ts.num_populations==ts.num_individuals==0 and not ts.tables.metadata
        and not ts.num_provenances and (ts.tables.nodes.population==-1).all()
        and (ts.tables.nodes.individual==-1).all()
        and all(len(table.metadata)==0 for table in [ts.tables.nodes,ts.tables.edges,ts.tables.sites,ts.tables.mutations])
        and nodes.tolist()==[x['observed_node_id'] for x in rows]
        and np.array_equal(ts.nodes_time[nodes],ages),'Full sanitized inferred topology/sample dates required')
    g=np.load(bound(m['genotypes_ref']),mmap_mode='r',allow_pickle=False)
    pos=np.load(bound(m['positions_ref']),allow_pickle=False)
    require(g.dtype==np.int8 and g.shape==(len(pos),h) and len(pos)==ts.num_sites
        and np.array_equal(pos,ts.tables.sites.position),'Exact full phased observed genotype axes required')
    allele_count=[]
    with bound(m['alleles_ref']).open() as stream:
        for line in stream:allele_count.append(len(json.loads(line)))
    allele_count=np.asarray(allele_count,dtype=np.int8)
    require(allele_count.shape==(len(pos),) and ((allele_count>0)&(allele_count<=4)).all(),'All declared nucleotide allele rows required')
    focals=registry(focal_registry_ref)
    require(focals['parent_id'].item()==m['parent_id'] and focals['class_order'].tolist()==m['class_order']
        and (focals['sample_indices']<h).all()
        and np.array_equal(nodes[focals['sample_indices']],focals['sample_nodes'])
        and np.isin(focals['sample_indices'],query_columns).all() and (labels[focals['sample_indices']]==-1).all(),'Exact same native focal query/sample identities required')
    breaks=np.asarray(list(ts.breakpoints()),dtype=np.float64)
    require(((focals['positions_bp0']>=0)&(focals['positions_bp0']<ts.sequence_length)).all(),'No focal coordinate extrapolation')
    focal_tree=np.searchsorted(breaks,focals['positions_bp0'],side='right')-1
    return dict(manifest=m,rows=rows,labels=labels,query_columns=query_columns,original_reference_counts=counts,ages=ages,ts=ts,nodes=nodes,genotype=g,
        positions=pos,allele_count=allele_count,focals=focals,focal_tree=focal_tree,breaks=breaks,spans=np.diff(breaks))


def build_plan(config):
    require(config['schema']==DEFAULTS['schema'] and all(type(config[x]) is int and config[x]>0
        for x in ('max_trees_per_block','max_nodes_per_block','max_sites_per_block')),'Positive whole-tree block caps required')
    state=load_observation(config['observation_ref'],config['focal_registry_ref'],config)
    ts=state['ts'];pos=state['positions'];blocks=[];start=0;nodes=0;sites=0;tree_count=0
    for tree in ts.trees():
        count=sum(1 for _ in tree.nodes());lo,hi=np.searchsorted(pos,[tree.interval.left,tree.interval.right])
        site_count=int(hi-lo)
        require(count<=config['max_nodes_per_block'] and site_count<=config['max_sites_per_block'],'One whole tree exceeds cap; new explicit plan required')
        if tree_count and (tree_count==config['max_trees_per_block'] or nodes+count>config['max_nodes_per_block'] or sites+site_count>config['max_sites_per_block']):
            blocks.append(dict(start_tree=start,stop_tree=tree.index,nodes=nodes,sites=sites))
            start=tree.index;nodes=sites=tree_count=0
        nodes+=count;sites+=site_count;tree_count+=1
    blocks.append(dict(start_tree=start,stop_tree=ts.num_trees,nodes=nodes,sites=sites))
    return dict(schema='american_c3_whole_tree_block_index_v1',parent_id=state['manifest']['parent_id'],
        observation_ref=config['observation_ref'],focal_registry_ref=config['focal_registry_ref'],config=config,
        class_order=config['class_order'],haplotypes=len(state['rows']),original_reference_counts=state['original_reference_counts'],effective_reference_counts=[int((state['labels']==c).sum()) for c in range(len(config['class_order']))],
        sequence_length_bp=ts.sequence_length,trees=ts.num_trees,sites=ts.num_sites,focal_records=len(state['focals']['record_ids']),blocks=blocks,
        all_trees_and_samples_retained=True,targets_opened=False,genomic_context='two_pass_full_parent_state_carry',
        feature_tensor_byte_planning_bound_per_block=256*config['max_nodes_per_block']+32*config['max_trees_per_block']*len(state['rows'])
            +(len(state['rows'])+256)*config['max_sites_per_block'])


class AmericanC3BlockV1(Data):
    def __init__(self,bank=None,**metadata):
        super().__init__()
        if bank is not None:
            for name in BANK_FIELDS:self[name]=getattr(bank,name)
            for name,value in metadata.items():self[name]=value
            self.num_nodes=len(bank.parent)
    def to_model_inputs(self):
        return MutationRoutedWholeTreeInputsV1(**{k:tuple(self[k]) if k in _LEVEL_FIELDS else self[k] for k in BANK_FIELDS})
    def pin_memory(self,*args):
        for name in BANK_FIELDS+BLOCK_TENSORS:
            value=self[name];self[name]=tuple(t.pin_memory() for t in value) if name in _LEVEL_FIELDS else value.pin_memory()
        return self
    def __inc__(self,key,value,*args,**kwargs):return 0
    def __cat_dim__(self,key,value,*args,**kwargs):return None


class AmericanC3WholeTreeDatasetV1(Dataset):
    def __init__(self,plan_ref,*,direction='forward',worker_tmp_directory):
        require(direction in ('forward','reverse'),'Explicit complete chronological scan order')
        self.plan_ref=dict(plan_ref);self.plan=read(plan_ref);self.direction=direction
        require(self.plan['schema']=='american_c3_whole_tree_block_index_v1' and self.plan['all_trees_and_samples_retained'] is True
            and self.plan['targets_opened'] is False,'Exact complete-tree source plan required')
        self.worker_tmp_directory=str(local(worker_tmp_directory));self._pid=None;self._state=None
    def __len__(self):return len(self.plan['blocks'])
    def __getstate__(self):
        return dict(self.__dict__,_pid=None,_state=None)
    def __getitem__(self,index):
        if self._pid!=os.getpid():
            self._state=load_observation(self.plan['observation_ref'],self.plan['focal_registry_ref'],self.plan['config']);self._pid=os.getpid()
        state=self._state
        index=int(index) if self.direction=='forward' else len(self)-1-int(index)
        spec=self.plan['blocks'][index];a,b=spec['start_tree'],spec['stop_tree'];breaks=state['breaks']
        lo,hi=np.searchsorted(state['positions'],[breaks[a],breaks[b]])
        g=np.asarray(state['genotype'][lo:hi])
        require(self.plan['config']['allow_missing'] or (g>=0).all(),'Registered no-genotype-error reconstruction requires complete calls')
        tree=state['ts'].at_index(a);trees=[];intervals=[]
        for i in range(a,b):
            require(tree.index==i,'Every original tree retained in chronological order')
            trees.append(native_tree(tree,state['nodes']));intervals.append(tuple(tree.interval))
            if i+1<b:require(tree.next(),'Truncated original tree traversal')
        bank,receipt=encode_block(trees,intervals,g,state['allele_count'][lo:hi],state['positions'][lo:hi],
            state['labels'],state['ages'],num_classes=len(self.plan['class_order']),age_split=self.plan['config']['donor_age_split'],query_columns=state['query_columns'])
        require(receipt['nodes']==spec['nodes'] and receipt['sites']==spec['sites'],'Plan/source whole-tree count drift')
        spans=state['spans'];left=np.zeros(b-a);right=np.zeros(b-a)
        if a>0:left[0]=.5*(spans[a-1]+spans[a])
        if b<len(spans):right[-1]=.5*(spans[b-1]+spans[b])
        if b-a>1:
            left[1:]=.5*(spans[a+1:b]+spans[a:b-1]);right[:-1]=left[1:]
        geometry=np.stack((np.log1p(spans[a:b]),np.log1p(left),np.log1p(right)),-1)/math.log(2000001.)
        rows=np.flatnonzero((state['focal_tree']>=a)&(state['focal_tree']<b))
        return AmericanC3BlockV1(bank,parent_id=self.plan['parent_id'],plan_ref=self.plan_ref,block_index=index,
            start_tree=a,stop_tree=b,sequence_length_bp=state['ts'].sequence_length,complete_parent=a==0 and b==self.plan['trees'],
            genomic_geometry=torch.as_tensor(geometry,dtype=torch.float32),
            focal_rows=torch.as_tensor(rows,dtype=torch.long),
            focal_hold=torch.as_tensor(state['focal_tree'][rows]-a,dtype=torch.long),
            focal_sample_column=torch.as_tensor(state['focals']['sample_indices'][rows],dtype=torch.long),
            observations_receipt=receipt,worker_pid=os.getpid())


_ORIGINAL_ADDRESS=mp_connection.arbitrary_address
def _ipc_address(family):
    if family=='AF_UNIX' and sys.platform.startswith('linux'):return b'\0arggnn-c3-'+str(os.getpid()).encode()+b'-'+uuid.uuid4().hex.encode()
    return _ORIGINAL_ADDRESS(family)
def _worker_init(worker_id):
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1);mp_connection.arbitrary_address=_ipc_address
    base=local(get_worker_info().dataset.worker_tmp_directory);directory=local(base/('worker_'+str(os.getpid())))
    directory.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(directory);tempfile.tempdir=str(directory)
def _identity(block):return block
def make_dataloader(dataset,*,num_workers=2,pin_memory=True):
    require(isinstance(dataset,AmericanC3WholeTreeDatasetV1) and type(num_workers) is int and num_workers>=0,'Explicit CPU block Dataset/workers required')
    args=dict(batch_size=None,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,collate_fn=_identity)
    if num_workers:args.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=1,worker_init_fn=_worker_init)
    return DataLoader(dataset,**args)
def shutdown_dataloader(loader):
    it=getattr(loader,'_iterator',None)
    if it is None:return dict(getattr(loader,'_american_c3_worker_drain',dict(drained=True,worker_pids=[])))
    workers=tuple(it._workers);it._shutdown_workers();loader._iterator=None
    receipt=dict(drained=not any(w.is_alive() for w in workers),worker_pids=[w.pid for w in workers],exitcodes=[w.exitcode for w in workers])
    loader._american_c3_worker_drain=receipt
    return receipt


def execute(config,writer):
    plan=build_plan(config);plan_ref=writer.json('block_index.json',plan)
    return writer.json('completed.json',dict(status='completed_C3_observation_only_block_plan',block_index_ref=plan_ref,
        parent_id=plan['parent_id'],blocks=len(plan['blocks']),trees=plan['trees'],sites=plan['sites'],targets_opened=False))

DEPENDENCIES=('src/arggnn/american_c3_evaluate_v1.py','src/arggnn/american_c3_gnn_observations_v1.py',
    'src/arggnn/mutation_routed_tree_observations_v1.py','src/arggnn/mutation_routed_whole_tree_gnn_v1.py')
if __name__=='__main__':cli(SOURCE,DEFAULTS,execute,dependencies=DEPENDENCIES)
