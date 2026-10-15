#!/usr/bin/env python3
"""Thin prospective1611 wrapper around frozen C4 PyG equations.

Numeric preparation/fit/predict are explicit commands. No model/data action occurs
on import; original observation/model/native/generation sources stay frozen.
"""
from __future__ import annotations
import argparse
import collections
from dataclasses import fields
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
NATIVE=ROOT/'research/reviews/ARGMix_prospective1611_native_adapters_20261016_v1'
GENERATION=ROOT/'research/reviews/ARGMix_prospective1611_generation_20261016_v1'
LABELS=('Ana','CHG','WHG','EHG')
HAPS=3222
QUERIES=2972
PURE_COUNTS=(44,26,96,84)
TRAIN=tuple('published_recon_train_%03d'%i for i in range(5))
VAL=('published_recon_val_000',)
IID=tuple('published_recon_iid_%03d'%i for i in range(3))


def read(path):return json.loads(Path(path).read_text())
def jsonl(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def require(v,message):
    if not v:raise ValueError(message)
def digest(v):require(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v),'invalid SHA256');return v

def ref(path):
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def verify_ref(r):
    require(set(r)=={'path','bytes','sha256'},'exact file path/bytes/SHA required')
    p=Path(r['path']);require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'artifact identity differs: '+str(p));return p

def verify_sources():
    pins=read(HERE/'source_pins.v1.json')
    for r in pins['sources']:verify_ref({'path':str(ROOT/r['path']),'bytes':r['bytes'],'sha256':r['sha256']})
    return len(pins['sources'])

def source_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def metadata_modules(*,truth=False):
    verify_sources();sys.path.insert(0,str(NATIVE))
    panel=source_module('panel_adapter',NATIVE/'panel_adapter.py')
    generation=source_module('_prospective1611_generation',GENERATION/'generation.py');generation.verify_sources()
    ancestry=source_module('_prospective1611_external_truth',NATIVE/'truth_adapter.py') if truth else None
    return panel,generation,ancestry


def class_roles(roster):
    """Observed pure founder identities only; mixed/query ancestry remains unknown."""
    refs=[];ages=[];query=[];identities=[]
    for i,r in enumerate(roster['rows']):
        identities.append(r['haplotype_id']);ages.append(r['age_generations'])
        c=r['cohort'];refs.append(LABELS.index(c) if c in LABELS else -1)
        if c in ('NF','Yam','PostNeolithic','Present'):query.append(i)
    require(len(identities)==HAPS and len(set(identities))==HAPS and len(query)==QUERIES,'full3222/2972 identity condition required')
    require(tuple(refs.count(c) for c in range(4))==PURE_COUNTS,'pure founder class counts44/26/96/84 required')
    require(all(refs[q]==-1 for q in query),'query ancestry must be unknown')
    return refs,ages,query


def target_indices(targets,roster,intervals):
    """Exact tree/haplotype join, no nearest-grid or physical-span relabeling."""
    _,_,queries=class_roles(roster)
    by_hap={roster['rows'][col]['haplotype_id']:q for q,col in enumerate(queries)}
    out=[]
    for r in targets:
        index=r['tree_index'];require(0<=index<len(intervals),'target tree index outside inferred bank')
        require(tuple(map(float,intervals[index]))==(float(r['left']),float(r['right'])),'target interval differs from actual inferred tree')
        require(r['haplotype_id'] in by_hap,'target must be an unknown query')
        out.append((index,by_hap[r['haplotype_id']]))
    require(len(out)==len(set(out)),'duplicate target tree/query pair')
    return out


def dense_native_forecast(trees,queried_haplotypes):
    require(type(trees) is int and trees>0 and type(queried_haplotypes) is int and queried_haplotypes>0,'positive context counts required')
    per_query=1+(max(trees-1024,0)+511)//512
    chunks=per_query*queried_haplotypes
    return {'status':'source_derived_expected_counts_requires_actual_native_emission_confirmation',
            'native_context_rows_if_one_per_tree_query':trees*queried_haplotypes,
            'native_chunks_1024_stride512':chunks,'final_dense_chunk_tensor_bytes':chunks*672768,
            'chunk_list_plus_stack_transient_bytes_at_least':chunks*1345536,
            'converted_context_tensor_bytes_at_least':trees*queried_haplotypes*640,
            'extra_not_included':'decoded Python JSON/tensors, simultaneous TRAIN+VAL, kept-chunk loader copies, batch/device/model activations'}


class Progress:
    def __init__(self,path,cap):self.path=Path(path);self.cap=cap;self.bytes=0;self.start=time.monotonic();self.stream=self.path.open('x')
    def emit(self,event,**values):
        row={'event':event,'elapsed_seconds':time.monotonic()-self.start,**values}
        line=json.dumps(row,sort_keys=True,allow_nan=False)+'\n';self.bytes+=len(line.encode())
        require(self.bytes<=self.cap,'progress log cap exceeded');self.stream.write(line);self.stream.flush();print(line,end='',flush=True)
    def close(self):self.stream.close()


def root_request(path,kind):
    verify_sources();request=read(path)
    require(request.get('schema')=='prospective1611_pyg_'+kind+'_request_v1' and request.get('root_approved') is True,'explicit root-approved '+kind+' request required')
    require(request.get('wrapper_source_sha256')==sha(HERE/'pyg_adapter.py') and request.get('source_pins_sha256')==sha(HERE/'source_pins.v1.json'),'wrapper/source closure binding differs')
    runtime=read(verify_ref(request['runtime_binding']));resources=read(verify_ref(request['resource_binding']))
    require(runtime.get('schema')=='prospective1611_pyg_runtime_binding_v1' and runtime.get('root_qualified') is True,'root-qualified existing runtime required')
    digest(runtime.get('python_executable_sha256'));digest(runtime.get('qualification_receipt_sha256'))
    require(set(runtime.get('modules',{}))=={'numpy','torch','torch_geometric','tskit'},'explicit numerical module bindings required')
    for r in runtime['modules'].values():digest(r.get('sha256'));require(isinstance(r.get('version'),str) and r['version'] and isinstance(r.get('file'),str),'module version/file required')
    require(resources.get('schema')=='prospective1611_pyg_resource_binding_v1' and resources.get('root_approved') is True and resources.get('enforced_by_root_wrapper') is True,'root-enforced resource binding required')
    for key in ('cpu_threads','max_cpu_rss_bytes','max_gpu_reserved_bytes','max_prepared_tensor_bytes','max_parent_nodes','max_parent_trees','max_sites','max_output_bytes','max_log_bytes','wall_seconds'):
        require(type(resources.get(key)) is int and resources[key]>0,'positive resource '+key+' required')
    digest(resources.get('wrapper_source_sha256'));digest(resources.get('admission_receipt_sha256'))
    require(not Path(request['output_dir']).exists(),'fresh output directory required')
    return request,runtime,resources


def numeric_modules(binding):
    import numpy
    import torch
    import torch_geometric
    import tskit
    require(sha(Path(sys.executable).resolve())==binding['python_executable_sha256'],'qualified Python executable differs')
    for name,module in {'numpy':numpy,'torch':torch,'torch_geometric':torch_geometric,'tskit':tskit}.items():
        r=binding['modules'][name]
        require(module.__version__==r['version'] and Path(module.__file__).resolve()==Path(r['file']).resolve() and sha(module.__file__)==r['sha256'],'loaded runtime '+name+' differs')
    return numpy,torch,tskit


def tensor_bytes(bank):
    size=0
    for field in fields(bank):
        value=getattr(bank,field.name)
        for tensor in (value if isinstance(value,tuple) else (value,)):size+=tensor.numel()*tensor.element_size()
    return size


def read_shared_observations(request):
    panel,generation,_=metadata_modules()
    observed=read(verify_ref(request['observed_generation_receipt']));inference=read(verify_ref(request['inference_receipt']))
    require(observed.get('schema')=='prospective1611_shared_observed_generation_v1' and observed.get('complete') is True,'complete shared observation receipt required')
    require(observed.get('generator_source_sha256')==sha(GENERATION/'generation.py') and observed.get('science_plan_sha256')==sha(generation.PLAN),'shared generation source/condition differs')
    condition=read(HERE/'data_condition.v1.json')
    require(observed.get('map_derivation_receipt_sha256')==condition['primary_map_derivation_receipt']['sha256'] and observed['map_binding']['source_interval_bp']==condition['source_interval_bp'] and observed['map_binding']['length_bp']==condition['length_bp'],'exact root-selected full chr22 condition required; no silent downscale')
    parent=observed['parent_id'];require(parent in TRAIN+VAL+IID and request['parent_id']==parent,'registered common parent required')
    require(inference.get('schema')=='prospective1611_inferred_bank_receipt_v1' and inference.get('complete') is True and inference.get('parent_id')==parent,'complete inferred common parent required')
    require(inference.get('observed_generation_receipt_sha256')==request['observed_generation_receipt']['sha256'],'inference observed realization differs')
    require(inference.get('true_genealogy_used_as_input') is False and inference.get('map_derivation_receipt_sha256')==observed['map_derivation_receipt_sha256'],'observed-only exact-map inference required')
    generation.validate_inference_coordinates(inference,observed)
    require(set(inference.get('files',{}))=={'observed_trees','roster','observed_node_mapping'},'exact observed inferred-bank refs required')
    generation.verify_manifest(observed['files']);generation.verify_manifest(inference['files'])
    original=read(observed['files']['roster']['path']);roster=read(inference['files']['roster']['path'])
    require(canonical(panel.bind_roster(original,read(inference['files']['observed_node_mapping']['path'])))==canonical(roster),'actual observed mapping/roster differs')
    bound=panel.Panel(roster);class_roles(roster)
    targets=panel.read_jsonl(verify_ref(request['targets']))
    panel.validate_targets(targets,bound,parent_id=parent)
    # There is deliberately no private generation/truth reference in this interface.
    return panel,observed,inference,roster,targets


def read_haps(np,observed,roster,length,progress,resources):
    """One already-shared observed matrix; no re-error, refiltering or phasing."""
    manifest_path=observed['files']['haplotypes.tsv']['path']
    with Path(manifest_path).open() as f:
        header=f.readline().strip().split('\t');rows=[line.strip().split('\t') for line in f if line.strip()]
    require(header==['haplotype_column_zero_based','haplotype_id','diploid_id','phase','cohort','age_generations'] and len(rows)==HAPS,'explicit full phased export-column mapping required')
    for col,(row,expected) in enumerate(zip(rows,roster['rows'])):
        require(row==list(map(str,(col,expected['haplotype_id'],expected['diploid_id'],expected['phase'],expected['cohort'],expected['age_generations']))),'export phase/sample metadata differs from roster')
    sites=observed['observed_sites'];require(type(sites) is int and 0<sites<=resources['max_sites'],'site count cap exceeded')
    g=np.empty((sites,HAPS),dtype=np.int8);positions=np.empty(sites,dtype=np.float64)
    with Path(observed['files']['observed.haps']['path']).open() as f:
        count=0
        for line in f:
            parts=line.split();require(count<sites and len(parts)==5+HAPS and parts[0]=='22' and parts[3:5]==['A','C'],'shared binary Oxford haps row invalid')
            x=int(parts[2])-1;require(0<=x<length and (count==0 or positions[count-1]<x),'explicit POS-1 normalized coordinate invalid')
            require(all(v in ('0','1') for v in parts[5:]),'complete known-phase binary calls required')
            g[count]=[int(v) for v in parts[5:]];positions[count]=x;count+=1
            if count%10000==0:progress.emit('observed_genotype_rows',parent_id=observed['parent_id'],sites_loaded=count,total_sites=sites)
    require(count==sites,'every shared observed variant must survive')
    return g,positions


def prepare(request_path):
    request,runtime,resources=root_request(request_path,'prepare')
    require(set(request)=={'schema','root_approved','wrapper_source_sha256','source_pins_sha256','runtime_binding','resource_binding','output_dir','parent_id','observed_generation_receipt','inference_receipt','targets'},'preparation accepts observations/targets only; labels/private truth forbidden')
    panel,observed,inference,roster,targets=read_shared_observations(request)
    np,torch,tskit=numeric_modules(runtime);torch.set_num_threads(resources['cpu_threads'])
    sys.path.insert(0,str(ROOT/'src'))
    from arggnn.american_c3_gnn_observations_isolated_v2 import native_tree,encode_block
    from arggnn.mutation_routed_whole_tree_gnn_v1 import validate_inputs_v1
    out=Path(request['output_dir']);out.mkdir(parents=True,exist_ok=False);log=Progress(out/'progress.jsonl',resources['max_log_bytes'])
    parent=observed['parent_id'];log.emit('prepare_started',parent_id=parent,targets=len(targets),haplotypes=HAPS,queries=QUERIES)
    ts=tskit.load(inference['files']['observed_trees']['path']);length=observed['map_binding']['length_bp']
    require(float(ts.sequence_length)==length and int(ts.num_trees)<=resources['max_parent_trees'],'inferred length/tree cap differs')
    panel_runtime=source_module('_native1611_runtime_metadata',NATIVE/'native_runtime.py')
    panel_runtime.check_observed_mapping(ts,panel.Panel(roster),targets)
    reference,age,queries=class_roles(roster);nodes=[r['node_id'] for r in roster['rows']]
    genotype,positions=read_haps(np,observed,roster,length,log,resources)
    trees=[];intervals=[];node_count=0
    for tree in ts.trees():
        native=native_tree(tree,nodes);node_count+=len(native.parent)
        require(node_count<=resources['max_parent_nodes'],'complete-parent node-occurrence cap exceeded; no truncation allowed')
        trees.append(native);intervals.append([float(tree.interval.left),float(tree.interval.right)])
        if len(trees)%100==0 or len(trees)==int(ts.num_trees):log.emit('observed_tree_rows',parent_id=parent,trees_loaded=len(trees),total_trees=int(ts.num_trees),node_occurrences=node_count)
    require(intervals[0][0]==0 and intervals[-1][1]==length,'all normalized original holds required')
    pairs=target_indices(targets,roster,intervals)
    cohort_by_hap={r['haplotype_id']:r['cohort'] for r in roster['rows']}
    target_cohorts=dict(collections.Counter(cohort_by_hap[r['haplotype_id']] for r in targets))
    forecast=dense_native_forecast(len(trees),len({r['haplotype_id'] for r in targets}))
    log.emit('label_free_feature_counts',parent_id=parent,trees=len(trees),node_occurrences=node_count,sites=len(genotype),haplotypes=HAPS,queries=QUERIES,pure_founder_counts=list(PURE_COUNTS),targets=len(targets),native_dense_forecast=forecast)
    bank,encoding=encode_block(trees,np.asarray(intervals),genotype,np.full(len(genotype),2,dtype=np.int64),positions,np.asarray(reference,dtype=np.int64),np.asarray(age,dtype=np.float64),num_classes=4,age_split=350.,query_columns=np.asarray(queries,dtype=np.int64))
    structure=validate_inputs_v1(bank,expected_haplotypes=HAPS);size=tensor_bytes(bank)
    require(size<=resources['max_prepared_tensor_bytes'],'complete bank tensor cap exceeded')
    payload={'schema':'prospective1611_pyg_observed_bank_v1','parent_id':parent,'bank':bank,'target_ids':[r['row_id'] for r in targets],
             'target_tree':torch.tensor([p[0] for p in pairs],dtype=torch.long),'target_query_slot':torch.tensor([p[1] for p in pairs],dtype=torch.long)}
    torch.save(payload,out/'observed_bank.pt')
    meta={'schema':'prospective1611_pyg_preparation_v1','complete':True,'request':ref(request_path),'runtime_binding':request['runtime_binding'],'resource_binding':request['resource_binding'],'parent_id':parent,'split':observed['split'],
          'bank':ref(out/'observed_bank.pt'),'targets':request['targets'],'roster':inference['files']['roster'],
          'observed_generation_receipt':request['observed_generation_receipt'],'inference_receipt':request['inference_receipt'],
          'wrapper_source_sha256':sha(HERE/'pyg_adapter.py'),'source_pins_sha256':sha(HERE/'source_pins.v1.json'),
          'feature_counts':{**structure,'tensor_bytes':size,'raw_float64_node_occurrences':node_count,'target_cohort_counts':target_cohorts},'native_dense_forecast':forecast,
          'common_observed_metadata':{'parent_id':parent,'target_rows':len(targets),'targets_sha256':request['targets']['sha256'],
              'roster_sha256':inference['files']['roster']['sha256'],'observed_trees_sha256':inference['files']['observed_trees']['sha256']},
          'method_label':'Mutation-routed PyG C4 prospective1611','query_ancestry_labels_read':False,'truth_or_private_generation_read':False,
          'mixed_role_representation':'NF/Yam unknown graph/genotype context; pure founder reference frequencies only; no six-category donor-token claim',
          'physical_geometry':'unchanged model span/log(2000001) encoding; normalized0-based physical bp; no total-length rescaling',
          'encoding':encoding,'paper_reproduction':False}
    write(out/'preparation_receipt.json',meta);log.emit('prepare_completed',parent_id=parent,tensor_bytes=size,bank_sha256=meta['bank']['sha256']);log.close()
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=resources['max_output_bytes'],'prepared output byte cap exceeded')


def load_preparation(entry,*,with_labels):
    panel,_,truth=metadata_modules(truth=with_labels)
    required={'parent_id','preparation_receipt'}|({'labels','supervision_contract'} if with_labels else set())
    require(set(entry)==required,'exact fit/predict metadata boundary required')
    meta=read(verify_ref(entry['preparation_receipt']))
    require(meta.get('schema')=='prospective1611_pyg_preparation_v1' and meta.get('complete') is True and meta['parent_id']==entry['parent_id'],'complete exact preparation required')
    require(meta['wrapper_source_sha256']==sha(HERE/'pyg_adapter.py') and meta['source_pins_sha256']==sha(HERE/'source_pins.v1.json'),'preparation source differs')
    for key in ('bank','targets','roster','observed_generation_receipt','inference_receipt'):verify_ref(meta[key])
    targets=panel.read_jsonl(meta['targets']['path']);roster=read(meta['roster']['path'])
    panel.validate_targets(targets,panel.Panel(roster),parent_id=entry['parent_id']);class_roles(roster)
    labels=None;supervision={}
    if with_labels:
        labels=panel.validate_labels(panel.read_jsonl(verify_ref(entry['labels'])),targets)
        contract=read(verify_ref(entry['supervision_contract']))
        supervision=truth.validate_supervision_contract(contract,parent_id=entry['parent_id'],targets_path=meta['targets']['path'],roster_path=meta['roster']['path'],labels_path=entry['labels']['path'],observed_metadata=meta['common_observed_metadata'])
    common=meta['common_observed_metadata']
    signature={k:common[k] for k in ('parent_id','targets_sha256','roster_sha256','observed_trees_sha256')};signature.update(supervision)
    return meta,targets,labels,signature


def load_bank(torch,meta,targets,resources):
    data=torch.load(meta['bank']['path'],map_location='cpu',weights_only=False)
    require(data.get('schema')=='prospective1611_pyg_observed_bank_v1' and data['parent_id']==meta['parent_id'],'observed bank schema/parent differs')
    require(data['target_ids']==[r['row_id'] for r in targets],'exact target row order differs')
    roster=read(meta['roster']['path']);_,_,query_columns=class_roles(roster)
    query_slot={roster['rows'][col]['haplotype_id']:q for q,col in enumerate(query_columns)}
    require(data['target_tree'].tolist()==[r['tree_index'] for r in targets] and data['target_query_slot'].tolist()==[query_slot[r['haplotype_id']] for r in targets],'prepared target gather differs from exact shared IDs')
    from arggnn.mutation_routed_whole_tree_gnn_v1 import validate_inputs_v1
    stats=validate_inputs_v1(data['bank'],expected_haplotypes=HAPS)
    require(stats=={k:meta['feature_counts'][k] for k in stats} and tensor_bytes(data['bank'])==meta['feature_counts']['tensor_bytes']<=resources['max_prepared_tensor_bytes'],'bank counts/byte cap differs')
    return data


def fit_config():
    config=read(HERE/'finite_fit_config.v1.json')
    require(config['model']=={'hidden_dim':32,'expected_haplotypes':HAPS,'site_chunk_size':256,'checkpoint_sites':True},'unchanged C4 model configuration required')
    return config


def fit(request_path):
    request,runtime,resources=root_request(request_path,'fit')
    require(set(request)=={'schema','root_approved','wrapper_source_sha256','source_pins_sha256','runtime_binding','resource_binding','output_dir','train_inputs','val_inputs','device','fit_config_sha256'},'exact finite fit request required')
    config=fit_config();require(request['fit_config_sha256']==sha(HERE/'finite_fit_config.v1.json'),'finite fit config differs')
    entries=[];all_ids=set();signatures={};totals={}
    for split,key,expected in [('train','train_inputs',TRAIN),('val','val_inputs',VAL)]:
        bundle=read(verify_ref(request[key]));require(bundle.get('schema')=='prospective1611_pyg_input_bundle_v1','input bundle schema differs')
        require(tuple(r['parent_id'] for r in bundle['parents'])==expected,'exact ordered fiveTRAIN/oneVAL parents required')
        signatures[split]=[];totals[split]=0
        for entry in bundle['parents']:
            meta,targets,labels,signature=load_preparation(entry,with_labels=True)
            require(meta['split']==split and len(targets)==100000 and set(meta['feature_counts']['target_cohort_counts'])=={'NF','Yam','PostNeolithic','Present'},'exact shared100000 targets with all query cohorts per parent required')
            ids={r['row_id'] for r in targets};require(not ids&all_ids,'cross-parent/split target identity collision');all_ids|=ids
            entries.append((split,meta,targets,labels));signatures[split].append(signature);totals[split]+=len(targets)
    np,torch,_=numeric_modules(runtime);torch.set_num_threads(resources['cpu_threads']);sys.path.insert(0,str(ROOT/'src'))
    from arggnn.mutation_routed_pyg_gnn_v1 import MutationRoutedPyGGNNV1
    from torch.nn import functional as F
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=torch.device(request['device']);model=MutationRoutedPyGGNNV1(**config['model']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    out=Path(request['output_dir']);out.mkdir(parents=True,exist_ok=False);log=Progress(out/'progress.jsonl',resources['max_log_bytes'])
    steps=0;best=math.inf;best_epoch=0;best_ref=None;left=config['early_stop_nonimproving_val_epochs'];history=[]
    log.emit('fit_started',method='Mutation-routed PyG C4 prospective1611',train_targets=totals['train'],val_targets=totals['val'],max_epochs=config['epochs'],updates_per_epoch=5,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),initialization='fresh_seed101_no_checkpoint')
    for epoch in range(1,config['epochs']+1):
        scores={s:{'loss_sum':0.,'correct':0,'total':0} for s in ('train','val')}
        for split,meta,targets,labels in entries:
            training=split=='train';model.train(training)
            log.emit('parent_started',epoch=epoch,split=split,parent_id=meta['parent_id'],optimizer_step=steps,feature_counts=meta['feature_counts'])
            data=load_bank(torch,meta,targets,resources);bank=data['bank'].to(device)
            ti=data['target_tree'].to(device);qi=data['target_query_slot'].to(device)
            y=torch.tensor([labels[rid] for rid in data['target_ids']],dtype=torch.long,device=device)
            adam_min=adam_max=None
            with torch.set_grad_enabled(training):
                if training:optimizer.zero_grad(set_to_none=True)
                # Frozen full-parent graph + bidirectional recurrence; no tile resets.
                logits=model(bank,control='aligned')[ti,qi]
                loss=F.cross_entropy(logits,y)
                require(bool(torch.isfinite(loss)),'nonfinite common-target categorical loss')
                correct=int((logits.argmax(-1)==y).sum().item());n=len(targets);value=float(loss.detach().item());grad=None
                if training:
                    loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm']).item())
                    require(math.isfinite(grad),'nonfinite gradient norm');optimizer.step();steps+=1
                    adam_steps=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
                    require(bool(adam_steps),'optimizer produced no actual parameter state')
                    adam_min=min(adam_steps);adam_max=max(adam_steps)
            scores[split]['loss_sum']+=value*n;scores[split]['correct']+=correct;scores[split]['total']+=n
            log.emit('parent_completed',epoch=epoch,split=split,parent_id=meta['parent_id'],optimizer_step=steps,targets=n,categorical_NLL=value,correct=correct,total=n,gradient_norm_before_clip=grad,adam_parameter_step_min=adam_min,adam_parameter_step_max=adam_max,learning_rate=float(optimizer.param_groups[0]['lr']))
            del data,bank,ti,qi,y,logits,loss
        row={'epoch':epoch,'optimizer_steps':steps,**{s+'_NLL':scores[s]['loss_sum']/scores[s]['total'] for s in scores},**{s+'_accuracy':scores[s]['correct']/scores[s]['total'] for s in scores}}
        history.append(row);log.emit('epoch_completed',**row)
        if row['val_NLL']<best:
            best=row['val_NLL'];best_epoch=epoch;left=config['early_stop_nonimproving_val_epochs']
            path=out/('checkpoint_epoch_%03d.pt'%epoch)
            torch.save({'schema':'prospective1611_pyg_checkpoint_v1','model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'epoch':epoch,'optimizer_steps':steps,'fit_config':config,'class_order':LABELS,'wrapper_source_sha256':sha(HERE/'pyg_adapter.py')},path)
            best_ref=ref(path);log.emit('best_VAL_checkpoint',epoch=epoch,val_NLL=best,checkpoint_sha256=best_ref['sha256'])
        else:
            left-=1
            if left<=0:break
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=resources['max_output_bytes'],'fit output byte cap exceeded')
    require(best_ref is not None,'no finite VAL-selected checkpoint')
    summary={'schema':'prospective1611_pyg_fit_summary_v1','request':ref(request_path),'runtime_binding':request['runtime_binding'],'resource_binding':request['resource_binding'],'method':'pyg','method_label':'Mutation-routed PyG C4 prospective1611','class_order':LABELS,
             'train_shared_signatures':signatures['train'],'val_shared_signatures':signatures['val'],'logical_train_targets':totals['train'],'logical_val_targets':totals['val'],
             'best_epoch':best_epoch,'best_val_NLL':best,'epochs_ran':len(history),'optimizer_steps':steps,'history':history,'checkpoint':best_ref,
             'fit_config_sha256':sha(HERE/'finite_fit_config.v1.json'),'wrapper_source_sha256':sha(HERE/'pyg_adapter.py'),'source_pins_sha256':sha(HERE/'source_pins.v1.json'),
             'context':'complete original parent graph and full bidirectional recurrence; no extra labels or sampled native targets','paper_reproduction':False}
    write(out/'summary.json',summary);log.emit('fit_completed',best_epoch=best_epoch,best_val_NLL=best,optimizer_steps=steps);log.close()
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=resources['max_output_bytes'],'complete fit output byte cap exceeded')


def predict(request_path):
    request,runtime,resources=root_request(request_path,'predict')
    require(set(request)=={'schema','root_approved','wrapper_source_sha256','source_pins_sha256','runtime_binding','resource_binding','output_dir','inputs','fit_summary','device'},'prediction inputs must omit labels/contracts')
    summary=read(verify_ref(request['fit_summary']));config=fit_config()
    require(summary.get('schema')=='prospective1611_pyg_fit_summary_v1' and summary['wrapper_source_sha256']==sha(HERE/'pyg_adapter.py') and summary['fit_config_sha256']==sha(HERE/'finite_fit_config.v1.json') and summary['source_pins_sha256']==sha(HERE/'source_pins.v1.json'),'fit/checkpoint source condition differs')
    verify_ref(summary['checkpoint']);bundle=read(verify_ref(request['inputs']))
    require(bundle.get('schema')=='prospective1611_pyg_input_bundle_v1' and tuple(e['parent_id'] for e in bundle['parents'])==IID,'exact threeIID label-free parents required')
    loaded=[load_preparation(e,with_labels=False) for e in bundle['parents']]
    require(all(len(t)==100000 and m['split']=='iid' for m,t,_,_ in loaded),'exact shared IID support required')
    _,torch,_=numeric_modules(runtime);torch.set_num_threads(resources['cpu_threads']);sys.path.insert(0,str(ROOT/'src'))
    from arggnn.mutation_routed_pyg_gnn_v1 import MutationRoutedPyGGNNV1
    device=torch.device(request['device']);checkpoint=torch.load(summary['checkpoint']['path'],map_location='cpu',weights_only=False)
    require(checkpoint.get('schema')=='prospective1611_pyg_checkpoint_v1' and canonical(checkpoint['fit_config'])==canonical(config) and tuple(checkpoint['class_order'])==LABELS and checkpoint.get('wrapper_source_sha256')==sha(HERE/'pyg_adapter.py'),'strict checkpoint configuration/order required')
    model=MutationRoutedPyGGNNV1(**config['model']).to(device);model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    out=Path(request['output_dir']);out.mkdir(parents=True,exist_ok=False);log=Progress(out/'progress.jsonl',resources['max_log_bytes']);pred=out/'probabilities.jsonl';count=0;seen=set()
    with pred.open('x') as stream,torch.no_grad():
        for meta,targets,_,_ in loaded:
            log.emit('predict_parent_started',parent_id=meta['parent_id'],feature_counts=meta['feature_counts'])
            data=load_bank(torch,meta,targets,resources);bank=data['bank'].to(device)
            logits=model(bank,control='aligned')[data['target_tree'].to(device),data['target_query_slot'].to(device)]
            probabilities=logits.softmax(-1).cpu().tolist()
            for rid,p in zip(data['target_ids'],probabilities):
                require(rid not in seen and len(p)==4 and all(math.isfinite(v) and v>=0 for v in p) and math.isclose(sum(p),1.,abs_tol=1e-5),'unique normalized shared prediction required')
                seen.add(rid);stream.write(json.dumps({'row_id':rid,'labels':LABELS,'probabilities':p},separators=(',',':'))+'\n');count+=1
            stream.flush();require(pred.stat().st_size<=resources['max_output_bytes'],'prediction byte cap exceeded')
            log.emit('predict_parent_completed',parent_id=meta['parent_id'],target_rows=len(targets),predictions_written=count)
            del data,bank,logits,probabilities
    require(count==300000,'all threeIID targets must survive')
    write(out/'prediction_receipt.json',{'schema':'prospective1611_pyg_predictions_v1','request':ref(request_path),'runtime_binding':request['runtime_binding'],'resource_binding':request['resource_binding'],'method':'pyg','target_rows':count,'predictions':ref(pred),'shared_signatures':[s for _,_,_,s in loaded],'checkpoint_sha256':summary['checkpoint']['sha256'],'labels_or_truth_read':False,'paper_reproduction':False})
    log.emit('predict_completed',target_rows=count);log.close()
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=resources['max_output_bytes'],'complete prediction output byte cap exceeded')


def compare_fit_inputs(native_summary_path,pyg_summary_path):
    verify_sources();native=read(native_summary_path);pyg=read(pyg_summary_path)
    require(native.get('schema')=='prospective1611_native_fit_summary_v1' and native.get('method') in ('argmix','ap') and pyg.get('schema')=='prospective1611_pyg_fit_summary_v1','native and prospective PyG fit summaries required')
    keys=('train_shared_signatures','val_shared_signatures','logical_train_targets','logical_val_targets')
    require(all(canonical(native[k])==canonical(pyg[k]) for k in keys),'method observations/targets/labels/truth provenance differ')
    for split in ('train_shared_signatures','val_shared_signatures'):
        require(all({'labels_sha256','supervision_contract_sha256','truth_definition_sha256'}<=set(r) for r in pyg[split]),'explicit biological truth signatures required')
    return {'schema':'prospective1611_native_PyG_common_fit_data_check_v1','same_supervised_data_verified':True,'native_method':native['method'],'pyg_method':'pyg','signature_sha256':canonical({k:pyg[k] for k in keys})}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','fit','predict'):
        p=sub.add_parser(name);p.add_argument('--request',required=True)
    c=sub.add_parser('compare-fit-inputs');c.add_argument('--native-summary',required=True);c.add_argument('--pyg-summary',required=True)
    sub.add_parser('verify-sources');args=parser.parse_args()
    if args.command=='compare-fit-inputs':print(json.dumps(compare_fit_inputs(args.native_summary,args.pyg_summary),sort_keys=True))
    elif args.command=='verify-sources':print(json.dumps({'source_pins_verified':verify_sources(),'numeric_paths_executed':False}))
    else:globals()[args.command](args.request)

if __name__=='__main__':main()
