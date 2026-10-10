"""Pinned official ARGformer components and observed-path interoperability.

No training launch, installed dependency substitution, target loader or full-model
reproduction is implemented here. ARG node tokens keep their source namespace.
The optional four-class similarity readout is explicitly downstream adaptation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tskit

from .resources import ROOT

COMMIT = 'f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596'
SOURCE = ROOT/'.runtime/author_sources/argformer_f7545be7bca0'
TREE_SHA256 = '2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b'
ARCHIVE_SHA256 = '44c1b76f7806dfc3063d707eae1950601eb4caa3ebe2fa20277bb98e77f1890c'
SPECIAL = {'[PAD]': 0, '[MASK]': 1, '[CLS]': 2, '[SEP]': 3}
DEPENDENCIES = ('torch', 'numpy', 'tskit', 'omegaconf', 'transformers', 'mosaicml',
                'mosaicml-streaming', 'torchmetrics', 'einops', 'flash-attn', 'triton', 'torch-optimi')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _positive_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:raise ValueError(name+' must be an integer')
    return value


def validate_source(source=SOURCE):
    source=Path(source)
    if source.resolve()!=SOURCE.resolve():raise ValueError('Only the assigned pinned author source is accepted')
    receipt=json.loads((source/'ARG_GNN_SOURCE_RECEIPT.json').read_text())
    if receipt.get('commit')!=COMMIT or receipt.get('archive_sha256')!=ARCHIVE_SHA256:
        raise ValueError('Pinned author archive identity differs')
    records=[]
    for path in sorted(source.rglob('*')):
        if path.is_symlink():raise ValueError('Author source symlink is not admitted')
        if path.is_file() and path.name!='ARG_GNN_SOURCE_RECEIPT.json':
            records.append({'path':path.relative_to(source).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    actual=hashlib.sha256(_canonical(records)).hexdigest()
    if actual!=TREE_SHA256:raise ValueError('Unchanged pinned author source bytes required')
    return {'repository':'https://github.com/AI-sandbox/ARGformer','commit':COMMIT,
            'source_tree_sha256':actual,'files':records,'full_model_import_performed':False}


def dependency_inventory():
    result={}
    for name in DEPENDENCIES:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


@dataclass(frozen=True)
class PathCorpus:
    namespace: str
    paths: tuple[tuple[int,...],...]
    occurrences: tuple[tuple[int,int,float,float],...]
    sample_ids: tuple[int,...]
    domain_bp: tuple[float,float]

    def __post_init__(self):
        if not isinstance(self.namespace,str) or not self.namespace.strip():raise ValueError('Explicit ARG namespace required')
        if not self.paths or len(set(self.paths))!=len(self.paths):raise ValueError('Nonempty unique full paths required')
        if len(set(self.sample_ids))!=len(self.sample_ids):raise ValueError('Unique sample order required')
        for path in self.paths:
            if not path or len(set(path))!=len(path) or any(type(n) is not int or n<0 for n in path):
                raise ValueError('Acyclic integer node-ID full paths required')
            if path[0] not in self.sample_ids:raise ValueError('Path leaf is outside sample roster')
        for path_id,leaf,left,right in self.occurrences:
            if type(path_id) is not int or not 0<=path_id<len(self.paths) or self.paths[path_id][0]!=leaf:
                raise ValueError('Occurrence/full-path leaf identity differs')
            if not self.domain_bp[0]<=left<right<=self.domain_bp[1]:raise ValueError('Occurrence coverage differs')

    def readout_path_ids(self,positions_bp,query_ids):
        positions=np.asarray(positions_bp,dtype=np.float64);queries=tuple(query_ids)
        if positions.ndim!=1 or not np.isfinite(positions).all() or np.any(np.diff(positions)<=0):
            raise ValueError('Finite strictly ordered physical positions required')
        if not len(positions) or np.any(positions<self.domain_bp[0]) or np.any(positions>=self.domain_bp[1]):
            raise ValueError('Readouts must lie in observed half-open coverage')
        if not queries or len(set(queries))!=len(queries) or any(type(q) is not int or q not in self.sample_ids for q in queries):
            raise ValueError('Explicit original query IDs required')
        result=np.full((len(positions),len(queries)),-1,dtype=np.int64)
        for path_id,leaf,left,right in self.occurrences:
            if leaf in queries:
                column=queries.index(leaf);mask=(positions>=left)&(positions<right)
                if np.any(result[mask,column]!=-1):raise ValueError('Repeated readout occurrence')
                result[mask,column]=path_id
        if np.any(result<0):raise ValueError('No observed full path for a readout')
        return result


def extract_observed_paths(ts,*,namespace,sample_ids):
    """Author-order unique full leaf-to-root paths, with physical occurrence joins.

    Uses node IDs and parent topology only. No node populations, dates, mutations,
    census labels or target ancestry are returned or used. Caller supplies an
    independently reviewed observed tree sequence, never a planned held-out read.
    """
    if not isinstance(ts,tskit.TreeSequence):raise ValueError('Observed tskit TreeSequence required')
    samples=tuple(sample_ids)
    if not samples or len(set(samples))!=len(samples) or any(type(s) is not int or s not in ts.samples() for s in samples):
        raise ValueError('Unique observed sample node IDs required')
    paths=[];known={};occurrences=[]
    for leaf in samples:
        for tree in ts.trees():
            nodes=[];u=leaf
            while u!=tskit.NULL:
                nodes.append(int(u));u=tree.parent(u)
            key=tuple(nodes)
            if key not in known:known[key]=len(paths);paths.append(key)
            occurrences.append((known[key],leaf,float(tree.interval.left),float(tree.interval.right)))
    return PathCorpus(namespace,tuple(paths),tuple(occurrences),samples,(0.,float(ts.sequence_length)))


@dataclass(frozen=True)
class ArgVocabulary:
    namespace: str
    node_ids: tuple[int,...]

    def __post_init__(self):
        if not isinstance(self.namespace,str) or not self.namespace.strip():raise ValueError('Explicit vocabulary ARG namespace required')
        if not self.node_ids or tuple(sorted(set(self.node_ids)))!=self.node_ids or any(type(n) is not int or n<0 for n in self.node_ids):
            raise ValueError('Exactly sorted unique source node IDs required')

    def official_values(self):
        return {'special':dict(SPECIAL),'node_id_tokens':{f'node_{n}':i+4 for i,n in enumerate(self.node_ids)}}

    def fingerprint(self):
        return hashlib.sha256(_canonical({'namespace':self.namespace,'vocab':self.official_values()})).hexdigest()


def build_vocabulary(corpus):
    if not isinstance(corpus,PathCorpus):raise ValueError('Actual observed-path corpus required')
    return ArgVocabulary(corpus.namespace,tuple(sorted({node for path in corpus.paths for node in path})))


def _author_nodes(filename,names):
    """Only reviewed pure definitions, no upstream module imports/main/config I/O."""
    validate_source()
    tree=ast.parse((SOURCE/filename).read_text());found={n.name:n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    definitions=[]
    for name in names:
        if '.' in name:
            parent,method=name.split('.',1)
            node=next(n for n in found[parent].body if isinstance(n,ast.FunctionDef) and n.name==method)
            node=ast.FunctionDef(name=method,args=node.args,body=node.body,decorator_list=[],returns=node.returns,type_comment=node.type_comment)
        else:node=found[name]
        definitions.append(node)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*definitions],type_ignores=[])
    ast.fix_missing_locations(module)
    ns={'torch':torch,'nn':nn,'F':F,'math':math,'logger':logging.getLogger(__name__),
        'IMPL_USE_FLASH2':False,'IMPL_USE_FLASH3':False}
    exec(compile(module,str(SOURCE/filename),'exec'),ns)
    return ns


def tokenize_official(corpus,vocabulary,*,max_seq_len=128,skip_extant_tokens=True):
    """Execute unchanged author tokenization methods on owned in-memory paths.

    Cross-ARG vocabulary use and unseen IDs reject rather than falling back to
    PAD or assigning another parent's biological meaning to an integer token.
    Full-path deduplication always precedes skip/truncation.
    """
    if not isinstance(corpus,PathCorpus) or not isinstance(vocabulary,ArgVocabulary) or corpus.namespace!=vocabulary.namespace:
        raise ValueError('Path/vocabulary ARG namespace differs')
    _positive_int(max_seq_len,'max_seq_len',minimum=3)
    if type(skip_extant_tokens) is not bool:raise ValueError('Typed skip-extant setting required')
    namespace=_author_nodes('src/arg_data.py',('ARGDataset._tokenize_raw_sequence_from_vocab','ARGDataset.__getitem__'))
    values=vocabulary.official_values()
    obj=SimpleNamespace(sequences=[list(p) for p in corpus.paths],_data_is_pretokenized=False,
        return_metadata=True,skip_extant_tokens=skip_extant_tokens,max_seq_len=max_seq_len,pad_sequences=True,
        pad_token_id=0,cls_token_id=2,sep_token_id=3,_node_id_token_map=values['node_id_tokens'])
    obj._tokenize_raw_sequence_from_vocab=lambda row:namespace['_tokenize_raw_sequence_from_vocab'](obj,row)
    rows=[namespace['__getitem__'](obj,i) for i in range(len(corpus.paths))]
    return {key:torch.tensor([r[key] for r in rows],dtype=torch.long) for key in rows[0]} | {
        'arg_namespace':corpus.namespace,'vocabulary_sha256':vocabulary.fingerprint(),
        'max_seq_len':max_seq_len,'skip_extant_tokens':skip_extant_tokens,
        'full_paths_before_truncation':corpus.paths,'published_task_reproduction':False}


def author_component_cpu_probe():
    """Real pinned author global SDPA component, not the released RoPE/MLM model."""
    components=_author_nodes('src/bert_layers/attention.py',('FlexBertAttentionBase','FlexBertPaddedAttention'))
    config=SimpleNamespace(hidden_size=16,num_attention_heads=4,attention_probs_dropout_prob=0.,
        attn_qkv_bias=False,attn_out_bias=False,attn_out_dropout_prob=0.,use_fa2=False,
        deterministic_fa2=True,use_sdpa_attn_mask=True,global_attn_every_n_layers=-1,sliding_window=-1)
    torch.manual_seed(17);model=components['FlexBertPaddedAttention'](config,layer_id=0).eval()
    x=torch.randn(2,5,16,requires_grad=True);mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]])
    y=model(x,mask);y.sum().backward()
    if y.shape!=(2,5,16) or not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():raise ValueError('Author SDPA component failed')
    config.sliding_window=128
    try:components['FlexBertPaddedAttention'](config,layer_id=0)
    except ValueError as error:sliding_rejection=str(error)
    else:raise AssertionError('Author sliding-window fallback unexpectedly accepted')
    return {'author_component_forward_backward':True,'shape':list(y.shape),'device':'cpu',
        'configuration':'author global padded SDPA attention; no RoPE, no sliding window, hidden16/heads4',
        'full_official_model_forward':False,'published_task_reproduction':False,
        'released_sliding_window_stock_fallback_rejection':sliding_rejection}


def downstream_reference_similarity_logits(query_embeddings,reference_embeddings,reference_labels,*,
        query_ids,reference_ids,query_namespace,reference_namespace,temperature,class_order):
    """Explicit downstream ancestry adapter; not an author supplied classifier.

    Mean exponentiated cosine similarity per class gives four finite raw readout
    scores. Temperature and aggregation must be frozen using development only.
    Inputs are extracted official path embeddings and observed reference labels,
    never query ancestry targets. This API does not fit or calibrate anything.
    """
    if query_namespace!=reference_namespace or not isinstance(query_namespace,str) or not query_namespace:
        raise ValueError('Reference and query embeddings must share one ARG/checkpoint namespace')
    if list(class_order)!=['ANA','CHG','WHG','EHG']:raise ValueError('Explicit downstream four-source meaning required')
    q=torch.as_tensor(query_embeddings,dtype=torch.float64);r=torch.as_tensor(reference_embeddings,dtype=torch.float64)
    labels=torch.as_tensor(reference_labels)
    if q.ndim!=2 or r.ndim!=2 or q.shape[1]!=r.shape[1] or not len(q) or not len(r) or not torch.isfinite(q).all() or not torch.isfinite(r).all():
        raise ValueError('Finite aligned extracted embedding matrices required')
    if q.device.type!='cpu' or r.device.type!='cpu' or labels.dtype!=torch.int64 or labels.shape!=(len(r),):raise ValueError('CPU embeddings and integer observed labels required')
    if len(query_ids)!=len(q) or len(reference_ids)!=len(r) or set(query_ids)&set(reference_ids):raise ValueError('Reference/query identity separation required')
    if type(temperature) not in (float,int) or not math.isfinite(temperature) or temperature<=0:raise ValueError('Pre-frozen positive temperature required')
    if not torch.isfinite(q.norm(dim=-1)).all() or not torch.isfinite(r.norm(dim=-1)).all() or torch.any(q.norm(dim=-1)==0) or torch.any(r.norm(dim=-1)==0):raise ValueError('Finite nonzero embedding norms required')
    if set(labels.tolist())!={0,1,2,3}:raise ValueError('Every source class needs observed reference support')
    similarity=F.normalize(q,dim=-1)@F.normalize(r,dim=-1).T/temperature
    logits=torch.stack([torch.logsumexp(similarity[:,labels==c],dim=-1)-math.log(int((labels==c).sum())) for c in range(4)],dim=-1)
    if not torch.isfinite(logits).all():raise ValueError('Downstream raw scores exceed float64 range')
    return {'raw_readout_logits':logits,'class_order':list(class_order),'temperature':temperature,
        'official_local_ancestry_classifier':False,'kind':'explicit_downstream_reference_similarity_adapter',
        'query_ancestry_targets_used':False,'published_task_reproduction':False}


def author_contrastive_cpu_probe():
    """Execute original contrastive loss with an explicitly toy encoder/pooler.

    This establishes the author's loss/gradient path only. The toy encoder is
    never returned or used as ARGformer and cannot establish model reproduction.
    """
    components=_author_nodes('sequence_contrastive.py',('ContrastiveModel',))
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(20,8)
        def bert(self,input_ids,attention_mask,token_type_ids=None):
            return self.embedding(input_ids)*attention_mask[...,None]
        def head(self,output):return output.mean(1)
    torch.manual_seed(17);backbone=ToyBackbone()
    model=components['ContrastiveModel'](SimpleNamespace(model=backbone),normalize=True,
        temperature=.3,similarity='dot',class_weights={0:1.,1:1.})
    batch={'input_ids':torch.tensor([[2,4,3],[2,5,3],[2,8,3],[2,9,3]]),
           'attention_mask':torch.ones((4,3),dtype=torch.long),'labels':torch.tensor([0,0,1,1])}
    result=model(batch);result['loss'].backward()
    if not torch.isfinite(result['loss']) or not torch.isfinite(backbone.embedding.weight.grad).all():
        raise ValueError('Original contrastive component loss/gradient failed')
    return {'actual_author_contrastive_loss_executed':True,'loss':float(result['loss'].detach()),
        'encoder_and_pooler':'explicit synthetic toy, never a supplied ARGformer substitute',
        'population_labels':'synthetic observed-reference labels only','full_official_model_forward':False,
        'training_loop_or_biological_result':False}


def author_embedding_retrieval_cpu_probe():
    """Unchanged pooling/retrieval functions on synthetic hidden states/vectors."""
    pooling=_author_nodes('embeddings.py',('get_embeddings',))['get_embeddings']
    class SyntheticBackbone(nn.Module):
        def __init__(self):super().__init__();self.marker=nn.Parameter(torch.zeros(()))
        def bert(self,input_ids,attention_mask):
            return torch.arange(input_ids.numel()*4,dtype=torch.float32).reshape(*input_ids.shape,4)+self.marker
    wrapper=nn.Module();wrapper.model=SyntheticBackbone();wrapper.tokenizer=SimpleNamespace(mask_token_id=1)
    batch={'input_ids':torch.tensor([[2,4,3,0],[2,5,6,3]]),
           'attention_mask':torch.tensor([[1,1,1,0],[1,1,1,1]])}
    first=pooling(wrapper,batch,pooling_strategy='first_node')
    mean=pooling(wrapper,batch,pooling_strategy='mean')
    if not torch.equal(first,torch.tensor([[4.,5.,6.,7.],[20.,21.,22.,23.]])):
        raise ValueError('Author first-node pooling differs')
    if not torch.equal(mean,torch.tensor([[4.,5.,6.,7.],[22.,23.,24.,25.]])):
        raise ValueError('Author nonpadding mean pooling differs')
    retrieval=_author_nodes('retrieve.py',('topk_indices_streaming',))
    class QuietProgress:
        def __init__(self,**kwargs):pass
        def update(self,n):pass
        def close(self):pass
    retrieval['tqdm']=QuietProgress
    scores,indices=retrieval['topk_indices_streaming'](torch.eye(4)[:2],torch.eye(4),torch.device('cpu'),k=1,q_batch=1,e_chunk=2)
    if indices.tolist()!=[[0],[1]] or scores.tolist()!=[[1.],[1.]]:raise ValueError('Author streaming retrieval differs')
    return {'actual_author_pooling_and_streaming_retrieval':True,
        'first_node_with_skipped_leaf_means_first_ancestor':True,'mean_includes_special_nonpadding_tokens':True,
        'encoder_hidden_states':'explicit handmade tensors, not official encoder predictions',
        'retrieval_indices':indices.tolist(),'retrieval_scores':scores.tolist(),
        'full_official_model_forward':False,'published_task_reproduction':False}
