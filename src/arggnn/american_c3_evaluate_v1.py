"""Common C3 evaluator and observation-only native top1 seal before external truth."""
from .american_c3_common_v1 import CLASSES,require,bound,read,cli
SOURCE='src/arggnn/american_c3_evaluate_v1.py'
AUTHOR_BASE='.runtime/author_sources/argformer_f7545be7bca0/'
DEFAULTS=dict(schema='american_c3_external_evaluation_v1',execution_enabled=False,mode='score',
    prediction_ref=None,focal_registry_ref=None,external_truth_ref=None,method=None,output_cap_bytes=10000000,
    support='focal_unique_path_first_occurrence_midpoint_RECONSTRUCTION',
    preparation_ref=None,native_query_embeddings_ref=None,native_corpus_embeddings_ref=None,
    native_retrieval_ref=None,expected_pooling='head_norm',expected_splits=['train','val'])


def metrics(confusion):
    import numpy as np
    c=np.asarray(confusion,dtype=np.int64);require(c.shape==(3,3) and (c>=0).all() and c.sum()>0,'Nonempty C3 confusion matrix')
    result={}
    for i,name in enumerate(CLASSES):
        tp=int(c[i,i]);actual=int(c[i].sum());predicted=int(c[:,i].sum())
        p=tp/predicted if predicted else 0.;r=tp/actual if actual else 0.;f=2*tp/(actual+predicted) if actual+predicted else 0.
        result[name]=dict(precision=p,recall=r,F1=f,actual_count=actual,predicted_count=predicted)
    return dict(per_class=result,macro_F1=float(np.mean([r['F1'] for r in result.values()])),
        accuracy=float(np.trace(c)/c.sum()),confusion=c.tolist(),zero_division=0,unit='one_prespecified_focal_record')


def load_npz(record):
    import numpy as np
    with np.load(bound(record),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def registry(record):
    import numpy as np
    r=load_npz(record);n=len(r.get('record_ids',[]))
    require({'record_ids','sample_nodes','sample_indices','positions_bp0','parent_id','class_order'}<=set(r),
        'Prespecified observation-only evaluation registry required')
    require(n>0 and all(r[k].dtype==np.int64 and r[k].shape==(n,) for k in ('record_ids','sample_nodes','sample_indices'))
        and len(set(r['record_ids'].tolist()))==n and (r['sample_indices']>=0).all()
        and r['positions_bp0'].dtype==np.float64 and r['positions_bp0'].shape==(n,)
        and np.isfinite(r['positions_bp0']).all() and (r['positions_bp0']>=0).all()
        and r['parent_id'].shape==() and str(r['parent_id'].item())
        and r['class_order'].tolist()==list(CLASSES),'Unique stable complete registry identities/coordinates/parent/class order required')
    return r


def verify_prediction(p,r):
    import numpy as np
    identity=('record_ids','sample_nodes','sample_indices','positions_bp0','parent_id','class_order')
    require(set(p)==set(identity)|{'probabilities'} and all(np.array_equal(p[k],r[k]) for k in identity),
        'All prespecified identities/coordinates/parent/classes must be sealed before truth; no eligibility revision')
    prob=p['probabilities'];n=len(r['record_ids'])
    require(prob.shape==(n,3) and prob.dtype==np.float32 and np.isfinite(prob).all() and (prob>=0).all()
        and np.allclose(prob.sum(-1),1,atol=1e-6,rtol=1e-5),'Complete raw C3 probabilities or native one-hot; no floor/renormalization')
    return prob.argmax(-1)


def native_top1_seal(cfg,w,r):
    """Join native released row order to full raw path IDs without opening truth."""
    import numpy as np,torch
    require(cfg['external_truth_ref'] is None and cfg['prediction_ref'] is None,'Native seal accepts no private truth/prediction input')
    prep=read(cfg['preparation_ref'])
    require(prep['parent_id']==r['parent_id'].item() and prep['class_order']==list(CLASSES)
        and prep['focals_ref']==cfg['focal_registry_ref'] and prep['query_embedding_view_prespecified_focals_only'] is True
        and prep['contrastive_labels_reference_only'] is True and prep['query_ancestry_truth_opened'] is False,
        'Exact preparation/focal/observation-only contract required')
    def ordered(view):
        pieces=[load_npz(prep['full_path_id_order_refs'][f'{view}/{sp}']) for sp in cfg['expected_splits']]
        require(all(x['parent_id'].item()==r['parent_id'].item() and x['class_order'].tolist()==list(CLASSES) for x in pieces),
            'Exact native row-order parent/class identity required')
        names=['record_ids','sample_nodes','sample_indices']+(['reference_classes_0based'] if view=='contrastive' else [])
        out={k:np.concatenate([x[k] for x in pieces]) for k in names}
        n=len(out['record_ids'])
        require(n>0 and all(a.dtype==np.int64 and a.shape==(n,) for a in out.values())
            and len(set(out['record_ids'].tolist()))==n,'Unique complete native full-path row-order metadata required')
        return out
    q=ordered('query_paths');c=ordered('contrastive')
    require(cfg['expected_splits']==['train','val'] and cfg['expected_pooling']=='head_norm',
        'Registered native embedding split/pooling reconstruction contract required')
    require(set(q['record_ids'].tolist())==set(r['record_ids'].tolist()),'Embed every and only prespecified query focal')
    def tensor_array(value):
        require(torch.is_tensor(value) and value.device.type=='cpu','Native CPU tensor output required')
        return value.detach().numpy()
    def embedding(record,order,labelled):
        out=torch.load(bound(record),map_location='cpu',weights_only=True,mmap=True)
        meta=out['metadata'];vectors=out['embeddings'];n=len(order['record_ids'])
        require(meta['num_samples']==n and meta['splits']==cfg['expected_splits']
            and meta['pooling_strategy']==cfg['expected_pooling'] and vectors.ndim==2
            and vectors.shape[0]==n and vectors.shape[1]==meta['embedding_dim'],'Native complete unmodified embedding row count/order settings required')
        require('extant_node_ids' in out and np.array_equal(tensor_array(out['extant_node_ids']).reshape(-1),order['sample_nodes']),
            'Native extant-node output must confirm sidecar sample row order')
        if labelled:
            labels=tensor_array(out['labels']).reshape(-1)
            require(np.array_equal(labels,order['reference_classes_0based']) and (labels>=0).all() and (labels<3).all(),
                'Native corpus labels must equal known prepared reference labels')
        else:require('labels' not in out,'No query ancestry labels in embeddings')
        return out
    query=embedding(cfg['native_query_embeddings_ref'],q,False)
    corpus=embedding(cfg['native_corpus_embeddings_ref'],c,True)
    require(query['embeddings'].shape[1]==corpus['embeddings'].shape[1],'Equal native embedding dimension')
    native=torch.load(bound(cfg['native_retrieval_ref']),map_location='cpu',weights_only=True,mmap=True)
    require(native['k']==1 and 'query_labels' not in native,'Registered native top1 retrieval; no query truth labels')
    ix=tensor_array(native['topk_indices']);scores=tensor_array(native['topk_scores'])
    top=tensor_array(native['top1_indices']).reshape(-1);top_scores=tensor_array(native['top1_scores']).reshape(-1)
    require(ix.dtype==np.int64 and ix.shape==(len(q['record_ids']),1) and scores.shape==ix.shape
        and np.array_equal(ix[:,0],top) and np.array_equal(scores[:,0],top_scores)
        and np.isfinite(scores).all() and (top>=0).all() and (top<len(c['record_ids'])).all(),'Exact unmodified native top1 index/score products required')
    names=corpus.get('metadata',{}).get('int_to_label')
    if isinstance(names,dict):
        mapping={int(k):str(v) for k,v in names.items()}
        expected=[mapping.get(int(v),str(int(v))) for v in c['reference_classes_0based']]
    else:expected=[str(int(v)) for v in c['reference_classes_0based']]
    require(native.get('corpus_labels')==expected,'Native retrieve corpus label mapping must equal prepared reference row order')
    lookup={int(k):i for i,k in enumerate(q['record_ids'])};join=np.asarray([lookup[int(k)] for k in r['record_ids']])
    require(all(np.array_equal(q[k][join],r[k]) for k in ('record_ids','sample_nodes','sample_indices')),
        'Full raw-path identity join must precede sealing; truncated input/embedding values cannot identify paths')
    calls=c['reference_classes_0based'][top][join]
    prediction={k:r[k] for k in ('record_ids','sample_nodes','sample_indices','positions_bp0','parent_id','class_order')}
    prediction['probabilities']=np.eye(3,dtype=np.float32)[calls];verify_prediction(prediction,r)
    prediction_ref=w.npz('predictions.npz',**prediction)
    native_ref=w.npz('native_top1_preserved.npz',record_ids=q['record_ids'],top1_indices=top,top1_scores=top_scores,
        parent_id=r['parent_id'],class_order=r['class_order'])
    return w.json('completed.json',dict(status='sealed_native_C3_top1',prediction_ref=prediction_ref,
        native_top1_preserved_ref=native_ref,focal_registry_ref=cfg['focal_registry_ref'],
        native_retrieval_ref=cfg['native_retrieval_ref'],preparation_ref=cfg['preparation_ref'],
        native_query_embeddings_ref=cfg['native_query_embeddings_ref'],native_corpus_embeddings_ref=cfg['native_corpus_embeddings_ref'],
        parent_id=r['parent_id'].item(),class_order=list(CLASSES),records=len(calls),
        scores_and_retrieval_indices_modified=False,query_ancestry_truth_opened=False,author_Table1_exact_reproduction=False))


def execute(cfg,w):
    import numpy as np
    require(cfg['schema']==DEFAULTS['schema'] and cfg['method'],'Registered method/score contract required')
    r=registry(cfg['focal_registry_ref'])
    if cfg['mode']=='native_top1_seal':return native_top1_seal(cfg,w,r)
    require(cfg['mode']=='score','Registered score or native_top1_seal mode required')
    calls=verify_prediction(load_npz(cfg['prediction_ref']),r)
    # No private truth file is opened until every prespecified prediction passes.
    truth=load_npz(cfg['external_truth_ref'])
    require(set(truth)=={'intervals_bp','query_sample_indices','source_classes_0based','parent_id','class_order'}
        and np.array_equal(truth['parent_id'],r['parent_id']) and truth['class_order'].tolist()==list(CLASSES),'Exact external parent/class pulse truth schema')
    intervals=truth['intervals_bp'];samples=truth['query_sample_indices'];labels=truth['source_classes_0based']
    require(intervals.dtype==np.float64 and len(intervals)>0 and intervals.shape[1:]==(2,)
        and np.isfinite(intervals).all() and intervals[0,0]==0 and (intervals[:,1]>intervals[:,0]).all()
        and samples.dtype==np.int64 and samples.ndim==1 and len(set(samples.tolist()))==len(samples)
        and labels.dtype==np.uint8 and labels.shape==(len(intervals),len(samples)) and (labels<3).all()
        and np.array_equal(intervals[:-1,1],intervals[1:,0]),'Complete half-open C3 truth domain')
    positions=r['positions_bp0'];require(((positions>=intervals[0,0])&(positions<intervals[-1,1])).all(),'No coordinate extrapolation')
    lookup={int(u):i for i,u in enumerate(samples)}
    require(all(int(u) in lookup for u in r['sample_indices']),'Only externally registered original genotype-column query identities')
    rows=np.searchsorted(intervals[:,1],positions,side='right')
    true=labels[rows,[lookup[int(u)] for u in r['sample_indices']]]
    confusion=np.zeros((3,3),dtype=np.int64);np.add.at(confusion,(true,calls),1)
    return w.json('completed.json',dict(status='completed_external_C3_score',method=cfg['method'],support=cfg['support'],
        prediction_ref=cfg['prediction_ref'],focal_registry_ref=cfg['focal_registry_ref'],external_truth_ref=cfg['external_truth_ref'],
        parent_id=r['parent_id'].item(),class_order=list(CLASSES),records=len(calls),result=metrics(confusion),
        probabilities_modified=False,probability_floor=False,author_Table1_exact_reproduction=False))


if __name__=='__main__':cli(SOURCE,DEFAULTS,execute,dependencies=(
    'src/arggnn/american_c3_paths_v1.py',AUTHOR_BASE+'embeddings.py',AUTHOR_BASE+'retrieve.py'))
