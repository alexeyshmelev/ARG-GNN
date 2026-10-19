"""Literal original native seal with the one explicit RAM-load argument."""
from arggnn.american_c3_evaluate_v1 import CLASSES, require, bound, read, load_npz, verify_prediction

def native_top1_seal(cfg,w,r,load_embedding):
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
        out=load_embedding(record)
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

