"""Independent handmade source tests. No real arrays, truth, models or fits."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import io
import json
import pickle
import zipfile
import numpy as np
import pytest
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SPEC=importlib.util.spec_from_file_location('retrieval_independent_under_review',ROOT/'scripts/score_official_argformer_retrieval_biological_v1.py')
r=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

@pytest.fixture(autouse=True)
def hidden_cuda():
    torch.set_num_threads(4)


def test_streamed_integer_dot_oracle_across_chunks_and_permuted_path_IDs():
    q=np.array([[2,0,1],[0,1,-1],[-1,2,0],[0,0,0]],dtype=np.float32)
    e=np.array([[1,0,1],[0,1,1],[-1,0,2],[0,-1,-1],[1,1,1],[2,0,-1],[-2,1,0],[1,0,1],[0,0,0]],dtype=np.float32)
    ids=np.array([74,13,82,20,19,51,17,6,35],dtype=np.int64)
    labels=np.array([0,1,2,3,3,0,1,2,0],dtype=np.int64)
    full=q@e.T
    expected=np.array([[ids[i] for i in sorted(range(len(ids)),key=lambda c:(-float(row[c]),int(ids[c])))[:6]] for row in full])
    expected_scores=np.array([[row[np.flatnonzero(ids==pid)[0]] for pid in out] for row,out in zip(full,expected)],dtype=np.float32)
    tiecounts=np.array([np.bincount(labels[row==row.max()],minlength=4) for row in full])
    for qb,ec in [(1,1),(2,2),(3,4),(4,9)]:
        score,out,ties=r.exhaustive_neighbors(q,e,ids,labels,qb,ec)
        assert np.array_equal(out,expected)
        assert np.array_equal(score,expected_scores)
        assert np.array_equal(ties,tiecounts)


def test_all_zero_scores_keep_all_distinct_path_rows_and_class_ties():
    ids=np.array([40,3,7,11,5,19,1,8],dtype=np.int64)
    labels=np.array([3,0,1,2,1,0,2,3],dtype=np.int64)
    score,out,ties=r.exhaustive_neighbors(np.zeros((2,3),np.float32),np.zeros((8,3),np.float32),ids,labels,1,3)
    assert out.tolist()==[[1,3,5,7,8,11]]*2
    assert np.array_equal(ties,np.array([[2,2,2,2]]*2))
    assert np.count_nonzero(score)==0
    p=r.hard_votes(out,ids,labels,5)
    assert np.array_equal(p,np.array([[.2,.4,.2,.2]]*2,np.float32))


def test_corpus_has_unique_full_paths_without_class_or_span_balancing():
    paths=((10,100),(10,101),(11,100),(12,100),(13,100),(20,100),(20,101),(30,100))
    ids,labels=r.corpus_roles(paths,np.array([20]),np.array([10,11,12,13]),np.arange(4))
    assert ids.tolist()==[0,1,2,3,4]
    assert labels.tolist()==[0,0,1,2,3]
    joins=dict(local_positions_bp=np.array([0.,5.,9.999]),query_nodes=np.array([20]),reference_nodes=np.array([10,11,12,13]),
        query_path_ids=np.array([[5],[6],[6]]),reference_path_ids=np.array([[0,2,3,4],[1,2,3,4],[1,2,3,4]]))
    occurrences=((0,10,0.,5.),(1,10,5.,10.),(2,11,0.,10.),(3,12,0.,10.),(4,13,0.,10.),
        (5,20,0.,5.),(6,20,5.,10.),(7,30,0.,10.))
    assert r.validate_support(paths,occurrences,'ARG:A','ARG:A',joins,10.)
    with pytest.raises(ValueError,match='physical partition'):
        r.validate_support(paths,occurrences[:-1]+((7,30,0.,9.),),'ARG:A','ARG:A',joins,10.)
    with pytest.raises(ValueError,match='occurrence join'):
        r.validate_support(paths,occurrences,'ARG:A','ARG:A',{**joins,'query_path_ids':np.array([[5],[5],[6]])},10.)


def test_complete_projection_repeats_paths_and_rejects_missing_support():
    uid=np.array([5,19,31],np.int64)
    p=np.eye(4,dtype=np.float32)[[3,0,2]]
    cells=np.array([[31,5,19],[5,31,5]],np.int64)
    projected=r.project_grid(uid,p,cells)
    assert projected.shape==(2,3,4)
    assert np.array_equal(projected,p[[[2,0,1],[0,2,0]]])
    with pytest.raises(ValueError,match='shrink common support'):
        r.project_grid(uid,p,np.array([[5,6,31]],np.int64))


def test_npz_header_reader_does_not_read_payload(tmp_path,monkeypatch):
    stream=io.BytesIO()
    np.lib.format.write_array_header_1_0(stream,{'descr':'<f4','fortran_order':False,'shape':(100,768)})
    header=stream.getvalue()
    path=tmp_path/'handmade_header_only.npz'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('embeddings.npy',header+b'NOT_AN_ARRAY_PAYLOAD')
    calls=[]
    original=zipfile.ZipExtFile.read
    def counted(self,n=-1):
        calls.append(n)
        return original(self,n)
    monkeypatch.setattr(zipfile.ZipExtFile,'read',counted)
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No payload loader')))
    assert r.npz_headers(path)=={'embeddings':{'descr':'<f4','fortran_order':False,'shape':(100,768)}}
    assert sum(calls)==len(header) and all(n>=0 for n in calls)


def test_primitive_path_unpickler_allows_metadata_and_rejects_global_classes(tmp_path):
    good=tmp_path/'handmade_primitive.pkl'
    good.write_bytes(pickle.dumps({'paths':((1,20),(2,20)),'occurrences':((0,1,0.,10.),),'namespace':'A'},protocol=4))
    assert r.primitive_load(good)['namespace']=='A'
    bad=tmp_path/'handmade_nonprimitive.pkl'
    bad.write_bytes(pickle.dumps(Path('handmade'),protocol=4))
    with pytest.raises(pickle.UnpicklingError,match='primitive'):
        r.primitive_load(bad)


def test_handmade_full_280_leaf_seal_retains_both_k_all_120_query_centers(tmp_path,monkeypatch):
    prepared_dir=tmp_path/'handmade_prepared'
    prepared_dir.mkdir()
    (prepared_dir/'embedding/all').mkdir(parents=True)
    paths=tuple((leaf,1000) for leaf in range(280))
    corpus={'paths':paths,'occurrences':tuple((leaf,leaf,0.,10.) for leaf in range(280)),'namespace':'HANDMADE:ARG'}
    corpuspath=prepared_dir/'corpus.pkl'
    corpuspath.write_bytes(pickle.dumps(corpus,protocol=4))
    rawpath=prepared_dir/'embedding/all/raw_all_sequences_and_vocab.pkl'
    rawpath.write_bytes(pickle.dumps({'raw_sequences':[list(p) for p in paths],'vocab':{}},protocol=4))
    vectors=torch.zeros(280,768,dtype=torch.float32)
    for leaf in range(280): vectors[leaf,(leaf%4 if leaf<120 else (leaf-120)//24 if leaf<216 else 0)]=1
    embeddingpath=tmp_path/'handmade_embeddings.pt'
    torch.save({'embeddings':vectors,'native_pooling':'trained_author_mean_head'},embeddingpath)
    query=np.arange(120,dtype=np.int64)
    refs=np.arange(120,216,dtype=np.int64)
    joins={'query_nodes':query,'reference_nodes':refs,'reference_classes':np.repeat(np.arange(4),24),
        'local_positions_bp':np.array([2.,7.]),'query_path_ids':np.tile(query,(2,1)),'reference_path_ids':np.tile(refs,(2,1))}
    state={'prepared':{'unique_full_paths':280,'namespace':'HANDMADE:ARG','files_sha256':{'corpus.pkl':r.sha(corpuspath),
        'embedding/all/raw_all_sequences_and_vocab.pkl':r.sha(rawpath)}},'records':{'prepared_manifest':{'path':str(prepared_dir/'prepared_manifest.json')},
        'fit_receipt':{'path':'handmade','sha256':'0'*64}},'joins':joins,'receipt':{'readout':{'native_embedding_sha256':r.tensor_sha(vectors)}}}
    parent={'parent_id':'HANDMADE','observation':{'nodes':np.arange(280)},'coverage_original':np.array([0.,10.]),
        'positions':np.array([2.,7.]),'edges':np.array([0.,5.,10.]),'retained_indexes':np.arange(2),
        'query_slots':query,'raw_query_nodes':query+5000}
    metadata={'reference_full_path_count':96,'embeddings':{'path':str(embeddingpath),'sha256':r.sha(embeddingpath)},
        'original_mean_readout':{'path':'handmade_metadata_only','sha256':'0'*64}}
    output=tmp_path/'sealed_handmade'
    output.mkdir()
    seal=r.seal_parent(parent,state,{},metadata,output,torch)
    assert seal['all_common_retained_support'] and seal['truth_opened'] is False and seal['both_prespecified_k']==[1,5]
    assert r.sha(ROOT/seal['path'])==seal['sha256']
    assert r.sha(ROOT/seal['neighbors']['path'])==seal['neighbors']['sha256']
    with np.load(ROOT/seal['path'],allow_pickle=False) as z:
        expected=np.tile(np.eye(4,dtype=np.float32)[query%4],(2,1,1))
        assert np.array_equal(z['C4_probabilities_raw_k1'],expected)
        assert np.array_equal(z['C4_probabilities_raw_k5'],expected)
        assert np.array_equal(z['query_path_ids'],joins['query_path_ids'])
    with np.load(ROOT/seal['neighbors']['path'],allow_pickle=False) as z:
        expected=np.array([[120+24*(leaf%4)+offset for offset in range(6)] for leaf in query])
        assert np.array_equal(z['neighbor_full_path_ids'],expected)
        assert np.array_equal(z['cosine_scores_float32'],np.ones((120,6),np.float32))
    assert not torch.cuda.is_initialized()
    def payload_forbidden(*args,**kwargs):
        raise AssertionError('Cost/count admission must precede tensor load')
    monkeypatch.setattr(torch,'load',payload_forbidden)
    with pytest.raises(RuntimeError,match='count/cost admission'):
        r.seal_parent(parent,state,{},dict(metadata,reference_full_path_count=95),output,torch)
    monkeypatch.setitem(r.RULE,'max_exhaustive_pairs_per_parent',100)
    with pytest.raises(RuntimeError,match='count/cost admission'):
        r.seal_parent(parent,state,{},metadata,output,torch)


def test_run_seals_both_VAL_all_k_before_any_scoring(tmp_path,monkeypatch):
    import arggnn.resources
    monkeypatch.setattr(arggnn.resources,'require_storage',lambda headroom:123)
    events=[]
    @contextmanager
    def fake_libraries(cfg):
        yield np,SimpleNamespace(__version__='HANDMADE',get_num_threads=lambda:4,cuda=SimpleNamespace(is_initialized=lambda:False)),[]
    monkeypatch.setattr(r,'cpu_libraries',fake_libraries)
    parents={name:{'parent_id':name} for name in r.VAL}
    legacy=SimpleNamespace(validate_inputs=lambda *a:None)
    monkeypatch.setattr(r,'load_legacy',lambda cfg:(legacy,{'external_truth_manifest':{}},None,parents,parents))
    def seal(parent,*args):
        events.append(('seal',parent['parent_id']))
        return {'parent_id':parent['parent_id'],'both_prespecified_k':[1,5]}
    monkeypatch.setattr(r,'seal_parent',seal)
    def score(parent,seal,common,helper,legacy,aggregate):
        assert events[:2]==[('seal',name) for name in r.VAL]
        data=json.loads((ROOT/aggregate['path']).read_text())
        assert data['all_k']==[1,5] and data['truth_scoring_started'] is False
        assert [s['parent_id'] for s in data['predictions']]==list(r.VAL)
        events.append(('score',parent['parent_id']))
        return {'parent_id':parent['parent_id'],'handmade':True}
    monkeypatch.setattr(r,'score_parent',score)
    monkeypatch.setattr(r,'policy',lambda *a:None)
    monkeypatch.setattr(r,'metadata_preflight',lambda *a:None)
    monkeypatch.setattr(r,'bound',lambda *a,**k:None)
    output=tmp_path/'handmade_orchestration'
    assert r.run({},parents,output)==0
    assert events==[('seal',name) for name in r.VAL]+[('score',name) for name in r.VAL]
