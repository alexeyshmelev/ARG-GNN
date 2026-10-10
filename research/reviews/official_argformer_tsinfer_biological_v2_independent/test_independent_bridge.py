import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/official_argformer_tsinfer_biological_v2_independent"
spec=importlib.util.spec_from_file_location("independent_bridge",ROOT/"src/arggnn/official_argformer_tsinfer_biological_v2.py")
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def test_actual_four_parent_JSON_metadata_only():
    report=b.metadata_report(b.policy())
    assert list(report["parents"])==b.FIXED_CONFIG["parent_ids"]
    assert len({r["namespace"] for r in report["parents"].values()})==4
    assert all(r["native_ID_preservation"] and r["retained_center_count"]>0 for r in report["parents"].values())
    assert report["arrays_opened"] is False and report["model_or_training_executed"] is False

def test_no_activated_config_accepted_and_all_public_execute_paths_reject():
    for metadata in (False,True):
        with pytest.raises(RuntimeError,match="inactive"):
            b.main(["--execute"]+(["--metadata"] if metadata else []))
    c=copy.deepcopy(b.policy());c["execution_enabled"]=True
    with pytest.raises(RuntimeError,match="inactive"):
        b.execution_policy(c,b.digest(b.CONFIG))

def test_all_nonpure_capsule_functions_reject_before_native_imports():
    c=b.policy();ns=b.native_capsule(c)
    for name in set(b.NATIVE_FUNCTIONS)-{"sha","canonical","json_new","observed_path","load_frozen"}:
        with pytest.raises(RuntimeError,match="inactive"):ns[name]()
    core,_=b.preparation_core(c)
    with pytest.raises(RuntimeError,match="inactive"):core()

def test_internal_json_new_is_unguarded_but_no_write_occurs(monkeypatch):
    class WriteReached(RuntimeError):pass
    original=Path.open
    reached=[]
    def spy(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get("mode","r")
        if "x" in mode:
            reached.append(str(path));raise WriteReached("ungated internal writer reached")
        return original(path,*args,**kwargs)
    ns=b.native_capsule(b.policy())
    monkeypatch.setattr(Path,"open",spy)
    with pytest.raises(WriteReached):ns["json_new"](OWN/"never_created.json",{})
    assert reached==[str(OWN/"never_created.json")]
    assert not (OWN/"never_created.json").exists()

def test_internal_load_frozen_reaches_import_gate_without_import(monkeypatch):
    class ImportReached(RuntimeError):pass
    ns=b.native_capsule(b.policy())
    path=ROOT/"src/arggnn/official_argformer_tsinfer_biological_v2.py"
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    def spy(*args,**kwargs):raise ImportReached("ungated internal importer reached")
    monkeypatch.setattr(importlib.util,"spec_from_file_location",spy)
    with pytest.raises(ImportReached):ns["load_frozen"](path,digest,"never_imported")

@pytest.mark.parametrize("counts",[
    {"unique_full_paths":11000,"MLM_train":1000,"MLM_val":10000,"pure_reference_train":5000,"pure_reference_val":500,"query_full_paths_scored":4000},
    {"unique_full_paths":11000,"MLM_train":10000,"MLM_val":1000,"pure_reference_train":5000,"pure_reference_val":2000,"query_full_paths_scored":4000},
    {"unique_full_paths":11000,"MLM_train":10000,"MLM_val":1000,"pure_reference_train":5000,"pure_reference_val":500,"query_full_paths_scored":11001},
])
def test_document_count_consistency_gap_current_inactive_api(counts):
    result=b.corpus_budget(b.policy(),counts)
    assert result["actual_counts"]==counts
    assert (counts["pure_reference_train"]>counts["MLM_train"] or
            counts["pure_reference_val"]>counts["MLM_val"] or
            counts["query_full_paths_scored"]>counts["unique_full_paths"]-
                counts["pure_reference_train"]-counts["pure_reference_val"])

def handmade_corpus():
    source=ast.parse((ROOT/"src/arggnn/official_argformer_bridge.py").read_text())
    names={"PathCorpus","ArgVocabulary","extract_observed_paths","build_vocabulary"}
    defs=[copy.deepcopy(n) for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    class Tree:
        def __init__(self,left,right,tip):
            self.interval=SimpleNamespace(left=left,right=right);self.tip=tip
        def parent(self,u):
            if u in (4000,4002):return 7000
            if 7000<=u<7260:return u+2
            if u==7260:return self.tip
            return -1
    class FakeTS:
        sequence_length=100.
        def samples(self):return (4000,4002)
        def trees(self):
            return iter((Tree(0.,50.,9000),Tree(50.,75.,9002),Tree(75.,100.,9000)))
    ns={"dataclass":dataclass,"tskit":SimpleNamespace(TreeSequence=FakeTS,NULL=-1),"SPECIAL":{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3},"hashlib":hashlib,"_canonical":lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()}
    tree=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*defs],type_ignores=[])
    ast.fix_missing_locations(tree);exec(compile(tree,"handmade_extraction","exec"),ns)
    corpus=ns["extract_observed_paths"](FakeTS(),namespace="ARG_A:hashA",sample_ids=[4000,4002])
    return ns,corpus

def test_complete_path_dedup_keeps_native_IDs_occurrences_and_colliding_truncations():
    ns,c=handmade_corpus()
    assert len(c.paths)==4 and len(c.occurrences)==6
    assert [x[0] for x in c.occurrences]==[0,1,0,2,3,2]
    assert all(len(p)>128 for p in c.paths)
    assert c.paths[0][1:127]==c.paths[1][1:127]
    assert c.paths[0]!=c.paths[1]
    vocab=ns["build_vocabulary"](c)
    values=vocab.official_values()
    assert set(values["special"].values())=={0,1,2,3}
    assert set(values["node_id_tokens"])=={"node_"+str(n) for p in c.paths for n in p}
    other=ns["PathCorpus"]("ARG_B:hashB",c.paths,c.occurrences,c.sample_ids,c.domain_bp)
    assert ns["build_vocabulary"](other).fingerprint()!=vocab.fingerprint()

def test_exhaustive_retrieval_matches_brute_force_with_cross_chunk_ties():
    import numpy as np
    ns=b.retrieval_core(b.policy())
    rng=np.random.default_rng(312)
    corpus=rng.normal(size=(19,7)).astype(np.float32)
    corpus[0]=corpus[6]=corpus[13]
    query=np.concatenate((corpus[:1],rng.normal(size=(3,7)).astype(np.float32)))
    corpus/=np.maximum(np.linalg.norm(corpus,axis=1,keepdims=True),np.float32(1e-12))
    query/=np.maximum(np.linalg.norm(query,axis=1,keepdims=True),np.float32(1e-12))
    ids=np.array([80,4,7,9,11,25,3,42,19,22,21,17,50,2,41,20,18,29,13],dtype=np.int64)
    labels=np.arange(19,dtype=np.int64)%4
    values,neighbors,_=ns["exhaustive_neighbors"](query,corpus,ids,labels,2,3,save_k=6)
    scores=query@corpus.T
    expected=np.array([ids[np.lexsort((ids,-row))[:6]] for row in scores])
    assert np.array_equal(neighbors,expected)
    for k in (1,5):
        votes=ns["hard_votes"](neighbors,ids,labels,k)
        lookup=dict(zip(ids,labels))
        expected_votes=np.array([[sum(lookup[i]==c for i in row[:k])/k for c in range(4)] for row in neighbors],np.float32)
        assert np.array_equal(votes,expected_votes)

def test_pure96_entire_corpus_eligibility_is_independent_of_split_query_and_mixed_rows():
    import numpy as np
    ns=b.retrieval_core(b.policy())
    refs=np.array([4000+2*i for i in range(96)])
    labels=np.repeat(np.arange(4,dtype=np.int64),24)
    queries=np.array([9000+2*i for i in range(120)])
    paths=tuple([(int(n),7001) for n in refs]+[(int(n),7003) for n in refs]+[(int(n),7001) for n in queries]+[(6000,7001)])
    ids,classes=ns["corpus_roles"](paths,queries,refs,labels)
    assert ids.tolist()==list(range(192)) and classes.tolist()==labels.tolist()*2
