"""Targeted CPU synthetic I/O parity checks; no real corpus/model/checkpoint.

Pinned author retrieval helper source is compiled into an isolated namespace and
called on synthetic CPU tensors. The original project seal is run with injected
in-memory readers, without monkeypatching global filesystem or Torch functions.
"""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,types
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
BASE=Path("research/reviews/C3_native_post_MLM_root_handoff_20261015_v1")
path=BASE/"ram_downstream_v1.py"
spec=importlib.util.spec_from_file_location("_synthetic_RAM_adapter",path)
ram=importlib.util.module_from_spec(spec);spec.loader.exec_module(ram)
assert not torch.cuda.is_initialized()
events=[]
guard=lambda event:events.append(event)
parent="synthetic_RAM_IO_parent"
counts={"datasets":{}}
orders={}
datasets={}
sizes={("query_paths","train"):9001,("query_paths","val"):999,("contrastive","train"):2,("contrastive","val"):1}
q_offset=c_offset=0
for (view,split),n in sizes.items():
    start=q_offset if view=="query_paths" else c_offset
    classes=(np.arange(start,start+n)%3).astype(np.int64)
    ids=np.arange(start,start+n,dtype=np.int64)+(100000 if view=="query_paths" else 700000)
    nodes=np.arange(start,start+n,dtype=np.int64)%100+(600 if view=="query_paths" else 0)
    order=dict(record_ids=ids,sample_nodes=nodes,sample_indices=nodes.copy(),
        parent_id=np.asarray(parent),class_order=np.asarray(ram.CLASSES))
    if view=="contrastive":order["reference_classes_0based"]=classes.copy()
    orders[view+"/"+split]=order;counts["datasets"][view+"/"+split]={"rows":n}
    inputs=torch.zeros((n,128),dtype=torch.long);inputs[:,0]=torch.from_numpy(classes)+10
    inputs[:,1]=torch.arange(start,start+n)%7
    batches=dict(input_ids=inputs,attention_mask=torch.ones((n,128),dtype=torch.long),
        extant_node_id_raw=torch.from_numpy(nodes))
    if view=="contrastive":batches["labels"]=torch.from_numpy(classes)
    datasets[(view,split)]=batches
    if view=="query_paths":q_offset+=n
    else:c_offset+=n
loader_calls=[];embedding_calls=[]
class FakeModel:
    def eval(self):return self
def build_loader(**kwargs):
    loader_calls.append(kwargs.copy())
    batches=datasets[(kwargs["local"],kwargs["split"])]
    for start in range(0,len(batches["input_ids"]),kwargs["batch_size"]):
        yield {k:v[start:start+kwargs["batch_size"]] for k,v in batches.items()}
def get_embeddings(**kwargs):
    embedding_calls.append((kwargs["layer_index"],kwargs["pooling_strategy"]))
    batch=kwargs["batch"];out=torch.zeros((len(batch["input_ids"]),768),dtype=torch.float32)
    classes=batch["input_ids"][:,0]-10
    out[torch.arange(len(out)),classes]=1.
    return out
fake_emb=types.SimpleNamespace(_build_arg_split_dataloader=build_loader,get_embeddings=get_embeddings)
parts={};combined={};combined_order={}
for view in ("query_paths","contrastive"):
    parts[view]={}
    for split in ("train","val"):
        cfg=dict(train_loader=dict(local=view,max_seq_len=128,skip_extant_tokens=True),
            model=dict(model_config=dict(max_seq_len=128)))
        parts[view][split]=ram.embed_single_split_ram_v1(torch=torch,embeddings_module=fake_emb,
            model=FakeModel(),cfg=cfg,view=view,split=split,
            expected_rows=sizes[(view,split)],fast_guard=guard)
    combined[view],combined_order[view]=ram.concatenate_embeddings_ram_v1(torch=torch,view=view,
        parts=parts[view],orders={sp:orders[view+"/"+sp] for sp in ("train","val")},
        counts=counts,parent_id=parent,fast_guard=guard)
assert len(loader_calls)==4 and all(c["batch_size"]==256 and c["num_workers"]==0
    and c["return_metadata"] is True and c["skip_extant_tokens"] is True for c in loader_calls)
assert all(call==(None,"head_norm") for call in embedding_calls)
assert combined["query_paths"]["embeddings"].shape==(10000,768)
assert combined["query_paths"]["input_ids"].dtype==combined["query_paths"]["attention_masks"].dtype==torch.int64
assert combined["query_paths"]["embeddings"].dtype==torch.float32
assert "labels" not in combined["query_paths"]
for view in combined:
    for key,value in combined[view].items():
        if key=="metadata":continue
        expected=torch.cat([parts[view][sp][key] for sp in ("train","val")],dim=0)
        assert torch.equal(value,expected),(view,key)
class NoProgress:
    def __init__(self,*args,**kwargs):pass
    def update(self,*args):pass
    def close(self):pass
author_path=Path(".runtime/author_sources/argformer_f7545be7bca0/retrieve.py")
author_text=author_path.read_text();author_tree=ast.parse(author_text)
names={"labels_from_metadata","ensure_l2_normalized","topk_indices_streaming"}
selected=[n for n in author_tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(selected)==3
native_ns=dict(torch=torch,F=F,tqdm=NoProgress,Dict=dict,List=list,Optional=__import__("typing").Optional,
    Tuple=tuple)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(author_path)+":synthetic_helpers","exec"),native_ns)
native=types.SimpleNamespace(**{name:native_ns[name] for name in names})
retrieval_calls=[]
original_topk=native.topk_indices_streaming
def recorded_topk(Q,E,**kwargs):
    retrieval_calls.append(kwargs.copy());return original_topk(Q,E,**kwargs)
native.topk_indices_streaming=recorded_topk
result=ram.retrieve_ram_v1(retrieve_module=native,query=combined["query_paths"],
    corpus=combined["contrastive"],device=torch.device("cpu"),fast_guard=guard)
assert retrieval_calls==[dict(device=torch.device("cpu"),k=1,q_batch=1024,e_chunk=100000)]
assert result["top1_indices"].shape==(10000,1) and result["top1_scores"].shape==(10000,1)
assert torch.equal(result["top1_indices"][:,0],torch.arange(10000)%3)
assert torch.equal(result["top1_scores"],torch.ones((10000,1)))
assert result["corpus_labels"]==["0","1","2"] and "query_labels" not in result
permutation=np.roll(np.arange(9999,-1,-1,dtype=np.int64),113)
qorder=combined_order["query_paths"]
registry={key:qorder[key][permutation] for key in ("record_ids","sample_nodes","sample_indices")}
registry.update(positions_bp0=np.arange(10000,dtype=np.float64)+.5,
    parent_id=np.asarray(parent),class_order=np.asarray(ram.CLASSES))
prep=dict(parent_id=parent,class_order=list(ram.CLASSES),focals_ref="synthetic_focal_ref",
    query_embedding_view_prespecified_focals_only=True,contrastive_labels_reference_only=True,
    query_ancestry_truth_opened=False,full_path_id_order_refs={key:key for key in orders})
cfg=dict(external_truth_ref=None,prediction_ref=None,preparation_ref="synthetic_prep_ref",
    focal_registry_ref="synthetic_focal_ref",expected_splits=["train","val"],expected_pooling="head_norm",
    native_query_embeddings_ref=None,native_corpus_embeddings_ref=None,native_retrieval_ref=None,
    native_RAM_tensor_manifest_ref="synthetic_manifest_ref")
class MemoryWriter:
    def __init__(self):self.outputs={}
    def npz(self,name,**arrays):self.outputs[name]={k:v.copy() for k,v in arrays.items()};return {"synthetic_record":name}
    def json(self,name,value):self.outputs[name]=copy.deepcopy(value);return {"synthetic_record":name}
writer=MemoryWriter()
ram.native_top1_seal_ram_v1(cfg,writer,registry,prep=prep,orders=orders,
    query_arrays=combined["query_paths"],corpus_arrays=combined["contrastive"],native=result,torch=torch)
expected=np.eye(3,dtype=np.float32)[(np.arange(10000)%3)[permutation]]
assert np.array_equal(writer.outputs["predictions.npz"]["probabilities"],expected)
assert np.array_equal(writer.outputs["native_top1_preserved.npz"]["record_ids"],qorder["record_ids"])
assert np.array_equal(writer.outputs["native_top1_preserved.npz"]["top1_indices"],np.arange(10000)%3)
# Run the original project seal with isolated in-memory readers, changing no math.
original_path=Path("src/arggnn/american_c3_evaluate_v1.py")
original_text=original_path.read_text();original_tree=ast.parse(original_text)
node=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name=="native_top1_seal")
node=copy.deepcopy(node);node.body=[n for n in node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
class TorchReader:
    def __getattr__(self,name):return getattr(torch,name)
    def load(self,key,**kwargs):return {"query":combined["query_paths"],"corpus":combined["contrastive"],"retrieval":result}[key]
original_ns=dict(np=np,torch=TorchReader(),CLASSES=ram.CLASSES,require=ram.require,
    read=lambda record:prep,bound=lambda record:record,load_npz=lambda record:orders[record],
    verify_prediction=ram.verify_prediction)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
    str(original_path)+":synthetic_injected_readers","exec"),original_ns)
file_cfg=dict(cfg,native_query_embeddings_ref="query",native_corpus_embeddings_ref="corpus",native_retrieval_ref="retrieval")
original_writer=MemoryWriter()
original_ns["native_top1_seal"](file_cfg,original_writer,registry)
for name in ("predictions.npz","native_top1_preserved.npz"):
    assert writer.outputs[name].keys()==original_writer.outputs[name].keys()
    assert all(np.array_equal(writer.outputs[name][key],original_writer.outputs[name][key]) for key in writer.outputs[name])
# Negatives target identity, leakage, native tensor dtype and index bounds.
negative_checks=[]
def rejects(name,operation):
    try:operation()
    except ValueError:negative_checks.append(name)
    else:raise AssertionError("Expected rejection: "+name)
bad_query=dict(combined["query_paths"],labels=torch.zeros(10000,dtype=torch.long))
rejects("query_label_leak",lambda:ram.native_top1_seal_ram_v1(cfg,MemoryWriter(),registry,
    prep=prep,orders=orders,query_arrays=bad_query,corpus_arrays=combined["contrastive"],native=result,torch=torch))
bad_result=dict(result,topk_indices=torch.full((10000,1),3,dtype=torch.long),
    top1_indices=torch.full((10000,1),3,dtype=torch.long))
rejects("retrieval_index_out_of_range",lambda:ram.native_top1_seal_ram_v1(cfg,MemoryWriter(),registry,
    prep=prep,orders=orders,query_arrays=combined["query_paths"],corpus_arrays=combined["contrastive"],native=bad_result,torch=torch))
bad_orders={key:dict(value) for key,value in orders.items()}
bad_orders["query_paths/train"]["record_ids"]=bad_orders["query_paths/train"]["record_ids"].copy()
bad_orders["query_paths/train"]["record_ids"][0]=bad_orders["query_paths/train"]["record_ids"][1]
rejects("duplicate_full_path_ID",lambda:ram.native_top1_seal_ram_v1(cfg,MemoryWriter(),registry,
    prep=prep,orders=bad_orders,query_arrays=combined["query_paths"],corpus_arrays=combined["contrastive"],native=result,torch=torch))
bad_parts=dict(parts["query_paths"]);bad_parts["train"]=dict(bad_parts["train"])
bad_parts["train"]["input_ids"]=bad_parts["train"]["input_ids"].to(torch.int32)
rejects("token_dtype_downcast",lambda:ram.concatenate_embeddings_ram_v1(torch=torch,view="query_paths",
    parts=bad_parts,orders={sp:orders["query_paths/"+sp] for sp in ("train","val")},
    counts=counts,parent_id=parent,fast_guard=guard))
manifest=ram.tensor_manifest_v1(torch=torch,values={"query":combined["query_paths"],"corpus":combined["contrastive"],
    "retrieval":result},fast_guard=guard)
tensor=combined["query_paths"]["embeddings"]
assert manifest["bundles"]["query"]["tensors"]["embeddings"]["raw_C_sha256"]==hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
assert not torch.cuda.is_initialized()
receipt=dict(schema="native_RAM_adapter_synthetic_source_check_v1",status="pass",
    scope="synthetic_CPU_IO_parity_only_not_native_training_or_ancestry_accuracy",
    synthetic_query_rows=10000,synthetic_reference_rows=3,loader_calls=len(loader_calls),
    embedding_batches=len(embedding_calls),native_retrieval_parameters=dict(k=1,q_batch=1024,e_chunk=100000),
    parity_with_original_project_seal=True,all_native_tensor_values_and_dtypes_preserved=True,
    focal_permutation_restored=True,negative_checks=negative_checks,raw_tensor_hash_parity=True,
    real_corpus_checkpoint_or_truth_payloads_opened=False,real_models_loaded=False,CUDA_initialized=False,
    admissions_accounting_Git_mutations=0,adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    author_retrieval_source_sha256=hashlib.sha256(author_text.encode()).hexdigest(),
    original_project_seal_source_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
    torch_version=torch.__version__,numpy_version=np.__version__)
(BASE/"RAM_synthetic_checks.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps(receipt,sort_keys=True))
