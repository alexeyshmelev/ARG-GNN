"""Outer namespace adapters; no CLI, launcher, fits or weights are created at import.

These callable integration sources need independent review, real root admission,
and genuine parent observations/recipes/runtime qualification. They are not
evidence that missing full952 fits, checkpoints, recipes or jobs already exist.
"""
import ast,copy,hashlib,importlib,inspect,io,json,os
from pathlib import Path
from .outer_prediction_validator_v2 import (
    ROOT,HERE,C4,PARENTS,METHODS,ObservedReaderV2,need,record,ref,cell_directory,probability,observed_support,reviewed_source)
def _pin_module(module):
    p=Path(module.__file__).resolve();name=str(p.relative_to(ROOT))
    manifest=json.loads((ROOT/HERE/"source_manifest.v1.json").read_text())
    expected=next(r for r in manifest["refs"] if r["path"]==name)
    need(hashlib.sha256(p.read_bytes()).hexdigest()==expected["sha256"],"Exact existing native source before isolated adapter")
def _clone(module,name,replacements=(),overrides=None):
    _pin_module(module);function=getattr(module,name);text=inspect.getsource(function)
    for before,after in replacements:
        need(text.count(before)==1,"Reviewed exact native I/O literal changed: "+name);text=text.replace(before,after)
    namespace=dict(function.__globals__);namespace.update(overrides or {})
    exec(compile("from __future__ import annotations\n"+text,function.__code__.co_filename+":European_outer8_v2","exec"),namespace)
    return namespace[name]
def _publish_npz(path,arrays,guard):
    import numpy as np
    need(path.is_relative_to(ROOT) and path.resolve()==path,"Canonical admitted output")
    guard(sum(v.nbytes for v in arrays.values())+2*2**20)
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue();guard(len(data))
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    guard(0);return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
def _publish_json(path,value,guard):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(path.is_relative_to(ROOT) and path.resolve()==path and len(data)<=2**20,"Compact canonical admitted metadata")
    guard(len(data));fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    guard(0);return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
def exact_checkpoint_identity_v3(producer,checkpoint_ref):
    record(checkpoint_ref)
    need(producer.actual_ref(checkpoint_ref["path"])==checkpoint_ref,"Exact relative-string selected checkpoint reference")
    return ROOT/checkpoint_ref["path"]
def _authority(request_ref,method,parent,guard):
    reader=ObservedReaderV2(guard);q=reader.decode_json(request_ref)
    from datetime import datetime,timezone
    need(q["schema"]=="European_outer_native_adapter_ROOT_request_v2" and q["root_authorizer"]=="root_agent"
        and q["status"]=="active" and q["method_id"]==method and q["parent_id"]==parent
        and q["adapter_ref"]==ref(HERE+"/outer_adapters_v2.py") and q["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")
        and q["scientific_execution_authorized"] is True and q["outer_truth_access_authorized"] is False,"Exact separate native adapter ROOT admission")
    expires=datetime.fromisoformat(q["expires_at_utc"]);need(expires.tzinfo and datetime.now(timezone.utc)<expires,"Current actual native adapter authority")
    reviewed_source(q["adapter_ref"],q["source_review_ref"],reader)
    manifest=reader.decode_json(q["source_manifest_ref"])
    for source in manifest["refs"]:reader.read(source,maximum=32*2**20)
    qualified=reader.decode_json(q["actual_runtime_and_driver_qualification_ref"])
    need(qualified["root_authorizer"]=="root_agent" and qualified["status"]=="qualified" and qualified["unresolved_items"]==[]
        and qualified["method_id"]==method and qualified["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json"),"Missing runtime/driver/recipes remain blockers")
    need(q["output_directory"]==cell_directory(q["run_id"],method,parent),"Exact distinct cell namespace")
    return q,reader
def ap_outer_functions_v2():
    """Reuse original native AP functions; only outer scope/metadata changes."""
    ap=importlib.import_module("arggnn.expanded_native_ancestralpaths_v1")
    read=_clone(ap,"read_completed_parent_v1",[
        ('roles.get(parent_id) in ("train", "val")','roles.get(parent_id) == "outer"')])
    lineage=_clone(ap,"_lineage",[("outer_included=False","outer_included=True")])
    prepare=_clone(ap,"prepare_parent_v1",overrides={"_lineage":lineage})
    seal=_clone(ap,"seal_parent_predictions_v1",[
        ('roles.get(features["parent_id"]) == features["split"] == "val"','roles.get(features["parent_id"]) == features["split"] == "outer"'),
        ('split="val"','split="outer"'),("outer_included=False","outer_included=True")])
    return ap,read,prepare,seal
def predict_ap_outer_v2(request_ref,completion_ref,*,guard):
    q,reader=_authority(request_ref,"AncestralPaths_native_Dense5",json_parent(request_ref,guard),guard)
    need(q["stage"]=="predict","Explicit AP native preparation/prediction stage")
    ap,read,prepare,seal=ap_outer_functions_v2();cfg=ap.default_config_v1()
    inventory=reader.decode_json(ref(HERE+"/frozen_method_ids.v1.json"));fixed=inventory["methods"]["AncestralPaths_native_Dense5"]
    observation=read(q["parent_id"],completion_ref,config=cfg)
    feature=prepare(observation,ROOT/q["output_directory"]/"features",config=cfg,guard=guard)
    result=seal(fixed["fit_receipt_ref"],feature["manifest_ref"],ROOT/q["output_directory"]/"predictions",config=cfg,guard=guard)
    # Original native filenames/math stay intact. The cell contract binds this exact
    # relative product directory; the root publication step must place it canonically.
    return result
def json_parent(request_ref,guard):
    p=ObservedReaderV2(guard).decode_json(request_ref)["parent_id"];need(p in PARENTS,"Registered outer parent");return p
def argmix_outer_functions_v2(frames,guard):
    """Same stock exporter/Rust/GRU/refinement; no development truth re-reader."""
    expanded=importlib.import_module("arggnn.expanded_native_argmix_v1")
    native=importlib.import_module("arggnn.official_argmix_biological")
    need(callable(guard),"Actual original resource callback");roles={p:"outer" for p in PARENTS}
    observe=_clone(expanded,"prepare_observed_parent_v1",overrides={"PARENT_SPLITS":roles})
    infer=_clone(native,"infer_and_seal",[
        ('"official_argmix_biological_stock_fit_v1"','"expanded_native_argmix_stock_fit_v1"'),
        ('prepared = verify_prepared(fit["native_prepared"], cfg)','prepared = None'),
        ('frames = verify_frames(prepared["frames"], cfg)','frames = dict(parents=_outer_frames)'),
        ('metadata_info = reference_metadata_amendment(prepared, cfg)','metadata_info = copy.deepcopy(fit["metadata_reference_counting"])'),
        ('if frame["split"] != "val":','if frame["split"] != "outer":'),
        ('len(coords["grid_indexes"]), 120, 4','len(coords["grid_indexes"]), 686, 4'),
        ('    verify_prepared(fit["native_prepared"], cfg)\n','    _outer_guard(0)\n'),
        ('if len(seals) != 2:','if len(seals) != 1:'),
        ('Both observed VAL parents must be sealed before external scoring','One complete observed outer parent must be sealed before external scoring'),
        ('all_VAL_predictions_sealed.json','one_outer_predictions_sealed.json')],
        overrides={"stage":expanded._stage,"_outer_frames":frames,"copy":copy,"_outer_guard":guard})
    return expanded,observe,infer
def predict_argmix_outer_v2(request_ref,entry_ref,cfg,modules,*,guard):
    q,reader=_authority(request_ref,"ARGMix_current_native",json_parent(request_ref,guard),guard)
    need(q["stage"]=="predict","Explicit ARGMix native preparation/prediction stage")
    inventory=reader.decode_json(ref(HERE+"/frozen_method_ids.v1.json"));fixed=inventory["methods"]["ARGMix_current_native"]
    expanded,observe,_=argmix_outer_functions_v2({},guard)
    need(expanded.load_config_v1(cfg["config_ref"])==cfg,"Exact original native ARGMix config")
    frame_ref=observe(q["parent_id"],entry_ref,ROOT/q["output_directory"]/"frame",cfg,guard=guard)
    frame=reader.decode_json(frame_ref)
    _,_,infer=argmix_outer_functions_v2({q["parent_id"]:frame},guard)
    native_output=expanded._new_output(ROOT/q["output_directory"]/"native")
    result=infer(fixed["fit_receipt_ref"],native_output,cfg,modules)
    need(len(result)==1 and result[0]["parent_id"]==q["parent_id"],"One exact native author seal")
    native_ref=ref(q["output_directory"]+"/native/"+q["parent_id"]+"/prediction_seal.json")
    axis_ref=_publish_json(ROOT/q["output_directory"]/"adapter_axis_receipt.json",dict(
        schema="European_outer_ARGMix_axis_receipt_v3",parent_id=q["parent_id"],method_id="ARGMix_current_native",
        adapter_source_ref=ref(HERE+"/outer_adapters_v2.py"),native_completion_ref=native_ref,
        fit_receipt_ref=fixed["fit_receipt_ref"],checkpoint_ref=fixed["checkpoint_ref"],config_ref=cfg["config_ref"],
        authored_source_ref=ref("src/arggnn/expanded_native_argmix_v1.py"),class_order=list(cfg["class_order"])),guard)
    guard(0);return dict(native_author_seals=result,adapter_axis_receipt_ref=axis_ref)
def predict_h050_outer_v2(request_ref,ctx,*,guard):
    """Frozen pair only; same native holds/softmax and exact original grid mapping.

    Caller must first supply reviewed full outer bank/episode/corrected cut recipes.
    This function performs no selection, optimization, truth access or recipe fit.
    """
    m=ctx["method_id"];need(m in METHODS[:2],"Only the fixed H050 pair")
    q,reader=_authority(request_ref,m,json_parent(request_ref,guard),guard)
    need(q["stage"]=="predict","Explicit native prediction stage")
    expected=observed_support(q["parent_id"],ctx["observation_entry_ref"],reader)
    import numpy as np
    import torch
    producer=importlib.import_module("research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.producer_v1")
    _pin_module(producer);model,hold=producer.model,producer.hooks
    _pin_module(model);_pin_module(hold);_pin_module(producer.h45)
    producer.h45.science()
    from .h050_observed_bridge_v3 import reconstruct_h050_bundle_v3
    bundle,parent,provenance=reconstruct_h050_bundle_v3(q,ctx,expected,producer,reader,guard)
    inventory=reader.decode_json(ref(HERE+"/frozen_method_ids.v1.json"));fixed=inventory["methods"][m];arm=fixed["arm"]
    cp_path=exact_checkpoint_identity_v3(producer,fixed["checkpoint_ref"])
    cp=torch.load(cp_path,map_location="cpu",weights_only=False)
    need(cp["schema"]=="H050_direct_readout_checkpoint_v1" and cp["arm"]==arm and cp["pass_index"]==fixed["selected_new_pass"]
        and cp["seed"]==101 and cp["phase"]=="train" and cp["parameter_count"]==model.PARAMETERS[arm]
        and cp["scientific_protocol"]==hold.fixed_protocol_v1(),"Unchanged selected H050 checkpoint/protocol")
    network=(model.OrderedCladeEpisodeGNNV1(**model.MODEL_KWARGS,arm="own_state") if arm==model.CONTINUATION
        else model.DirectReadoutCutMemoryGNNV1(memory_arm=arm)).to(ctx["device"]).eval().requires_grad_(False)
    network.load_state_dict(cp["model_state"],strict=True);bundle.validate()
    bundle,_=producer.device_parent(ctx,bundle)
    value=hold.predict_holds_v1(arm,network,bundle).softmax(-1).detach().cpu().numpy()
    value=probability(value,(len(bundle.bank.spans),686,4))
    indexes=bundle.bank.readout_hold.detach().cpu().numpy()
    need(indexes.shape==(len(parent["positions"]),) and indexes.dtype.kind in "iu" and np.all((indexes>=0)&(indexes<len(value))),"Exact native-hold to original grid join")
    result=dict(C4_probabilities=value[indexes],positions_bp=parent["positions"],bin_edges_bp=parent["edges"],
        retained_fixed100bp_indexes=parent["retained_indexes"],query_sample_indexes=parent["query_slots"])
    output=ctx["output"];guard(int(result["C4_probabilities"].nbytes)+65536)
    product=_publish_npz(ROOT/q["output_directory"]/"probabilities.npz",result,guard)
    need(product["path"]==q["output_directory"]+"/probabilities.npz","Canonical cell publication")
    return output.save("prediction_seal.json",dict(schema="European_outer_H050_fixed_checkpoint_prediction_v2",parent_id=parent["parent_id"],
        arm=arm,pass_index=fixed["selected_new_pass"],checkpoint_ref=fixed["checkpoint_ref"],
        training_completion_ref=fixed["training_completion_ref"],selection_completion_ref=fixed["selection_completion_ref"],
        query_keys=expected["query_keys"],class_order=C4,predictions_ref=product,optimization_performed=False,targets_read=False,
        bank_provenance=provenance))
def argformer_outer_functions_v2(resource_caps):
    """Source integration for the full native recipe; genuine qualification is pending."""
    need(isinstance(resource_caps,dict) and resource_caps,"Actual full952 caps, never reduced fixtures")
    inner=importlib.import_module("arggnn.expanded_native_argformer_tsinfer_v1")
    def outer_config():
        cfg=copy.deepcopy(inner.default_config_v1())
        cfg.update(primary_fit_parent_ids=list(PARENTS),primary_fitting_scope="eight_independent_transductive_OUTER_ARGs",
            TRAIN_ARGformer_role="registry_metadata_not_checkpoint_training_for_OUTER",
            all40_CPU_inference_role="historical_inner_provenance_only")
        cfg["observation"]["parent_order"]=list(PARENTS)
        return cfg
    registry=_clone(inner,"_registry",overrides={"default_config_v1":outer_config})
    completed=_clone(inner,"_completed",[('roles.get(parent_id) in ("train", "val")','roles.get(parent_id) == "outer"')],{"_registry":registry})
    observation=_clone(inner,"observe_parent_v1",[("outer_included=False","outer_included=True")],{"_completed":completed})
    read=_clone(inner,"read_observed_parent_v1",[('roles[parent_id] in ("train", "val")','roles[parent_id] == "outer"')],
        {"_completed":completed,"default_config_v1":outer_config,"_registry":registry})
    # Extend the existing AST namespace adapter only at the registered-split guard.
    insert='''def visit_Compare(self,node):
            if (len(node.ops)==1 and isinstance(node.ops[0],ast.In) and len(node.comparators)==1
                    and isinstance(node.comparators[0],ast.Tuple)
                    and all(isinstance(x,ast.Constant) for x in node.comparators[0].elts)
                    and tuple(x.value for x in node.comparators[0].elts)==("train","val")
                    and isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Attribute)
                    and isinstance(node.left.func.value,ast.Name) and node.left.func.value.id=="roles"
                    and node.left.func.attr=="get"):
                node.comparators[0]=ast.Tuple(elts=[ast.Constant("outer")],ctx=ast.Load())
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if node.arg=="outer_included" and isinstance(node.value,ast.Constant) and node.value.value is False:
                node.value=ast.Constant(True)
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):'''
    namespace=_clone(inner,"_native_namespace",[("def visit_FunctionDef(self, node):",insert)],
        {"read_observed_parent_v1":read,"_NATIVE":None,"default_config_v1":outer_config,"_registry":registry})(resource_caps)
    need(namespace["_prepared"] is not None,"Existing complete native corpus path retained")
    return dict(config=outer_config(),observe=observation,read=read,native=namespace,cpu_runtime=inner._CPU_runtime)
def run_argformer_outer_stage_v2(request_ref,stage,args,*,guard,resource_caps,gpu_authorizer=None):
    q,_=_authority(request_ref,"ARGformer_native_within_ARG",json_parent(request_ref,guard),guard)
    need(stage in ("observe","prepare","fit","predict") and q["stage"]==stage,"One explicitly admitted native stage")
    api=argformer_outer_functions_v2(resource_caps);cfg=api["config"];p=q["parent_id"]
    expected=ROOT/q["output_directory"]/dict(observe="observe",prepare="prepared",fit="fit",predict="prediction")[stage]
    need(Path(args["output"])==expected,"Exact admitted stage output")
    if stage=="prepare":record(args["observation_ref"],q["output_directory"]+"/observe/observation_manifest.json")
    if stage in ("fit","predict"):record(args["prepared_ref"],q["output_directory"]+"/prepared/prepared_manifest.json")
    if stage=="predict":record(args["fit_ref"],q["output_directory"]+"/fit/fit_manifest.json")
    if stage=="observe":return api["observe"](p,args["completion_ref"],args["output"],config=cfg,guard=guard,resource_caps=resource_caps)
    if stage=="prepare":
        api["cpu_runtime"]("prepare",cfg)
        return api["native"]["prepare_parent_v1"](p,args["observation_ref"],args["output"],config=cfg,guard=guard)
    need(callable(gpu_authorizer),"Actual qualified GPU/runtime authority required; no fabricated fits")
    if stage=="fit":return api["native"]["fit_parent_v1"](args["prepared_ref"],args["output"],config=cfg,guard=guard,gpu_authorizer=gpu_authorizer)
    return api["native"]["infer_parent_v1"](args["fit_ref"],args["prepared_ref"],args["output"],config=cfg,guard=guard,gpu_authorizer=gpu_authorizer)
