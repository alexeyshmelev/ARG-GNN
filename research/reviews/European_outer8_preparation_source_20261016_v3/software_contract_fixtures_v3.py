"""Tiny CPU software contracts only; no experiments, models or original payloads."""
from pathlib import Path
import ast,copy,hashlib,importlib,io,json,math,sys,types
import numpy as np
BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/"src")]
V=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_prediction_validator_v2")
A=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_adapters_v2")
class MemoryReader:
    """Only this synthetic dictionary can be read; paths never reach the filesystem."""
    def __init__(self):self.data={};self.maximum=0
    def put(self,path,value):
        payload=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(",",":")).encode()
        self.data[path]=payload;self.maximum=max(self.maximum,sum(map(len,self.data.values())))
        assert self.maximum<900000,"Software fixture corpus must stay below1MB"
        return dict(path=path,sha256=hashlib.sha256(payload).hexdigest())
    def npz(self,path,arrays):
        out=io.BytesIO();np.savez_compressed(out,**arrays);return self.put(path,out.getvalue())
    def read(self,r,**kwargs):
        data=self.data[r["path"]];assert hashlib.sha256(data).hexdigest()==r["sha256"];return data
    def decode_json(self,r):return json.loads(self.read(r))
COHORTS={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26,"WHG":102,"EHG":86,"Ana":50,"CHG":28}
CLASSES={"Ana":0,"CHG":1,"WHG":2,"EHG":3};OLD={"WHG":12,"EHG":10,"Ana":2,"CHG":10}
def rows(native=False):
    result=[];i=0
    for cohort,count in COHORTS.items():
        for j in range(count):
            node=(10000+i if native else i)
            age=0 if cohort=="present_day" else 400 if cohort in OLD and j<OLD[cohort] else 50
            result.append(dict(sample_index=i,sample_id="tsk_%d"%(i//2),haplotype=i%2,
                observed_cohort=cohort,age_generations=age,role="pure_source_reference" if cohort in CLASSES else "query",
                observed_reference_class=CLASSES.get(cohort),observed_node_id=node,inferred_tree_node_id=node,source_observed_node_id=i))
            i+=1
    return result
def parent():
    source=rows();q=source[:686]
    return dict(parent_id=V.PARENTS[0],split="outer",entry_ref=dict(path="SOFTWARE_FIXTURE/entry.json",sha256="0"*64),
        source=source,coverage=np.array([0.,200.]),positions=np.array([50.,150.]),edges=np.array([0.,100.,200.]),
        retained_indexes=np.array([0,1],dtype=np.int64),query_slots=np.arange(686,dtype=np.int64),
        raw_query_nodes=np.arange(686,dtype=np.int64),query_nodes=np.arange(686,dtype=np.int64),
        cohorts=np.asarray([r["observed_cohort"] for r in q]),query_keys=[[r["sample_id"],r["haplotype"]] for r in q])
def products(p,method):
    value=np.full((2,686,4),.25,dtype=np.float32)
    common=dict(positions_bp=p["positions"],bin_edges_bp=p["edges"])
    if method.startswith("H050"):return dict(common,C4_probabilities=value,retained_fixed100bp_indexes=p["retained_indexes"],query_sample_indexes=p["query_slots"])
    if method=="AncestralPaths_native_Dense5":
        return dict(common,source_probabilities=value,query_node_ids=p["query_nodes"],query_sample_indices=p["query_slots"],
            retained_fixed100bp_indexes=p["retained_indexes"],coverage_original_bp0=p["coverage"],
            local_to_original_offset_bp=np.asarray(0.,dtype=np.float64),raw_query_nodes=p["raw_query_nodes"])
    if method=="ARGMix_current_native":
        return dict(common,probabilities=value,supported=np.ones((2,686),dtype=bool),grid_indexes=p["retained_indexes"],
            local_positions_bp=p["positions"],local_site_positions_bp=np.array([1,101],dtype=np.int64),
            original_site_positions_bp=np.array([1.,101.]),query_nodes=p["query_nodes"])
    return dict(common,probabilities=value,SNP_probabilities=value.copy(),site_positions_bp=np.array([1.,101.]),
        query_nodes=np.arange(10000,10686,dtype=np.int64),query_sample_indices=p["query_slots"],raw_query_nodes=p["raw_query_nodes"],
        retained_fixed100bp_indexes=p["retained_indexes"],coverage_original_bp0=p["coverage"])
REPORT=[];MAX_BYTES=0
def passed(name):REPORT.append(dict(name=name,status="PASS"))
def reject(name,call):
    try:call()
    except (ValueError,KeyError,TypeError,AssertionError):passed(name);return
    raise AssertionError("Missing rejection: "+name)
def support_reader(p):
    r=MemoryReader();inference=r.put("SOFTWARE_FIXTURE/inference.json",dict(post_five_iterations=dict(site_join=[
        dict(local_position_0based=1,original_position_0based=1.),dict(local_position_0based=101,original_position_0based=101.)])))
    p["entry"]=dict(original_native_inference_receipt=inference)
    roster=r.put("SOFTWARE_FIXTURE/argformer_roster.json",rows(True))
    return r,roster
def wait(r,cell,completion,stage=""):
    base=V.cell_directory(cell["run_id"],cell["method_id"],cell["parent_id"])+("/"+stage if stage else "")+"/evidence"
    execution=dict(cell,evidence_directory=base)
    source=r.put("SOFTWARE_FIXTURE/driver.py",b"# software fixture; no process is launched\n")
    review=r.put("SOFTWARE_FIXTURE/driver_review.json",dict(verdict="SOURCE_PASS",source_ref=source,software_fixture_only=True))
    launch=r.put(base+"/launch.json",dict(pid=111,start_ticks=222,software_fixture_only=True))
    admission=r.put(base+"/input.json",dict(schema="European_outer_native_adapter_ROOT_request_v2",status="active",scientific_execution_authorized=True,
        outer_truth_access_authorized=False,source_manifest_ref=V.ref(V.HERE+"/source_manifest.v1.json"),
        stage={"observation":"prepare_bank","cuts":"prepare_cuts","fit":"fit","":"predict"}[stage],
        root_authorizer="root_agent",parent_id=cell["parent_id"],method_id=cell["method_id"],adapter_ref=cell["actual_inference_adapter_ref"],software_fixture_only=True))
    actual=r.put(base+"/actual_wait.json",dict(schema="European_outer_native_actual_wait_v2",returncode=0,actual_OS_wait_status=0,
        actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None,completion_ref=completion,
        input_ref=admission,launch_ref=launch,driver_source_ref=source,pid=111,start_ticks=222,group_observations=[dict(absent=True)],
        adopted_waits=[],software_fixture_only=True))
    terminal=r.put(base+"/ROOT_terminal_receipt.json",dict(root_authorizer="root_agent",actual_tool_terminal_consumed=True,
        actual_tool_exit_code=0,actual_wait_ref=actual,completion_ref=completion,input_ref=admission,software_fixture_only=True))
    execution.update(actual_wait_ref=actual,launch_ref=launch,input_ref=admission,wait_driver_source_ref=source,wait_driver_review_ref=review,ROOT_terminal_receipt_ref=terminal)
    return execution
def native_case(method):
    p=parent();r,roster=support_reader(p);run="SOFTWARE_ONLY"
    base=V.cell_directory(run,method,p["parent_id"])
    adapter=r.put("SOFTWARE_FIXTURE/adapter.py",b"# software fixture source only\n")
    cell=dict(run_id=run,method_id=method,parent_id=p["parent_id"],actual_inference_adapter_ref=adapter)
    checkpoint=dict(path="SOFTWARE_FIXTURE/checkpoint.bin",sha256="1"*64);identity=dict(checkpoint_ref=checkpoint)
    n=dict(parent_id=p["parent_id"])
    if method.startswith("H050"):
        n.update(schema="European_outer_H050_fixed_checkpoint_prediction_v2",arm="fixture",pass_index=30,checkpoint_ref=checkpoint,
            query_keys=p["query_keys"],optimization_performed=False,class_order=V.C4)
        identity.update(arm="fixture",selected_new_pass=30)
        identity["training_completion_ref"]=r.put("SOFTWARE_FIXTURE/training.json",dict(checkpoint_groups={"30":{"fixture":checkpoint}}))
        identity["selection_completion_ref"]=r.put("SOFTWARE_FIXTURE/selection.json",dict(selected={"fixture":{"checkpoint":checkpoint}}))
        n.update(training_completion_ref=identity["training_completion_ref"],selection_completion_ref=identity["selection_completion_ref"])
        geom=r.npz(base+"/observation/native_hold_geometry.npz",dict(intervals_bp0=np.array([[0.,100.],[100.,200.]]),
            positions_bp=p["positions"],query_sample_indexes=p["query_slots"],readout_hold=np.array([0,1],dtype=np.int64)))
        array_ref=dict(path="SOFTWARE_FIXTURE/bank.npz",sha256="2"*64)
        cache=r.put(base+"/observation/cache/manifest.json",dict(parents={p["parent_id"]:dict(array_ref=array_ref)},
            entry_refs={p["parent_id"]:p["entry_ref"]},targets_cached=False,observations_only=True))
        prep=r.put(base+"/observation/bank_preparation.json",dict(schema="European_outer_H050_bank_preparation_v3",parent_id=p["parent_id"],
            original_observed_entry_ref=p["entry_ref"],producer_source_ref=V.ref(V.HERE+"/h050_observed_bridge_v3.py"),cache_ref=cache,
            native_hold_geometry_ref=geom,targets_read=False))
        cut_receipt=r.put("SOFTWARE_FIXTURE/cut.json",dict(original_cache_array_ref=array_ref,original_observed_entry_ref=p["entry_ref"],lossless_roundtrip_verified=True))
        cuts=r.put("SOFTWARE_FIXTURE/cuts.json",dict(observation_cache_ref=cache,entry_refs={p["parent_id"]:p["entry_ref"]},
            targets_read=False,observations_only=True,parents={p["parent_id"]:dict(receipt_ref=cut_receipt)}))
        n["bank_provenance"]=dict(bank_preparation_ref=prep,cache_ref=cache,native_hold_geometry_ref=geom,cache_array_ref=array_ref,
            cut_preparation_completion_ref=cuts,bank_execution=wait(r,cell,prep,"observation"),cut_execution=wait(r,cell,cuts,"cuts"))
        completion=base+"/prediction_seal.json"
    elif method=="AncestralPaths_native_Dense5":
        identity["fit_receipt_ref"]=r.put("SOFTWARE_FIXTURE/AP_fit.json",dict(checkpoint_ref=checkpoint,epochs=5,VAL_selection=False))
        focal=r.npz("SOFTWARE_FIXTURE/focal.npz",dict(native_probabilities=np.full((2,6),1/6,dtype=np.float32),
            site_row=np.array([0,1]),query_column=np.array([0,1]),cohort=p["cohorts"][:2],
            positions_bp=np.array([1.,101.]),query_sample_indices=np.array([0,1]),query_node_ids=np.array([0,1])))
        n.update(schema="expanded_native_ap_prediction_seal_v1",split="outer",checkpoint_ref=checkpoint,fit_ref=identity["fit_receipt_ref"],
            query_keys=p["query_keys"],targets_read=False,C4_collapse_before_argmax="p1+p5,p2+p6,p3,p4",
            native6_focal_ref=focal,C4_class_order=V.C4);completion=base+"/predictions/prediction_seal.json"
    elif method=="ARGMix_current_native":
        # Literal actual author seal deliberately has no class_order/C4_class_order.
        owned={"src/arggnn/expanded_native_argmix_v1.py":"3"*64}
        source=r.put("src/arggnn/expanded_native_argmix_v1.py",b"# software fixture authored source\n");owned[source["path"]]=source["sha256"]
        config=r.put("SOFTWARE_FIXTURE/ARGMix_config.json",dict(schema="expanded_native_argmix_v1",hypothesis_id="H038",owned_code_sha256=owned,class_order=["ANA","CHG","WHG","EHG"]))
        identity["fit_receipt_ref"]=r.put("SOFTWARE_FIXTURE/ARGMix_fit.json",dict(checkpoint=checkpoint,config_sha256=config["sha256"],owned_code_sha256=owned))
        n.update(checkpoint=checkpoint,frame=dict(source_observed_entry_ref=p["entry_ref"]),full_support_complete=True,
            missing_grid_query_cells=0,native_replay_max_probability_error=0.,original_raw_logits_are_per_chunk_not_log_of_averaged_probability=True,
            native_final_estimand="source arithmetic average of overlapping-chunk softmax probabilities",
            original_chunk_logits=dict(path="SOFTWARE_FIXTURE/logits.npz",sha256="4"*64),
            stock_base_HDF5=dict(path="SOFTWARE_FIXTURE/base.h5",sha256="5"*64),
            stock_refined_HDF5=dict(path="SOFTWARE_FIXTURE/refined.h5",sha256="6"*64))
        completion=base+"/native/"+p["parent_id"]+"/prediction_seal.json"
        cell["native_completion_ref"]=r.put(completion,n)
        cell["adapter_axis_receipt_ref"]=r.put(base+"/adapter_axis_receipt.json",dict(schema="European_outer_ARGMix_axis_receipt_v3",
            parent_id=p["parent_id"],method_id=method,adapter_source_ref=adapter,native_completion_ref=cell["native_completion_ref"],
            fit_receipt_ref=identity["fit_receipt_ref"],checkpoint_ref=checkpoint,config_ref=config,authored_source_ref=source,class_order=["ANA","CHG","WHG","EHG"]))
    else:
        tree=dict(path="SOFTWARE_FIXTURE/observed.trees",sha256="7"*64);namespace=p["parent_id"]+":"+tree["sha256"]
        prepared=r.put(base+"/prepared/prepared_manifest.json",dict(parent_id=p["parent_id"],split="outer",namespace=namespace,
            inferred_tree_ref=tree,true_query_ancestry_read=False,input_samples=952,query_haplotypes=686,reference_haplotypes=266,
            query_keys=p["query_keys"],MLM_optimizer_updates=5,contrastive_optimizer_updates=1,files={"observed_roster.json":roster}))
        phases={}
        for key,epochs,batch,seed,precision,directory in [("MLM",5,512,17,"amp_bf16","mlm"),("contrastive",1,4096,36,"fp32","contrastive")]:
            phases[key]=dict(checkpoint_ref=dict(path=base+"/fit/"+directory+"/composer_final.pt",sha256="8"*64),
                epochs=epochs,logical_batch=batch,seed=seed,precision=precision,optimizer_updates=5 if key=="MLM" else 1,
                query_truth_used=False,same_ARG_only=True,source_batch_reduced=False,gradient_cache_used=False)
        fit=r.put(base+"/fit/fit_manifest.json",dict(parent_id=p["parent_id"],prepared_ref=prepared,namespace=namespace,
            fresh_same_ARG_fit=True,old_checkpoint_reused=False,query_truth_used=False,final_checkpoint_ref=phases["contrastive"]["checkpoint_ref"],**phases))
        n.update(schema="expanded_native_argformer_tsinfer_prediction_seal_v1",class_order=V.C4,fit_ref=fit,prepared_ref=prepared,
            namespace=namespace,primary_k=1,sensitivity_k=5,raw_vote_zero_probabilities_retained=True,no_query_truth_read=True)
        cell.update(actual_checkpoint_refs=dict(mlm=phases["MLM"]["checkpoint_ref"],contrastive=phases["contrastive"]["checkpoint_ref"]),
            native_query_roster_ref=roster,fit_execution=wait(r,cell,fit,"fit"))
        completion=base+"/prediction/prediction_seal.json"
    if method!="ARGMix_current_native":cell["native_completion_ref"]=r.put(completion,n)
    if method!="ARGformer_native_within_ARG":cell["actual_checkpoint_refs"]=dict(selected=checkpoint)
    cell.update(wait(r,cell,cell["native_completion_ref"]))
    return p,r,cell,identity
def main():
    global MAX_BYTES
    p=parent()
    for method in V.METHODS:
        r,roster=support_reader(p);z=products(p,method);product=r.npz("SOFTWARE_FIXTURE/product.npz",z)
        V.validate_product(product,p,method,r,argformer_roster_ref=roster);passed("native_archive_success_"+method)
        key="C4_probabilities" if method.startswith("H050") else "source_probabilities" if method=="AncestralPaths_native_Dense5" else "probabilities"
        for label,edit in [("nan",lambda a:a.__setitem__((0,0,0),np.nan)),("negative",lambda a:a.__setitem__((0,0,0),-.1)),
                           ("unnormalized",lambda a:a.__setitem__((0,0,0),.8))]:
            bad=copy.deepcopy(z);edit(bad[key]);badref=r.npz("SOFTWARE_FIXTURE/bad_"+label+".npz",bad)
            reject("reject_"+method+"_"+label,lambda:V.validate_product(badref,p,method,r,argformer_roster_ref=roster))
        bad=copy.deepcopy(z);bad["positions_bp"]=bad["positions_bp"]+1;badref=r.npz("SOFTWARE_FIXTURE/bad_support.npz",bad)
        reject("reject_"+method+"_support",lambda:V.validate_product(badref,p,method,r,argformer_roster_ref=roster))
        for label,change in [
            ("probability_shape",lambda d:d.__setitem__(key,d[key][:,:685])),
            ("probability_dtype",lambda d:d.__setitem__(key,d[key].astype(np.float64))),
            ("bin_edges",lambda d:d.__setitem__("bin_edges_bp",d["bin_edges_bp"]+1)),
            ("missing_schema_field",lambda d:d.pop("positions_bp"))]:
            bad=copy.deepcopy(z);change(bad);badref=r.npz("SOFTWARE_FIXTURE/bad_"+label+".npz",bad)
            reject("reject_"+method+"_"+label,lambda:V.validate_product(badref,p,method,r,argformer_roster_ref=roster))
        query_key=("query_sample_indexes" if method.startswith("H050") else "query_nodes" if method=="ARGMix_current_native" else "query_sample_indices")
        bad=copy.deepcopy(z);bad[query_key]=bad[query_key][::-1].copy();badref=r.npz("SOFTWARE_FIXTURE/permuted_query.npz",bad)
        reject("reject_"+method+"_query_order",lambda:V.validate_product(badref,p,method,r,argformer_roster_ref=roster))
        p2,r2,cell,identity=native_case(method);V.native_lineage(cell,p2,identity,r2);passed("native_lineage_success_"+method)
        if method!="ARGMix_current_native":
            broken=r2.decode_json(cell["native_completion_ref"]);axis="C4_class_order" if method=="AncestralPaths_native_Dense5" else "class_order"
            broken[axis]=list(reversed(V.C4));r2.put(cell["native_completion_ref"]["path"],broken)
            broken_cell=dict(cell,native_completion_ref=dict(path=cell["native_completion_ref"]["path"],sha256=hashlib.sha256(r2.data[cell["native_completion_ref"]["path"]]).hexdigest()))
            reject("reject_"+method+"_class_axis",lambda:V.native_lineage(broken_cell,p2,identity,r2))

        MAX_BYTES=max(MAX_BYTES,r.maximum,r2.maximum)
    method="ARGMix_current_native";p2,r,cell,identity=native_case(method)
    missing=dict(cell);missing.pop("adapter_axis_receipt_ref");reject("ARGMix_axis_receipt_required",lambda:V.native_lineage(missing,p2,identity,r))
    r,roster=support_reader(p);z=products(p,method);z["supported"][0,0]=False
    reject("ARGMix_missing_support",lambda:V.validate_product(r.npz("SOFTWARE_FIXTURE/no_support.npz",z),p,method,r))
    query=p["query_slots"];readout=np.array([0,1]);spans=np.array([100.,100.],dtype=np.float32);intervals=np.array([[0.,100.],[100.,200.]])
    V.validate_h050_bank_geometry_v3(query,readout,spans,(2,952),intervals,p);passed("H050_actual_bank_geometry_success")
    reject("H050_bank_query_permutation",lambda:V.validate_h050_bank_geometry_v3(query[::-1],readout,spans,(2,952),intervals,p))
    reject("H050_bank_readout_permutation",lambda:V.validate_h050_bank_geometry_v3(query,readout[::-1],spans,(2,952),intervals,p))
    reject("H050_reduced_panel",lambda:V.validate_h050_bank_geometry_v3(query,readout,spans,(2,951),intervals,p))
    reject("H050_changed_native_spans",lambda:V.validate_h050_bank_geometry_v3(query,readout,spans+1,(2,952),intervals,p))
    fixture_dir=BASE/"software_fixture_files";fixture_dir.mkdir(exist_ok=True);(fixture_dir/"bytes.bin").write_bytes(b"software bytes only")
    source=(PROJECT/"research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/producer_v1.py").read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="actual_ref")
    env=dict(ROOT=fixture_dir,hashlib=hashlib);exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_ref:software_fixture","exec"),env)
    actual=env["actual_ref"];fixed=actual("bytes.bin")
    A.exact_checkpoint_identity_v3(types.SimpleNamespace(actual_ref=actual),fixed);passed("H050_relative_reference_success")
    assert actual(fixture_dir/"bytes.bin")!=fixed;passed("H050_previous_absolute_Path_mismatch_reproduced")
    metrics=dict(calibration_bins=15,boundary_tolerance_bp=1000.,short_tract_cutoff_bp=10000.)
    view=V.metric_parent_view_v3(p);V.preflight_metric_contract_v3(view,metrics);passed("full_metric_parent_preflight_success")
    for field in ("samples","coverage_original","cohorts","query_slots","retained_indexes"):
        bad=dict(view);bad.pop(field)
        reject("metric_preflight_missing_"+field,lambda:V.preflight_metric_contract_v3(bad,metrics))
    helper_source=(PROJECT/"scripts/train_ancestralpaths_biological_v1.py").read_text();tree=ast.parse(helper_source)
    names={"intersections","boundary_match","tracts","length_summary","full_span_score","json_value"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env={"np":np,"math":math};exec(compile(ast.Module(body=nodes,type_ignores=[]),"inherited_metric:software_fixture","exec"),env)
    truth=types.SimpleNamespace(intervals=np.array([[0.,2000000.]]),c4=np.zeros((1,686),dtype=np.int64))
    result=env["full_span_score"](products(p,V.METHODS[0])["C4_probabilities"],view,truth,np.array([0,1]),metrics)
    assert result["query_count"]==2 and "boundaries" in result and "calibration_ece" in result;passed("actual_inherited_full_span_helper_software_contract_success")
    json.dumps(env["json_value"](result),allow_nan=False);passed("original_metric_JSON_serializer_contract")
    helper=next(n for n in nodes if n.name=="full_span_score")
    required={n.slice.value for n in ast.walk(helper) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=="parent" and isinstance(n.slice,ast.Constant)}
    assert required<={"samples","coverage_original","edges","query_slots","cohorts","retained_indexes"}
    passed("all_actual_inherited_parent_helper_fields_audited")
    assert V.ObservedReaderV2(lambda:None).root==PROJECT;passed("original_observed_reader_root_API")
    q=dict(parent_id=p["parent_id"],stage="predict",output_directory=V.cell_directory("SOFTWARE_ONLY","ARGMix_current_native",p["parent_id"]))
    trace=[];cfg=dict(config_ref=dict(path="SOFTWARE_FIXTURE/config.json",sha256="1"*64),class_order=["ANA","CHG","WHG","EHG"])
    fixed=dict(fit_receipt_ref=dict(path="SOFTWARE_FIXTURE/fit.json",sha256="2"*64),checkpoint_ref=dict(path="SOFTWARE_FIXTURE/checkpoint.bin",sha256="3"*64))
    reader=types.SimpleNamespace(decode_json=lambda r:dict(methods={"ARGMix_current_native":fixed}) if r["path"].endswith("frozen_method_ids.v1.json") else dict(parent_id=p["parent_id"]))
    def new_output(path):trace.append(("mkdir",str(path)));return path
    expanded=types.SimpleNamespace(load_config_v1=lambda r:cfg,_new_output=new_output)
    author=[dict(parent_id=p["parent_id"])]
    def infer(fit,path,config,modules):
        assert trace[-1]==("mkdir",str(path));trace.append(("infer",str(path)));return author
    def fake_api(frames,guard):return expanded,(lambda *a,**k:dict(path="SOFTWARE_FIXTURE/frame.json",sha256="4"*64)),infer
    published=[]
    namespace=dict(A.predict_argmix_outer_v2.__globals__,_authority=lambda *a:(q,reader),json_parent=lambda *a:p["parent_id"],
        argmix_outer_functions_v2=fake_api,ref=lambda name:dict(path=name,sha256="5"*64),
        _publish_json=lambda path,value,guard:(published.append(value) or dict(path=str(path),sha256="6"*64)))
    function=types.FunctionType(A.predict_argmix_outer_v2.__code__,namespace)
    value=function({},p["entry_ref"],cfg,None,guard=lambda *a:None)
    assert value["native_author_seals"]==author and "class_order" not in author[0] and published[0]["class_order"]==cfg["class_order"]
    passed("ARGMix_native_output_directory_created_before_native_API")
    passed("ARGMix_separate_axis_receipt_preserves_author_seal")

    ap_source=(PROJECT/"src/arggnn/ancestralpaths_native_bridge.py").read_text();tree=ast.parse(ap_source)
    ap_nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"_probabilities","_path_labels","native_accuracy_confusion"}]
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    env=dict(np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+ap_nodes,type_ignores=[])),"native_AP6:software_fixture","exec"),env)
    focal_source=(PROJECT/"src/arggnn/expanded_native_ancestralpaths_v1.py").read_text()
    focal=next(n for n in ast.parse(focal_source).body if isinstance(n,ast.FunctionDef) and n.name=="native_focal_metrics_v1")
    focal_env=dict(bridge=types.SimpleNamespace(native_accuracy_confusion=env["native_accuracy_confusion"]))
    exec(compile(ast.Module(body=[focal],type_ignores=[]),"native_focal:software_fixture","exec"),focal_env)
    outcome=focal_env["native_focal_metrics_v1"](np.full((2,6),1/6,dtype=np.float32),np.array([1,6]),cohorts=np.array(["present_day","present_day"]))
    assert outcome["n_examples"]==2 and outcome["confusion_predicted_rows_truth_columns"].shape==(6,6)
    passed("actual_native_AP6_focal_scorer_API_and_schema")
    paths={V.product_path("SOFTWARE_ONLY",m,par) for m in V.METHODS for par in V.PARENTS}
    assert len(paths)==40;passed("distinct_all40_canonical_primary_paths")
    reject("reject_namespace_traversal",lambda:V.cell_directory("../invalid",V.METHODS[0],V.PARENTS[0]))
    import inspect
    assert "bundle" not in inspect.signature(A.predict_h050_outer_v2).parameters and "parent" not in inspect.signature(A.predict_h050_outer_v2).parameters
    passed("H050_caller_cannot_supply_bank_bundle_or_parent")
    bridge_source=(BASE/"h050_observed_bridge_v3.py").read_text()
    writer_node=next(n for n in ast.parse(bridge_source).body if isinstance(n,ast.ClassDef) and n.name=="_CutWriterV3")
    writer_env=dict(Path=Path,ROOT=fixture_dir,need=V.need,record=V.record)
    exec(compile(ast.Module(body=[writer_node],type_ignores=[]),"cut_writer:software_fixture","exec"),writer_env)
    class SyntheticWriter:
        def save(self,name,value,kind=None):
            relative="software_cell/cuts/"+name
            destination=fixture_dir/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            payload=b"software cut bytes";destination.write_bytes(payload)
            return dict(path=relative,sha256=hashlib.sha256(payload).hexdigest())
    bounds=types.SimpleNamespace(extra_arrays={})
    writer=writer_env["_CutWriterV3"](SyntheticWriter(),bounds,"software_cell",100,lambda *a:None)
    cutref=writer.save("tiny.npz",{},"npz")
    assert bounds.extra_arrays[cutref["path"],cutref["sha256"]]==len(b"software cut bytes")
    passed("H050_cut_writer_registers_actual_canonical_archive_readback_bound")
    reject("H050_cut_writer_rejects_traversal",lambda:writer.save("../invalid.npz",{},"npz"))
    writer=writer_env["_CutWriterV3"](SyntheticWriter(),bounds,"software_cell",1,lambda *a:None)
    reject("H050_cut_writer_rejects_archive_over_admitted_cap",lambda:writer.save("overcap.npz",{},"npz"))
    entry=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.check_prediction_barrier_v2")
    from contextlib import redirect_stdout
    captured=io.StringIO()
    with redirect_stdout(captured):status=entry.main([])
    assert status==2 and json.loads(captured.getvalue())["status"]=="INACTIVE_SOURCE_ONLY"
    passed("default_entry_is_inactive_without_resource_or_payload_calls")
    called=[]
    dispatch=types.FunctionType(entry.main.__code__,dict(entry.main.__globals__,run_v3=lambda r,stage:(called.append((r,stage)) or {})))
    for flag,stage in (("--validate-ref","validate"),("--score-ref","score")):
        with redirect_stdout(io.StringIO()):assert dispatch([flag,"{}"])==0
        assert called[-1]==({},stage)
        passed("future_ROOT_entry_dispatch_"+stage)
    assert "torch" not in sys.modules;passed("software_fixtures_import_no_Torch")
    assert MAX_BYTES+sum(f.stat().st_size for f in fixture_dir.rglob("*") if f.is_file())<900000
    report=dict(schema="European_outer8_v3_tiny_software_contract_results",software_fixtures_only=True,
        source_contract_successes_and_rejections=REPORT,cases=len(REPORT),max_serialized_in_memory_fixture_bytes=MAX_BYTES,
        disk_fixture_bytes=sum(f.stat().st_size for f in fixture_dir.rglob("*") if f.is_file()),no_original_data_model_truth_or_checkpoint_access=True,
        no_Torch_model_or_resource_RPC_or_admissions=True,scientific_accuracy_or_runtime_qualification_claimed=False)
    (BASE/"software_contract_results.v3.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(dict(status="SOFTWARE_CONTRACT_FIXTURES_PASS",cases=len(REPORT),max_fixture_bytes=MAX_BYTES)))
if __name__=="__main__":main()
