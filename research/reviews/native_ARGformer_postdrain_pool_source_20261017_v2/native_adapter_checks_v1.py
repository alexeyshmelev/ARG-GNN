"""Explicit author RAM/static integration checks. No production module imports."""
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import ast,copy,hashlib,json
HERE=Path(__file__).parent;ROOT=HERE.parents[2]
def need(value,message):
    if not value:raise ValueError(message)
def extract(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[copy.deepcopy(n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    need({n.name for n in nodes}==set(names),"Exact extracted source function set")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<RAM actual source>","exec"),namespace)
    return namespace
def rejected(fn):
    try:fn()
    except (ValueError,TypeError,KeyError,AssertionError):return True
    return False

def run_v1():
    parsed={p.name:ast.parse(p.read_text(),filename=str(p))for p in HERE.glob("*.py")}
    constant=extract(HERE/"native_common_v1.py",["cardinality_v1"],dict(require=need))
    derive=constant["cardinality_v1"]
    derivations=[]
    for n in (512,513,1024,1041,23346222):
        for drop in (True,False):
            row=derive(n,drop);need(row["expected_optimizer_updates"]==5*row["batches_per_epoch"],"Five exact full loader epochs")
            derivations.append(row)
    need(derive(513,True)["expected_optimizer_updates"]!=derive(1024,True)["expected_optimizer_updates"],"Different complete parents do not inherit one fixed budget")
    need(rejected(lambda:derive(True,True))and rejected(lambda:derive(0,True)),"Malformed/empty original cardinality rejected")
    class Optimizer:
        def __init__(self,preloaded=False):self.preloaded=preloaded
        def state_dict(self):return {"state":{0:{"step":1}}if self.preloaded else {}}
    def author_module(*,rows=1041,workers=8,drop=True,preloaded=False):
        loader=SimpleNamespace(dataset=list(range(rows)),drop_last=drop,batch_size=512,num_workers=workers,sampler=SimpleNamespace(num_replicas=1,rank=0))
        class SizedLoader:
            def __getattr__(self,name):return getattr(loader,name)
            def __len__(self):return rows//512 if drop else (rows+511)//512
        module=SimpleNamespace(builder_calls=0,trainer_calls=0,fit_calls=0)
        def builder(config,*args,**kwargs):module.builder_calls+=1;return SizedLoader()
        def trainer(*args,**kwargs):
            module.trainer_calls+=1
            return SimpleNamespace(state=SimpleNamespace(timestamp=SimpleNamespace(epoch=0,batch=0),optimizers=[Optimizer(preloaded)]))
        def main(config,return_trainer=False):
            train=module.build_dataloader(config)
            result=module.Trainer();module.fit_calls+=1;return result
        module.build_dataloader=builder;module.Trainer=trainer;module.main=main
        return module
    c=SimpleNamespace(require=need)
    native=extract(HERE/"native_child_v1.py",["_int_time","optimizer_counter_evidence","validated_composer_timestamp_v1","run_original_checked_v1","_final_checkpoint"],dict(c=c,timedelta=timedelta,Path=Path))
    config=SimpleNamespace(dataset=SimpleNamespace(split="train"));protocol=dict(cardinality=derive(1041,True))
    module=author_module();builder=module.build_dataloader;trainer=module.Trainer
    _,loader_receipt,fresh=native["run_original_checked_v1"](module,config,protocol)
    need(module.builder_calls==module.trainer_calls==module.fit_calls==1,"Exactly one original builder/Trainer/fit call; no qualification replay")
    need(module.build_dataloader is builder and module.Trainer is trainer,"Original author bindings restored after metadata-only wrappers")
    need(loader_receipt["actual_dataset_cardinality"]==1041 and fresh["optimizer_state_empty"],"Actual original parent loader and fresh optimizer attested")
    for bad in (dict(rows=1042),dict(workers=4),dict(drop=False),dict(preloaded=True)):
        module=author_module(**bad)
        need(rejected(lambda:native["run_original_checked_v1"](module,config,protocol)),"Wrong parent/cardinality/workers/drop_last or preloaded optimizer rejected")
        need(module.fit_calls==0,"Rejected metadata never reaches fake fit")
    keys={"iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch"}
    walls={"total_wct","iteration_wct","epoch_wct","batch_wct"}
    timestamp={k:0 for k in keys};timestamp.update({k:timedelta(0)for k in walls});timestamp.update(epoch=5,batch=10)
    need(native["validated_composer_timestamp_v1"]({"timestamp":{"Timestamp":timestamp}})==timestamp,"Reviewed installed nested Timestamp accepted")
    need(rejected(lambda:native["validated_composer_timestamp_v1"]({"timestamp":timestamp})),"Flat timestamp remains rejected")
    class FakePath:
        name="ep5-ba10-rank0.pt"
        def lstat(self):return SimpleNamespace(st_size=100)
    path=FakePath();state=dict(timestamp={"Timestamp":timestamp},model={"w":1},optimizers={"DecoupledAdamW":{"state":{0:dict(step=10,exp_avg=0,exp_avg_sq=0)}}})
    native["_checkpoint_entries"]=lambda directory:([path],None)
    torch=SimpleNamespace(load=lambda *a,**k:dict(state=state))
    final=native["_final_checkpoint"](None,epochs=5,updates=10,max_files=2,max_bytes=1000,torch=torch,reference=lambda p:dict(path=p.name,sha256="a"*64))
    need(final["actual_optimizer_updates"]==10 and final["completed_epochs"]==5,"Actual saved full epoch/optimizer counters validated against derived budget")
    state["optimizers"]["DecoupledAdamW"]["state"][0]["step"]=9
    need(rejected(lambda:native["_final_checkpoint"](None,epochs=5,updates=10,max_files=2,max_bytes=1000,torch=torch,reference=lambda p:dict(path=p.name,sha256="a"*64))),"Wrong saved optimizer counter rejected")
    # All registered original parents are required before an evaluator may even
    # receive a completed pretruth barrier. This fixture reads JSON metadata only.
    registry=json.loads((HERE/"parent_registry.v1.json").read_text())
    fake_c=SimpleNamespace(require=need,ref=lambda p:p,PARENT_REGISTRY="registry",read=lambda r:registry)
    ns=extract(HERE/"benchmark_barrier_v1.py",["all_parent_pretruth_barrier_v1"],dict(c=fake_c,PARENTS=tuple(registry["parents"])))
    records={p:dict(parent_id=p,query_focal_count=10000,evaluation_length_bp=30000000,query_focals_ref=r["query_focals_ref"],observation_ref=r["observation_ref"],query_ancestry_truth_opened=False,full_author_per_parent_pipeline_completed=True,source_bound_prediction_bytes_verified=True,genuine_completion_direct0_owned_drain_outer0_verified=True)for p,r in registry["parents"].items()}
    barrier=ns["all_parent_pretruth_barrier_v1"](records,lambda x:x)
    need(barrier["ancestry_truth_access_authorized"] is False and barrier["all_five_native_predictions_verified"],"Pretruth barrier grants no truth access or score by itself")
    subset=dict(records);subset.pop("american_c3_val_004")
    need(rejected(lambda:ns["all_parent_pretruth_barrier_v1"](subset,lambda x:x)),"Missing one original parent rejects")
    changed=copy.deepcopy(records);changed["american_c3_val_001"]["query_focals_ref"]=registry["parents"]["american_c3_val_000"]["query_focals_ref"]
    need(rejected(lambda:ns["all_parent_pretruth_barrier_v1"](changed,lambda x:x)),"Cross-parent focal transplant rejects")
    # No fixed VAL000 count or checkpoint is permitted in native executable code.
    for name in ("native_common_v1.py","native_child_v1.py","native_root_driver_v1.py","native_admission_v1.py"):
        need("227990"not in (HERE/name).read_text(),"No fixed VAL000 optimizer count")
    old="research/reviews/GPU0_postdrain_two_slot_pool_source_20261017_v2";new=str(HERE.relative_to(ROOT))
    for name in ("producer_v1.py","observations_gpu_v1.py","context_factory_v1.py"):
        before=(ROOT/old/name).read_text().replace(old,new).replace(old.replace("/","."),new.replace("/","."))
        need(ast.dump(ast.parse(before),include_attributes=False)==ast.dump(parsed[name],include_attributes=False),"Sealed H058 science namespace-only")
    # Exact original loader implementation and release YAML remain fully pinned.
    author=ROOT/".runtime/author_sources/argformer_f7545be7bca0"
    need(hashlib.sha256((author/"src/arg_data.py").read_bytes()).hexdigest()=="2e7df2c955ece89ced3f05c609a4450f997250df5a7cb06ff4a2ca957e49ffdf","Exact original loader source")
    loader_ast=next(n for n in ast.parse((author/"src/arg_data.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="build_arg_dataloader")
    need(ast.unparse(loader_ast).count("drop_last=cfg.drop_last")==2,"Original sampler and loader drop_last preserved at world1")
    return dict(schema="native_original_parent_postdrain_author_RAM_checks_v1",status="PASS_STATIC_RAM_ONLY",python_AST_files=len(parsed),original_parent_cardinality_cases=derivations,exact_original_loader_and_fresh_Trainer_called_once=True,wrong_parent_workers_drop_last_or_optimizer_rejected_before_fit=True,nested_Composer_Timestamp_and_genuine_saved_optimizer_counter_checks=True,all_five_pretruth_barrier_and_focal_identity_negatives=True,H058_scientific_AST_namespace_only=True,production_imports=0,scientific_models_or_numeric_payloads=0,real_process_device_guard_probe_admission_journal_Git_operations=0,fresh_independent_whole_join_review_required=True)
if __name__=="__main__":
    result=run_v1();out=HERE/"native_adapter_checks.final.v1.json"
    with out.open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps(dict(status=result["status"],result_path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),sort_keys=True))
