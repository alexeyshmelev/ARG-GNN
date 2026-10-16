"""Synthetic source fixtures only. No Torch/PyG/model or project payload reads."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1"
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/HERE)]
import evaluation_contract_v1 as c
import worker_entry_v1 as worker
import root_driver_v1 as root_driver
import numpy as np
from arggnn import american_c3_evaluate_v1 as common
def bad(call):
    try:call()
    except (ValueError,KeyError):return True
    raise AssertionError("Invalid synthetic condition accepted")
checks=[]
def check(name,test):assert test;checks.append(name)
# Metadata-only restored prefix / fixed six-candidate qualification.
migration=dict(migrated_boundary_ref=dict(path="synthetic/restored_epoch5.pt",sha256="a"*64),
 successor_protocol_sha256="p",observation_seal_sha256="o",target_join_sha256="t",actual_global_step=100,
 migrated_checkpoint_receipts=[dict(global_step=100,migrated_ref=dict(path="synthetic/restored_epoch5.pt",sha256="a"*64))])
training=dict(status="completed_C3_checkpointed_complete_parent_training",
 schema="american_c3_checkpointed_complete_parent_training_v1",epochs=30,complete_parent_updates=600,
 focal_exposures=6000000,trained_parent_ids=["american_c3_short_train_%03d"%i for i in range(20)],
 train_sequence_lengths_bp=[2000000]*20,VAL_used_for_optimization=False,TEST_deferred=True,
 H043_extension=None,purity_filter=False,resume_ref=migration["migrated_boundary_ref"],
 protocol_sha256="p",observation_seal_sha256="o",target_join_sha256="t",
 checkpoint_candidates=[dict(epoch=e,global_step=e*20,checkpoint_ref=(migration["migrated_boundary_ref"] if e==5 else
 dict(path=c.TRAIN_OUTPUT+"/checkpoint_epoch_%03d.pt"%e,sha256=str(e%10)*64))) for e in c.EPOCHS])
check("all600_metadata_and_restored_prefix",c._candidate_metadata_v1(training,migration)==training)
for key,value in [("complete_parent_updates",599),("VAL_used_for_optimization",True),("epochs",29),
                  ("protocol_sha256","wrong"),("schema","wrong")]:
    copy1=copy.deepcopy(training);copy1[key]=value
    check("reject_"+key,bad(lambda:c._candidate_metadata_v1(copy1,migration)))
copy1=copy.deepcopy(training);copy1["checkpoint_candidates"][0]["checkpoint_ref"]=dict(path="synthetic/unrestored.pt",sha256="a"*64)
check("reject_unrestored_original_epoch5",bad(lambda:c._candidate_metadata_v1(copy1,migration)))
copy1=copy.deepcopy(training);copy1["checkpoint_candidates"]=copy1["checkpoint_candidates"][:-1]
check("reject_missing_epoch30",bad(lambda:c._candidate_metadata_v1(copy1,migration)))
cmdref=dict(path=HERE+"/fixtures/synthetic_wait.json",sha256="0"*64)
job=dict(logical_job=c.TRAIN_JOB,exit_code=0,state="succeeded",leader_wait_status_known=True,
 owned_drain_observed=True,cancel_requested=False,resource_abort=None,pid=12345,leader_start_ticks=456,
 requested=dict(input_ref=c.TRAIN_ENVELOPE_REF,job_id=c.TRAIN_JOB))
command=dict(client_returncode=0,stderr="",stdout=json.dumps(job))
terminal=dict(actual_terminal_consumed=True,actual_exit_code=0,result_path=cmdref["path"])
check("genuine_wait_contract_synthetic_values",c._logical_wait_values_v1(command,terminal,cmdref)==job)
for key,value in [("exit_code",1),("owned_drain_observed",False),("leader_wait_status_known",False),("state","running")]:
    j=copy.deepcopy(job);j[key]=value;x=dict(command,stdout=json.dumps(j))
    check("reject_wait_"+key,bad(lambda:c._logical_wait_values_v1(x,terminal,cmdref)))
# Original unchanged matrix function body with synthetic metadata only.
source=(ROOT/"src/arggnn/american_c3_gnn_selection_isolated_v2.py").read_text()
tree=ast.parse(source);names=("validation_matrix_v1","score_validation_matrix_v1")
nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names]
module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
records={};bound_calls=[];truth_calls=[]
matrix=dict(status="sealed_C3_all30_original_VAL_predictions",training_completion_ref="synthetic_train",
 protocol_sha256="p",validation_parent_ids=list(c.VAL),checkpoint_epochs=list(c.EPOCHS),
 all_predictions_sealed_before_VAL_truth=True,metrics="original_C3_external_evaluator_unchanged",cells=[])
for e in c.EPOCHS:
    for p in c.VAL:
        name="synthetic_seal_%d_%s"%(e,p);prediction="synthetic_prediction_"+name;focal="synthetic_focals_"+p
        cp=next(x["checkpoint_ref"] for x in training["checkpoint_candidates"] if x["epoch"]==e)
        cell=dict(epoch=e,parent_id=p,prediction_seal_ref=name,prediction_ref=prediction,focal_registry_ref=focal,
                  focal_identity_sha256="synthetic_identity_"+p,checkpoint_ref=cp)
        records[name]=dict(status="sealed_C3_original_heldout_predictions",split="val",records=10000,
            training_completion_ref="synthetic_train",protocol_sha256="p",private_truth_opened=False,purity_filter=False,**{k:v for k,v in cell.items() if k!="prediction_seal_ref"})
        matrix["cells"].append(cell)
records["synthetic_matrix"]=matrix
namespace=dict(require=c.require,read=lambda r:copy.deepcopy(records[r]),bound=lambda r:bound_calls.append(r),
 completed_training_v1=lambda r:copy.deepcopy(training),VAL=c.VAL,CHECKPOINT_EPOCHS=c.EPOCHS)
exec(compile(module,"<unchanged_original_C3_matrix_and_score_function_bodies>","exec"),namespace)
namespace["validation_matrix_v1"]("synthetic_matrix")
check("original30_cell_barrier_binds_every_prediction_and_registry",len(bound_calls)==60)
class Writer:
    def save(self,*args,**kwargs):raise AssertionError("No scoring output before valid barrier")
for kind in ("missing","duplicate","wrong_checkpoint","truth_opened"):
    records["synthetic_matrix"]=copy.deepcopy(matrix)
    if kind=="missing":records["synthetic_matrix"]["cells"].pop()
    elif kind=="duplicate":records["synthetic_matrix"]["cells"][-1]=copy.deepcopy(matrix["cells"][0])
    elif kind=="wrong_checkpoint":records["synthetic_matrix"]["cells"][0]["checkpoint_ref"]={"path":"bad","sha256":"b"*64}
    else:
        records[matrix["cells"][0]["prediction_seal_ref"]]["private_truth_opened"]=True
    def resolve(p):truth_calls.append(p);raise AssertionError("No truth resolver allowed for invalid barrier")
    check("original_barrier_rejects_"+kind,bad(lambda:namespace["score_validation_matrix_v1"](
        "synthetic_matrix",writer=Writer(),resolve_val_truth=resolve,runtime_guard=lambda event:None)))
    records[matrix["cells"][0]["prediction_seal_ref"]]["private_truth_opened"]=False
check("all_invalid_barriers_before_any_truth_resolution",truth_calls==[])
# Original common identity join and metric equations using four handmade rows.
r=dict(record_ids=np.array([11,12,13,14],dtype=np.int64),sample_nodes=np.array([60,60,61,61],dtype=np.int64),
 sample_indices=np.array([600,600,601,601],dtype=np.int64),positions_bp0=np.array([1.,5.,7.,9.9],dtype=np.float64),
 parent_id=np.asarray("synthetic_fixture_parent"),class_order=np.asarray(["EUR","EAS","AFR"]))
calls=np.array([0,2,0,1]);p=dict(r,probabilities=np.eye(3,dtype=np.float32)[calls])
check("original_identity_prediction_acceptance",np.array_equal(common.verify_prediction(p,r),calls))
for field in ("record_ids","sample_nodes","sample_indices","positions_bp0","parent_id","class_order"):
    x={k:v.copy() for k,v in p.items()}
    if x[field].shape==():x[field]=np.asarray("wrong_parent")
    elif field=="class_order":x[field]=x[field][::-1]
    else:x[field][0]+=1
    check("reject_prediction_identity_"+field,bad(lambda:common.verify_prediction(x,r)))
x=dict(p,probabilities=np.eye(3,dtype=np.float64)[calls])
check("reject_changed_probability_dtype",bad(lambda:common.verify_prediction(x,r)))
x=dict(p,probabilities=p["probabilities"]*.9)
check("reject_probability_renormalization_input",bad(lambda:common.verify_prediction(x,r)))
run=1
while (ROOT/HERE/("fixtures_run_%02d"%run)).exists():run+=1
base=ROOT/HERE/("fixtures_run_%02d"%run);base.mkdir(exist_ok=False)
def npz(name,arrays):
    path=base/name;np.savez_compressed(path,**arrays)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
rr=npz("synthetic_registry.npz",r);pr=npz("synthetic_prediction.npz",p)
truth=dict(intervals_bp=np.array([[0.,5.],[5.,10.]],dtype=np.float64),
 query_sample_indices=np.array([600,601],dtype=np.int64),source_classes_0based=np.array([[0,1],[2,0]],dtype=np.uint8),
 parent_id=r["parent_id"],class_order=r["class_order"])
tr=npz("synthetic_truth.npz",truth);scores=[]
class Sink:
    def json(self,name,value):scores.append(value);return dict(synthetic_only=True)
common.execute(dict(common.DEFAULTS,mode="score",method="SYNTHETIC_SOURCE_FIXTURE_ONLY",
 prediction_ref=pr,focal_registry_ref=rr,external_truth_ref=tr),Sink())
check("original_common_scorer_halfopen_join",scores[0]["result"]["confusion"]==[[2,1,0],[0,0,0],[0,0,1]])
check("original_common_scorer_tiny_accuracy",scores[0]["result"]["accuracy"]==.75)
# Operational wrapper must preserve object identity and original order.
objects=[object() for _ in range(4)];events=[]
class Loader:
    dataset="synthetic_dataset"
    def __iter__(self):return iter(objects)
wrapped=worker.GuardedLoaderV1(Loader(),events.append)
got=list(wrapped)
check("guarded_loader_same_objects_and_order",all(a is b for a,b in zip(got,objects)) and len(got)==4)
check("guarded_loader_checks_every_block",events.count("before_original_whole_tree_inference_block")==4
      and events.count("after_original_whole_tree_inference_block")==4)
check("exact_reviewed_CPU_supervisor_can_bind_without_launch",callable(root_driver.bind_reviewed_supervisor_v1()))
check("no_Torch_or_PyG_or_model_import","torch" not in sys.modules and "torch_geometric" not in sys.modules)
result=dict(schema="C3_evaluation_tiny_source_fixture_v1",status="PASSED",synthetic_only=True,checks=checks,
 count=len(checks),original_common_scorer_equations_reused=True,original_pretruth_barrier_function_bodies_reused=True,
 scientific_payload_or_checkpoint_or_model_run=False,actual_fit_or_ancestry_score_claim=False,
 jobs_GPU_accounting_packages_Git=False)
(ROOT/HERE/"tiny_source_fixture.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status="PASSED_SYNTHETIC_ONLY",checks=len(checks),Torch_imported=False)))
