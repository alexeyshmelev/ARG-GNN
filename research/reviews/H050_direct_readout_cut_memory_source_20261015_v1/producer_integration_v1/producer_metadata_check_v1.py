"""Focused pure metadata/checkpoint-schema check; no scientific training."""
from pathlib import Path
import copy,hashlib,io,json,sys
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1 import producer_v1 as p
HERE=Path(__file__).resolve().parent

def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return True
    raise AssertionError("Expected rejection")

cfg=json.loads((HERE/"config_train_evaluate.proposed.inactive.json").read_text())
p.validate_protocol(cfg,active=False)
assert rejects(lambda:p.validate_protocol(cfg))
active=copy.deepcopy(cfg);active.update(execution_enabled=True,root_resource_caps_reviewed=True)
active.update({k:1 for k in p.RESOURCE_CAPS}) # handmade metadata values, never an issued capability
p.validate_protocol(active)
class Reader:
    def __init__(self):self.bytes={};self.metadata={}
    def read(self,r):return self.bytes[r["path"]]
    def decode_json(self,r):return self.metadata[r["path"]]
r=Reader();config_ref=dict(path="handmade/config.json",sha256="0"*64);r.metadata[config_ref["path"]]=active
reg=cfg["registration_ref"];r.metadata[reg["path"]]=json.loads((ROOT/reg["path"]).read_text())
assert rejects(lambda:p.bind_registration(cfg,r))
train=dict(schema="H050_four_arm_train_completion_v1",hypothesis_id="H050",status="completed_fixed30pass_four_arm_TRAIN",
    arms=list(p.ARMS),steps={a:960 for a in p.ARMS},common_initial_predictor_verified=True,VAL_truth_read=False,
    outer_truth_read=False,scientific_protocol=p.hooks.fixed_protocol_v1(),registration_ref=active["registration_ref"],
    producer_ref=active["producer_ref"],parameters_per_arm=p.model.PARAMETERS,first_step_qualification={a:{} for a in p.ARMS},
    history=[dict(pass_index=i,steps={a:32*i for a in p.ARMS}) for i in range(1,31)],config_ref=config_ref,
    checkpoint_groups={},initial_state_sha256={a:a for a in p.ARMS},input_ref=dict(path="handmade/input.json",sha256="1"*64),
    observations_seal=dict(path="handmade/observations.json",sha256="2"*64),warmstart_checkpoint_ref=p.model.WARM_CHECKPOINT_REF,
    warmstart_completion_ref=p.model.WARM_COMPLETION_REF,cut_preparation_completion_ref=None,cut_preparation_actual_wait_ref=None)
for at in p.PASSES:
    train["checkpoint_groups"][str(at)]={}
    for arm in p.ARMS:
        cp=dict(schema="H050_direct_readout_checkpoint_v1",hypothesis_id="H050",phase="train",arm=arm,pass_index=at,
            additional_optimizer_steps=32*at,seed=101,parameter_count=p.model.PARAMETERS[arm],optimizer_state_reset=True,
            teacher_used=False,initial_state_sha256=arm,observation_seal=train["observations_seal"],model_state={"handmade":torch.zeros(1)})
        for key in ("input_ref","config_ref","registration_ref","producer_ref","scientific_protocol","warmstart_checkpoint_ref", "warmstart_completion_ref","cut_preparation_completion_ref","cut_preparation_actual_wait_ref"):cp[key]=train[key]
        stream=io.BytesIO();torch.save(cp,stream);data=stream.getvalue()
        ref=dict(path="handmade/%s_%d.pt"%(arm,at),sha256=hashlib.sha256(data).hexdigest())
        r.bytes[ref["path"]]=data;train["checkpoint_groups"][str(at)][arm]=ref
states,refs=p.load_checkpoints(r,train,dict(cfg=active));assert len(states)==len(refs)==24
missing=copy.deepcopy(train);del missing["checkpoint_groups"]["30"][p.model.CONTINUATION]
assert rejects(lambda:p.load_checkpoints(r,missing,dict(cfg=active)))
wrong=copy.deepcopy(train);wrong["observations_seal"]=dict(path="handmade/wrong",sha256="3"*64)
assert rejects(lambda:p.load_checkpoints(r,wrong,dict(cfg=active)))
full={a:dict(cells=[dict(parent_id=pid,groups={"primary_present_day_182":dict(accuracy=.96,boundaries=dict(f1=.18))}) for pid in p.VAL]) for a in p.ARMS}
for c in full[p.ARMS[0]]["cells"]:c["groups"]["primary_present_day_182"].update(accuracy=.963,boundaries=dict(f1=.21))
assert p.paired_criteria(full)["meets_primary"] and p.paired_criteria(full)["meets_secondary_boundary"]
for c in full[p.model.CONTINUATION]["cells"]:c["groups"]["primary_present_day_182"]["accuracy"]=.964
assert not p.paired_criteria(full)["meets_primary"] and not p.paired_criteria(full)["meets_secondary_boundary"]
assert not torch.cuda.is_initialized()
receipt=dict(schema="H050_producer_pure_metadata_check_v1",status="PASS",inactive_protocol_pass=True,unactivated_protocol_rejected=True,
    unregistered_draft_rejected=True,all24_handmade_checkpoint_records_validated=True,missing_continuation_checkpoint_rejected=True,
    changed_observation_seal_rejected=True,all3_comparators_in_utility_rules=True,actual_checkpoint_source_authentication="separate_genuine_CPU_receipt",
    handmade_checkpoint_schema_test_only=True,no_actual_H050_model_state_or_training_claim=True,optimizer_steps=0,GPU_initialized=False,
    scientific_observation_or_truth_payload_read=False,guards_or_jobs=False,Git=False)
with (HERE/"producer_metadata_check_receipt_v1.json").open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(receipt,indent=2))
