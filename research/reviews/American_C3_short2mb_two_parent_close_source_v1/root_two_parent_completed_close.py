"""Root-only two-parent adapter to the unchanged frozen four-stage close helper."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_short2mb_two_parent_close_source_v1"
SOURCE=REL+"/root_two_parent_completed_close.py"
PRIMITIVE_REF={"path":"research/reviews/American_C3_short2mb_completed_close_source_v1/root_close_completed_stages.py","sha256":"d85ef0b3811545eff5dc149dd4de144d7873bb7914e5d1c847feba654e8f6259"}
REGISTRY_REF={"path":"research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json","sha256":"c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37"}
WAVE_REF={"path":"research/reviews/American_C3_short_train2mb_wave_supervision_v2/root_wave_driver.py","sha256":"f47820e031a156c7711161bb544074a1dddbb9b3eae13fb2b56fb937e7c805f6"}
WAVE="research/reviews/American_C3_short_train2mb_wave_supervision_v2"
OUTPUT="experiments/results/American_C3_short_train2mb_wave_v2"
CONDITION="C3_independent20x2Mb_TRAIN_1600haps_inferred_unary_v1"
UNARY="direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1"
PARENTS=tuple("american_c3_short_train_%03d"%i for i in range(20))
def primitive():
    path=ROOT/PRIMITIVE_REF["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PRIMITIVE_REF["sha256"]:
        raise ValueError("Frozen original four-stage close helper changed")
    spec=importlib.util.spec_from_file_location("_C3_frozen_four_stage_close",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
def bind_input(m,input_ref):
    batch=m.read(input_ref);parents=batch["parent_ids"]
    m.require(batch.get("schema")=="American_C3_exact_two_registered_parent_close_input_v1"
        and batch.get("scientific_condition_id")==CONDITION
        and batch.get("registry_ref")==REGISTRY_REF and batch.get("wave_driver_ref")==WAVE_REF
        and type(parents) is list and len(parents)==2 and len(set(parents))==2
        and parents==sorted(parents) and set(parents)<=set(PARENTS),
        "Exactly two fixed registered independent TRAIN IDs000-019 required")
    tags=[p[-3:] for p in parents];batch_tag="batch_"+"_".join(tags)
    m.require(input_ref["path"]==REL+"/"+batch_tag+"/two_parent_input.json"
        and type(batch["supervisor_pid"]) is int and batch["supervisor_pid"]>1
        and type(batch["supervisor_session_id"]) is int and batch["supervisor_session_id"]>0,
        "Fixed canonical batch input and named original supervisor/session required")
    registry=m.read(REGISTRY_REF)
    selected=[row for row in registry["parents"] if row["parent_id"] in parents]
    m.require(registry["scientific_condition_id"]==CONDITION
        and registry["new_independent_TRAIN_count"]==20
        and [r["parent_id"] for r in selected]==parents
        and all(r["split"]=="train" and r["sequence_length_bp"]==2000000 for r in selected)
        and batch["registered_parent_rows"]==selected and m.ref(WAVE_REF["path"])==WAVE_REF,
        "Exact registered TRAIN rows and frozen original wave driver required")
    # Only explicit metadata bindings change. Every original function is retained.
    m.SOURCE=SOURCE
    m.HERE=ROOT/REL/batch_tag
    m.CHOICES=tuple(tag+"-"+stage for tag in tags for stage in ("simulate","infer"))
    return batch
def build_evidence(m,input_ref,batch):
    stages={};all_pids=set();all_groups=set()
    for parent in batch["parent_ids"]:
        packet=ROOT/WAVE/"admitted"/parent
        prep_ref=m.ref(packet/"parent_preparation_completed.json");prep=m.read(prep_ref)
        m.require(prep["source_registry_ref"]==REGISTRY_REF
            and prep["root_supervisor_pid"]==batch["supervisor_pid"]
            and prep["scientific_condition_id"]==CONDITION
            and prep["split"]=="train" and prep["sequence_length_bp"]==2000000,
            "Original preparation must match registered parent and exact completed supervisor")
        for stage,suffix,cap in (("simulate","simulation",1000000000),("infer","inference",2000000000)):
            tag=parent[-3:]+"-"+stage;where=packet/stage
            admission_ref=m.ref(where/"admission.json");ad=m.read(admission_ref)
            admitted_ref=m.ref(where/"admitted.json");admitted=m.read(admitted_ref)
            launch_ref=m.ref(where/"launch.json");launch=m.read(launch_ref)
            wait_ref=m.ref(where/"actual_wait.json");wait=m.read(wait_ref)
            completed_ref=wait["completed_ref"];completed=m.read(completed_ref)
            config_ref=m.ref(where/"config.json");own=ad["storage"]["reservation"]
            output=str(ROOT/OUTPUT/parent/suffix);aux=[str(where/"worker_aux")]
            m.require(admitted["admission_ref"]==admission_ref and admitted["driver_ref"]==WAVE_REF
                and admitted["parent_id"]==parent and admitted["stage"]==stage
                and ad["driver_ref"]==wait["driver_ref"]==WAVE_REF
                and ad["config_ref"]==config_ref and ad["scientific_condition_id"]==CONDITION
                and own["key"]=="H041/American-C3-short2mb-"+parent+"-"+stage+"-v2"
                and own["output"]==output and own["auxiliary_scopes"]==aux and own["reserved_bytes"]==cap
                and completed_ref["path"]==OUTPUT+"/"+parent+"/"+suffix+"/completed.json"
                and launch["no_helper_launcher"] is True
                and launch["worker_process_group"]==wait["worker_process_group"]==launch["pid"]
                and launch["supervisor_pid"]==batch["supervisor_pid"],
                "Only exact original four-stage admission/config/launch/wait/completion scopes permitted")
            m.require(launch["owned_process_roster"]==wait["owned_process_roster"],
                "Original direct worker roster must remain identical")
            pids={batch["supervisor_pid"],launch["pid"]};groups={launch["worker_process_group"]}
            for owned in launch["owned_process_roster"]:
                pids.add(owned["pid"])
                if "process_group" in owned:groups.add(owned["process_group"])
            identity=dict(status="completed_C3_simulation_reconstruction",parent_id=parent,
                sample_unit="haploid",class_order=["EUR","EAS","AFR"]) if stage=="simulate" else dict(
                status="completed_current_C3_inference",tsinfer_num_threads=8,unary_preprocessing_policy=UNARY)
            identity["observation_ref"]=completed["observation_ref"]
            m.require(identity["observation_ref"]["path"]==OUTPUT+"/"+parent+"/"+suffix+"/observed/manifest.json"
                and all(completed.get(k)==v for k,v in identity.items()),
                "Exact completed producer identity and observation metadata required")
            if stage=="infer":
                m.require(prep["inferred_model_observation_ref"]==completed["observation_ref"],"Preparation inference identity")
            else:
                m.require(prep["external_TRAIN_truth_ref"]==completed["private_truth_ref"],"Preparation external truth reference")
            stages[tag]=dict(parent_id=parent,stage=stage,key=own["key"],output=output,
                auxiliary_scopes=aux,reserved_bytes=cap,admission_ref=admission_ref,
                admitted_ref=admitted_ref,config_ref=config_ref,launch_ref=launch_ref,
                wait_ref=wait_ref,completed_ref=completed_ref,completed_identity=identity,
                preparation_ref=prep_ref,required_owned_pids=sorted(pids),required_owned_worker_groups=sorted(groups))
            all_pids.update(pids);all_groups.update(groups)
    return dict(schema="American_C3_short_wave_completed_stage_evidence_v1",
        batch_input_ref=input_ref,original_close_primitive_ref=PRIMITIVE_REF,registry_ref=REGISTRY_REF,
        wave_driver_ref=WAVE_REF,parent_ids=batch["parent_ids"],stage_order=list(m.CHOICES),
        supervisor_pid=batch["supervisor_pid"],supervisor_session_id=batch["supervisor_session_id"],
        required_owned_pids=sorted(all_pids),required_owned_worker_groups=sorted(all_groups),
        source_receipts_only=True,scientific_payloads_opened=False,stages=stages)
def approval_draft(m,input_ref,batch,evidence,evidence_ref):
    return dict(schema="American_C3_short_wave_root_close_approval_v1",active=False,root_authorizer="root_agent",
        close_helper_ref=m.ref(SOURCE),original_close_primitive_ref=PRIMITIVE_REF,batch_input_ref=input_ref,
        parent_ids=batch["parent_ids"],stage_order=list(m.CHOICES),stage_evidence_ref=evidence_ref,
        stage_evidence=evidence["stages"],root_registered_exact_two_parent_batch=False,
        supervisor_actual_wait=dict(supervisor_pid=batch["supervisor_pid"],session_id=batch["supervisor_session_id"],
            returncode=0,root_consumed_terminal_result=False),
        all_genuinely_owned_processes_and_groups_listed=False,
        all_prior_input_config_storage_source_launch_resume_write_capabilities_revoked=False,
        preserve_all_files_and_historical_journals=True,
        root_owned_pids=evidence["required_owned_pids"],
        root_owned_process_groups=evidence["required_owned_worker_groups"],
        root_must_expand_roster_if_other_owned_supervisors_or_descendants_exist=True)
def prepare(input_ref):
    m=primitive();batch=bind_input(m,input_ref);evidence=build_evidence(m,input_ref,batch)
    evidence_ref=m.save(m.HERE,"stage_evidence_refs.json",evidence);m.EVIDENCE_REF=evidence_ref
    for which in m.CHOICES:m.evidence_for(which)
    draft_ref=m.save(m.HERE,"root_close_approval_inactive.json",approval_draft(m,input_ref,batch,evidence,evidence_ref))
    print(json.dumps(dict(evidence_ref=evidence_ref,inactive_approval_ref=draft_ref,
        guard_imported=False,process_checks=0,closes_applied=0,scope_inventories=0)),flush=True)
def close_stage(which,input_ref,evidence_ref,approval_ref):
    m=primitive();batch=bind_input(m,input_ref)
    m.require(evidence_ref["path"]==(m.HERE/"stage_evidence_refs.json").relative_to(ROOT).as_posix(),
        "Fixed exact two-parent evidence location required")
    evidence=m.read(evidence_ref)
    m.require(evidence==build_evidence(m,input_ref,batch),"Unchanged exact original four-stage evidence required")
    m.EVIDENCE_REF=evidence_ref
    for stage in m.CHOICES:m.evidence_for(stage)
    cert=m.read(approval_ref)
    m.require(cert.get("original_close_primitive_ref")==PRIMITIVE_REF and cert.get("batch_input_ref")==input_ref
        and cert.get("parent_ids")==batch["parent_ids"] and cert.get("stage_order")==list(m.CHOICES)
        and cert.get("root_registered_exact_two_parent_batch") is True
        and cert.get("supervisor_actual_wait")==dict(supervisor_pid=batch["supervisor_pid"],
            session_id=batch["supervisor_session_id"],returncode=0,root_consumed_terminal_result=True)
        and type(cert["supervisor_actual_wait"]["returncode"]) is int,
        "Root actual supervisor wait0 and fixed registered two-parent certificate required")
    m.require(which in m.CHOICES,"Only this exact four-stage bundle permitted")
    m.approval(approval_ref,evidence)
    for earlier in m.CHOICES[:m.CHOICES.index(which)]:
        packet=m.HERE/("closed_"+earlier)
        observation=m.read(m.ref(packet/"root_observation.json"))
        m.read(m.ref(packet/"close_receipt.json"))
        m.require(observation["key"]==evidence["stages"][earlier]["key"]
            and observation["root_approval_ref"]==approval_ref and observation["source_ref"]==m.ref(SOURCE),
            "Earlier original stage close receipt required; close sequentially")
    # Original close owns all targeted checks, mutex, sole inventory and mutation.
    return m.close(which,approval_ref)
def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-evidence",action="store_true");mode.add_argument("--close")
    p.add_argument("--batch-input-ref");p.add_argument("--batch-input-sha256")
    p.add_argument("--evidence-ref");p.add_argument("--evidence-sha256")
    p.add_argument("--root-approval-ref");p.add_argument("--root-approval-sha256");a=p.parse_args()
    if a.prepare_evidence or a.close:
        if not a.batch_input_ref or not a.batch_input_sha256:raise ValueError("Fixed root batch input path/SHA required")
        input_ref=dict(path=a.batch_input_ref,sha256=a.batch_input_sha256)
        if a.prepare_evidence:prepare(input_ref);return 0
        if not all((a.evidence_ref,a.evidence_sha256,a.root_approval_ref,a.root_approval_sha256)):
            raise ValueError("Exact evidence and active root approval path/SHA required")
        close_stage(a.close,input_ref,dict(path=a.evidence_ref,sha256=a.evidence_sha256),
            dict(path=a.root_approval_ref,sha256=a.root_approval_sha256));return 0
    print(json.dumps(dict(status="source_only_inactive_exact_two_parent_adapter",registered_parent_range="000-019",
        stages_per_bundle=4,root_active_approval_required=True,process_checks=0,scope_inventories=0,closes_applied=0)))
    return 0
if __name__=="__main__":raise SystemExit(main())
