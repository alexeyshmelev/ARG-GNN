"""Fresh per-parent native admission. Inert unless ROOT explicitly invokes."""
from pathlib import Path
import sys
from . import native_common_v1 as c
from . import common_v1 as h

def admit_v1(request_ref):
    c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact -B -s ROOT admission")
    q,cfg,_,protocol,_=c.validate_request_v1(request_ref)
    packet=c.local(q["packet_directory"]);output=c.local(q["output_directory"])
    stage=c.local(q["root_preparation_directory"])
    c.require(stage==c.local(request_ref["path"]).parent and stage.is_dir()
        and q["root_preparation_directory"].startswith("research/reviews/native_postdrain_ROOT_preparation_")
        and not packet.exists() and not output.exists() and not stage.is_relative_to(packet)
        and not stage.is_relative_to(output),"Existing charged ROOT staging and absent complete new scopes")
    g=c.guard_v1(q,consumer=False)
    from .root_driver_v1 import mutation_memo_v1
    memo=mutation_memo_v1(g,q)
    closure=c.save(stage,"native_consumer_closure.json",dict(schema="native_whole_parent_consumed_closure_v1",
        root_request_ref=request_ref,adapter_manifest_ref=c.ref(c.MANIFEST),parent_id=q["parent_id"],
        config_ref=q["config_ref"],runtime_ref=q["runtime_ref"],preparation_ref=q["preparation_ref"],counts_ref=q["counts_ref"],
        scientific_protocol=protocol,author_source_manifest_ref=c.AUTHOR_MANIFEST_REF,
        source_sha256={r["path"]:r["sha256"]for r in c.source_refs_v1()}))
    implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
    state=memo.read_state_v1(expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    needed=[c.ref(c.MANIFEST),c.ref(c.DRIVER),c.ref(c.DIRECT),request_ref,q["config_ref"],q["runtime_ref"],closure,
        q["source_review_ref"],q["direct_attachment_authority_ref"],c.SELECTOR_REF]
    consumers=c.unique([*state[3].values(),*needed])
    c.require(len(consumers)<=64,"Actual unchanged direct-role union64 checked before reserve")
    reserve=dict(key="native-postdrain/"+q["job_id"],output=str(output),auxiliary_scopes=[str(packet)],
        reserved_bytes=c.CAPS["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    auth=c.save(stage,"native_reserve_authority.json",authority)
    receipt=memo.reserve_v1(auth,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    c.require(not packet.exists() and not output.exists(),"Complete fresh namespaces absent through actual reserve")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
        job_id=q["job_id"],output_directory=q["output_directory"],reserved_new_bytes=c.CAPS["reservation_bytes"],
        reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],created_at_utc=q["created_at_utc"],
        expires_at_utc=q["expires_at_utc"],completed_overlay_policy_ref=c.POLICY,storage_policy_selector_ref=c.SELECTOR_REF,
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD))
    g.budget_check_v9(storage,boundary=True,required_write_bytes=2*2**20)
    packet.mkdir()
    storage_ref=c.save(packet,"storage.json",storage)
    s=dict(schema="native_original_parent_postdrain_input_v1",status="active",root_authorizer="root_agent",execution_authorized=True,
        phase="mlm",job_id=q["job_id"],parent_id=q["parent_id"],output_directory=q["output_directory"],
        worker_auxiliary_scope=q["packet_directory"]+"/worker_aux",root_stage_request_ref=request_ref,
        main_source_ref=c.ref(c.DRIVER),consumer_manifest_ref=closure,
        source_sha256={r["path"]:r["sha256"]for r in c.source_refs_v1()},refs=dict(config=q["config_ref"],runtime=q["runtime_ref"],storage=storage_ref),
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],physical_gpu=0,device="cuda",
        max_cpu_rss_bytes=c.CAPS["max_cpu_rss_bytes"],max_cuda_reserved_bytes=c.CAPS["max_cuda_reserved_bytes"])
    input_ref=c.save(packet,"input.json",s);c.validate_input_v1(input_ref)
    admitted=c.save(packet,"admitted.json",dict(schema="native_whole_parent_postdrain_admitted_v1",input_ref=input_ref,
        reserve_receipt=receipt,consumer_binding_count=len(consumers),science_or_GPU_started=False,
        whole_scope_reservation_bytes=c.CAPS["reservation_bytes"],automatic_execution_or_close_or_credit=False))
    return dict(input_ref=input_ref,admitted_ref=admitted,execute_only_via_authenticated_pool_submission=True)
