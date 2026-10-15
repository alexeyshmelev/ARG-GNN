"""Pure H049 prepare layout derived from the original completed count planner.

No guard imports, data payload reads, admission or execution. The root issues
the plan BEFORE reservation, then a separate lease AFTER full admission. This
avoids a reservation/plan/lease hash cycle.
"""
from __future__ import annotations
from bounded_phase_writer_v1 import NPZ_MEMBERS,OVERHEAD,ROUND,charge_cap,require

READONLY_ENV=("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME",
              "TORCH_HOME","CUDA_CACHE_PATH","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR",
              "MPLCONFIGDIR","HF_HOME","NUMBA_CACHE_DIR")

def layout_fields_v1(count_plan,cfg,parents):
    require(count_plan["phase"]=="prepare" and count_plan["parent_count"]==40
        and len(parents)==40 and set(count_plan["parents"])==set(parents),"Only original40 prepare counts")
    files={};rows={};recipe_total=0
    for parent in parents:
        b=count_plan["parents"][parent];compact=b["compact_cut_recipe"]
        npz=compact["compressed_NPZ_upper_bytes"];js=compact["recipe_JSON_planning_bytes"]
        require(type(npz) is int and 0<npz<=cfg["cut_recipe_archive_cap_bytes"] and type(js) is int and js>0,"Original compact per-parent bounds")
        for suffix,kind,cap,schema in [(".cut_recipe.npz","npz",npz,None),(".cut_recipe.json","json",js,"H049_compact_cut_recipe_v1")]:
            rule={"kind":kind,"max_payload_bytes":cap,"group":"recipes"}
            if schema:rule["schema"]=schema
            else:rule["members"]=list(NPZ_MEMBERS)
            files[parent+suffix]=rule;recipe_total+=charge_cap(cap)
        rows[parent]={"native_nodes":b["native_nodes"],"native_holds":b["native_holds"]}
    metadata=cfg["fixed_metadata_cap_bytes"]
    require(type(metadata) is int and metadata>4*OVERHEAD,"Original combined fixed metadata cap")
    for name,schema in [("started.json","H049_order2_started_v1"),("completed.json","H049_order2_observation_preparation_v1"),("failed.json","H049_cut_preparation_failure_v1")]:
        files[name]={"kind":"json","max_payload_bytes":metadata-2*OVERHEAD,"group":"phase_metadata","schema":schema}
    # Fixed preparation has exactly three directories: output, aux and aux/ro.
    directories=3*16384;lease=charge_cap(8192)
    terminal_tail=6*cfg["log_cap_bytes"]+65536+16384
    # Scratch contains the current payload plus a zero-byte owned final placeholder.
    # The payload charge is already included in recipe/group final caps.
    # Reserve one additional zero-byte placeholder counter during atomic rename.
    scratch=OVERHEAD
    peak=recipe_total+metadata+directories+lease+terminal_tail+scratch
    return {"files":files,"parents":rows,"normal_output_sequence":["started.json"]+[name for p in parents for name in [p+".cut_recipe.npz",p+".cut_recipe.json"]]+["completed.json"],
        "group_charge_caps":{"recipes":recipe_total,"phase_metadata":metadata},
        "readonly_aux_directories":["ro"],"readonly_environment_keys":list(READONLY_ENV),
        "per_file_overhead_bytes":OVERHEAD,"allocation_round_bytes":ROUND,
        "per_directory_charge_cap_bytes":16384,"phase_lease_charge_cap_bytes":lease,
        "reserved_terminal_tail_bytes":terminal_tail,"scratch_charge_bound_bytes":scratch,
        "peak_phase_charge_bound_bytes":peak}

def proposed_plan_v1(*,count_plan,cfg,parents,job_id,output,auxiliary,reservation_key,reservation_bytes,nonce,expires_unix,source_sha256,required_reserved_refs):
    fields=layout_fields_v1(count_plan,cfg,parents)
    require(fields["peak_phase_charge_bound_bytes"]<=reservation_bytes<=cfg["output_cap_bytes"],"New full reservation/config must cover scratch, all aux/log and original file bounds")
    require(len(nonce)==32 and all(c in "0123456789abcdef" for c in nonce),"Root-generated fixed nonce")
    return dict(schema="H049_prepare_bounded_phase_plan_v1",status="inactive_source_only_root_review_required",phase="prepare",
        job_id=job_id,output=output,auxiliary=auxiliary,reservation_key=reservation_key,
        reservation_bytes=reservation_bytes,nonce=nonce,expires_unix=expires_unix,
        phase_lease_name="phase_lease.json",source_sha256=source_sha256,
        required_reserved_refs=required_reserved_refs,**fields)

def validate_count_layout_v1(plan,count_plan,cfg,parents):
    fields=layout_fields_v1(count_plan,cfg,parents)
    require(all(plan[k]==v for k,v in fields.items()),"Exact original count-derived file/member/metadata/aux/log/scratch bounds")
    require(plan["peak_phase_charge_bound_bytes"]<=plan["reservation_bytes"]<=cfg["output_cap_bytes"],"Unchanged full root storage cap")
