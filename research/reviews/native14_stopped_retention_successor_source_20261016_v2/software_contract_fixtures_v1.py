"""Tiny synthetic filesystem contracts only. No original scope or accounting calls."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,socket,stat,types,uuid
from datetime import datetime,timezone,timedelta
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("native14_fixture_measurement",BASE/"retained_nodes_v1.py")
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
RUN=BASE/"software_fixtures"/("run_"+uuid.uuid4().hex)
RUN.mkdir(parents=True)
REPORT=[]
def passed(name):REPORT.append(dict(case=name,passed=True))
def rejected(name,fn):
    try:fn()
    except (ValueError,OSError):passed(name);return
    raise AssertionError("Expected contract rejection: "+name)
def roots(name):
    root=RUN/name;paths=[root/x for x in ("output","auxiliary","temporary")]
    for p in paths:p.mkdir(parents=True)
    return root,paths
def measure(root,paths,callback=lambda p:True):
    return H.measure_retained_nodes_v1(root,paths,expected_paths=[str(p) for p in paths],socket_is_absent=callback)
def integer(value,label,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(label)
    return value
def main():
    root,paths=roots("accepted")
    (paths[0]/"checkpoint.bin").write_bytes(b"synthetic payload")
    (paths[0]/"latest.pt").symlink_to("checkpoint.bin")
    (paths[1]/"external_alias").symlink_to("/external/target/must/not/be/followed")
    sock=socket.socket(socket.AF_UNIX);socket_path=paths[2]/"inactive.sock"
    previous=Path.cwd()
    try:os.chdir(paths[2]);sock.bind("inactive.sock")
    finally:os.chdir(previous);sock.close()
    original_open=H.os.open;opened=[];checked=[]
    def directories_only(path,flags,*a,**k):
        assert flags&os.O_DIRECTORY and flags&os.O_NOFOLLOW,"No payload file may be opened"
        opened.append(str(path));return original_open(path,flags,*a,**k)
    H.os.open=directories_only
    try:result=measure(root,paths,lambda p:(checked.append(p) or True))
    finally:H.os.open=original_open
    assert result["entries"]==7 and result["per_scope"][0]["node_types"]["symlink"]==1
    assert result["per_scope"][1]["node_types"]["symlink"]==1 and result["per_scope"][2]["node_types"]["socket"]==1
    assert result["conservative_scope_bytes"]>=7*4096 and len(checked)==3
    assert stat.S_ISLNK((paths[0]/"latest.pt").lstat().st_mode) and socket_path.exists()
    passed("all_three_roots_regular_link_socket_metadata_counted")
    passed("only_readonly_nofollow_directory_descriptors_opened")
    passed("external_alias_target_never_opened_or_resolved")
    passed("stored_latest_alias_and_inactive_socket_preserved")
    rejected("kernel_listener_callback_denies_retained_socket",lambda:measure(root,paths,lambda p:False))
    rejected("missing_parent_is_not_omitted",lambda:H.measure_retained_nodes_v1(root,paths[:2],expected_paths=list(map(str,paths)),socket_is_absent=lambda p:True))
    rejected("reordered_or_substituted_parents_rejected",lambda:H.measure_retained_nodes_v1(root,paths[::-1],expected_paths=list(map(str,paths)),socket_is_absent=lambda p:True))
    rejected("overlapping_scope_roots_rejected",lambda:H.measure_retained_nodes_v1(root,[paths[0],paths[0]/"child",paths[2]],
        expected_paths=list(map(str,[paths[0],paths[0]/"child",paths[2]])),socket_is_absent=lambda p:True))
    r,ps=roots("hardlinks");(ps[0]/"first").write_bytes(b"x");os.link(ps[0]/"first",ps[0]/"second")
    rejected("multiply_linked_regular_file_rejected",lambda:measure(r,ps))
    r,ps=roots("fifo");os.mkfifo(ps[0]/"pipe")
    rejected("FIFO_unsafe_node_rejected",lambda:measure(r,ps))
    r,ps=roots("linked_ancestor");(r/"alias").symlink_to(ps[0],target_is_directory=True)
    bad=[r/"alias",ps[1],ps[2]]
    rejected("symlinked_scope_root_rejected_without_following",lambda:measure(r,bad))
    r,ps=roots("between_passes");(ps[0]/"value").write_bytes(b"a")
    original_scandir=H.os.scandir;calls=[0]
    def change_file(fd):
        calls[0]+=1
        if calls[0]==4:(ps[0]/"value").write_bytes(b"changed")
        return original_scandir(fd)
    H.os.scandir=change_file
    try:rejected("file_metadata_change_between_complete_passes_rejected",lambda:measure(r,ps))
    finally:H.os.scandir=original_scandir
    r,ps=roots("membership");calls=[0]
    def change_membership(fd):
        calls[0]+=1
        if calls[0]==4:(ps[0]/"new_file").write_bytes(b"x")
        return original_scandir(fd)
    H.os.scandir=change_membership
    try:rejected("directory_membership_race_rejected",lambda:measure(r,ps))
    finally:H.os.scandir=original_scandir
    r,ps=roots("final_revalidation");(ps[0]/"value").write_bytes(b"x")
    original_leaf=H._leaf_metadata;calls=[0]
    def final_change(*a,**k):
        calls[0]+=1
        if calls[0]==1:(ps[0]/"value").write_bytes(b"changed_at_final_boundary")
        return original_leaf(*a,**k)
    H._leaf_metadata=final_change
    try:rejected("final_descriptor_relative_change_rejected",lambda:measure(r,ps))
    finally:H._leaf_metadata=original_leaf
    r,ps=roots("link_text");(ps[0]/"alias").symlink_to("abcd")
    original_readlink=H.os.readlink;calls=[0]
    def changing_link(*a,**k):
        calls[0]+=1
        return "abcd" if calls[0]==1 else "efgh"
    H.os.readlink=changing_link
    try:rejected("stored_link_text_digest_change_rejected",lambda:measure(r,ps))
    finally:H.os.readlink=original_readlink
    oldlimit=H.MAX_ENTRIES;H.MAX_ENTRIES=3
    try:rejected("finite_metadata_inventory_limit_enforced",lambda:measure(root,paths))
    finally:H.MAX_ENTRIES=oldlimit
    r,ps=roots("cross_device");(ps[0]/"value").write_bytes(b"x")
    original_stat=H.os.stat
    def foreign_stat(path,*a,**k):
        s=original_stat(path,*a,**k)
        if path=="value":
            names=("st_dev","st_ino","st_mode","st_uid","st_gid","st_nlink","st_size","st_blocks","st_mtime_ns","st_ctime_ns")
            values={n:getattr(s,n) for n in names};values["st_dev"]+=1
            return types.SimpleNamespace(**values)
        return s
    H.os.stat=foreign_stat
    try:rejected("cross_device_node_rejected",lambda:measure(r,ps))
    finally:H.os.stat=original_stat

    r,ps=roots("ambiguous_namespace");(ps[0]/"line\nbreak").write_bytes(b"x")
    rejected("control_character_namespace_rejected_before_kernel_path_matching",lambda:measure(r,ps))
    # Generic helper callback tests above preserve socket metadata. Production
    # v2 has no qualified bind-identity map and must reject socket certificates.
    safe_root,safe_paths=roots("production_accept_no_socket")
    (safe_paths[0]/"synthetic.bin").write_bytes(b"ok")
    result=measure(safe_root,safe_paths)
    source=(BASE/"native14_stopped_retention_v1.py").read_text();tree=ast.parse(source)
    names={"require","_stamp","consumer_bindings_v1","_native_validate_targets","_native_measurement","_native_recorded_proof"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env=dict(copy=copy,datetime=datetime,timezone=timezone,NATIVE_RESERVED=15032385536,NATIVE_ROOTS=result["scope_paths"],
        budget=types.SimpleNamespace(v4=types.SimpleNamespace(encoded=lambda x:json.dumps(x,sort_keys=True,separators=(",",":")).encode(),integer=integer)),
        _raw=lambda r:None,_native_manifest=lambda:{},_paths=lambda own:[Path(x) for x in result["scope_paths"]])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"native14_pure_contracts:synthetic","exec"),env)
    assert env["_native_measurement"](result,dict(reserved_bytes=15032385536))==result["conservative_scope_bytes"]
    passed("actual_retention_certificate_consumer_accepts_zero_socket_helper_schema")
    socket_certificate=copy.deepcopy(result)
    socket_certificate["per_scope"][0]["node_types"]["regular"]-=1
    socket_certificate["per_scope"][0]["node_types"]["socket"]+=1
    rejected("production_v2_rejects_unqualified_retained_socket_certificate",
        lambda:env["_native_measurement"](socket_certificate,dict(reserved_bytes=15032385536)))
    for name,change in [
        ("over_original_cap",lambda z:z.update(conservative_scope_bytes=15032385537)),
        ("omitted_parent",lambda z:z.update(scope_paths=z["scope_paths"][:2])),
        ("normalization_claim",lambda z:z.update(namespace_normalization_performed=True)),
        ("followed_target_claim",lambda z:z.update(symlink_targets_followed=True)),
        ("missing_parent_summary",lambda z:z.update(per_scope=z["per_scope"][:2]))]:
        bad=copy.deepcopy(result);change(bad)
        rejected("retention_certificate_rejects_"+name,lambda:env["_native_measurement"](bad,dict(reserved_bytes=15032385536)))
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    m=dict(known_PID_start_rows=[dict(pid=10001,start_ticks="UNKNOWN"),dict(pid=10002,start_ticks=1234)],known_groups=[10001],
        recorded_filesystem_endpoints=[dict(path="synthetic/stale.sock")])
    facts=dict(PID_observations=[[10001,"UNKNOWN",True,stamp],[10002,1234,True,stamp]],group_observations=[[10001,True,stamp]],
        endpoint_observations=[["synthetic/stale.sock","absent_path",stamp]],
        abstract_namespace_observation=["@arggnn-c3mlm-v12-","no_exact_prefix_kernel_endpoint",stamp])
    assert env["_native_validate_targets"](facts,m,now-timedelta(seconds=1),now,fresh=True)
    passed("targeted_contract_preserves_unknown_starts_and_absent_endpoint_facts")
    bad=copy.deepcopy(facts);bad["endpoint_observations"][0][1]="genuine_stale_socket_metadata_preserved"
    rejected("production_v2_rejects_unqualified_present_historical_socket_observation",
        lambda:env["_native_validate_targets"](bad,m,now-timedelta(seconds=1),now,fresh=True))
    for name,change in [
        ("missing_known_PID",lambda z:z["PID_observations"].pop()),
        ("invented_historical_start",lambda z:z["PID_observations"][0].__setitem__(1,9999)),
        ("numeric_reuse_not_absent",lambda z:z["group_observations"][0].__setitem__(1,False)),
        ("boolean_coercion",lambda z:z["PID_observations"][0].__setitem__(2,1)),
        ("missing_abstract_prefix",lambda z:z.__setitem__("abstract_namespace_observation",[]))]:
        bad=copy.deepcopy(facts);change(bad)
        rejected("targeted_contract_rejects_"+name,lambda:env["_native_validate_targets"](bad,m,now-timedelta(seconds=1),now,fresh=True))
    rev_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_native_revocation")
    certificate=dict(schema="SOFTWARE_REVOCATION_ONLY",active=True,revoked_at_utc=(now-timedelta(seconds=1)).isoformat())
    rev_env=dict(env,AUDIT_REF={},_json=lambda r:certificate,
        stopped_revocation_draft_v1=lambda:dict(schema="SOFTWARE_REVOCATION_ONLY",active=False,revoked_at_utc=None),
        _completed_json=lambda r:dict(observed_utc=(now-timedelta(seconds=2)).isoformat()))
    exec(compile(ast.Module(body=[rev_node],type_ignores=[]),"actual_revocation_time_contract:synthetic","exec"),rev_env)
    rev_env["_native_revocation"]({})
    passed("permanent_ROOT_revocation_occurs_after_sealed_stop_audit")
    certificate["revoked_at_utc"]=(now-timedelta(seconds=3)).isoformat()
    rejected("backdated_pre_audit_revocation_rejected",lambda:rev_env["_native_revocation"]({}))
    env["_native_manifest"]=lambda:m
    env["_native_revocation"]=lambda r:dict(revoked_at_utc=(now-timedelta(seconds=1)).isoformat())
    proof=dict(schema="SOFTWARE_PROOF_ONLY",permanent_revocation_ref=dict(path="SOFTWARE_FIXTURE/revocation.json",sha256="0"*64),
        pre_measurement_targets=copy.deepcopy(facts),post_measurement_targets=copy.deepcopy(facts))
    env["_native_recorded_proof"](proof,copy.deepcopy(proof),stamp)
    passed("recorded_source_proof_preserves_target_identity_and_chronology")
    broken=copy.deepcopy(proof);broken["post_measurement_targets"]["endpoint_observations"][0][1]="genuine_stale_socket_metadata_preserved"
    rejected("endpoint_presence_churn_across_measurement_rejected",lambda:env["_native_recorded_proof"](broken,proof,stamp))
    broken=copy.deepcopy(proof);broken["post_measurement_targets"]["PID_observations"][0][-1]=(now-timedelta(seconds=2)).isoformat()
    rejected("post_measurement_fact_cannot_predate_certificate",lambda:env["_native_recorded_proof"](broken,proof,stamp))
    roleplan=json.loads((BASE/"direct_role_plan.v1.json").read_text())
    realmanifest=json.loads((BASE/"source_manifest.v1.json").read_text())
    binding=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_bindings")
    paths={r["path"]:r for r in roleplan["authored_direct_refs"]}
    fake=lambda name:dict(path="SOFTWARE_FIXTURE/"+name+".json",sha256="1"*64)
    statepolicy=dict(root_activation_ref=fake("activation"),permanent_revocation_ref=fake("revocation"),source_review_ref=fake("review"))
    env.update(_native_manifest=lambda:realmanifest,_ref=lambda path:paths[path],SOURCE=str(BASE.relative_to(BASE.parents[2])/"native14_stopped_retention_v1.py"),
        MANIFEST=str(BASE.relative_to(BASE.parents[2])/"source_manifest.v1.json"),MEASURE_SOURCE=str(BASE.relative_to(BASE.parents[2])/"retained_nodes_v1.py"),
        _POLICY_REF=fake("policy"))
    exec(compile(ast.Module(body=[binding],type_ignores=[]),"actual_role_assembly:software_fixture","exec"),env)
    parentbindings={"ancestor_%02d"%i:r for i,r in enumerate(roleplan["current_Census23_refs"])}
    bound=env["_bindings"](parentbindings,statepolicy)
    unique=env["consumer_bindings_v1"](list(bound.values()))
    assert len(unique)==62 and all(r in unique for r in roleplan["historical45_refs"])
    passed("actual_binding_source_retains_all_historical45_and_current23")
    assert len(env["consumer_bindings_v1"](unique,[fake("observation"),fake("operation")]))==64
    passed("complete_retention_invocation_actual_assembly_fits64")
    rejected("retention_invocation_with_additional_unbound_role_rejected",
        lambda:env["consumer_bindings_v1"](unique,[fake("observation"),fake("operation"),fake("extra")]))
    refs=[dict(path="SOFTWARE_FIXTURE/source_%02d.json"%i,sha256=("%064x"%i)) for i in range(65)]
    assert len(env["consumer_bindings_v1"](refs[:64],refs[:4]))==64
    passed("unchanged_direct_binding_dedup_accepts_exact64")
    rejected("unchanged_direct_binding_rejects65",lambda:env["consumer_bindings_v1"](refs))
    original=(BASE.parents[0]/"prospective980_Census_fit2_retention_source_20261016_v1"/"prospective_980gb_Census_fit2_v1.py").read_text()
    def bodies(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    first,second=bodies(original),bodies(source)
    for name in ("apply_reserve_v1","apply_close_completed_v1","check_storage_locked_v1","budget_check_v9","validate_storage_v9","consumer_bindings_v1","_proof","_measurement"):
        assert first[name]==second[name]
    passed("ordinary_reserve_close_check_and64role_AST_bodies_unchanged")
    assert "torch" not in __import__("sys").modules
    regular_bytes=sum(p.lstat().st_size for p in RUN.rglob("*") if stat.S_ISREG(p.lstat().st_mode))
    assert regular_bytes<1000000
    report=dict(schema="native14_tiny_synthetic_filesystem_software_contract_results_v1",cases=len(REPORT),results=REPORT,
        regular_fixture_payload_bytes=regular_bytes,fixture_directory=str(RUN),original_scope_measurements=0,real_process_or_device_probes=0,
        accounting_calls_mutations_admissions_or_jobs=0,original_payload_model_or_truth_reads=0,
        aliases_sockets_and_failed_fixture_files_preserved=True,scientific_or_capacity_evidence=False)
    (BASE/"software_contract_results.v1.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(dict(status="TINY_SYNTHETIC_SOFTWARE_CONTRACTS_PASS",cases=len(REPORT),regular_fixture_bytes=regular_bytes)))
if __name__=="__main__":main()
