from pathlib import Path
import copy, datetime, hashlib, importlib.util, json, stat, xml.etree.ElementTree as ET
root=Path("/workspace-SR003.nfs2/ARG-GNN")
own=root/"research/reviews/biological_completed_candidates_storage_settlement_v1_independent/storage_revision_v2"
author=root/"research/reviews/biological_completed_candidates_storage_settlement_v1_author/storage_revision_v2"
v1=author.parent
read_paths=[]
def read(p):
    p=Path(p)
    assert any(base in p.parents for base in [root/"research/reviews",root/"src",root/"configs"]),str(p)
    assert p.resolve()==p and p.is_file(),str(p)
    data=p.read_bytes();read_paths.append(str(p.relative_to(root)));return data
def digest(p):return hashlib.sha256(read(p)).hexdigest()
def ref(p):return {"path":str(p.relative_to(root)),"sha256":digest(p)}
def load(p):
    def pairs(rows):
        d={}
        for k,v in rows:
            assert k not in d,"duplicate JSON";d[k]=v
        return d
    return json.loads(read(p),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite JSON")))
def bound(r):
    p=root/r["path"];assert digest(p)==r["sha256"],str(p);return load(p)
pins={"audit_and_settle.py":"0892242146949b46005337cd583f659867c6e108b415d864152c6f76bd8cef40","config.json":"bee8263c0fad30d385bb5bf17324ebaac71ea0a4b1367365b6c4cd0f1f3a994d","review_receipt.json":"78097e0f81e7a6d483e761a27d3679d2d468b89a7db219b1290730807e9dd9b8"}
for p,sha in pins.items():assert digest(author/p)==sha
c=load(author/"config.json")
inspection_path=author/"root_inspection/inspection.json";inspection_ref=ref(inspection_path)
assert inspection_ref["sha256"]=="7759198b8b75d6a60e4fdd7475a7ea14b15001c9fb50b0a21148dfdcc8b204e1"
inspection=bound(inspection_ref)
assert inspection["kind"]=="biological_completed_candidates_storage_settlement_v1_inspect_receipt" and inspection["applied"] is False and inspection["root_authorizer"]=="root_agent"
assert inspection["procedure"]==ref(author/"audit_and_settle.py") and inspection["config"]==ref(author/"config.json")
assert inspection["prior_inventory"]==c["prior_inventory"] and inspection["before_ledger_sha256"]==c["before_ledger_sha256"]
keys=[t["key"] for t in c["targets"]]
assert inspection["scope_keys"]==keys and set(inspection["inventories"])==set(keys) and set(inspection["terminal"])==set(keys)
assert inspection["operator_frozen_output_policy"]==c["operator_frozen_output_policy"] and c["unknown_writer_absence_claimed"] is False
for k in ["known_saved_identity_absence_does_not_prove_unknown_writer_absence","two_full_byte_stat_inventory_passes_under_scheduler_mutex","no_product_files_deleted","all_other_ledger_fields_charges_baseline_external_preserved","historical_projection_not_global_remeasurement"]:assert inspection[k] is True
for k,value in [("old_bytes_each",29281048576),("retained_bytes_each",15000000000),("reduction_bytes",28562097152)]:assert inspection[k]==value
prior_receipt=bound(c["prior_inventory"])
metadata=[bound(r) for r in c["prior_metadata"]]
assert metadata[0]["ledger"]==metadata[1]["ledger"] and all(m["ledger_sha256"]==c["before_ledger_sha256"] for m in metadata)
for pin in [c[k] for k in ["prior_GST_procedure","prior_GST_independent_review","pool_source","candidate_source","candidate_config"]]:assert digest(root/pin["path"])==pin["sha256"]
for receipt in [load(author/"review_receipt.json"),load(v1/"review_receipt.json")]:
    for path,sha in receipt["artifact_sha256"].items():assert digest(root/path)==sha,path
assert digest(root/"research/reviews/biological_completed_candidates_storage_settlement_v1_independent/review_receipt_v1.json")=="2cdbbaad7b606070efe4060072067c88220917e459e38832bb6d7efef1e2eead"
spec=importlib.util.spec_from_file_location("independent_v2_saved_only",author/"audit_and_settle.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
assert h.exact(h.policy(),c)
assert len(inspection["inventory_refs"])==2
summaries={};allrefs=[]
fields={"scope","relative_path","entry_type","device","inode","mode","size_bytes","allocated_bytes","mtime_ns","ctime_ns","link_count","uid","gid","retained_charge_bytes"}
for t in c["targets"]:
    k=t["key"];passes=[]
    prior=[bound(r) for r in t["prior_stat_passes"]]
    assert prior[0]["entries"]==prior[1]["entries"]
    for index,row in enumerate(inspection["inventory_refs"],1):
        assert set(row)==set(keys);r=row[k]
        expected=author/"root_inspection"/("seed"+str(t["seed"])+"_full_inventory_pass"+str(index)+".json")
        assert r["path"]==str(expected.relative_to(root))
        inv=bound(r);allrefs.append(r);passes.append(inv)
    assert h.exact(passes[0],passes[1]) and h.exact(passes[1],inspection["inventories"][k])
    inv=passes[1]
    assert inv["key"]==k and inv["seed"]==t["seed"] and inv["output"]==str(root/t["output"]) and inv["retained_pool_log"]==str(root/t["retained_pool_log"]) and inv["opaque_byte_hashes_only"] is True
    entries=[{a:b for a,b in e.items() if a!="sha256"} for e in inv["entries"]]
    assert sorted(entries,key=lambda e:(e["scope"],e["relative_path"]))==sorted(prior[0]["entries"],key=lambda e:(e["scope"],e["relative_path"]))
    seen=set();filecount=0;dircount=0;logcount=0;charge=0
    for e in inv["entries"]:
        p=Path(e["relative_path"]);assert not p.is_absolute() and ".." not in p.parts
        assert (e["scope"],e["relative_path"]) not in seen;seen.add((e["scope"],e["relative_path"]))
        for a in fields-{"scope","relative_path","entry_type"}:assert type(e[a]) is int and e[a]>=0,(k,a)
        assert e["retained_charge_bytes"]==max(e["size_bytes"],e["allocated_bytes"]);charge+=e["retained_charge_bytes"]
        if e["entry_type"]=="regular_file":
            assert set(e)==fields|{"sha256"} and stat.S_ISREG(e["mode"]) and e["link_count"]==1
            assert len(e["sha256"])==64 and all(ch in "0123456789abcdef" for ch in e["sha256"]);filecount+=1
        else:
            assert e["entry_type"]=="directory" and set(e)==fields and stat.S_ISDIR(e["mode"]) and e["scope"]=="output";dircount+=1
        if e["scope"]=="retained_pool_log":
            assert e["entry_type"]=="regular_file" and e["relative_path"]==Path(t["retained_pool_log"]).name;logcount+=1
        else:assert e["scope"]=="output"
    assert charge==inv["conservative_retained_charge_bytes"] and charge<15000000000 and logcount==1
    terminal=inspection["terminal"][k]
    sealed=[r["terminal_job"] for r in metadata[1]["candidates"] if r["key"]==k]
    assert len(sealed)==1
    projected,extras=h.terminal_projection(c,t,terminal["terminal_job"],sealed[0])
    assert projected==terminal["sealed_terminal_projection"]==sealed[0] and extras==terminal["terminal_metadata_extras"] and terminal["exact_prior_projection_fields"]==sorted(projected)
    job=terminal["terminal_job"]
    assert job["state"]=="succeeded" and job["exit_code"]==0 and job["requested"]["seed"]==t["seed"] and job["requested"]["module"]=="arggnn.biological_candidate" and job["requested"]["storage_bytes"]==29281048576
    checks=terminal["saved_identity_checks"]
    assert len(checks)==len(job["process_identities"]) and all(e["saved_identity_present"] is False for e in checks)
    assert [(e["pid"],e["saved_start_ticks"]) for e in checks]==[(e["pid"],e["start_ticks"]) for e in job["process_identities"]]
    assert terminal["receipt"]["path"]==str(Path(t["output"])/"receipt.json")
    receiptentry=[e for e in inv["entries"] if e["scope"]=="output" and e["relative_path"]=="receipt.json"]
    assert len(receiptentry)==1 and receiptentry[0]["sha256"]==terminal["receipt"]["sha256"]
    assert terminal["registry_path"]==t["registry"] and len(terminal["registry_sha256"])==64
    summaries[k]={"total_entries":len(inv["entries"]),"regular_files_including_retained_log":filecount,"directories":dircount,"conservative_retained_charge_bytes":charge,"retained_charge_bytes":15000000000,"retained_margin_bytes":15000000000-charge,"sealed_field_count":len(projected),"full_job_field_count":len(job),"known_saved_identity_checks":len(checks),"all_saved_identity_checks_absent":True,"prior_complete_stat_membership_exact":True,"both_full_hash_stat_inventories_exact":True,"receipt_hash_matches_inventoried_file":True,"validated_recorded_historical_extras":extras}
state=metadata[1]["ledger"];proposed=copy.deepcopy(state)
for t in c["targets"]:
    assert state["jobs"][t["key"]]=={"output":str(root/t["output"]),"reserved_bytes":29281048576}
    proposed["jobs"][t["key"]]["reserved_bytes"]=15000000000
inverse=copy.deepcopy(proposed)
for t in c["targets"]:inverse["jobs"][t["key"]]["reserved_bytes"]=29281048576
assert h.exact(inverse,state)
def account(s):
    reserve=s["external_reserve_bytes"]+sum(x["reserved_bytes"] for coll in ["jobs","controllers"] for x in s[coll].values())
    return {"snapshot_sha256":s["snapshot_sha256"],"baseline_bytes":s["baseline_bytes"],"reserve_bytes":reserve,"projected_bytes":s["baseline_bytes"]+reserve,"limit_bytes":s["limit_bytes"],"persistent_reservations":len(s["jobs"])}
assert inspection["before_account"]==account(state) and inspection["after_account"]==account(proposed)
suite=ET.parse(own/"pytest.xml").getroot().find("testsuite")
tests={k:int(suite.attrib[k]) for k in ["tests","failures","errors","skipped"]}
assert tests=={"tests":125,"failures":0,"errors":0,"skipped":0} and load(own/"test_execution.json")["blocked_events"]==[]
evidence={"kind":"candidate_storage_settlement_v2_independent_saved_inspection_evidence","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"inspection":inspection_ref,"procedure":ref(author/"audit_and_settle.py"),"config":ref(author/"config.json"),"four_inventory_refs":allrefs,"scope_summaries":summaries,"before_account":account(state),"after_account":account(proposed),"exact_two_reserved_bytes_delta_inverse_preserves_every_other_ledger_field":True,"all_author_v1_artifact_hashes_preserved":True,"prior_independent_v1_BLOCKED_preserved":True,"tests":tests,"source_enforced_candidate_receipt_scientific_provenance_and_three_positive_fit_gates":True,"actual_product_receipts_arrays_checkpoint_or_tree_decode_or_scan_by_reviewer":False,"live_ledger_registry_PID_GPU_access_by_reviewer":False,"actual_apply_performed":False,"read_metadata_and_source_paths":sorted(set(read_paths))}
with (own/"inspection_evidence.json").open("x") as stream:json.dump(evidence,stream,indent=2);stream.write("\n")
print(json.dumps({"inspection":inspection_ref,"scope_charges_and_margins":{k:{a:b for a,b in v.items() if a in ["conservative_retained_charge_bytes","retained_margin_bytes","total_entries"]} for k,v in summaries.items()},"evidence":ref(own/"inspection_evidence.json"),"tests":tests},indent=2))
