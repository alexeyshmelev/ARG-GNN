"""Fixed VAL002 recovery successor: original path validator and complete source-default path checks."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/'research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v2'
SOURCE='research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v2/recover_val002_infer_v2.py'
PINS_REF={'path': 'research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v2/pins.json', 'sha256': 'd21d3f4822eab4f781c2da21f10f6d9e6c6986dff05306ffeb8d6a35a20637a3'}
def need(ok,message):
    if not ok:raise ValueError(message)
def load(record,name):
    p=ROOT/record["path"]
    need(p.resolve()==p and p.is_relative_to(ROOT) and p.suffix==".py" and p.stat().st_size<=262144,"Exact bounded pinned source")
    need(hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],"Frozen source hash")
    ms=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m);return m
raw=(ROOT/PINS_REF["path"]).read_bytes()
need(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==PINS_REF["sha256"],"Exact successor pins")
pins=json.loads(raw)
packet=load(pins["original_driver"],"_VAL002_original_packet_ref_v2")
prior=load(pins["prior_recovery_source"],"_VAL002_preserved_recovery_v1")
def original_bounded_ref(value):
    p=packet.local(value)
    need(p.suffix in (".py",".json",".sh") and p.is_file() and p.stat().st_size<=262144,
        "Only bounded actual original JSON/Python/shell binding source")
    return packet.ref(p)
prior.ref=original_bounded_ref
prior.RECOVERY=HERE
prior.RECOVERY_SOURCE=SOURCE
prior.PINS_REF=PINS_REF
original_inspect=prior.inspect
def complete_source_inspect():
    values=original_inspect()
    p,m,frozen,cfg,partial,output,job,old,observed=values
    failure=prior.read(p["prior_recovery_failure"]);revoked=prior.read(p["prior_cooperative_revocation"])
    need(failure["returncode"]==1 and failure["phase"]=="cooperative_old_authority_revocation"
        and failure["source_ref"]==p["prior_recovery_source"]
        and failure["error"]["type"]=="ValueError" and failure["error"]["message"]=="Bounded JSON/source only",
        "Preserve exact prior actual failed call1 before authority/reserve")
    need(revoked["active"] is True and revoked["old_authority_ref"]==p["original_authority"]
        and revoked["old_bytes_preserved"] is True and revoked["old_authority_must_never_be_applied"] is True,
        "Preserve original actual cooperative revocation")
    for r in old["request"]["consumer_bindings"]:
        need(original_bounded_ref(r["path"])==r,"Every actual original direct binding validated in default mode")
    for r in p.values():
        if isinstance(r,dict) and set(r)=={"path","sha256"}:
            need(original_bounded_ref(r["path"])==r,"Every actual successor evidence/source path validated in default mode")
    for name in ("reserve_authority_recovery1.json","reserve_receipt.json","admission.json","admitted.json",
                 "launch.json","actual_wait.json","worker_aux"):
        target=packet.local(partial/name)
        need(not target.exists(),"No fresh authority, reserve, admission or worker exists; never blind retry")
    for name in ("root_old_authority_cooperative_revocation_v1.json","root_recovery_admit_actual_terminal_v1.json"):
        target=packet.local(HERE/name)
        need(not target.exists(),"Fresh successor evidence destination; preserve earlier attempt")
    need(len(old["request"]["consumer_bindings"])+6<=64,"All original40 plus existing recovery6 fit unchanged64 limit")
    return values
prior.inspect=complete_source_inspect
def main():
    if "--admit-recovery" in sys.argv:
        return prior.main()
    need(len(sys.argv)==1,"Only default metadata report or --admit-recovery")
    values=complete_source_inspect()
    print(json.dumps(dict(status="source_only_all_actual_binding_and_metadata_paths_validated",
        original_direct_binding_paths=len(values[7]["request"]["consumer_bindings"]),permitted_actual_source_suffixes=[".json",".py",".sh"],
        original_packet_ref_canonical_validation=True,prior_actual_failure1_preserved=True,
        current_original_partial_unchanged=True,observed_prefix=values[8]["observed_prefix"],
        guard_call_or_reserve_or_job=False),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
