#!/usr/bin/env python3
"""Preliminary readonly stat inventory; no settlement or ledger mutation."""
from pathlib import Path
import collections,datetime as dt,hashlib,json,math,os,stat

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
AUDIT=ROOT/"research/reviews/biological_completed_candidates_storage_inventory_v1"
PINS={
 "research/reviews/biological_gst_completed_storage_settlement_v1/storage_revision_v2/audit_and_settle.py":"9508b4c2f4e21d9a2d2d1e3b6a47e42c6296773dee43233da53e1633173e9066",
 "research/reviews/biological_gst_storage_settlement_v1_independent/review_receipt_v2.json":"e6ebd15ef989918abf74545bf43a02beeca8cae057a8a94f1bc50f5074c66bcb",
 "src/arggnn/biological_candidate.py":"6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3"}
TARGETS=[
 dict(seed=202,pool="biological_gpu2_v1",job="biological-candidate-seed202",identities=[(937999,433043178),(939505,433046553)]),
 dict(seed=303,pool="biological_gpu3_v1",job="biological-candidate-seed303",identities=[(938040,433043284),(939529,433046633)])]
for t in TARGETS:
 t.update(key=t["pool"]+"/"+t["job"],output=str(ROOT/("results/biological_candidate_pilot_v1_seed%d"%t["seed"])),
          log=str(ROOT/".runtime/gpup"/t["pool"]/"jobs"/(t["job"]+".log")))

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def save(name,value,plain=False):
 if Path(name).name!=name: raise ValueError("Write outside owned audit directory rejected")
 path=AUDIT/name
 with path.open("x",encoding="utf-8") as f: f.write(value if plain else json.dumps(value,sort_keys=True,indent=2)+"\n")
 return dict(path=str(path),sha256=sha(path.read_bytes()),bytes=path.stat().st_size)
def metadata_json(path):
 a=path.stat(); raw=path.read_bytes(); b=path.stat()
 if any(getattr(a,k)!=getattr(b,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")):
  raise RuntimeError("Metadata changed during read: "+str(path))
 return json.loads(raw),sha(raw)
def proc(pid,ticks):
 base=Path("/proc")/str(pid)
 try: uid=base.stat().st_uid; raw=(base/"stat").read_text()
 except FileNotFoundError: return dict(pid=pid,saved_start_ticks=ticks,saved_identity_present=False,reason="pid_absent_at_read")
 f=raw[raw.rfind(")")+2:].split()
 current=dict(pid=pid,uid=uid,state=f[0],ppid=int(f[1]),pgid=int(f[2]),sid=int(f[3]),start_ticks=int(f[19]))
 same=current["start_ticks"]==ticks
 return dict(pid=pid,saved_start_ticks=ticks,saved_identity_present=same,current=current,
             reason="same_saved_identity_present" if same else "pid_reused_different_start")
def metadata():
 ledger,digest=metadata_json(ROOT/".runtime/gpup/storage_reservations.json")
 if ledger["root"]!=str(ROOT) or ledger["schema"]!="gpu_pool_storage_ledger_v1": raise RuntimeError("Ledger identity differs")
 for k in ("baseline_bytes","external_reserve_bytes","limit_bytes"):
  if type(ledger[k]) is not int or ledger[k]<0: raise RuntimeError("Invalid ledger charge")
 for namespace in ("jobs","controllers"):
  for entry in ledger[namespace].values():
   if type(entry["reserved_bytes"]) is not int or entry["reserved_bytes"]<0: raise RuntimeError("Invalid reservation")
 projection=ledger["baseline_bytes"]+ledger["external_reserve_bytes"]+sum(e["reserved_bytes"] for e in ledger["jobs"].values())+sum(e["reserved_bytes"] for e in ledger["controllers"].values())
 candidates=[]
 for t in TARGETS:
  path=ROOT/".runtime/gpup"/t["pool"]/"registry.json"; r,rsha=metadata_json(path)
  jobs=r["jobs"]
  matches=[j for j in (jobs.values() if isinstance(jobs,dict) else jobs) if j.get("logical_job")==t["job"]]
  if len(matches)!=1: raise RuntimeError("Expected exactly one bound terminal job")
  j=matches[0]; req=j["requested"]
  if (j["logical_job"]!=t["job"] or j["state"]!="succeeded" or j["exit_code"]!=0 or r["safety_error"] is not None
      or req["job"]!=t["job"] or req["module"]!="arggnn.biological_candidate" or req["seed"]!=t["seed"]
      or req["output"]!=t["output"] or req["storage_bytes"]!=29281048576 or j["log_path"]!=t["log"]
      or j["module_source_sha256"]!=PINS["src/arggnn/biological_candidate.py"]):
   raise RuntimeError("Terminal candidate binding differs: "+t["key"])
  if sorted((x["pid"],x["start_ticks"]) for x in j["process_identities"])!=sorted(t["identities"]):
   raise RuntimeError("Saved worker identities differ")
  if (j["pid"],j["leader_start_ticks"])!=t["identities"][0]: raise RuntimeError("Leader identity differs")
  if ledger["jobs"][t["key"]]!=dict(output=t["output"],reserved_bytes=29281048576): raise RuntimeError("Candidate ledger binding differs")
  fields=("logical_job","state","exit_code","pid","leader_start_ticks","process_identities","log_path","log_bytes",
          "log_dropped_bytes","started_unix","finished_unix","module_source_sha256","requested","argv")
  candidates.append(dict(key=t["key"],registry_path=str(path),registry_sha256=rsha,lifecycle=r["lifecycle"],
                         safety_error=r["safety_error"],terminal_job={k:j.get(k) for k in fields},
                         pid_start_checks=[proc(pid,ticks) for pid,ticks in t["identities"]]))
 return dict(observed_utc=now(),ledger_sha256=digest,ledger=ledger,recomputed_projection_bytes=projection,
             remaining_headroom_bytes=ledger["limit_bytes"]-projection,candidates=candidates,
             gpu_queries_performed=False,global_process_scan_performed=False)
def fail(error): raise error
def record(path,scope,relative):
 s=path.lstat()
 if stat.S_ISLNK(s.st_mode) or not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):
  raise RuntimeError("Symlink or nonregular entry rejected: "+str(path))
 return dict(scope=scope,relative_path=relative,entry_type="directory" if stat.S_ISDIR(s.st_mode) else "regular_file",
             mode=s.st_mode,size_bytes=s.st_size,allocated_bytes=s.st_blocks*512,retained_charge_bytes=max(s.st_size,s.st_blocks*512),
             device=s.st_dev,inode=s.st_ino,link_count=s.st_nlink,uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def inventory(t):
 started=now(); out=Path(t["output"]); log=Path(t["log"])
 for path in (out,log):
  if path.is_symlink() or path.resolve(strict=True)!=path: raise RuntimeError("Canonical path identity differs")
 if not stat.S_ISDIR(out.lstat().st_mode): raise RuntimeError("Output must be directory")
 rows=[]
 for current,dirs,files in os.walk(out,topdown=True,followlinks=False,onerror=fail):
  dirs.sort(); files.sort(); p=Path(current)
  rows.append(record(p,"output","." if p==out else str(p.relative_to(out))))
  for name in dirs:
   s=(p/name).lstat()
   if stat.S_ISLNK(s.st_mode) or not stat.S_ISDIR(s.st_mode): raise RuntimeError("Walk directory must be regular directory")
  for name in files: rows.append(record(p/name,"output",str((p/name).relative_to(out))))
 lr=record(log,"retained_pool_log",log.name)
 if lr["entry_type"]!="regular_file": raise RuntimeError("Retained log must be regular file")
 rows.append(lr); rows.sort(key=lambda r:(r["scope"],r["relative_path"]))
 ext_counts=collections.Counter(); ext_charges=collections.Counter()
 for r in rows:
  if r["entry_type"]=="regular_file":
   ext=Path(r["relative_path"]).suffix or "(no suffix)"; ext_counts[ext]+=1; ext_charges[ext]+=r["retained_charge_bytes"]
 return dict(schema="completed_candidate_stat_inventory_v1",seed=t["seed"],key=t["key"],output=t["output"],retained_pool_log=t["log"],
             started_utc=started,completed_utc=now(),method="fail-closed lstat-only traversal; no completed-output content reads",
             content_hashes_created=False,scheduler_mutex_held=False,logical_bytes=sum(r["size_bytes"] for r in rows),
             allocated_bytes=sum(r["allocated_bytes"] for r in rows),conservative_retained_charge_bytes=sum(r["retained_charge_bytes"] for r in rows),
             output_charge_bytes=sum(r["retained_charge_bytes"] for r in rows if r["scope"]=="output"),retained_pool_log_charge_bytes=lr["retained_charge_bytes"],
             counts=dict(collections.Counter(r["entry_type"] for r in rows)),regular_file_extension_counts=dict(ext_counts),
             regular_file_extension_charges=dict(ext_charges),multiple_link_regular_files=sum(r["entry_type"]=="regular_file" and r["link_count"]>1 for r in rows),
             largest_regular_files=sorted((r for r in rows if r["entry_type"]=="regular_file"),key=lambda r:(-r["retained_charge_bytes"],r["relative_path"]))[:8],entries=rows)
def main():
 started=now()
 for name,expected in PINS.items():
  if sha((ROOT/name).read_bytes())!=expected: raise RuntimeError("Approved/source pin differs: "+name)
 before=metadata(); refs=[save("metadata_before.json",before)]; first={}; second={}
 for t in TARGETS:
  first[t["seed"]]=inventory(t); refs.append(save("seed%d_stat_inventory_pass1.json"%t["seed"],first[t["seed"]]))
 for t in TARGETS:
  second[t["seed"]]=inventory(t); refs.append(save("seed%d_stat_inventory_pass2.json"%t["seed"],second[t["seed"]]))
 after=metadata(); refs.append(save("metadata_after.json",after))
 fields=("key","output","retained_pool_log","logical_bytes","allocated_bytes","conservative_retained_charge_bytes","entries")
 stable=all(all(first[t["seed"]][k]==second[t["seed"]][k] for k in fields) for t in TARGETS)
 terminal=all(before["candidates"][i]["terminal_job"]==after["candidates"][i]["terminal_job"] and before["candidates"][i]["safety_error"]==after["candidates"][i]["safety_error"] for i in range(2))
 gone=all(not p["saved_identity_present"] for m in (before,after) for c in m["candidates"] for p in c["pid_start_checks"])
 checks=dict(two_stat_passes_identical=stable,terminal_job_metadata_unchanged=terminal,saved_worker_identities_absent_before_and_after=gone,
             live_ledger_sha256_unchanged=before["ledger_sha256"]==after["ledger_sha256"],canonical_regular_entries_only=True,
             candidate_job_output_module_seed_bindings_match=True,approved_procedure_review_and_candidate_source_pins_match=True)
 proposals=[]
 if all(checks.values()):
  for t in TARGETS:
   measured=second[t["seed"]]["conservative_retained_charge_bytes"]
   margin=max(2000000000,(measured+3)//4)
   proposed=max(15000000000,((measured+margin+999999999)//1000000000)*1000000000)
   if proposed>=29281048576: raise RuntimeError("No conservative reduction justified")
   proposals.append(dict(seed=t["seed"],key=t["key"],output=t["output"],retained_pool_log=t["log"],current_reserved_bytes=29281048576,
                         observed_stat_charge_bytes=measured,proposed_retained_charge_bytes=proposed,growth_margin_bytes=proposed-measured,
                         growth_margin_fraction=(proposed-measured)/measured,reservation_reduction_bytes=29281048576-proposed,
                         proposed_charge_rule="at least 15 decimal GB; at least 25% or 2 decimal GB growth margin, rounded up to decimal GB"))
 reduction=sum(p["reservation_reduction_bytes"] for p in proposals); current=after["recomputed_projection_bytes"]; ledger=after["ledger"]
 preserved={k:ledger[k] for k in ("baseline_bytes","external_reserve_bytes","limit_bytes","snapshot_sha256","controllers")}
 keys={t["key"] for t in TARGETS}; preserved["other_jobs"]={k:v for k,v in ledger["jobs"].items() if k not in keys}
 delta=[
 "Bind both exact candidate keys, roots, retained logs, registry terminal jobs, seeds and all saved PID/start identities; replace GST hardcoded job/status/ledger bindings.",
 "Verify candidate receipt metadata only: kind biological_candidate_fit_v1; status completed_three_conditions_one_seed; matching seed/source/config/observed/prepared pins; exact expected three model keys with completed and optimizer_fit_completed=true. Do not decode arrays or labels.",
 "Freeze these stat artifacts as prior evidence and create opaque SHA256 file bindings for every output file and retained log with before/after stat checks. Stat-only evidence cannot substitute for the reviewed full-hash inventory.",
 "Require root contractual freeze of completed outputs/logs and trustworthy terminal registries/cooperating ledger writers. Under canonical .runtime/locks/scheduler.lock, create two full inventories, recheck terminal jobs/PID-start identities before and after, bind the fresh ledger, then use the reviewed atomic mutation.",
 "Independently review the candidate adaptation before apply. Change only reserved_bytes for the two existing bound keys and assert exact preservation of every other ledger field/reservation. Retain completed keys permanently.",
 "Retain all files, including every epoch probability NPZ, checkpoints, metrics and logs. Future fits use fresh output roots and reservations. Preserve mechanisms, compact workers/controllers, historical baseline and 130000000000-byte external reserve.",
 "Do not run the old hardcoded GST script for candidates or claim OS immutability/arbitrary unknown-writer absence. Preserve the qualified PASS contract explicitly."]
 receipt=dict(schema="completed_candidate_storage_inventory_receipt_v1",status="preliminary_stat_inventory_complete" if len(proposals)==2 else "preliminary_inventory_requires_reinspection",
              started_utc=started,completed_utc=now(),authority="readonly inventory and proposal only",settlement_authorized=False,settlement_applied=False,
              project_outputs_modified=False,live_ledger_modified=False,gpu_queries_performed=False,arrays_or_labels_decoded=False,output_receipts_decoded=False,
              scheduler_mutex_held=False,prior_and_source_pins=PINS,inventory_script_sha256=sha(Path(__file__).read_bytes()),checks=checks,
              before_ledger_sha256=before["ledger_sha256"],current_ledger_sha256=after["ledger_sha256"],current_projection_bytes=current,
              current_headroom_bytes=after["remaining_headroom_bytes"],proposals=proposals,proposed_total_reduction_bytes=reduction,
              proposed_projection_bytes=current-reduction,proposed_headroom_bytes=ledger["limit_bytes"]-current+reduction,
              preserved_ledger_fields_and_reservations=preserved,minimal_settlement_audit_delta=delta,artifact_refs=list(refs),
              limitations=["No completed-output content hashes; stat/membership inventory only.","Scheduler mutex was not held for this preliminary inspection.",
                           "Stat stability and absent saved identities do not prove arbitrary unknown writers absent.","Actual candidate receipt terminal gate remains for future settlement audit.",
                           "Any apply must rebind/recompute the live ledger under canonical mutex."])
 receipt_ref=save("inventory_receipt.json",receipt); refs.append(receipt_ref)
 lines=["# Completed candidates: preliminary retained-storage inventory","",
        "Readonly proposal only; no settlement authorization or apply. Writes were confined to this audit directory. Completed outputs/logs, live ledger, controllers and decision records were unchanged. No arrays/labels or output receipts were decoded; no GPU query/context was created.","",
        "| Seed | Files + directories | Retained stat charge | Proposed charge | Growth margin | Reduction |",
        "|---|---:|---:|---:|---:|---:|"]
 for p in proposals:
  count=second[p["seed"]]["counts"]
  lines.append("| %d | %d files + %d directories | %s B | %s B | %s B (%.1f%%) | %s B |"%(p["seed"],count.get("regular_file",0),count.get("directory",0),
               format(p["observed_stat_charge_bytes"],","),format(p["proposed_retained_charge_bytes"],","),format(p["growth_margin_bytes"],","),
               100*p["growth_margin_fraction"],format(p["reservation_reduction_bytes"],",")))
 lines.extend(["","Every retained file/directory and pool log is charged using max(st_size, st_blocks * 512). Both complete stat/membership passes matched. Terminal jobs remained succeeded with exit code 0; all four saved PID/start identities were absent at both checks. This does not prove absence of arbitrary unknown writers.","",
               "Fresh live ledger SHA256: "+after["ledger_sha256"]+". Recomputed current projection: "+format(current,",")+" B. Proposed combined reduction: "+format(reduction,",")+" B; proposed projection: "+format(current-reduction,",")+" B; headroom: "+format(receipt["proposed_headroom_bytes"],",")+" B under the 450,000,000,000 B limit.","",
               "The fresh ledger includes both 1,000,000,000 B census reservations. Preserve all unrelated jobs, all four 67,108,864 B controllers, historical baseline 167,336,640,512 B, external reserve 130,000,000,000 B, limit, snapshot and output bindings.","",
               "Proposed charges have a 15 decimal GB floor and at least 25%/2 decimal GB measured-growth margin, rounded up to decimal GB. All epoch probability NPZs remain included. Future fits need fresh output directories and separate reservations.","",
               "## Minimal delta to the reviewed GST settlement",""])
 lines.extend(str(i+1)+". "+item for i,item in enumerate(delta))
 lines.extend(["","## Evidence and limitations","",
               "Pins are recorded in inventory_receipt.json. Candidate source establishes the receipt schema at lines 733-750 and terminal completion at 867-880. Actual output receipt metadata verification remains outstanding because this preliminary task reads only terminal registries and filesystem stats. Opaque file hashing and two full inventories under the canonical mutex remain required before applying an independently reviewed adaptation. Contractual freeze and cooperating-writer assumptions remain explicit; the procedure does not enforce OS-level immutability.",""])
 refs.append(save("report.md","\n".join(lines),plain=True))
 for name in ("registration.json","inventory_completed_candidates.py"):
  path=AUDIT/name; refs.append(dict(path=str(path),sha256=sha(path.read_bytes()),bytes=path.stat().st_size))
 seal=save("artifact_hashes.json",dict(schema="audit_artifact_hashes_v1",completed_utc=now(),artifacts=refs))
 print(json.dumps(dict(status=receipt["status"],checks=checks,ledger_sha256=after["ledger_sha256"],current_projection_bytes=current,
                       proposed_total_reduction_bytes=reduction,proposed_projection_bytes=current-reduction,proposed_headroom_bytes=receipt["proposed_headroom_bytes"],
                       proposals=proposals,receipt=receipt_ref,artifact_hashes=seal),sort_keys=True))

if __name__=="__main__": main()
