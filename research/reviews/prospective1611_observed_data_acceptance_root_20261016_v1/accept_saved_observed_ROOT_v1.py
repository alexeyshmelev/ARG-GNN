"""ROOT accepts verified observed exports without changing the failed operational outcome."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/prospective1611_observed_data_acceptance_root_20261016_v1"
PINS={
"research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py":"491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468",
"research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json":"66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7",
"research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py":"ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635",
"research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json":"7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582",
"research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py":"17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f",
"src/arggnn/project_budget_v4.py":"a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47",
"research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py":"c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364",
"research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/storage.json":"76c8e8a49a1330937298d059633676ec7eb1f06bab5f16d134b94a95d8b70b37",
"research/reviews/prospective1611_observed_data_acceptance_root_20261016_v1/byte_integrity.root_v1.json":"d579f9709009c61fa37f7f4e0f5b8335e5fdb70ffaf9cfa501b2667b7100a8f7"}
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(path,name):
 assert ref(path)["sha256"]==PINS[path]
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(name,value):
 p=HERE/name
 with p.open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
 return ref(p)
def absent(pid):
 try:os.kill(pid,0)
 except ProcessLookupError:return dict(pid=pid,absent=True,observed_utc=datetime.now(timezone.utc).isoformat())
 raise RuntimeError("Known completed owned process is not absent")
def main():
 assert Path.cwd()==ROOT
 for path,digest in PINS.items():assert ref(path)["sha256"]==digest
 integrity=json.loads((HERE/"byte_integrity.root_v1.json").read_text())
 assert integrity["status"]=="PASS" and len(integrity["observed_products_independently_stream_hashed"])==7
 assert integrity["observed_bytes_stream_hashed"]==4052787961 and integrity["parent_outer_returncode"]==1 and integrity["numerical_child_returncode"]==0
 observations=[absent(pid) for pid in (3477332,3477878)]
 for row in integrity["observed_products_independently_stream_hashed"]:
  p=ROOT/row["path"];s=p.lstat();before=row["identity_before_and_after"]
  assert p.resolve()==p and s.st_nlink==1
  assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(before["device"],before["inode"],before["bytes"],before["mtime_ns"],before["ctime_ns"])
 revocation=save("future_write_revocation.root_active_v1.json",dict(
  schema="root_cooperative_completed_generation_namespace_revocation_v1",root_authorizer="root_agent",
  observed_utc=datetime.now(timezone.utc).isoformat(),actual_child_wait_ref=integrity["actual_child_wait_ref"],
  actual_outer_terminal_ref=integrity["actual_outer_terminal_ref"],observations=observations,
  scope="operational_successor_train000_live_boundary_root_v1 observed and private generated outputs",
  all_future_launch_resume_and_write_capabilities_revoked=True,read_only_reuse_preserved=True,
  full_original_charge_retained=True,all_original_files_preserved=True,
  complete_historical_writer_absence_not_asserted=True,no_signal_or_deletion_performed=True))
 sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
 renewal=load("research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py","_accept1611_renewal")
 renewal.install_renewal_v1(ref("research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json"))
 policy_path="research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json"
 policy=json.loads((ROOT/policy_path).read_text())
 g=load("research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py","_accept1611_original_guard")
 g.configure_v2(policy["marker_repair_authority_ref"],policy_ref=ref(policy_path))
 namespace=load("research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py","_accept1611_readonly_namespace")
 handle=namespace.install_consumer_v1(g.budget.v4,g.budget,consumer_role="read_only_consumer",
  expected_source_path=ROOT/"src/arggnn/project_budget_v4.py",expected_source_sha256=PINS["src/arggnn/project_budget_v4.py"])
 storage_path="research/reviews/ARGMix_prospective1611_first_train_H020v3_source_20261016_v1/admitted_train000_oct16_root_v1/storage.json"
 storage=json.loads((ROOT/storage_path).read_text())
 assert storage["reservation"]["reserved_bytes"]==17179869184
 checked=g.budget_check_v9(storage,boundary=True,required_write_bytes=0)
 strict=save("strict_postdrain_guard.root_v1.json",dict(
  status="PASS",schema="root_original_guard_strict_postdrain_check_v1",observed_utc=datetime.now(timezone.utc).isoformat(),
  storage_ref=ref(storage_path),guard_ref=ref("research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py"),
  read_only_consumer_memo_used=True,accounting_mutations=0,original_guard_result=checked))
 private=ROOT/"data/ARGMix_prospective1611_first_train_H020v3_generated_train000_oct16_root_v1/operational_successor_train000_live_boundary_root_v1/private_receipts/generation_receipt.json"
 assert ref(private)["sha256"]=="06dde89d638a1c161143560cba3c1af5addb272e34796d049c09483debe3b025"
 accepted=save("acceptance.root_v1.json",dict(
  schema="prospective1611_saved_observed_data_acceptance_root_v1",status="accepted_observed_data_for_downstream_inference",
  root_authorizer="root_agent",observed_utc=datetime.now(timezone.utc).isoformat(),source_ref=ref(Path(__file__)),
  accepted_for_observed_inference_only=True,observed_receipt_ref=integrity["observed_receipt_ref"],
  byte_integrity_ref=ref(HERE/"byte_integrity.root_v1.json"),strict_postdrain_guard_ref=strict,
  future_write_revocation_ref=revocation,actual_child_wait_ref=integrity["actual_child_wait_ref"],
  actual_outer_terminal_ref=integrity["actual_outer_terminal_ref"],private_generation_metadata_ref=ref(private),
  independent_source_metadata_audit_ref=ref("research/reviews/prospective1611_saved_generation_acceptance_audit_20261016_v1/source_handoff.v1.json"),
  parent_outer_returncode=1,numerical_child_returncode=0,original_monitor_failure_preserved=integrity["operational_failure_preserved"],
  full_original_charge_retained=True,original_reserved_bytes=17179869184,
  diploids=1611,haplotypes=3222,length_bp=35178458,observed_sites=311599,
  private_truth_payload_opened=False,independent_sample_spec_semantics_recomputed=False,full_run_peak_RSS_certified=False,
  published_table_reproduction=False,founder_labels_pending=True,ancestry_score=None,
  downstream_execution_requires_separate_admission=True,storage_close_or_credit_authorized=False,simulation_rerun=False))
 print(json.dumps(dict(status="accepted_observed_data_for_downstream_inference",acceptance_ref=accepted,strict_guard_ref=strict,
  retained_scope_bytes=checked["scope_measurement"]["conservative_scope_bytes"],original_charge_bytes=17179869184,parent_outer_exit=1,child_exit=0)),flush=True)
if __name__=="__main__":main()
