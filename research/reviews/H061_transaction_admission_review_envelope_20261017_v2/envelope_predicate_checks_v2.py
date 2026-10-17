"""Exact original reviewed(record) predicate, isolated over sealed metadata only."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,traceback,sys
BASE=Path("research/reviews/H061_transaction_admission_review_envelope_20261017_v2")
OLD=Path("research/reviews/H060_H061_transaction_admission_independent_20261017_v2/H061.integration_review.v1.json")
CONTRACT=Path("research/reviews/H061_postselection_execution_source_20261017_v1/root_contract_v1.py")
BINDING=Path("research/reviews/H061_postselection_execution_source_20261017_v1/scientific_binding.v1.json")
SRC=Path("research/reviews/H060_H061_transaction_admission_source_20261017_v2")
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 assert ref(CONTRACT)["sha256"]=="c01479a37ad9725f5a3ab7a1759f31db48b4d4d8678ce2b723f1226fb2e74d98"
 assert ref(BINDING)["sha256"]=="8e559d519f95da5042231568da27d0723bc86e25b219060fa97cf21bcabe003c"
 assert ref(OLD)["sha256"]=="da35f50429f9c47e1a83552ca1ee03945ec9e27809f634c3f78e62aee0b691a1"
 binding=json.loads(BINDING.read_text())
 for key in ("runner_source_ref","protocol_ref"):assert ref(binding[key]["path"])==binding[key]
 value=json.loads((BASE/"H061.integration_review.v2.json").read_text())
 source_text=(SRC/"root_driver_v1.py").read_text()
 tree=ast.parse(source_text)
 contract_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="contract")
 text=ast.get_source_segment(source_text,contract_fn)
 assert "c.SOURCE=c.DRIVER=SOURCE" in text and "c.MANIFEST=MANIFEST;c.HANDOFF=HANDOFF" in text
 assert 'c.binding=lambda:read(spec["scientific_binding_ref"])' in text
 source_bindings=json.loads((SRC/"source_bindings.v1.json").read_text())
 assert source_bindings["consumers"]["H061"]["scientific_binding_ref"]==ref(BINDING)
 definitions=ast.parse(CONTRACT.read_text())
 real=next(n for n in definitions.body if isinstance(n,ast.FunctionDef) and n.name=="reviewed")
 assert ast.unparse(real.args)=="record"
 current_record=None
 sentinel={"path":"handwritten-envelope-ref","sha256":"handwritten-only"}
 def read(record):
  assert record==sentinel
  return copy.deepcopy(current_record)
 def require(ok,message):
  if not ok:raise ValueError(message)
 ns=dict(read=read,binding=lambda:copy.deepcopy(binding),require=require,ref=ref,
     DRIVER=str(SRC/"root_driver_v1.py"),MANIFEST=str(SRC/"source_manifest.v1.json"),
     HANDOFF=str(SRC/"source_handoff.v1.json"))
 exec(compile(ast.Module(body=[copy.deepcopy(real)],type_ignores=[]),str(CONTRACT)+":isolated-reviewed","exec"),ns)
 current_record=value
 assert ns["reviewed"](sentinel) is None
 rejected=[]
 for field in ("independent","verdict","source_ref","source_manifest_ref","source_handoff_ref","runner_source_ref","protocol_ref","blockers"):
  current_record=copy.deepcopy(value);current_record.pop(field)
  try:ns["reviewed"](sentinel)
  except ValueError as exc:
   assert str(exc)=="Actual independent whole postselection source review"
   rejected.append(field)
  else:raise AssertionError("Required real-contract field unexpectedly absent: "+field)
 current_record=json.loads(OLD.read_text())
 try:ns["reviewed"](sentinel)
 except ValueError as exc:
  assert str(exc)=="Actual independent whole postselection source review"
  old_rejected=True
 else:raise AssertionError("Prior incomplete envelope unexpectedly accepted")
 return dict(schema="H061_corrected_envelope_actual_contract_predicate_checks_v2",source_only=True,
   corrected_envelope_ref=ref(BASE/"H061.integration_review.v2.json"),original_review_ref=ref(OLD),
   actual_contract_ref=ref(CONTRACT),actual_scientific_binding_ref=ref(BINDING),
   actual_reviewed_signature="reviewed(record)",actual_original_reviewed_predicate_extracted_unchanged=True,
   current_common_driver_identity_and_original_binding_join_proven=True,
   corrected_envelope_passes_actual_complete_predicate=True,old_incomplete_envelope_rejected=old_rejected,
   all_actual_required_field_omissions_rejected=rejected,
   runner_source_ref=binding["runner_source_ref"],protocol_ref=binding["protocol_ref"],
   real_contract_modules_entrypoints_guards_resources_requests_payloads_or_operations_executed=False)
if __name__=="__main__":
 try:print(json.dumps(main(),indent=2,sort_keys=True))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
