"""Inert by default; root supplies the separately authorized operational adapter."""
from pathlib import Path
import argparse,copy,importlib,json

HERE=Path(__file__).resolve().parent

def run_root_adapter_v1(adapter):
    protocol=json.loads((HERE/"protocol.source_frozen.inactive.v1.json").read_text())
    # prepare_v1 owns actual guard/admission, GPU0 mapping, target capabilities,
    # audited archive/catalogue/observation loader and a fresh writer directory.
    context=adapter.prepare_v1(copy.deepcopy(protocol))
    context["runtime_check"]("before_scientific_imports")
    from .trainer_v1 import run_primary_v1
    bundle=run_primary_v1(protocol=protocol,**context)
    # No label evaluator is imported/called here. Root explicitly handles VAL,
    # confirmation decision while TEST closed, and any later TEST release.
    return bundle

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute-root",action="store_true");p.add_argument("--root-adapter")
    a=p.parse_args()
    if not a.execute_root:
        print(json.dumps(dict(status="source_only_inert",Torch_imported=False,model_or_data_execution=False)));return 0
    if not a.root_adapter:raise ValueError("Explicit root operational adapter module required")
    adapter=importlib.import_module(a.root_adapter)
    bundle=run_root_adapter_v1(adapter)
    print(json.dumps(dict(status="primary_full_budget_predictions_sealed",prediction_seal_ref=bundle.seal_ref,
        fit_receipt_ref=bundle.fit_receipt_ref,heldout_metrics_computed=False)))
    return 0

if __name__=="__main__":raise SystemExit(main())
