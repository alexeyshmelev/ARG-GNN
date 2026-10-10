"""Reject fixture review with exact default config using only in-memory mock receipts."""
from pathlib import Path
import importlib.util,json
from datetime import datetime,timezone
import arggnn.census_native_qualification_runner_v1 as runner
OWN=runner.ROOT/'research/reviews/census_native_qualification_runner_v1_independent'
spec=importlib.util.spec_from_file_location('independent_guard_recipe',OWN/'test_independent.py');recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
def main():
    tmp=OWN/'actual_review_guard_fixture';tmp.mkdir()
    _,store,active,ref,put,finish,ts=recipe.envelope(tmp)
    shipped=json.loads((runner.ROOT/'configs/census_native_qualification_runner_v1.json').read_text())
    configref=put('fixtures/default_exact_config.json',shipped);active['refs']['runner_config']=configref
    review=json.loads(store[active['refs']['runner_review']['path']]);full=json.loads(store[review['review_receipt_ref']['path']]);assert full['fixture_mode'] is True
    full['artifact_sha256']['configs/census_native_qualification_runner_v1.json']=configref['sha256']
    review['runner_config_sha256']=configref['sha256'];review['review_receipt_ref']=put('fixtures/still_fixture_full_review.json',full)
    active['refs']['runner_review']=put('fixtures/rebound_fixture_review.json',review)
    # Default-mode entry cannot proceed. Nothing is persisted as active admission.
    try:runner.admit_runner_v1(finish(active),phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))
    except ValueError as error:
        assert 'Full independent review receipt' in str(error)
        result={'fixture_only_mock_records':True,'default_config_equals_shipped':shipped==json.loads((runner.ROOT/'configs/census_native_qualification_runner_v1.json').read_text()),'fixture_review_rejected_in_default_mode':True,'error':str(error),'actual_admission':False,'created_at_utc':datetime.now(timezone.utc).isoformat()}
    else:raise AssertionError('Fixture review admitted default mode')
    (OWN/'actual_review_guard_metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
