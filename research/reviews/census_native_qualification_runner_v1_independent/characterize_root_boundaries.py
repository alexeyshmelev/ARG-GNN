"""Characterize external root duties with mocked fixture receipts, never actual admission."""
from pathlib import Path
import importlib.util,json
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import arggnn.census_native_qualification_runner_v1 as runner
OWN=runner.ROOT/'research/reviews/census_native_qualification_runner_v1_independent'
spec=importlib.util.spec_from_file_location('independent_recipe_for_boundaries',OWN/'test_independent.py');recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
def main():
    tmp=OWN/'root_boundary_fixture';tmp.mkdir()
    config,store,active,ref,put,finish,ts=recipe.envelope(tmp)
    storage=json.loads(store[active['refs']['storage']['path']]);storage['expires_at_utc']=(datetime.now(timezone.utc)+timedelta(seconds=15)).isoformat()
    active['refs']['storage']=put('fixtures/early_expiry_storage.json',storage)
    admitted=runner.admit_runner_v1(put('fixtures/root_boundary_admission.json',active),phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))
    shorter_reservation_accepted=admitted[0]['job_id']==active['job_id']
    # No waiting or deliberately expired real run. Expiry-at-entry is the source contract.
    active['refs']['storage']=finish(active) and active['refs']['storage']
    inspection,_=runner.run_qualification_phase_v1(finish(active),phase='inspect',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__))
    assert inspection['status']=='inspection_complete_pending_root_freeze'
    numeric=recipe.numerical_admission(active,put,config,inspection,tmp);numeric['job_id']=active['job_id'];calls=[]
    result,_=runner.run_qualification_phase_v1(finish(numeric),phase='numerical',reader=runner.VerifiedReaderV1(read_bytes=store.__getitem__),numerical_probe=recipe.probe_function(calls))
    assert result['status']=='numerical_complete_no_fitting_or_predictive_claim'
    record={'fixture_only':True,'source_resource_boundaries':{'storage_expiry_before_admission_expiry_accepted_at_entry':shorter_reservation_accepted,'same_job_id_across_distinct_phase_admissions_accepted':result['job_id']==inspection['job_id'],'source_checks_admission_expiry_between_cells':True,'no_in_cell_deadline_or_memory_watchdog':True},'root_duties':['Choose distinct job IDs and separate fresh CPU launcher/storage reservations for inspection and numerical phases.','Keep reservation validity through planned phase duration; abort externally on expiration or exhausted resource headroom.','Supply externally audited actual manifest hashes, original observed roster hashes and observation-only closure. SHA256 binding establishes chosen byte identity, not root authorship.'],'created_at_utc':datetime.now(timezone.utc).isoformat()}
    (OWN/'root_boundary_characterization.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
