from pathlib import Path
from datetime import datetime,timezone
import importlib.util,hashlib,json,runpy
import torch,numpy as np
from arggnn.census_native_qualification_runner_v1 import ROOT,VerifiedReaderV1,run_qualification_phase_v1,numerical_cell_v1,_model,runtime_identity_v1
from arggnn.census_biological_bank_v1 import fingerprint_sha256_v1
owned=ROOT/'research/reviews/census_native_qualification_runner_v1_author';attempt=1
while (owned/f'collector_attempt_{attempt:02d}').exists():attempt+=1
tmp=owned/f'collector_attempt_{attempt:02d}';tmp.mkdir()
torch.set_num_threads(1)
tests=runpy.run_path(str(ROOT/'tests/test_census_native_qualification_runner_v1.py'))
config,store,active,ref,put,finish,ts=tests['fixture_store'](tmp)
inspection,record=run_qualification_phase_v1(ref,phase='inspect',reader=VerifiedReaderV1(read_bytes=store.__getitem__))
assert inspection['status']=='inspection_complete_pending_root_freeze'
spec=importlib.util.spec_from_file_location('sealed_author_metric_fixture',ROOT/'tests/test_census_biological_bank_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tables=ts.dump_tables();tables.sequence_length=1000.;tables.edges.left=np.where(tables.edges.left==0.,0.,500.);tables.edges.right=np.where(tables.edges.right==875050.,500.,1000.);tables.sort()
_,roster=module.fixture();cert=module.prepare(ts=tables.tree_sequence(),roster=roster)
states={d:fingerprint_sha256_v1(_model(config,d).state_dict()) for d in config['dtypes']}
checks=numerical_cell_v1(cert,cert.certificate_sha256,config,torch.tensor([2,3,1,0]),states)
full=module.prepare(full=True);full_checks=numerical_cell_v1(full,full.certificate_sha256,config,torch.arange(183,-1,-1),states)
shipped=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
metrics={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Handmade CPU fixtures only. Mock admissions/reviews/resources are fixture data, not actual admission.','runtime_identity':runtime_identity_v1(),
'actual_runtime_code_file_sha256':{p:sha(ROOT/p) for p in shipped['runtime_files']},'inspection_evidence_ref':record,
'handmade_TS':{'samples':ts.num_samples,'stored_nodes':ts.num_nodes,'original_trees':ts.num_trees,'tiles':len(inspection['cells']),'sum_tile_tree_holds':sum(c['num_trees'] for c in inspection['cells'])},
'event_collision_counts':{k:sum(c['event_collision_counts'][k] for c in inspection['cells']) for k in ('anchor_boundary','anchor_readout','boundary_readout','unequal_IEEE_anchor_collision')},
'prespecified_TRAIN_cells':inspection['prespecified_TRAIN_numerical_cells'],'real_small_cell_donor_range':float(cert.bank.donor_weights[0,0,0].max()-cert.bank.donor_weights[0,0,0].min()),
'real_small_cell_checks':checks,'full280_single_tree_checks':full_checks,'default_shipped_model_initial_state_sha256_by_dtype':{d:fingerprint_sha256_v1(_model(shipped,d).state_dict()) for d in shipped['dtypes']},
'actual_observation_model_target_arrays_opened':False,'actual_admission_created':False,'GPU_used':False,'fitting_performed':False}
(owned/'handmade_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'tile_count':32,'holds':metrics['handmade_TS']['sum_tile_tree_holds'],'real_small_cell_controls':len(checks['checks']),'donor_range':metrics['real_small_cell_donor_range'],'max_directional_abs_error':max(c['directional_float64']['absolute_error'] for c in checks['checks'] if c['directional_float64'])}))
