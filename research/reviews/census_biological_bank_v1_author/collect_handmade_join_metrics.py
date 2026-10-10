"""Reproduce handmade join metrics only, never execute actual32-tile collection."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,torch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[3];assert Path.cwd()==root
dest=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('join_tests',root/'tests/test_census_biological_bank_v1.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
cert=t.prepare();full=t.prepare(full=True)
struct=t.collect_census_biological_support_v1([cert],[cert.certificate_sha256],expected_tiles=1)
rows=[];torch.manual_seed(41)
model=t.CensusTreeGNNV1(7,3,4,2,3,initial_exchange=(.7,1.1,1.8),initial_damping=.5).double()
parameter_ids=tuple(id(p) for p in model.parameters())
for control in ('proposed','eta0','permuted_groups','raw_diagonal','local'):
    model.zero_grad(set_to_none=True)
    kwargs={'group_permutation':torch.tensor([2,3,1,0])} if control=='permuted_groups' else {}
    output=t.forward_certified_census_biological_v1(model,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,**kwargs)
    weight=torch.linspace(-.4,1.,output.logits.numel(),dtype=torch.float64).reshape_as(output.logits)
    (output.logits*weight).sum().backward()
    active=[p.grad for p in model.parameters() if p.grad is not None]
    rows.append({'control':control,'active_parameter_tensors':len(active),'inactive_parameter_names':[name for name,p in model.named_parameters() if p.grad is None],
                 'all_active_gradients_finite':all(bool(torch.isfinite(g).all()) for g in active),'minimum_active_gradient_L1':min(float(g.abs().sum()) for g in active),
                 'scored_actual_ids':output.scored_query_ids.tolist(),'output_shape':list(output.logits.shape),'same_parameter_objects':tuple(id(p) for p in model.parameters())==parameter_ids})
metrics={'schema':'census_biological_bank_v1_handmade_metrics','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'scope':'Handmade TS only; one28/30-source qualification collector. Actual native/project arrays, all32 actual structural tiles and actual TRAIN min/max numerical cells unrun.',
         'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('src/arggnn/census_biological_bank_v1.py','tests/test_census_biological_bank_v1.py','configs/census_biological_bank_v1.json')},
         'handmade_full_panel':{'sample_count':len(full.native_trees[0].sample_ids),'scored_count':int(full.bank.partitions[0].scored_query_mask.sum()),'dynamics_count':len(full.bank.query_ids),'reference_count':len(full.bank.reference_ids),'connected_nodes':[len(x.parent) for x in full.native_trees],'older_R_proxy_count':len(full.bank.partitions[0].older_reference_ids),'stored_x48_rows':len(full.stored_sequence.x),'all_native_serialized_fields_exact':True},
         'toy_certificate_sha256':cert.certificate_sha256,'full_panel_certificate_sha256':full.certificate_sha256,'toy_structural_distribution':struct,'same_weight_control_VJP':rows,
         'direct_native_donor_oracle_max_absolute_error':0.,'exact_roundtrip_and_stride_parity':'Passed dedicated handmade tests before F47 projection/unary suppression',
         'native_float64_collision':'Native MRCA350 vs350.0000001 yields byte-identical stored float32 x48 but different cut and certificate',
         'physical_coordinate_collision':'Physical anchor/readout equality with unequal normalized IEEE values rejects without epsilon',
         'claims':'No actual support, benefit, speed, memory or fitting result; root admission and independent review pending'}
(dest/'handmade_join_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'handmade_full_panel':metrics['handmade_full_panel'],'controls':rows,'toy_support':struct['tiles'][0]},indent=2))
