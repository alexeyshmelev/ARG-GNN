"""CPU-only evidence collection; existing jobs, predictions and checkpoints are read-only."""
from pathlib import Path
import ast
import copy
import datetime
import hashlib
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from research.reviews.frozen_benchmark_training_ready_20261017_v1.european_frozen_treeloom_v1 import FrozenEuropeanTreeLoomV1
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_execution import decode_parent_backward, _tile_logits
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_gru import install_chunked_gru
from research.repairs.benchmark_recovery_20261020.training.decoder import geometry_from_spans

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/comparability_audit_20261028/model'
FIT = Path('experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v2/other/AP__all_train__treeloom_fit')
JOB = Path('.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json')
COMPARE = Path('experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json')
PROD = Path('experiments/results/paper_campaign_20261024/ap_treeloom_predictions/native/products')

def ref(path):
    path = Path(path)
    full = ROOT / path
    return dict(path=str(path), sha256=hashlib.sha256(full.read_bytes()).hexdigest(), bytes=full.stat().st_size,
                mtime_utc=datetime.datetime.fromtimestamp(full.stat().st_mtime, datetime.timezone.utc).isoformat())

def read(path):
    return json.loads((ROOT / path).read_text())

def verified(reference):
    actual = ref(reference['path'])
    assert actual['sha256'] == reference['sha256']
    return actual

def refs_in(value):
    if isinstance(value,dict):
        if set(('path','sha256')) <= set(value):
            yield {k:value[k] for k in ('path','sha256')}
        for item in value.values():
            yield from refs_in(item)
    elif isinstance(value,list):
        for item in value:
            yield from refs_in(item)

job = read(JOB)
spec = job['call']['spec']['scientific_spec']
completion = read(FIT / 'native/training_completed.json')
checkpoint_ref = verified(completion['checkpoint_ref'])
saved = torch.load(ROOT / checkpoint_ref['path'], map_location='cpu', weights_only=True)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(101)
    initial = FrozenEuropeanTreeLoomV1(condition='AP_paper_scale_200Mb_native6')
initial_state = initial.state_dict()
names = [name for name, _ in initial.named_parameters()]
assert names == list(saved['model_state'])
ids = saved['optimizer_state']['param_groups'][0]['params']
assert len(ids) == len(names) == 49
param_evidence = {}
for name,index in zip(names,ids):
    value = saved['model_state'][name]
    opt = saved['optimizer_state']['state'][index]
    delta = value - initial_state[name]
    param_evidence[name] = dict(shape=list(value.shape), elements=value.numel(),
        changed_elements=int(torch.count_nonzero(delta)), delta_max_abs=float(delta.abs().max()),
        delta_l2=float(delta.norm()), step=float(opt['step']),
        finite=bool(torch.isfinite(value).all() and torch.isfinite(opt['exp_avg']).all() and torch.isfinite(opt['exp_avg_sq']).all()),
        first_moment_l2=float(opt['exp_avg'].norm()), second_moment_sum=float(opt['exp_avg_sq'].sum()))
    if name == 'classifier.2.weight':
        param_evidence[name].update(delta_l2_by_class=delta.norm(dim=1).tolist(),
            first_moment_l2_by_class=opt['exp_avg'].norm(dim=1).tolist(),
            second_moment_sum_by_class=opt['exp_avg_sq'].sum(dim=1).tolist())
    if name == 'classifier.2.bias':
        param_evidence[name].update(value=value.tolist(), initial_value=initial_state[name].tolist(),
            delta=delta.tolist(), first_moment=opt['exp_avg'].tolist(), second_moment=opt['exp_avg_sq'].tolist())

training = {}
for parent,item in spec['train_inputs'].items():
    target = read(item['training_targets_ref']['path'])
    registry = read(item['sampled_registry_ref']['path'])
    target_ref = verified(target['arrays_ref'])
    with np.load(ROOT / target_ref['path'], allow_pickle=False) as arrays:
        labels = arrays['native_labels_0based'].copy()
    records = registry['records']
    assert len(labels) == len(records)
    cohorts = np.asarray([row['cohort'] for row in records])
    training[parent] = dict(target_metadata=verified(item['training_targets_ref']),
        registry=verified(item['sampled_registry_ref']), target_array=target_ref, rows=len(labels),
        class_counts=np.bincount(labels,minlength=6).tolist(),
        counts_by_cohort={str(cohort):np.bincount(labels[cohorts==cohort],minlength=6).tolist() for cohort in np.unique(cohorts)})

predictions = {}
for path in sorted((ROOT / PROD).glob('*.json')):
    product = read(path)
    array_ref = verified(product['predictions_ref'])
    assert product['checkpoint_ref'] == completion['checkpoint_ref']
    with np.load(ROOT / array_ref['path'],allow_pickle=False) as arrays:
        probabilities = arrays['probabilities'].copy()
    predictions[product['parent_id']] = dict(product=ref(path.relative_to(ROOT)), arrays=array_ref,
        checkpoint_ref=product['checkpoint_ref'], rows=len(probabilities),
        class_argmax_counts=np.bincount(probabilities.argmax(1),minlength=6).tolist(),
        mean_probability_by_class=probabilities.mean(0).tolist(), max_probability_by_class=probabilities.max(0).tolist(),
        sum_error_max=float(np.abs(probabilities.sum(1)-1).max()),
        execution_refs={k:v for k,v in product.items() if k.endswith('source_ref')})

events = [read(path.relative_to(ROOT)) for path in sorted((ROOT / FIT / 'native/updates').glob('update_*.json'))]
assert len(events) == 90 and [event['optimizer_updates'] for event in events] == list(range(1,91))
epochs = {}
for epoch in range(1,31):
    batch = [event for event in events if event['epoch'] == epoch]
    epochs[epoch] = dict(losses=[event['loss'] for event in batch],
        mean_loss=float(np.mean([event['loss'] for event in batch])),
        gradient_norms_before_clipping=[event['gradient_norm'] for event in batch],
        clipped_updates=sum(event['gradient_norm']>1 for event in batch))
source_refs = list(refs_in(job))
source_checks = []
for reference in {item['path']:item for item in source_refs if item['path'].endswith('.py')}.values():
    actual = ref(reference['path'])
    source_checks.append(dict(expected=reference, actual=actual, matches=actual['sha256']==reference['sha256']))

# Synthetic decoder-only backward probe: no optimizer, no real scientific input, no CUDA.
plain = copy.deepcopy(initial)
plain.load_state_dict(saved['model_state'],strict=True)
chunked = copy.deepcopy(plain)
install_chunked_gru(chunked,chunk_holds=3)
torch.manual_seed(7281)
context_a = torch.randn(7,9,32,requires_grad=True)
context_b = context_a.detach().clone().requires_grad_(True)
geometry = geometry_from_spans(torch.tensor([10.,35.,16.,62.,21.,77.,15.]))
holds = torch.tensor([0,1,2,3,4,5,6,2,0,3,4,6])
columns = torch.tensor([0,1,2,3,4,5,0,1,4,5,1,2])
labels = torch.tensor([0,1,2,3,4,5,0,1,2,3,4,5])
loss_a = F.cross_entropy(_tile_logits(plain,context_a,geometry,holds,columns),labels,reduction='mean')
loss_a.backward()
loss_b, gradient_b = decode_parent_backward(chunked,context_b,geometry,holds,columns,labels,tile_haplotypes=3)
context_b.backward(gradient_b)
probe_params = {}
for (name,a),(other,b) in zip(plain.named_parameters(),chunked.named_parameters()):
    assert name == other
    if a.grad is not None or b.grad is not None:
        assert a.grad is not None and b.grad is not None
        probe_params[name] = float((a.grad-b.grad).abs().max())
probe = dict(description='Monolithic original GRU versus actual tiled decoder with three-hold connected recurrent chunks; repeated focal draws, empty tile, all six labels; no optimizer step.',
    complete_loss=float(loss_a), tiled_loss=float(loss_b), loss_abs_error=float(abs(loss_a-loss_b)),
    context_gradient_max_abs_error=float((context_a.grad-context_b.grad).abs().max()),
    parameter_gradient_max_abs_errors=probe_params,
    classifier_output_gradient_l2_by_class=chunked.classifier[2].weight.grad.norm(dim=1).tolist(),
    untouched_checkpoint_sha256=ref(checkpoint_ref['path'])['sha256'],cuda_initialized=torch.cuda.is_initialized())
assert probe['loss_abs_error'] < 1e-5 and probe['context_gradient_max_abs_error'] < 1e-5
assert max(probe_params.values()) < 1e-5 and not torch.cuda.is_initialized()

result = dict(schema='european_treeloom_readonly_model_audit_v1',
    generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    comparison=ref(COMPARE), fit_job=ref(JOB), fit_completion=ref(FIT/'native/training_completed.json'),
    checkpoint=checkpoint_ref, checkpoint_fields={key:saved[key] for key in ('schema','protocol_id','epoch','optimizer_updates','settings','class_order','checkpoint_selection','TEST_truth_read')},
    saved_optimizer_group={k:v for k,v in saved['optimizer_state']['param_groups'][0].items() if k!='params'},
    model_parameters=param_evidence,train_targets=training,predictions=predictions,
    training_events=events,training_by_epoch=epochs,
    fit_python_source_checks=source_checks,
    fit_refs_containing_20261027=[item for item in source_refs if '20261027' in item['path']],
    decoder_probe=probe,audit_source=ref(Path(__file__).relative_to(ROOT)))
output=OUT/'evidence.v1.json'
with output.open('x') as stream:
    json.dump(result,stream,indent=2,sort_keys=True)
    stream.write('\n')
print(json.dumps(dict(output=ref(output.relative_to(ROOT)),checkpoint=checkpoint_ref,
    all_parameter_steps=sorted(set(value['step'] for value in param_evidence.values())),
    unchanged_parameter_names=[name for name,value in param_evidence.items() if not value['changed_elements']],
    target_counts={p:d['class_counts'] for p,d in training.items()},
    source_pin_mismatches=[item for item in source_checks if not item['matches']],
    output_head=param_evidence['classifier.2.weight'],decoder_probe=probe),indent=2))
