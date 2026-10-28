"""Read-only source and checkpoint-lineage audit; no model execution or CUDA."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/comparability_audit_20261028/model'
def read(path):return json.loads((ROOT/path).read_text())
def ref(path):
 p=ROOT/path
 return dict(path=str(path),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
  mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat())
def collect(value):
 if isinstance(value,dict):
  if 'path' in value and 'sha256' in value:yield {k:value[k] for k in ('path','sha256')}
  for child in value.values():yield from collect(child)
 elif isinstance(value,list):
  for child in value:yield from collect(child)
def methods(path):
 tree=ast.parse((ROOT/path).read_text())
 return {n.name:n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
a=Path('research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_execution.py')
b=Path('src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py')
x,y=methods(a),methods(b)
method_comparison={name:ast.dump(x[name])==ast.dump(y[name]) for name in sorted(x.keys()&y.keys())}
c=read(Path('experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json'))
lineage={}
for parent,reference in c['predictions']['treeloom'].items():
 actual=ref(Path(reference['path']))
 assert actual['sha256']==reference['sha256']
 product=read(Path(reference['path']))
 for key in ('checkpoint_ref','fit_completion_ref','prediction_wrapper_source_ref','static_executor_source_ref','recurrent_execution_source_ref','predictions_ref'):
  verified=ref(Path(product[key]['path']))
  assert verified['sha256']==product[key]['sha256']
 lineage[parent]=dict(product=reference,**{key:product[key] for key in ('checkpoint_ref','fit_completion_ref','prediction_wrapper_source_ref','static_executor_source_ref','recurrent_execution_source_ref')})
american_job=Path('.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt8.json')
aj=read(american_job)
american_checkpoint=Path('experiments/results/paper_american_20261024/treeloom_fit/native/exact_preprocessing_from_update281_memory_selected_adaptive2_attempt8/checkpoints/latest.pt')
ac=torch.load(ROOT/american_checkpoint,map_location='cpu',weights_only=True)
american_refs=list({v['path']:v for v in collect(aj)}.values())
fit=Path('experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v2/other/AP__all_train__treeloom_fit')
report=dict(schema='european_treeloom_source_lineage_review_v1',lineage=lineage,
 european_fit_receipt_files=[ref(fit/name) for name in ('result.json','actual_wait.json','native/training_completed.json','native/checkpoints/epoch_30.pt')],
 european_source_refs=[ref(path) for path in (a,b,
 Path('research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_stage.py'),
 Path('research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_observations.py'),
 Path('research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_gru.py'),
 Path('research/repairs/benchmark_full_vram_20261022/training/ap_fit_v2.py'),
 Path('src/arggnn/benchmarking/european_sampling.py'),
 Path('src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_history.py'),
 Path('research/reviews/TreeLoom_GPU_history_source_20261017_v1/stable_leaf_history_v1.py'),
 Path('research/benchmark_rebuild_20261024/american/treeloom_speed/stable_leaf_history_v2.py'),
 Path('research/benchmark_rebuild_20261024/american/treeloom_speed/parallel_history_window.py'))],
 training_vs_prediction_shared_method_AST_equal=method_comparison,
 reviewed_differences=dict(sites='Equivalent local alias bank=self.bank; h=bank.sample_index.shape[1] versus self.bank.sample_index.shape[1].',
  layout='Training uses128-hold blocks,32-haplotype decoder tiles and block_indices; prediction uses16-hold blocks,8-haplotype tiles and explicit equivalent level grouping.',
  decoder='Training accumulates tile CE sum/total labels and manual complete-context VJP; prediction copies all sampled row logits. Both use exact same classifier and full bidirectional GRU.',
  history='Training uses Oct17 sequential stable_leaf_history_v1 with128-hold windows; prediction uses Oct24 affine-prefix stable_leaf_history_v2 with1024-hold windows. Both implement the same recurrence with changed FP32 arithmetic order. Full-parent parity was not rerun.'),
 american=dict(job=ref(american_job),call_module=aj['call']['module'],checkpoint=ref(american_checkpoint),
  checkpoint_schema=ac['schema'],completed_updates=ac['complete_parent_updates'],settings=ac['settings'],
  classifier_shape=list(ac['model_state']['classifier.2.weight'].shape),
  model_parameter_count=sum(v.numel() for v in ac['model_state'].values()),
  source_refs_from_20261027=[r for r in american_refs if '20261027' in r['path']]),
 cuda_initialized=torch.cuda.is_initialized(),audit_source=ref(Path(__file__).relative_to(ROOT)))
assert not torch.cuda.is_initialized()
p=OUT/'source_review.v1.json'
with p.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(output=ref(p.relative_to(ROOT)),AST_equal=method_comparison,
 american_checkpoint=report['american']['checkpoint'],american_source_refs_20261027=len(report['american']['source_refs_from_20261027']),
 european_fit_receipt_files=report['european_fit_receipt_files']),indent=2))
