"""Pure metadata/synthetic checks; no generation, inference, GPU or truth reads."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from .validation_generation_v2 import generate_validation_parent
from .validation_sampling import build_validation_registry
from arggnn.benchmarking.european_recipes import load_recipe
from arggnn.benchmarking.european_sampling import build_sampled_registry


def need(ok,message):
    if not ok:raise ValueError(message)


def main():
    started=time.monotonic();root=Path('/workspace-SR003.nfs2/ARG-GNN');folder=Path(__file__).parent
    plan=json.loads((folder/'independent_validation_plan_v2.json').read_text())
    sample_plan=json.loads((folder/'validation_sampling_plan_v1.json').read_text());parent=plan['parents'][0]
    need(plan['execution_enabled'] is False and plan['jobs_started']==0,'Unlaunched immutable validation plan')
    raw=(root/'experiments/results/benchmark_campaign_20261018_workflow/release_v8/inherited/AP__ancestralpaths_european_train_000__ap_infer/native/observed_sample_manifest.json').read_bytes()
    need(hashlib.sha256(raw).hexdigest()=='2d86ba8414e5c111925dfd44138b1478be2a01da81a779f2b622bd4b949b525d','Exact public roster')
    rows=json.loads(raw)
    fake_ref=dict(path='synthetic_only/no_scientific_data',sha256='0'*64)
    inferred=dict(sample_index_to_inferred_tree_node_id=[dict(sample_index=row['sample_index'],inferred_tree_node_id=row['observed_node_id']) for row in rows],
        coverage_original_bp0=[0,200000000],local_to_original_offset_bp=0,
        site_join=[dict(native_site_id=i,local_position_0based=1000+10000*i,original_position_0based=1000+10000*i) for i in range(1000)])
    first=build_validation_registry(sample_plan,parent['parent_id'],rows,inferred,observation_ref=fake_ref)
    second=build_validation_registry(sample_plan,parent['parent_id'],rows,inferred,observation_ref=fake_ref)
    need(first==second and first['examples']==10000 and first['split']=='validation','Deterministic independent10000row sampler')
    # Reference algorithm is exercised only on synthetic metadata under a new
    # fictitious parent. No existing TEST parent, array or target is accessed.
    reference_plan=dict(sample_plan,schema='ap_paper_sampled_plan_v1',parents={parent['parent_id']:dict(split='test',cohort_quotas={c:2000 for c in ('present_day','Bronze_Age','BAA','Neo','Yam')})})
    reference=build_sampled_registry(reference_plan,parent['parent_id'],rows,inferred,observation_ref=fake_ref)
    need(first['records']==reference['records'] and first['query_identities']==reference['query_identities'],
         'Original scientific sampling algorithm unchanged on synthetic input')
    class DeliberateStop(Exception):pass
    def stop():raise DeliberateStop('Preparation only; no simulation execution')
    checks=[]
    try:
        generate_validation_parent(load_recipe('ancestralpaths_european'),dict(parent,seeds={k:parent['seeds'][k] for k in ('ancestry','mutations')}),
            folder/'prospective_validation'/parent['parent_id'],guard=lambda value:checks.append(value),
            execution_check=stop,write_headroom_bytes=2**30)
    except DeliberateStop:pass
    else:raise ValueError('Execution callback must prevent simulation')
    need(not checks and 'msprime' not in sys.modules and not (folder/'prospective_validation').exists(),
         'Execution callback stops before runtime import, guard, outputs or simulation')
    for path in folder.glob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            need(not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('exec','eval','compile')),
                 'No generated executable source loader')
    report=dict(schema='european_validation_preparation_CPU_check_v2',passed=True,
        deterministic10000_synthetic_rows=True,exact_original_sampling_algorithm=True,
        generator_stopped_before_scientific_runtime_or_output=True,original_TEST_scientific_data_opened=False,
        msprime_imported=False,simulation_started=False,inference_started=False,CUDA_initialized=False,
        source_AST_checks_passed=True,wall_seconds=time.monotonic()-started)
    with (folder/'validation_preparation_CPU_check_v2.json').open('x') as handle:json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
