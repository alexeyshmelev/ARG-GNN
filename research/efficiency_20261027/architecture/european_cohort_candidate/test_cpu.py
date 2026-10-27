"""Bounded CPU cohort binding and initial algebra check, without scientific truth."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import torch
from .model import build_model, CONDITION, COHORTS, PARAMETERS, PANEL_BINDING, need
from .observations import cohort_tensor
from .execution import CohortStagedExecution
from research.reviews.frozen_benchmark_training_ready_20261017_v1.european_frozen_treeloom_v1 import FrozenEuropeanTreeLoomV1
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_execution import EuropeanStagedExecutionV1


def main():
    started=time.monotonic();torch.set_num_threads(1)
    root=Path('/workspace-SR003.nfs2/ARG-GNN');folder=Path(__file__).parent
    path=root/'experiments/results/benchmark_campaign_20261018_workflow/release_v8/inherited/AP__ancestralpaths_european_train_000__ap_infer/native/observed_sample_manifest.json'
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    need(len(raw)<2**20 and digest=='2d86ba8414e5c111925dfd44138b1478be2a01da81a779f2b622bd4b949b525d','Frozen public TRAIN sample manifest')
    rows=json.loads(raw);labels=torch.tensor([-1 if row['observed_reference_class'] is None else row['observed_reference_class'] for row in rows])
    ages=torch.tensor([row['age_generations'] for row in rows],dtype=torch.float32)
    queries=torch.tensor([row['sample_index'] for row in rows if row['role']=='query'])
    nodes=[row['observed_node_id'] for row in rows]
    cohorts=cohort_tensor(rows,reference_class=labels,sample_age=ages,query_columns=queries,observed_nodes=nodes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101);old=FrozenEuropeanTreeLoomV1(condition=PANEL_BINDING).float()
    new=build_model();old.output_classes=new.output_classes
    old_state=old.state_dict();new_state=new.state_dict()
    need(old_state.keys()==new_state.keys(),'Exact original state tensor names')
    for key,value in old_state.items():
        candidate=new_state[key][:,:38] if key=='leaf_projection.0.weight' else new_state[key]
        need(torch.equal(value,candidate),'Identical original fresh weights: '+key)
    need(bool((new.leaf_projection[0].weight[:,38:]==0).all()),'New cohort weights start at zero')
    h=952;t=2;n=(h+1)*t
    parent=torch.arange(n)//(h+1)*(h+1)+h
    parent[h::h+1]=-1
    sample=torch.stack((torch.arange(h),torch.arange(h)+h+1))
    bank=SimpleNamespace(parent=parent,node_time=torch.cat((ages,torch.tensor([4000.]),ages,torch.tensor([4000.]))),
        sample_index=sample,reference_class=labels,sample_age=ages,
        edge_geometry=torch.zeros(n,4),branch_support_true=torch.zeros(n,5))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70027);site_leaf=torch.randn(t,h,32)*.1
    old_executor=SimpleNamespace(model=old)
    original=EuropeanStagedExecutionV1._base(old_executor,bank,site_leaf)
    cohort_executor=SimpleNamespace(model=new,cohort_device=cohorts)
    candidate=CohortStagedExecution._base(cohort_executor,bank,site_leaf)
    errors=[float((a-b).abs().max()) for a,b in zip(original,candidate)]
    need(all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(original,candidate)),
         'Nested original initial base/gates/degree math within declared FP32 tolerance')
    loss=candidate[0].square().mean();loss.backward()
    cg=new.leaf_projection[0].weight.grad[:,38:]
    need(bool(torch.isfinite(cg).all()) and bool((cg.abs().sum(0)>0).all()),'Every cohort channel has connected nonzero gradient')
    query_baa=next(i for i,row in enumerate(rows) if row['observed_cohort']=='BAA')
    need(labels[query_baa]==-1 and cohorts[query_baa,COHORTS.index('BAA')]==1,'BAA cohort does not become an ancestry donor')
    with torch.no_grad():
        new.leaf_projection[0].weight[:,38+COHORTS.index('BAA')].fill_(.1)
        altered=new.base_from_sites(bank,site_leaf,cohorts)[0]
    changed=(altered-candidate[0]).abs().sum(-1)>0
    expected=torch.zeros(n,dtype=torch.bool)
    baa=torch.tensor([row['observed_cohort']=='BAA' for row in rows])
    expected[sample[:,baa].flatten()]=True
    need(torch.equal(changed,expected),'Controlled cohort weights affect exactly all sampled BAA occurrences before propagation')
    for name in ('_sites','_first','_forcing','_finite_own_episode','_second','context_forward'):
        need(getattr(CohortStagedExecution,name) is getattr(EuropeanStagedExecutionV1,name),'Unchanged complete execution stage '+name)
    for source in folder.glob('*.py'):ast.parse(source.read_text())
    report=dict(schema='european_cohort_bounded_CPU_test_v1',passed=True,condition=CONDITION,
        sample_manifest_ref=dict(path=str(path.relative_to(root)),sha256=digest),
        samples=952,queries=686,cohort_counts=dict(Counter(row['observed_cohort'] for row in rows)),
        parameters=PARAMETERS,parameter_tensors=49,initial_original_parameters_bitwise_equal=True,
        new_cohort_weights_initially_zero=True,base_gate_degree_max_abs_errors=errors,
        tolerance=dict(atol=1e-6,rtol=1e-6),all8_cohort_gradients_nonzero=True,
        cohort_gradient_l2=float(cg.norm()),controlled_BAA_change_all28_occurrences=True,
        original_execution_stages_inherited_without_changes=True,synthetic_tree_holds=2,
        CUDA_initialized=torch.cuda.is_initialized(),targets_opened=False,full_parent_prepared=False,
        full_model_or_training_parity_claimed=False,wall_seconds=time.monotonic()-started,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    need(not report['CUDA_initialized'],'CPU-only test')
    output=folder/'CPU_test_v1.json'
    with output.open('x') as handle:json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
