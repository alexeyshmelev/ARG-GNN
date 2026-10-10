"""CPU-only synthetic config-binding reproducer. Source project_env.sh first.
Run once in a fresh ARGGNN_AUDIT_TMP directory inside the project. No Git subprocess.
Production count validator is replaced only to permit a one-parent fixture.
"""
from pathlib import Path
import json, os, sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path.insert(0,str(ROOT/'tests'))
from test_anchor_temporal_baselines import prepared
from test_train_anchor import protocol
import arggnn.train_anchor_development as runner
BASE=Path(os.environ.get('ARGGNN_AUDIT_TMP', str(ROOT/'.runtime/tmp/development_integration_audit_v1/candidate_config_drift_repeat'))).resolve()
if ROOT not in BASE.parents:
    raise ValueError('Audit writes must remain inside the project')
BASE.mkdir(parents=True)
data=BASE/'data';data.mkdir();manifest=data/'manifest.json';manifest.write_text('{}')
pack=prepared(anchors=8);groups={}
for split in ('train','val'):
    labels=torch.arange(14).reshape(7,2).remainder(4)
    tile=SimpleNamespace(bank=pack.bank,labels=labels,parent_position_indices=torch.arange(7))
    case=SimpleNamespace(tiles=[tile],labels=labels,positions=pack.bank.positions,query_node_ids=pack.bank.query_node_ids,metadata={'num_classes':4,'replicate':0})
    groups[split]=[case]
dataset=SimpleNamespace(manifest_sha256=runner.sha(manifest),cases=lambda split:groups[split])
settings=protocol();settings.update(num_classes=4,training_seeds=[101,202,303],models=['plain'],model_specs={'plain':settings['model_specs']['plain']},model_learning_rates={'plain':.001},boundary_dataset_manifest_sha256=runner.sha(manifest),source_dataset_manifest_sha256='c'*64)
config=BASE/'config.json';config.write_text(json.dumps(settings));initial_hash=runner.sha(config)
def load_and_change(*_args,**_kwargs):
    changed=dict(settings);changed['hidden_dim']=16;config.write_text(json.dumps(changed));return dataset
original_path=runner.project_path
def audit_project_path(value):
    path=Path(value)
    if str(path).startswith('checkpoints/'):
        return original_path(BASE/'checkpoints')
    return original_path(value)
argv=['audit','--config',str(config),'--expected-config-sha256',initial_hash,'--data',str(data),'--output',str(BASE/'run'),'--device','cpu']
with patch.object(sys,'argv',argv),patch.object(runner,'checked_development_protocol',return_value=settings),patch.object(runner,'load_boundary_dataset',side_effect=load_and_change),patch.object(runner,'require_storage',return_value=0),patch.object(runner.subprocess,'check_output',return_value='AUDIT_GIT_DISABLED\n'),patch.object(runner,'project_path',side_effect=audit_project_path):
    runner.main()
result=json.loads((BASE/'run/results.json').read_text());copied=json.loads((BASE/'run/protocol.json').read_text());model=result['models']['plain']['model_configuration']
assert copied['hidden_dim']==16 and model['hidden_dim']==8
fact={'initial_required_config_sha256':initial_hash,'saved_results_protocol_sha256':result['protocol_sha256'],'copied_protocol_width':copied['hidden_dim'],'actual_fitted_model_width':model['hidden_dim'],'completed_despite_midload_config_mutation':True,'git_subprocess_disabled':True,'production_count_validator_replaced_for_tiny_synthetic_fit':True}
(BASE/'reproducer_facts.json').write_text(json.dumps(fact,indent=2)+'\n')
print(json.dumps(fact,indent=2))
