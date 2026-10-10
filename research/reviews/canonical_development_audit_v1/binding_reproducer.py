"""Read-only synthetic probes of ending source checks and family lock checking."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');sys.path.insert(0,str(ROOT/'tests'))
from test_development_training_bindings import completed_facts
from arggnn.development_training_bindings import make_configs,check_live,sha
from arggnn.train_canonical_gru import select_family_configuration
TMP=ROOT/'.runtime/tmp/canonical_development_audit_v1/binding_probe';TMP.mkdir()
completion,path=completed_facts(TMP)
for stage,fact in completion['stage_manifests'].items():
    folder=TMP/stage;folder.mkdir()
    target=folder/'manifest.json';target.write_bytes(Path(fact['path']).read_bytes())
    fact['path']=str(target)
path.write_text(json.dumps(completion));settings=make_configs(completion,path,sha(path))['gru']
config=TMP/'config.json';config.write_text(json.dumps(settings));config_sha=sha(config)
check_live(settings,config,config_sha)
inferred=Path(completion['stage_manifests']['inferred']['path'])
inferred.write_text('{"mutated_source":true}')
check_live(settings,config,config_sha)
facts={'development_check_live_accepts_changed_inferred_manifest':True,'manifest_mutation_is_synthetic':True,'no_completion_or_data_loader_called':True}
settings=json.loads((ROOT/'configs/canonical_inferred_cohort64_pilot_v1.json').read_text())
rows=[{'configuration_id':c['id'],'training_seed':seed,'validation_nll':1.,'parameters':1,
       'protocol_sha256':'f'*64,'dataset_manifest_sha256':settings['dataset']['manifest_sha256'],
       'implementation_sha256':settings['implementation_sha256'],'configuration':{'id':c['id'],'hidden_dim':999,'dropout':.5,'learning_rate':1.}}
      for c in settings['configurations'] for seed in settings['training_seeds']]
selection=select_family_configuration(rows,settings)
facts['canonical_family_accepts_arbitrary_common_protocol_hash_and_wrong_record_configuration']=True
facts['canonical_selected']=selection['selected']
(ROOT/'research/reviews/canonical_development_audit_v1/binding_reproducer.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts,indent=2))
