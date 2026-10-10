"""Synthetic canonical synthesis caller/settings drift, without any fitting."""
import json,hashlib
from copy import deepcopy
from pathlib import Path
from arggnn.resources import ROOT
from arggnn.train_canonical_gru import select_family_configuration
from arggnn.canonical_gru import CanonicalTreeGNNGRU
p=ROOT/'configs/canonical_inferred_cohort64_pilot_v1.json'
original=p.read_bytes();expected=hashlib.sha256(original).hexdigest();settings=json.loads(original)
original_width=settings['configurations'][0]['hidden_dim']
settings['configurations'][0]['hidden_dim']=16
rows=[]
for spec in settings['configurations']:
 count=sum(v.numel() for v in CanonicalTreeGNNGRU(settings['input_dim'],settings['num_classes'],spec['hidden_dim'],settings['num_layers'],settings['temporal_layers'],spec['dropout']).parameters())
 for seed in settings['training_seeds']:
  rows.append({'configuration_id':spec['id'],'configuration':deepcopy(spec),'training_seed':seed,'validation_nll':1.,'parameters':count,'protocol_sha256':expected,'dataset_manifest_sha256':settings['dataset']['manifest_sha256'],'implementation_sha256':settings['implementation_sha256'],'source_artifact_sha256':settings['dataset']['source_artifact_sha256']})
facts={'synthetic_only':True,'original_config_sha256':expected,'source_config_unchanged':p.read_bytes()==original,'passed_expected_hash_of_real_original_config':True,'provided_settings_and_records_changed_coherently':True,'changed_field':'configurations[0].hidden_dim','original_width':original_width,'changed_width':16}
try:
 facts['selection']=select_family_configuration(rows,settings,expected,config_bytes=original)
 facts['accepted']=True
except (ValueError,RuntimeError,TypeError) as error:
 facts.update(accepted=False,rejection={'type':type(error).__name__,'message':str(error)})
(ROOT/'research/reviews/integrity_fix_reaudit_v1/coherent_settings_drift_post_repair.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts,indent=2))
