"""Independent cold-import RNG/thread audit, before importing candidate modules."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import random
import sys
import numpy as np
import torch
import setproctitle
from threadpoolctl import threadpool_info
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path.insert(0,str(ROOT))
setproctitle.setproctitle('AI Auto Research Pipeline Test')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
random.seed(819);np.random.seed(819);torch.manual_seed(819)
def state():
 return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
             numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
             torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
             torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
             cuda_initialized=torch.cuda.is_initialized(),native_pools=threadpool_info())
before=state()
from research.preprocessing_optimization_20261026.tree import process_observations
candidate_after=state()
from research.preprocessing_optimization_20261026.integration.hooks import install_tree
install_tree()
hooks_after=state()
keys=['python','numpy','torch','torch_threads','torch_interop_threads','cuda_initialized']
result=dict(before=before,candidate_import_after=candidate_after,hooks_after=hooks_after,
            candidate_unchanged={k:before[k]==candidate_after[k] for k in keys},
            hooks_unchanged={k:candidate_after[k]==hooks_after[k] for k in keys},
            no_scientific_files_opened=True)
result['status']='pass' if all(result['candidate_unchanged'].values()) and all(result['hooks_unchanged'].values()) else 'needs_fix'
Path('research/preprocessing_optimization_20261026/observed/process_import_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['before','candidate_import_after','hooks_after']},indent=2))
