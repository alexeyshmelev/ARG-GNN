"""Read-only TRAIN source memory profile; no cache writes, truth reads or GPU work."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import resource
import time
import psutil
import torch
from research.joint_evaluation_20261029.argmix.runtime import frozen, parent_item
from research.joint_evaluation_20261029.argmix.observations import ObservationSource
from research.joint_model_20261029.model.joint_model_v1 import prepare_observations, prepared_to_state
from research.joint_evaluation_20261029.american.state_codec_v2 import encode

BASE = Path('research/joint_evaluation_20261029/argmix/repair_20261030_memory_v1')
OUTPUT = BASE / 'single_source_profile.json'
assert not OUTPUT.exists()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
process = psutil.Process()
started = time.monotonic()
observations = []

def sample(stage, **extra):
    memory = process.memory_full_info()
    row = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(),
           'elapsed_seconds': time.monotonic() - started, 'rss_bytes': memory.rss,
           'pss_bytes': memory.pss, 'uss_bytes': memory.uss,
           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           **extra}
    observations.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)

sample('runtime_imported')
seal, protocol = frozen('research/joint_evaluation_20261029/argmix/freeze_v1.json')
items = {parent: parent_item(parent, protocol) for parent in ['published_recon_train_002', 'published_recon_train_004']}
sample('original_public_bindings_authenticated')
source = ObservationSource('published_recon_train_002', items['published_recon_train_002']['observation_ref'])
sample('largest_TRAIN_source_loaded', tree_bytes=source.ts.nbytes, nodes=source.ts.num_nodes,
       edges=source.ts.num_edges, trees=source.ts.num_trees, sites=source.ts.num_sites)
plans = source.plan()
sample('complete_source_plan', windows=len(plans), targets=len(source.records['record_ids']))
for index in [0, len(plans)//2, len(plans)-1]:
    packet = source.extract(plans[index])
    batch = prepare_observations(packet)
    payload = encode(prepared_to_state(batch), level=9)
    sample('representative_window_prepared', window_index=index, compressed_bytes=len(payload))
    del payload, batch, packet
    gc.collect()
source.check_stats()
sample('before_parent_replacement')
source = ObservationSource('published_recon_train_004', items['published_recon_train_004']['observation_ref'])
sample('second_large_parent_replaced', tree_bytes=source.ts.nbytes, nodes=source.ts.num_nodes,
       edges=source.ts.num_edges, trees=source.ts.num_trees, sites=source.ts.num_sites)
source.check_stats()
assert not torch.cuda.is_initialized()
result = {'schema': 'joint_argmix_single_source_memory_profile_v1', 'passed': True,
          'observations': observations, 'original_freeze_sha256': hashlib.sha256(Path('research/joint_evaluation_20261029/argmix/freeze_v1.json').read_bytes()).hexdigest(),
          'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'parents': list(items), 'public_bindings': items,
          'heldout_inputs_or_truth_opened': False, 'GPU_initialized': False,
          'cache_files_written': 0, 'original_files_modified': False}
with OUTPUT.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'result_path': str(OUTPUT), 'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}), flush=True)
