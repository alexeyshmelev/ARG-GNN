"""Root-authorized bounded CPU constructor/resource interface qualification."""
import hashlib,json
from pathlib import Path
import torch
from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1
from arggnn.marker_resolution_memory_bounds_v2 import constructor_interface_v2,logical_bounds_v2
torch.manual_seed(101)
model=AnonymousCladeFlagControlGNNV1(hidden_dim=32,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True)
interface=constructor_interface_v2(model)
assert sum(p.numel() for p in model.parameters())==30788
bounds=logical_bounds_v2(14099,1026,20000)
assert bounds["head_checkpoint_selected_state_and_feature_bytes"]==6585600000
assert bounds["exported_H042_state_bytes"]==4*1026*952*96
assert bounds["H042_exported_state_channels"]==96
source="src/arggnn/marker_resolution_memory_bounds_v2.py"
print(json.dumps(dict(schema="H044_memory_constructor_interface_checks_v2",status="passed",
    source_ref=dict(path=source,sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()),
    interface=interface,base_parameters=30788,corrected_bounds=bounds,
    native_model_constructor_only=True,forward_or_observations_executed=False,
    actual_targets_checkpoint_runtime_guards_GPU_queries=False),sort_keys=True))
