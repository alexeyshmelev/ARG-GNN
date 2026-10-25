"""Small CUDA check of the repaired recurrent execution, without training."""
from pathlib import Path
import json
import os
import time
import subprocess

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
GPU = 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'
assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU
from setproctitle import setproctitle
setproctitle('AI Auto Research Pipeline Test')
import torch
from arggnn.benchmarking import european_treeloom as european
from research.repairs.benchmark_recovery_20261025.european.ap_prediction_chunked_gru import install_model_execution

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.device_count() == 1
model = european.build_model('ancestralpaths_paper200mb_native6').to('cuda').eval()
torch.manual_seed(18273)
x = torch.randn(2,2051,35,device='cuda').transpose(0,1)
h = torch.randn(2,2,32,device='cuda')
with torch.inference_mode():
    expected, expected_state = model.genomic_gru(x,h)
    install_model_execution(model)
    actual, actual_state = model.genomic_gru(x,h)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(actual_state,expected_state,atol=1e-5,rtol=1e-5)
    result = dict(status='passed',observed_unix=time.time(),physical_gpu=5,
        gpu_uuid=GPU,input_shape=list(x.shape),input_noncontiguous=not x.is_contiguous(),
        complete_connected_bidirectional_chunks=3,
        output_max_abs_error=float((actual-expected).abs().max()),
        final_state_max_abs_error=float((actual_state-expected_state).abs().max()),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training_performed=False,production_checkpoint_modified=False,
        limitation='Small synthetic CUDA recurrent smoke, not full-parent prediction or ancestry accuracy.')
(ROOT/'research/repairs/benchmark_recovery_20261025/ap_cuda_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
