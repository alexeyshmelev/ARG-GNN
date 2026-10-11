"""Run the unchanged H038 CPU producer under the terminal storage guard."""
from pathlib import Path
import sys
sys.path.insert(0,"/workspace-SR003.nfs2/ARG-GNN")
import argparse,json,os
from arggnn import project_storage_terminal_scopes_v1 as accounting
from arggnn import prepare_expanded_full_panel_v1 as producer
SOURCE="research/reviews/expanded_full_panel_v1_root/prepare_expanded_terminal_root_v1.py"
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--expected-input-sha256",required=True);a=p.parse_args()
 ref=dict(path=a.input,sha256=a.expected_input_sha256);spec=accounting._bound(ref)
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU-only producer required")
 if accounting._ref(SOURCE) not in spec["storage"]["reservation"]["consumer_bindings"]:raise ValueError("Actual root wrapper not reserved")
 if spec["storage_adapter_ref"]!=accounting._ref(accounting.SOURCE):raise ValueError("Exact terminal adapter required")
 cfg=accounting._bound(spec["config_ref"])
 accounting.validate_storage_v9(spec,cfg,spec["storage"],source_ref=spec["source_ref"],config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
 result=producer.execute_parent_v1(cfg,spec["parent_id"],"all",spec,guard=lambda n=0:accounting.budget_check_v9(spec["storage"],boundary=True,required_write_bytes=int(n)))
 print(json.dumps(result),flush=True)
if __name__=="__main__":main()
