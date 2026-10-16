"""CPU-only namespace-memo bootstrap; invokes the unchanged original scientific CLI."""
from pathlib import Path
import json,os,runpy,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.American_C3_VAL003_simulate_H020v3_source_20261016_v1 import root_packet_v1 as packet

def main():
    packet.require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="",
                   "Exact CPU-only root execution")
    packet.require(len(sys.argv)==7 and sys.argv[2]=="--execute"
        and sys.argv[3]=="--admission" and sys.argv[5]=="--expected-admission-sha256",
        "Exact unchanged original scientific CLI arguments")
    record=dict(path=sys.argv[4],sha256=sys.argv[6])
    spec=packet.read(record);packet.select_parent(spec["parent_id"]);pins=packet.frozen()
    stage=spec["stage"]
    packet.require(stage in packet.STAGES and sys.argv[1]=="arggnn."+Path(packet.PRODUCERS[stage]).stem
        and spec["source_ref"]==pins[packet.PRODUCERS[stage]]
        and spec["source_closure_refs"]==packet.source_closure(stage,pins)
        and spec["driver_ref"]==packet.ref(packet.SOURCE)
        and spec["worker_bootstrap_ref"]==packet.ref(__file__)
        and spec["consumer_memo_ref"]==packet.MEMO_REF
        and spec["consumer_namespace_memo_ref"]==packet.NAMESPACE_MEMO_REF
        and spec["original_v4_ref"]==packet.V4_REF
        and spec["GPU_authorized"] is False and spec["execution_authorized"] is True,
        "Separately admitted pinned worker bootstrap and original science")
    guard=packet.load_guard_v1(consumer=True)
    packet.require(guard._remaining_VAL_memo2.receipt["installed"] is True
        and guard._remaining_VAL_memo2.receipt["cross_operation_cache"] is False
        and guard._remaining_VAL_memo2.receipt["namespace_extension_schema"]=="resource_guard_consumer_namespace_memo_v1",
        "Pinned operation-local consumer memo2 installed")
    module=sys.argv[1]
    sys.argv=[packet.PRODUCERS[stage],*sys.argv[2:]]
    runpy.run_module(module,run_name="__main__")

if __name__=="__main__":main()
