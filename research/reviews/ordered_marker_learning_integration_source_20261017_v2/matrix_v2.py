"""Original native-hold RAM scoring transport, explicit H058 complete-product barrier."""
from __future__ import annotations
import numpy as np
from arggnn.H040_latent_clade_execution_v1 import RAMMatrix, array_identity
from .protocol_v2 import HYPOTHESIS, ARMS, PARAMETERS, VAL, PASSES, require


class OrderedMarkerRAMMatrixV2(RAMMatrix):
    def __init__(self,output,reader,parents,banks,cfg,*,checkpoint_groups,arms):
        super().__init__(output,reader,parents,banks,cfg)
        self.arms=tuple(arms)
        require(self.arms == ARMS,
            "Exactly96 new placement-arm products")
        self.expected={(at,a,p) for at in PASSES for a in self.arms for p in VAL}
        require(set(parents)==set(banks)==set(VAL)
            and set(checkpoint_groups)=={str(p) for p in PASSES}
            and all(set(checkpoint_groups[str(p)])==set(self.arms) for p in PASSES),
            "All six fixed checkpoints and exact original eight VAL parents")
        self.checkpoint_groups=checkpoint_groups
        self.barrier_ref=None

    def seal(self,at,arm,pid,value,checkpoint):
        key=(at,arm,pid)
        require(key in self.expected and key not in self.values and not self.verified
            and checkpoint==self.checkpoint_groups[str(at)][arm], "Exact unique scheduled checkpoint product")
        bank,parent=self.banks[pid],self.parents[pid]
        planned=len(bank.spans)*686*4*4
        require(self.bytes+planned<=self.cfg["RAM_prediction_cap_bytes"], "Complete native RAM matrix preflight")
        value=np.array(value,dtype=np.float32,copy=True,order="C")
        require(value.shape==(len(bank.spans),686,4) and np.isfinite(value).all()
            and np.all(value>=0) and np.allclose(value.sum(-1),1,rtol=1e-5,atol=1e-6),
            "Complete original FP32 probability products")
        value.setflags(write=False)
        geometry={k:array_identity(parent[k]) for k in
            ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")}
        geometry["readout_hold"]=array_identity(bank.readout_hold.numpy())
        packet=dict(schema="ordered_marker_native_hold_RAM_seal_v2",hypothesis_id=HYPOTHESIS,pass_index=at,
            arm=arm,parent_id=pid,checkpoint_ref=checkpoint,probability=array_identity(value),
            geometry=geometry,parameters=PARAMETERS[arm],prediction_array_durable=False,
            targets_opened_before_seal=False,inference_backend=self.cfg["evaluation_backend"])
        record=self.output.save("%s.pass_%02d.%s.RAM_seal.json"%(arm,at,pid),packet)
        self.bytes+=value.nbytes
        self.values[key],self.records[key],self.packets[key]=value,record,packet
        self.lookup[record["path"]]=key

    def verify(self):
        require(not self.verified and set(self.values)==self.expected, "Every prescribed product before VAL truth")
        for key,value in self.values.items():
            require(not value.flags.writeable and array_identity(value)==self.packets[key]["probability"]
                and self.reader.decode_json(self.records[key])==self.packets[key],
                "Frozen probability bytes and actual JSON seal unchanged")
        required_new={(at,a,p) for at in PASSES for a in ARMS for p in VAL}
        self.barrier_ref=self.output.save("all_%d_ordered_marker_VAL_RAM_sealed_before_truth.json"%len(self.expected),dict(
            schema="ordered_marker_full_VAL_pretruth_barrier_v2",hypothesis_id=HYPOTHESIS,arms=list(self.arms),
            checkpoint_passes=list(PASSES),VAL_parent_ids=list(VAL),
            complete_native_RAM_prediction_count=len(self.expected),native_probability_bytes=self.bytes,
            all96_new_products_sealed_before_truth=required_new.issubset(self.values),
            all_required_products_sealed_before_truth=True,optimization_performed=False,
            inference_backend=self.cfg["evaluation_backend"],seals=[self.records[k] for k in sorted(self.records)]))
        self.verified=True
        return self.barrier_ref
