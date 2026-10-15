"""Concrete exclusive artifact writer; no reads/resume or operational guard."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os

def ref(path):
    p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

class DirectoryArtifactWriterV1:
    def __init__(self,directory):self.directory=Path(directory)
    def _json(self,name,value):
        p=self.directory/name
        with p.open("x") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
        return ref(p)
    def begin_run(self,protocol):
        self.directory.mkdir(parents=True,exist_ok=False)
        return self._json("protocol.executed.json",protocol)
    def save_initial(self,seed,state,metadata):
        import torch
        p=self.directory/("seed%d.common_initial.pt"%seed)
        with p.open("xb") as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        return self._json("seed%d.common_initial.json"%seed,dict(state_ref=ref(p),**metadata))
    def save_final(self,seed,arm,state,metadata,history):
        import torch
        p=self.directory/("seed%d.%s.final_epoch100.pt"%(seed,arm))
        with p.open("xb") as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        return self._json("seed%d.%s.training.json"%(seed,arm),dict(state_ref=ref(p),history=history,**metadata))
    def seal_predictions(self,scores,metadata):
        import numpy as np
        p=self.directory/"primary_all_heldout_predictions.npy"
        with p.open("xb") as f:np.save(f,scores.numpy(),allow_pickle=False);f.flush();os.fsync(f.fileno())
        seal=dict(schema="TGB_trade_H051_primary_all_country_predictions_sealed_v1",
            sealed_utc=datetime.now(timezone.utc).isoformat(),prediction_ref=ref(p),**metadata)
        return self._json("prediction_seal.json",seal)
    def write_fit_receipt(self,receipt):return self._json("fit_completed.json",receipt)
    def write_evaluation(self,split,result):return self._json("evaluation_%s.json"%split,result)
