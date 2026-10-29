"""Authenticated canonical targets; isolated from observation construction."""
import hashlib
import numpy as np
from research.redesign_20261028.data.window_data import small_json, path_for, require, CLASSES


class TargetTable:
    def __init__(self, parent_id, binding):
        self.parent_id=parent_id
        metadata=small_json(binding['refs']['training_targets_ref'])
        require(metadata['split']=='train' and metadata['registry_ref']==binding['refs']['sampled_registry_ref'],
                'Canonical TRAIN label/registry binding')
        require(tuple(metadata['native_path_names'])==CLASSES,'Exact native six-class order')
        path=path_for(metadata['arrays_ref']);raw=path.read_bytes()
        require(len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()==metadata['arrays_ref']['sha256'],
                'Authenticated bounded canonical label array')
        with np.load(path,allow_pickle=False) as archive:
            self.labels=archive['native_labels_0based'].astype(np.int64)
        require(self.labels.shape==(binding['canonical_examples'],) and np.isin(self.labels,np.arange(6)).all(),
                'Every canonical row has one valid native label')

    def select(self, example_indices):
        index=np.asarray(example_indices,dtype=np.int64)
        require(index.ndim==1 and ((index>=0)&(index<len(self.labels))).all(),'Canonical target index bounds')
        return self.labels[index].copy()
