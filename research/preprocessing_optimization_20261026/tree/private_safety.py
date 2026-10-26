"""Adversarial ctypes boundary checks, with no malformed call allowed to hang."""
import json
from pathlib import Path
import numpy as np
import torch
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
torch.set_num_threads(1)
from research.preprocessing_optimization_20261026.tree import kernels
checks=[]
def reject(name,call):
    try:call()
    except (ValueError,RuntimeError,IndexError,TypeError):checks.append(name)
    else:raise AssertionError(name+' accepted malformed input')
p=np.array([-1,0,0],dtype=np.int64)
row=np.array([1,2],dtype=np.int64)
head=np.array([2,-1,-1],dtype=np.int32)
sib=np.array([-1,-1,1],dtype=np.int32)
reject('negative_root',lambda:kernels._tree_clades(-1,row,p,head,sib))
reject('out_of_range_root',lambda:kernels._tree_clades(3,row,p,head,sib))
reject('short_head',lambda:kernels._tree_clades(0,row,p,head[:1],sib))
reject('out_of_range_child',lambda:kernels._tree_clades(0,row,p,np.array([8,-1,-1],dtype=np.int32),sib))
reject('sibling_cycle',lambda:kernels._tree_clades(0,row,p,head,np.array([-1,2,1],dtype=np.int32)))
reject('out_of_range_sample',lambda:kernels._tree_clades(0,np.array([1,7]),p,head,sib))
reject('duplicate_sample',lambda:kernels._tree_clades(0,np.array([1,1]),p,head,sib))
reject('forest_cycle',lambda:kernels._forest_index(np.array([1,0,-1]),np.array([[0,1,2]])))
reject('cut_shape_mismatch',lambda:kernels.cut_index(p,np.zeros(2,dtype=np.float32),np.array([0])))
reject('cut_root_oob',lambda:kernels.cut_index(p,np.zeros(3,dtype=np.float32),np.array([8])))
reject('cut_representative_oob',lambda:kernels.cut_groups(p,np.zeros(3,dtype=np.float32),np.array([0,9,0]),np.array([[1,2]]),np.zeros(2,dtype=bool)))
reject('direction_group_oob',lambda:kernels._direction_layouts([np.array([0,2])],range(1),2))
reject('direction_group_negative',lambda:kernels._direction_layouts([np.array([0,-1])],range(1),2))
result=dict(status='pass',rejected=checks)
Path('research/preprocessing_optimization_20261026/tree/private_safety.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
