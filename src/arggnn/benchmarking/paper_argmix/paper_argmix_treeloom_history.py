"""Qualified affine-prefix stable-leaf history for the European panel widths.

Frozen theta-to-length/rho/source transforms and original global CPU pooling
are retained. The shared bounded VJP keeps cross-window carry connected.
"""
from __future__ import annotations

import torch
from . import paper_argmix_generator as g

SPEED_SOURCE_REFS=(
 {'path':'research/benchmark_rebuild_20261024/american/treeloom_speed/parallel_history_window.py',
  'sha256':'adc8361e4cfda47f00155c8cb7d2eb01473194d237ee7579b89a590d1bb54b4e'},
 {'path':'research/benchmark_rebuild_20261024/american/treeloom_speed/stable_leaf_history_v2.py',
  'sha256':'891dc3a3994c61246191b36e548657ac691e83eb432ca6ddb8cb4437934ce21e'})
for source_ref in SPEED_SOURCE_REFS:
    g.verify_ref(source_ref)
from research.benchmark_rebuild_20261024.american.treeloom_speed.stable_leaf_history_v2 import _Direction,NullProfileV1


def stable_leaf_history(memory,forcing,theta,spans,plan,*,device,holds_per_window=1024,resource_check,profile=None):
    offsets=tuple(plan.group_offsets)
    g.need(memory.arm=='stable_leaf_memory' and forcing.device.type==theta.device.type==spans.device.type=='cpu' and
           forcing.dtype==theta.dtype==spans.dtype==torch.float32 and forcing.shape[1]==8 and
           type(holds_per_window) is int and 1<=holds_per_window<=1024 and
           plan.haplotypes in (952,3222) and len(offsets)==plan.native_holds+1 and
           offsets[0]==0 and offsets[-1]==len(forcing) and
           plan.leaf_groups.shape==(plan.native_holds,plan.haplotypes) and
           all(0<b-a<=plan.haplotypes for a,b in zip(offsets,offsets[1:])),
           'Complete original European FP32 cut-group partition and bounded history window required')
    profile=profile or NullProfileV1()
    def direction(reverse):
        ell=100.+(100000.-100.)*theta.sigmoid()
        rho=torch.exp(-spans/ell)
        source=-torch.expm1(-spans/ell)
        return _Direction.apply(forcing,rho,source,plan.leaf_groups,offsets,reverse,
                                holds_per_window,torch.device(device),resource_check,profile)
    left,right=direction(False),direction(True)
    with profile.stage('history_original_CPU_pool'):
        leaf=.5*(left+right)
        read=memory._mean(leaf.flatten(0,1),plan.leaf_groups.flatten(),plan.group_counts,forcing.shape[0])
        return read-forcing
