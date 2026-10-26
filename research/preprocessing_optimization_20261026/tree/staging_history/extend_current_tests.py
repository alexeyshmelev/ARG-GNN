from pathlib import Path
root=Path('/workspace-SR003.nfs2/ARG-GNN/research/preprocessing_optimization_20261026/tree')
p=root/'test_parity.py'
s=p.read_text()
s=s.replace('OUT = Path(', '''from research.repairs.benchmark_recovery_20261020.runtime import observations as current_observations
from research.repairs.benchmark_recovery_20261020.runtime import episodes as current_episodes
from research.repairs.benchmark_recovery_20261020.runtime import cuts as current_cut
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import indices_v1 as current_indices
OUT = Path(''')
s=s.replace("    p, s, tm = packed(trees)\n    return SimpleNamespace", "    p, s, tm = packed(trees)\n    depth=np.concatenate([original_layout._layout_with_isolated_samples(tr)[5] for tr in trees])\n    return SimpleNamespace(downward_levels=tuple(torch.from_numpy(level) for level in kernels._level_arrays(depth)[1:]),")
s=s.replace('SimpleNamespace(downward_levels=tuple(torch.from_numpy(level) for level in kernels._level_arrays(depth)[1:]),(parent=', 'SimpleNamespace(downward_levels=tuple(torch.from_numpy(level) for level in kernels._level_arrays(depth)[1:]),parent=')
s=s.replace("    increment('cut_complete')", """    active, active_meta = current_cut.prepare_cut_plan_v1(b, **kwargs)
    exact(active.tensors(), new.tensors(), 'current_cut.tensors')
    exact(active_meta, new_meta, 'current_cut.metadata')
    increment('cut_complete')
    increment('current_cut_complete')""")
s=s.replace("    compare_cut(bank(trees, spans), isolated)", """    b=bank(trees, spans)
    active_arrays, active_meta = current_episodes.build_clade_episode_arrays_v1(p,s,spans,
        _forest_index=lambda parent,samples: current_indices.forest_for_bank(b,parent,samples,isolated))
    exact((active_arrays,active_meta), episodes.build_clade_episode_arrays_v1(p,s,spans,**kwargs), 'current_episodes')
    active_obj, active_meta = current_episodes.build_episodes(b, max_payload_bytes=original_episodes.DEFAULT_PAYLOAD_CAP, **kwargs)
    new_obj, new_meta = episodes.build_episodes(b, max_payload_bytes=original_episodes.DEFAULT_PAYLOAD_CAP, **kwargs)
    for field in dataclasses.fields(active_obj):
        a,bvalue=getattr(active_obj,field.name),getattr(new_obj,field.name)
        if field.name=='flags':
            exact(a.node_flags,bvalue.node_flags,'current_episode_object.flags')
        else:exact(a,bvalue,'current_episode_object.'+field.name)
    exact(active_meta,new_meta,'current_episode_object.metadata')
    increment('current_episodes_complete')
    compare_cut(b, isolated)""")
s=s.replace("    increment('encoded_complete')", """    active, active_meta = current_observations.encode_block(*args, num_classes=3)
    for f in dataclasses.fields(active): exact(getattr(active, f.name), getattr(new, f.name), 'current_encoded.'+f.name)
    exact(active_meta,new_meta,'current_encoded.metadata')
    increment('encoded_complete')
    increment('current_encoded_complete')""")
p.write_text(s)
p=root/'cut.py'
s=p.read_text().replace('_fingerprint=_original._fingerprint','from research.reviews.workflow_progress_diagnosis_20261018.treeloom.indices_v1 import fingerprint as _fingerprint')
p.write_text(s)
