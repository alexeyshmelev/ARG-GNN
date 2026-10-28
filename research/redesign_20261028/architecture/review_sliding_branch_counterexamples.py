"""Tiny independent semantic checks; no fitting, labels, real data or CUDA."""
from pathlib import Path
import hashlib
import json
import numpy as np
from research.sliding_graph_convolution_20261028.branch_joint_conv import Tree, branch_overlap, build_graph


def main():
    root = Path('/workspace-SR003.nfs2/ARG-GNN')
    source_path = root / 'research/sliding_graph_convolution_20261028/branch_joint_conv.py'
    expected = 'f837adcd36c9a8e13d893fed5e54c5d8dcace184196d5f53498fa5213a22c3ae'
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert source_hash == expected
    ids = ('sample0', 'sample1', 'sample2', 'sample3')
    target = Tree(np.array([4, 4, 5, 5, 6, 6, -1]),
                  np.array([0., 0., 0., 0., 1., 1., 3.]), np.arange(4), ids).validate()
    broad_source = Tree(np.array([4, 4, 4, 5, 5, -1]),
                        np.array([0., 0., 0., 0., 1., 3.]), np.arange(4), ids).validate()
    overlap = branch_overlap(target, broad_source)
    broad_weight = overlap[(4, 4)] / target.masses()[4]
    assert broad_weight == 1.
    old_target = Tree(np.array([4, 4, 4, 4, -1]),
                      np.array([0., 0., 0., 0., 1000.]), np.arange(4), ids).validate()
    segmented_source = Tree(np.array([4, 4, 5, 5, 6, 6, -1]),
                            np.array([0., 0., 0., 0., 10., 10., 1000.]), np.arange(4), ids).validate()
    lifetime = branch_overlap(old_target, segmented_source)
    recent = lifetime[(0, 0)] / old_target.masses()[0]
    ancient = lifetime[(0, 4)] / old_target.masses()[0]
    assert recent == .01 and ancient == .99
    distance_results = []
    for distance in (.1, .999999, 1.):
        graph = build_graph([target, target], [0., distance], radius=1.)
        relation = graph.relations[3]
        distance_results.append(dict(distance=distance, right_leaf_edges=len(relation.weight),
                                     right_leaf_weights=relation.weight.tolist()))
    assert distance_results[0]['right_leaf_weights'] == distance_results[1]['right_leaf_weights'] == [1.] * 4
    assert distance_results[2]['right_leaf_edges'] == 0
    wrong_ids = Tree(target.parent, target.age, target.leaves, (ids[1], ids[0], ids[2], ids[3]))
    rejected = False
    try:
        build_graph([target, wrong_ids], [0., .5], radius=1.)
    except ValueError:
        rejected = True
    assert rejected
    report = dict(schema='independent_sliding_branch_semantic_checks', source_sha256=source_hash,
                  broad_source=dict(target_descendants=[0, 1], source_descendants=[0, 1, 2], normalized_weight=broad_weight),
                  chronological_average=dict(recent_0_to_10_weight=recent, old_10_to_1000_weight=ancient),
                  lone_neighbor_distance=distance_results, mismatched_present_day_sample_ids_rejected=rejected,
                  labels_used=False, ancestry_fitting=False, GPU_used=False,
                  meaning='These are intended operator properties and a repaired contract, not failed algebra tests.')
    output = Path(__file__).with_name('SLIDING_BRANCH_COUNTEREXAMPLES.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
