"""External exact half-open ancestry joins; held-out truth only after seals."""
import numpy as np
from .common import CLASSES, BASE, read, load_npz, require

TRUTH_KEYS = {'intervals_bp', 'query_sample_indices', 'source_classes_0based', 'parent_id', 'class_order'}


def join(records, truth):
    require(set(truth) == TRUTH_KEYS and np.array_equal(truth['parent_id'], records['parent_id']) and
            truth['class_order'].tolist() == list(CLASSES), 'Exact external same-parent native3 truth')
    intervals, samples, labels = truth['intervals_bp'], truth['query_sample_indices'], truth['source_classes_0based']
    require(intervals.dtype == np.float64 and intervals.ndim == 2 and intervals.shape[1] == 2 and len(intervals) > 0 and
            np.isfinite(intervals).all() and intervals[0, 0] == 0 and intervals[-1, 1] == 30000000 and
            (intervals[:, 1] > intervals[:, 0]).all() and np.array_equal(intervals[:-1, 1], intervals[1:, 0]) and
            samples.dtype == np.int64 and np.array_equal(samples, np.arange(600, 1600)) and
            labels.dtype == np.uint8 and labels.shape == (len(intervals), 1000), 'Complete canonical American ancestry domain')
    for start in range(0, len(labels), 32768):
        require((labels[start:start + 32768] < 3).all(), 'Native3 ancestry classes')
    positions, columns = records['positions_bp0'], records['sample_indices']
    require(((positions >= 0) & (positions < 30000000)).all() and ((columns >= 600) & (columns < 1600)).all(), 'No extrapolation or reference targets')
    return labels[np.searchsorted(intervals[:, 1], positions, side='right'), columns - 600].astype(np.int64)


def train_targets(parent_id, protocol, records):
    require(protocol['parents'][parent_id]['split'] == 'train', 'TRAIN truth boundary rejects held-out parent before any private read')
    bindings = read(protocol['private_TRAIN_bindings_ref'])['parents']
    require(set(bindings) == {p for p, row in protocol['parents'].items() if row['split'] == 'train'}, 'Exact TRAIN-only binding set')
    return join(records, load_npz(bindings[parent_id], cap=8 * 2**30))
