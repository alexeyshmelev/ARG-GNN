"""Observed-only row selection and separately checked four-path TRAIN labels."""
from __future__ import annotations
from collections.abc import Sequence

import numpy as np

from . import QUERY_COHORTS


def non_baa_row_indices(query_cohorts: Sequence[str]) -> np.ndarray:
    """Select existing focal rows by observed cohort, preserving original order.

    No label is accepted by this API. Using the frozen six-path focal registry
    keeps a paired support while removing BAA before opening TRAIN supervision.
    """
    cohorts = tuple(query_cohorts)
    allowed = set(QUERY_COHORTS) | {"BAA"}
    if not cohorts or any(not isinstance(c, str) or c not in allowed for c in cohorts):
        raise ValueError("nonempty declared query cohorts required")
    indices = np.asarray([i for i, c in enumerate(cohorts) if c != "BAA"], dtype=np.int64)
    if not len(indices):
        raise ValueError("four-path support has no eligible non-BAA queries")
    return indices


def four_path_targets(native_labels_0based, query_cohorts: Sequence[str]) -> np.ndarray:
    """Validate already selected labels 0..3. Never map native paths 5 or 6.

    Call only after non-BAA observed row selection. Native labels 4 and 5 are
    invalid here even though their founder ancestries could be merged into 0/1.
    This is distinct from the six-output model's derived C4 probability sum.
    """
    cohorts = tuple(query_cohorts)
    indices = non_baa_row_indices(cohorts)
    if len(indices) != len(cohorts):
        raise ValueError("BAA query rows must be excluded before reading four-path labels")
    labels = np.asarray(native_labels_0based)
    if (labels.shape != (len(cohorts),) or labels.dtype.kind not in "iu"
            or not ((labels >= 0) & (labels < 4)).all()):
        raise ValueError("four-path labels must be integer native indices 0..3, without collapsing 4/5")
    return labels.astype(np.int64, copy=True)
