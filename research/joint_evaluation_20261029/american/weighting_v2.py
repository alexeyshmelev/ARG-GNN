"""Expected canonical-target weighting for uniformly shuffled parent windows."""
from .common import require


def parent_batch_weight(parent_windows, total_batches, batch_windows, parent_batches, targets):
    require(min(parent_windows, total_batches, batch_windows, parent_batches, targets) > 0, 'Positive weighting counts')
    require(parent_batches == (parent_windows + 15) // 16, 'Exact parent batch count including its tail')
    require(batch_windows <= min(16, parent_windows), 'Registered window minibatch bound')
    return parent_windows * total_batches / (parent_batches * batch_windows * targets)


def analytic_tests():
    """Enumerate each window's possible positions, including every partial tail."""
    scenarios = [(1,), (16,), (17,), (15, 16, 17, 31, 33), (7908, 7949, 7875, 7952)]
    reports = []
    for counts in scenarios:
        targets = 200000
        total_batches = sum((w + 15) // 16 for w in counts)
        errors = []
        old_errors = []
        for windows in counts:
            sizes = [min(16, windows - start) for start in range(0, windows, 16)]
            # A specific window lands in each size-S batch with probability S/W.
            coefficient = sum(size / windows * parent_batch_weight(windows, total_batches, size, len(sizes), targets)
                              for size in sizes) / total_batches
            errors.append(abs(coefficient - 1 / targets))
            old_coefficient = sum(size / windows * sum(counts) / (size * targets) for size in sizes) / total_batches
            old_errors.append(abs(old_coefficient - 1 / targets))
        require(max(errors) < 1e-18, 'Each canonical target has expected coefficient1/N across all updates')
        reports.append(dict(parent_windows=list(counts), total_batches=total_batches,
                            maximum_absolute_error=max(errors), old_maximum_absolute_error=max(old_errors)))
    require(reports[3]['old_maximum_absolute_error'] > 1e-7, 'Tail-batch regression detects old global formula')
    return reports
