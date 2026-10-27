"""Scalar callback timing only; every existing resource check still executes."""
import copy
import threading


class ResourceTiming:
    def __init__(self):
        self.lock = threading.Lock()
        self.labels = {}
        self.calls = 0
        self.lock_wait_seconds = self.outer_callback_seconds = self.own_RSS_seconds = 0.
        self.maximum_outer_callback_seconds = 0.

    def record(self, label, lock_wait, outer, own_rss):
        label = label if type(label) is str else 'metadata_byte_reservation'
        with self.lock:
            if label not in self.labels and len(self.labels) >= 128:
                label = 'other_labels'
            row = self.labels.setdefault(label, dict(calls=0, outer_callback_seconds=0.,
                own_RSS_seconds=0., lock_wait_seconds=0., maximum_outer_callback_seconds=0.))
            self.calls += 1
            row['calls'] += 1
            for key, seconds in (('lock_wait_seconds', lock_wait),
                                 ('outer_callback_seconds', outer), ('own_RSS_seconds', own_rss)):
                row[key] += seconds
                setattr(self, key, getattr(self, key)+seconds)
            row['maximum_outer_callback_seconds'] = max(row['maximum_outer_callback_seconds'], outer)
            self.maximum_outer_callback_seconds = max(self.maximum_outer_callback_seconds, outer)

    def snapshot(self, *, labels=True):
        with self.lock:
            result = dict(calls=self.calls, lock_wait_seconds=self.lock_wait_seconds,
                outer_callback_seconds=self.outer_callback_seconds, own_RSS_seconds=self.own_RSS_seconds,
                maximum_outer_callback_seconds=self.maximum_outer_callback_seconds,
                checks_skipped=0, per_callback_IO_added=False,
                summed_thread_durations_are_not_wall_time=True)
            if labels:
                result['by_stage_label'] = copy.deepcopy(self.labels)
            return result
