"""Tiny allocation models for admission order; no scientific backend or fit claim."""
from collections import OrderedDict
import unittest

from arggnn.benchmarking import treeloom_training as training


class ParentMemoryAdmissionTest(unittest.TestCase):
    def test_cold_miss_evicts_before_prepare_and_preserves_fitting_entries(self):
        usage = [30]
        class Bundle:
            def __init__(self, size):
                self.size = size
                usage[0] += size
            def __del__(self):
                usage[0] -= self.size
        cache = OrderedDict((name, (Bundle(20), {}, 20)) for name in ("old_a", "old_b"))
        memory = training._memory_envelope(
            max_rss_bytes=100, preparation_headroom_bytes=50,
            training_headroom_bytes=40, observation_cache_bytes=40)
        events, preparations = [], []
        def prepare():
            # Both old entries total40. One must go *before* any cold allocation;
            # the other remains useful. Preparation peaks48 above current RSS.
            preparations.append(tuple(cache))
            self.assertEqual(tuple(cache), ("old_b",))
            self.assertEqual(usage[0], 50)
            self.assertLessEqual(usage[0] + 48, memory["max_rss_bytes"])
            return Bundle(20), {"parent_id": "new"}
        kwargs = dict(observation_cache_bytes=40, memory_envelope=memory,
                      rss_bytes=lambda: usage[0], tensor_bytes=lambda b: b.size,
                      resource_check=lambda *args: None,
                      event=lambda name, value: events.append((name, value)))
        bundle, metadata = training._load_cached_parent("new", cache, prepare, **kwargs)
        self.assertEqual(usage[0], 70)
        self.assertEqual(tuple(cache), ("old_b", "new"))
        again, _ = training._load_cached_parent(
            "new", cache, lambda: self.fail("Cache hit must reuse the complete bank"), **kwargs)
        self.assertIs(again, bundle)
        self.assertEqual(preparations, [("old_b",)])
        self.assertEqual(len(events), 1)
        # The active bundle remains pinned by the numerical caller. Admission
        # evicts only another bank before truth/autograd, retaining the active one.
        training._admit_parent_headroom(
            cache, memory_envelope=memory, stage="training",
            rss_bytes=lambda: usage[0], resource_check=lambda *args: None,
            event=lambda name, value: events.append((name, value)), keep_parent="new")
        self.assertEqual(tuple(cache), ("new",))
        self.assertEqual(usage[0], 50)
        self.assertIs(cache["new"][0], bundle)
        self.assertEqual(metadata["parent_id"], "new")
        self.assertEqual(events[-1][1]["admission_stage"], "training")

    def test_retained_allocator_RSS_denies_loader_after_exhausting_cache(self):
        cache = OrderedDict(old=(object(), {}, 10))
        calls = []
        memory = dict(max_rss_bytes=100, preparation_headroom_bytes=20)
        with self.assertRaisesRegex(training.c.ContractError, "larger admitted envelope"):
            training._load_cached_parent(
                "new", cache, lambda: calls.append("loaded"),
                observation_cache_bytes=50, memory_envelope=memory,
                # Freed tensors need not return allocator arenas to the OS.
                rss_bytes=lambda: 95, tensor_bytes=lambda b: 10,
                resource_check=lambda *args: None, event=lambda *args: None)
        self.assertEqual(calls, [])
        self.assertEqual(len(cache), 0)

    def test_no_cache_and_pinned_active_bank_cannot_fake_headroom(self):
        memory = dict(max_rss_bytes=100, preparation_headroom_bytes=30,
                      training_headroom_bytes=30)
        calls = []
        with self.assertRaises(training.c.ContractError):
            training._load_cached_parent(
                "new", OrderedDict(), lambda: calls.append("loaded"),
                observation_cache_bytes=0, memory_envelope=memory,
                rss_bytes=lambda: 80, tensor_bytes=lambda b: 0,
                resource_check=lambda *args: None, event=lambda *args: None)
        self.assertEqual(calls, [])
        cache = OrderedDict(active=(object(), {}, 50))
        with self.assertRaises(training.c.ContractError):
            training._admit_parent_headroom(
                cache, memory_envelope=memory, stage="training",
                rss_bytes=lambda: 80, resource_check=lambda *args: None,
                event=lambda *args: None, keep_parent="active")
        self.assertEqual(tuple(cache), ("active",))

    def test_cache_cap_and_post_preparation_RSS_are_independent_checks(self):
        cache = OrderedDict(old=(object(), {}, 15))
        memory = dict(max_rss_bytes=100, preparation_headroom_bytes=50)
        bundle = object()
        training._load_cached_parent(
            "new", cache, lambda: (bundle, {}),
            observation_cache_bytes=25, memory_envelope=memory,
            rss_bytes=lambda: 30, tensor_bytes=lambda b: 20,
            resource_check=lambda *args: None, event=lambda *args: None)
        self.assertEqual(tuple(cache), ("new",))
        reads = iter((30, 101))
        with self.assertRaisesRegex(training.c.ContractError, "preparation RSS cap"):
            training._load_cached_parent(
                "too_large", OrderedDict(), lambda: (object(), {}),
                observation_cache_bytes=25, memory_envelope=memory,
                rss_bytes=lambda: next(reads), tensor_bytes=lambda b: 20,
                resource_check=lambda *args: None, event=lambda *args: None)

    def test_memory_spec_rejects_missing_space_and_claims_no_analytical_peak(self):
        for change in (dict(max_rss_bytes=0), dict(preparation_headroom_bytes=-1),
                       dict(training_headroom_bytes=True), dict(observation_cache_bytes=70)):
            kwargs = dict(max_rss_bytes=100, preparation_headroom_bytes=40,
                          training_headroom_bytes=30, observation_cache_bytes=20)
            kwargs.update(change)
            with self.assertRaises(training.c.ContractError):
                training._memory_envelope(**kwargs)
        value = training._memory_envelope(
            max_rss_bytes=100, preparation_headroom_bytes=40,
            training_headroom_bytes=30, observation_cache_bytes=20)
        self.assertFalse(value["analytical_parent_peak_RSS_proven"])
        self.assertTrue(value["no_parent_reduction_or_partial_update"])


if __name__ == "__main__":
    unittest.main()
