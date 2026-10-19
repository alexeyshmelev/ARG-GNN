# Finite PSS cohort repair

The module changes process supervision in the new V8 runtime. Scientific methods, task limits and existing workers are unchanged.

## Binding

```python
from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
_core.OwnedGroup = repaired_owned_group_class(_core)
```

Bind once in the private V4 core used by V8. Do not mutate the frozen V4 or V7 source or their running namespaces. The only module dependency is the supplied core plus the Python standard library.

The factory retains the original constructor and method API. Inherited `stop_and_wait(actual_Popen)` still performs the real direct child wait and records its actual return code. The replacement signals individual verified process generations through Linux pidfds and reaps only identified adopted children. It never calls `killpg` or `waitpid(-1)`.

## Memory and admission contract

`measure()` returns the sum of successfully validated `smaps_rollup` PSS observations for the surviving finite cohort. It never returns an RSS fallback upper bound. The same return value is stored in `latest["enforcement_PSS_bytes"]` and `latest["accounted_bytes"]`.

A sample takes one initial cohort and one end snapshot. It does not wait for process membership to converge. A child exit or a numeric PID reused by a new eligible child generation invalidates only the old observation. Births, exits, generation transitions or unreadable PSS make the sample incomplete. The next scheduler poll samples the new cohort.

The runtime headroom consumer **must** use:

```python
known = tracker.latest["headroom_credit_bytes"]
credit += min(binding["max_rss_bytes"], known)
```

The old unconditional sum of PSS rows is incompatible with incomplete samples. Incomplete samples grant zero headroom credit and retain the existing full task reservation. The scheduler must continue charging each task's full configured cap separately.

The sampler attempts PSS at most twice per process. Persistent RSS-only readings appear in `summed_RSS_bytes` and member diagnostics. They contribute zero enforcement bytes and zero headroom credit. An RSS-only excess cannot trigger the existing memory comparison. Any hard-kill memory evidence comes from PSS, not duplicated shared-page RSS.

Incomplete PSS sums are partial observations, not measurements of total current group memory. Even complete samples remain sequential /proc observations, not atomic kernel memory limits. No delegated writable per-job cgroup was available in the inspected server namespace. The sampler cannot guarantee detection of every instantaneous memory spike or enforce an unobservable PSS cap.

## Identity lifetime

Audit history is keyed by `(pid, start_ticks)` and bounded to 256 rows. The event ring contains at most 64 events. Normal exits and new child generations do not raise the old reuse/stabilization exceptions.

The original root identity never changes. A genuine missing root allows its original orphan descendants to drain. If the numeric root PID resolves to a different exact identity, the entire numeric group is permanently excluded. This prevents foreign descendants in a recycled PGID/SID from being measured or signaled.

A known, still identical generation observed moving to another process group or session remains a supervision error. The module does not add general descendant tracing.

## Focused verification

Run from the repository root with the project interpreter and GPUs hidden:

```sh
PYTHONPATH=.:src:scripts CUDA_VISIBLE_DEVICES= .runtime/env/bin/python -B -s -m unittest research.reviews.benchmark_repairs_20261019.supervision.test_finite_pss_v1
```

Thirteen tests passed. Real CPU fixtures cover continuous child churn, private-memory growth after a fork, shared-page RSS duplication, persistent PSS denial, actual direct return code 7, adopted-child return code 9 and precise pidfd cleanup. Synthetic tests cover child PID reuse, reuse during a memory read, unrelated reused PIDs, root/group lifetime reuse, birth/exit uncertainty and bounded audit history.

The recorded churn fixture completed 30 finite observations with exactly two group captures each. Root identity is probed at both boundaries of each capture. The latest longest observation time is saved in focused_test_result.json. All 30 were incomplete and granted zero headroom credit. No scientific benchmark or GPU work was launched by these tests.
