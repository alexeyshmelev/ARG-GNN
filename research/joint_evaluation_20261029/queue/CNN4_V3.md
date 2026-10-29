# CNN four-class continuation, version 3

The existing version 2 scientific tasks, inputs, output paths, source references,
CPU runtimes, model policy, payload caps, and per-task RAM caps remain unchanged.
Version 3 changes the foreign RAM admission reservation from 1472 GiB to the
reviewed 1408 GiB. Its own maximum 64 GiB and 64 GiB host headroom remain additional.
The bound assumes the exact three-chain root GPU priority gate in successor V2.

The candidate template preserves two genuine completed preparations as external
aliases. Each alias authenticates its original direct-wait receipt, result, and
all referenced products within its output directory. The other 13 tasks retain
their exact version 2 definitions. No successful row is re-executed. The complete
640 MiB existing CNN allocation remains charged; no new allocation is requested.

`cnn4_v3_handoff.py --config research/joint_evaluation_20261029/queue/cnn4_v3_config.json --inspect`
is read-only. It checks source and plan hashes, the exact controller PID 2000574
and start ticks 592150161 at dispatch, absence of active task sessions, output
identity, and the prospective continuation. A non-inspection run belongs to ROOT.

The handoff uses a pidfd for that controller only. It checks saved state and live
descendants before SIGSTOP, checks again while frozen, and immediately resumes
on a launch race. SIGTERM is sent only after the second idle check; no worker is
signalled. The old controller exit is observed, never represented as a direct
wait by the new process. Original state, logs, source, plans, and receipts remain.
If version 2 progresses beyond the sealed aliases before the handoff, the
candidate fails closed and must be regenerated from its actual new successes.

The concrete plan is written to `cnn4_v3_records/plan.json` after the observed
idle shutdown. It includes `handoff_config_ref` for the exact reviewed config.
The new state is `cnn4_v3_records/state.json`. External followers should bind
that concrete plan to the config and read the new state for later CNN completion.

The scheduler source is an exact copy of the version 2 scheduler except for its
admission-module import. `cnn4_v3_tests.json` records six no-dispatch checks:
busy-controller zero signals, launch-race resume without termination, exact idle
shutdown with no fabricated wait, PID reuse rejection, authenticated aliases,
and rejection of scientific/resource changes. No training, handoff, signals, or
live source edits were performed while preparing or testing this candidate.
