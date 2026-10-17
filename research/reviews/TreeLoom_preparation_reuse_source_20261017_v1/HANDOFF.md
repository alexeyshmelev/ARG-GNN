# Exact observed-preparation reuse successor

This source-only candidate leaves the running qualification and every existing
live source untouched. It preserves H050 stable_leaf_memory, all49 FP32 tensors,
32132 parameters, original full30Mb observations and10000 focals, original parent
order,600 fresh optimizer updates, original truth barriers and fixed final epoch30.

## Actual correction

The original worker re-prepares each complete parent on all30 passes. This
successor's worker calls ParentLoaderV1. A fixed selected subset reads already
closed immutable observed-preparation archives in every original epoch. Other
parents use the exact optimized preparation. There is no LRU, eviction, change
of training order, truncation, imputation or dropped example. A single fixed
cached parent already avoids29 repeated preparations, despite19 other parents
appearing between its epoch visits.

Each replay eagerly loads only that current parent and is deleted by the
unchanged training loop. The whole20-parent corpus is never retained in RAM.
Archives never include model parameters, activations, optimizer, RNG, checkpoints
or query truth. Their known reference labels remain original observed inputs.

optimized_observations_v1 preserves list(tree.nodes()) local order. It replaces
per-node Python node()/parent() calls with the same node-time/parent C arrays and
a stamped global-to-local workspace. Connected and all-isolated zero-SNP holds
produce exact FP32 zeros in fields0..3 and ones in field4, bypassing the otherwise
unused per-hold Torch branch layout. Every nonempty hold uses the original
FP32 support operations in the original order. All geometry, levels, clade
episodes, cuts, donor arrays and tensor constructors remain original equations.
Preparation now calls finite resource ticks within native/encoding loops and
prints at most five short stage transitions per parent load.

## Serialization and source-derived sizes

cache_codec_v1 uses ZIP_STORED, numeric C-order NPY entries and bounded JSON.
There is no pickle or assumed compression. Each tensor has its exact dtype,
shape, nbytes and SHA256. The key binds parent, observed/focal/preparation refs,
this preparation source manifest and the fixed whole-parent recipe. Whole-file
SHA is authenticated before replay; every array header/size/SHA is checked before
use. New pointer/version stamps are created only after zero-conversion
torch.from_numpy reconstruction. Original complete recipe metadata is retained.

Let N be native nodes, R roots, T holds, H=1600 haplotypes, S sites, A original
allele width, M proper-clade memberships, E episodes, K cut-group occurrences,
G distinct cut episodes and Q=1000 query columns. Exact tensor storage is:

* bank:100N-24R+8TH+(H+18+28A)S+12T+12H+8Q bytes;
* episodes:2N+12M+4E+4T bytes;
* cut:13K+8TH+68GH+H bytes;
* two focal join vectors:160000 bytes.

These are serialized tensor bytes, not peak process RSS. The executable descriptor
sums the actual tensor nbytes, so it does not rely on rounded formulas. JSON is
bounded64MiB; each numeric NPY/ZIP entry has4096 bytes header allowance plus64KiB
archive overhead. Each cache scope adds1MiB for bounded completion metadata and
allocation rounding. The actual archive is checked against its declared bound.

Even ignoring SNPs, source maxima N=400M,T=100k can imply over62GB per parent,
so all20 cannot be promised under the existing1TB project limit or384GiB RSS.
No actual corpus tensor size, free space or throughput was measured here.

## Measured decision gates and bounded policy

cache_producer_v1.execute_in_admitted_context_v1 has the existing ROOT CPU bridge
signature. It validates exact current Census guard/source/runtime/config/actual
whole cache+aux reservation and review before constructing the original Writer.
It accepts only an explicitly admitted CPU context; it has no admission, reserve,
launch, close, cleanup or retry function. ROOT must supply the actual existing
owned CPU supervisor/context and later consume wait0/drain/ordinary close. This
package does not claim an unexecuted cache producer admission or controller.

mode=measure_only encodes the whole observed parent and emits only actual tensor
nbytes/serialization bound/cold seconds. It materializes no archive and makes no
replay or fit claim. size_candidates_v1 chooses a prospective fixed subset within
ROOT-supplied actual remaining capacity. Then serialize_and_verify writes only
within an actual separately charged scope and releases the original bundle before
exact replay. The complete genuine producer wait/drain/revocation/accepted close
must exist before full600 can consume the archive. Scope charges are retained;
no cleanup or invented credit is performed.

select_fixed_subset_v1 requires actual verified replay/size/RSS evidence. ROOT
derives actual_original_owned_RSS_cap_enforced only from a successful original
owned supervisor under an explicit <=384GiB cap; producer self-RSS does not by
itself claim aggregate ownership. If all20 actual charges fit, all20 may be fixed;
otherwise a smaller immutable subset remains useful each epoch. A zero-fit
decision yields no caching claim. An existing two-parent LRU is not used.

## Executable integration and finite full600 control

parent_v1/worker_v1 are separate narrow copies. The original borrowed GPU lease,
title, pidfd ownership, RSS/deadline/drain checks, synthetic parity, fresh seed70001,
AdamW, full600 training loop, genuine target join and output schemas remain.
The parent validates the new source/closed-cache extension before attaching.
A qualification using these copies must be cache-cold with no cache plan.
The already-running original qualification is also genuinely cache-cold; this
source does not stop it or reuse its weights. An accepted original qualification0
can remain the scientific gate of the continuation, with its real elapsed/peak
measurements disclosed. New source review is separately required.

full600_ROOT_reuse_continuation_v1 inherits compact all20/owner4 control. It keeps
actual qualification, all20 metadata barrier, per-parent encoding/replay before
own truth, fresh600 state, finite owner/root clocks and original accounting.
The measured plan must bind this actual cache plan, declare the cache assumption
truthfully, and explicitly review actual cache sizes/replay costs plus every
uncached parent's30 repeated preparations. The original measured work estimate
must still fit the remaining immutable <=7day window. No cache fit/speed claim
can replace that gate.

The new control source and manifest replace prior control refs one-for-one;
cache plan/source provenance is consolidated inside the existing real
ready_control_bundle. Original64 direct bindings and64MiB training primary+AUX
reservation stay unchanged. Cache bytes live in separately accounted retained
producer scopes and are read-only during training. ROOT must explicitly provide
the qualified source-extension fields; old active inputs are not modified.

This is a candidate plus bounded RAM harness. It performs no data access,
scientific training run, resource probe/accounting operation or operational
activation until separately reviewed and explicitly ROOT admitted.
