# Cross-domain benchmark plan: NTU60 skeleton action recognition

Status: protocol proposal, no downloads or training results yet. Sources checked
2026-10-10. This is a secondary test of ordered processing and interval
representation, not evidence for all ARG-specific structural claims.

## Dataset and access

- [Official NTU site](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
- [Authors' repository](https://github.com/shahroudy/NTURGB-D) provides metadata,
  official missing-skeleton lists and skeleton-only download links.
- NTU RGB+D 60 contains 56,880 clips and 60 action classes. Each frame has xyz
  coordinates of 25 body joints per tracked person.
- The official skeleton-only archive is 5.8 GB. The NTU120 extension adds 4.5 GB,
  but is not required for the initial benchmark.
- Exclude the official 302 clips with missing or incomplete skeletons, leaving
  56,578 usable clips. Select at most two relevant people and preserve masks.

The official terms restrict use to academic, noncommercial research. The site
requests registration and acceptance of its release agreement, and authors also
provide skeleton-only Drive links. Redistribution, derivation or generation of
new datasets, and commercial use require express ROSE Lab permission. Do not
redistribute raw or transformed clips as a new benchmark release. Record the
accepted agreement and download provenance before data acquisition. Publish
code, manifests and results only within applicable terms.

## Exact published splits

Cross-subject: **40,091 training and 16,487 test clips**. Training subject IDs:
`1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38`. All remaining subjects
are test subjects.

Cross-view: **37,646 training and 18,932 test clips**. Cameras 2 and 3 train,
camera 1 tests.

These counts were independently checked from the
[CTR-GCN available-clip manifest](https://github.com/Uason-Chen/CTR-GCN/blob/main/data/ntu/statistics/skes_available_name.txt)
and its [split implementation](https://github.com/Uason-Chen/CTR-GCN/blob/main/data/ntu/seq_transformation.py).
The official missing list contains 229 excluded cross-subject training clips
and 274 excluded cross-view training clips.

There is no separate official validation split. Freeze a held-out training-subject
validation manifest. Keep all synchronized camera recordings of an execution
together. For cross-view, group validation by setup, performer, repetition and
action so synchronized camera siblings cannot cross the tuning boundary.
After selecting hyperparameters, optionally refit using the whole official
training split. Never select checkpoints against the official test labels.

## Graph and interval interpretation

One frame is a graph with joint-type identities, skeleton edges and observed xyz
features. Person identity must be tracked within a clip or handled through a
consistent two-person convention with masks. Give all methods the same identity
information. The clip is the ordered collection of frame graphs.

Treat a frame graph as the piecewise-constant pose on [t_i,t_{i+1}). Its interval
mass is physical dwell time. Splitting this hold into two identical-pose copies
with the same total duration is a representation change, so fixed-coordinate
predictions should be identical. Dropping frames or interpolating different
poses changes the signal and is a separate robustness experiment. If coordinates
are normalized to clip length, retain original duration for speed-related tasks.

Official labels include sitting down versus standing up, putting on versus
taking off clothing, and walking towards versus apart. These motivate order
sensitivity, but ordinary class accuracy alone may exploit pose-distribution
shortcuts. Add shuffled-frame evaluation and a matched direction probe with
exactly the same raw pose multiset. Do not give precomputed velocities to an
unordered control, since they already contain order information.

Use deterministic held-out preprocessing and the same frame representation for
all models. A 64-frame protocol is an economical initial comparison, consistent
with the released CTR-GCN NTU60 configuration. A full or longer-sequence protocol
should be evaluated separately. Explicitly distinguish token serialization
changes, interval subdivision and actual motion reversal.

## Models and evaluation

Use chronological action classification as the main task. Report top-1 accuracy,
macro-F1, mean and spread over frozen seeds, class-pair confusion, subdivision
error, shuffle degradation, parameter count, peak memory and throughput.

Core controls: weighted DeepSets, late temporal Transformer, coordinate-aware
GST, leaf/joint-correspondence supra-graph MPNN, and a documented GraphSSM
adaptation. Include a domain specialist:
[CTR-GCN paper](https://arxiv.org/abs/2107.12213),
[official implementation](https://github.com/Uason-Chen/CTR-GCN), code licensed
CC BY-NC 4.0. Its single joint stream uses 64-frame windows and one GPU in the
released NTU60 configuration. Four-stream joint/bone/motion ensembles need their
own resource and modality comparison. CTR-GCN was inspected at commit
`e13d7582e281d06711eeecb380a472b278ae1663`.
[ST-GCN](https://arxiv.org/abs/1801.07455) with
[BSD-2-Clause code](https://github.com/yysijie/st-gcn) is a historical sanity check.

NTU mostly has fixed skeletal topology. It tests ordered reasoning, persistent
identity and the interval representation property. Changing tree topology,
graph-local anonymous nodes and set-conditional structural reasoning still need
ARG and controlled synthetic experiments.

## Resource plan

All 56,578 clips padded to 300 frames, two people, 25 joints, xyz float32 require
10.18404 GB. Store skeleton topology once and avoid repeating edge indices or
materializing one Python graph object per frame. A planned 100 GB budget covers
raw archives, coordinate arrays, manifests and a bounded checkpoint collection.
Measure actual decompression and preprocessing space before large writes.

All work remains under `/workspace-SR003.nfs2/ARG-GNN` on anogena. Total project
storage must remain below 500 GB, with a 450 GB warning threshold. Main owns
scheduling. At most two free physical GPUs from 2 through 5 may be used after
verification, through the project launcher with the required process title.
No runtime or GPU feasibility result is asserted by this plan.

## Execution gates

1. Obtain permitted access and record provenance without redistributing data.
2. Freeze exclusion, split, identity and preprocessing manifests.
3. Run CPU shape and invariance checks on representative examples.
4. Run a small measured pilot, then compare the frozen model suite.
5. Separate ordinary accuracy evidence from exact representation consistency
   and from ARG-specific changing-topology claims.
