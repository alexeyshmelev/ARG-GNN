# Current worker bottleneck observation

All three workers stayed alive and advanced counters during 24 snapshots over46.00s, beginning2026-10-11T04:43:01Z. Cached seed303 and Census seed101 had both logged epoch1 by then. Their logs are emitted only after full TRAIN and all VAL/model work, so the earlier quiet interval was not direct evidence of a stall.

| Worker | Observed phase evidence | Process CPU seconds | rchar read delta | GPU sample |
|---|---|---:|---:|---|
|1230680, uncached202|Epoch12 logged; VAL-bank FDs and new epoch13 VAL output path|49.33|23.99GB|GPU2 mean20.42%|
|1317405, cached303|Epoch1 logged; TRAIN-bank FD observed|46.05|1.22GB|Shared GPU3 mean60.17%|
|1317433, Census101|Epoch1 logged; finer function/tile unobserved|77.92|44.88MB|Same aggregate GPU3|

These windows differ by phase and are not a cache throughput comparison. Main-thread runnable scheduler waits were only6.9ms,0.36ms and3.96ms. Cached303's named autograd thread used25.64CPU seconds while its main thread used20.36s. This is direct CPU activity evidence, not a Python-function profile. Low read_bytes alone cannot prove page-cache hits on NFS.

A narrow future candidate is to reuse Census's already certified fixed CPU forest. Preparation constructs cert.forest, while the current forward passes cert.inputs and the encoder rebuilds the forest for every control forward. Both default time scales are10000. Preserve every source/certificate SHA and graph/role/topology check, neural operation/order, gradient/optimizer update, epochs and scoring. Require numerical parity and new bounded phase timings before accepting a future successor. The repeated operation is source-proven; its elapsed-time share and speedup are unmeasured.

The sample, exact source excerpts and diagnosis separate direct evidence from inference. No scientific output contents were read. No stack attachment, live patch, process CUDA context, GPU job, installation, budget mutation or Git action occurred. The existing one-tile cache experiment was not repeated. Current pools remain frozen.
