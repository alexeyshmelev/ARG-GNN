# Training interruption correction
The project disk-limit guard incorrectly made checkpoint writes depend on unrelated server process and descriptor stability. The last measured physical project usage was approximately 290 GB, below the user's authorized 1 TB. This was an implementation error, not exhausted storage.

New consumers, reserve-only admission and the new launcher charge original output allowances in full and inspect only each admitted output scope. Full allowances after H030 admission total 963,827,367,424 bytes, strictly below 1 TB. Global process inspection is removed from these storage decisions. GPU leases, at-most-two-device scheduling and scientific models remain unchanged.

H032 is executing real FP32 updates on GPU 2. H030's first startup found GPU 3 briefly occupied and exited before creating a worker. A subsequent startup succeeded after GPU 3 became free. The recovery worker's title and physical assignment were verified through NVTOP. The original failed startup and original partial training records remain preserved.

These corrections establish execution reliability only. They do not establish a scientific novelty or accuracy gain.
