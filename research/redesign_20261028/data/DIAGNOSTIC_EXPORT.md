# Registered diagnostic packet export

The exact registered 128 fitting draws from TRAIN000/001 and 64 development
draws from TRAIN002 were exported to `../diagnostic_packets/`. This preparation
ran no training and opened no TEST inputs. Original scientific inputs, split
registration and the pilot protocol remain unchanged.

Target RNG seeds are 20261028 for fitting and 20261029 for development. Each draw
has a deterministic context RNG constructed from SeedSequence([20261030,
split_code, draw_index]), with fit code 0 and development code 1. Context uses
the focal coordinate plus four uniform coordinates within the clipped 50-kb
radius. Complete containing trees and all their observed SNPs are retained.
Each draw has its own observation NPZ and separate scalar int64 label NPY.
Both sampled sets happen to contain unique target IDs for these fixed seeds;
no deduplication or class-dependent selection was performed.

| Quantity | Fitting | Development |
|---|---:|---:|
| Registered target draws | 128 | 64 |
| Unique target identities | 128 | 64 |
| Complete tree occurrences after within-window deduplication | 596 | 297 |
| Packed node occurrences | 1,134,188 | 565,191 |
| Observed SNP occurrences | 10,782 | 5,761 |
| Ana_nonBAA target count | 42 | 18 |
| CHG_nonBAA target count | 22 | 15 |
| WHG target count | 10 | 6 |
| EHG target count | 30 | 9 |
| Ana_viaBAA target count | 19 | 14 |
| CHG_viaBAA target count | 5 | 2 |

These class counts describe the preregistered sampled data. They are not
accuracy or learning results. The fitting support is the two-parent 33,334-row
pool and the development support is the original TRAIN002 16,666-row pool.

The export used two CPU fork workers and took 11.4696 seconds. Its complete output
is 138,344,025 bytes, below the 1-GiB limit. Maximum measured worker peak RSS was
3,121,428 KiB. Shared filesystem cache was uncontrolled; this RSS includes
accumulated mapped pages. The process did not construct a whole-parent graph,
full genotype matrix or large inference-site mapping.

A separate readback validated every one of the 192 packet hashes, 192 label
hashes, original target joins, split/draw identities and deterministic context
coordinates. It checked all retained native site IDs/positions, sample mappings,
core flags and truth separation, and independently measured the output scope.
The readback passed in 2.8662 seconds. Its receipt is
`diagnostic_packet_validation.json`.

The immutable manifest is `../diagnostic_packets/manifest.json`, SHA256
`eb9ba0cff26cd1479f8752ad8a2def2f5ebde1d9c03da24f79fc38fd6720ff51`.
The completion receipt SHA256 is
`b2be0c42b5bf4387ec7b07be4325734c91536a1206238978f3805d36688cc926`.
The full readback receipt SHA256 is
`f25b0f58cbf53f6c29babc3ef498e1dd2131ad7bf45175a5281c6e91a37d0264`.

`manifest['fit']` and `manifest['development']` are draw-order lists. Each row
contains draw_index, parent_id, example_index, packet_path, label_path,
packet_sha256 and label_sha256, plus observed provenance and context metadata.
Paths are relative to `diagnostic_packets`. The manifest includes top-level
protocol_sha256, archived exporter/loader/protocol source references, original
source bindings and exact target/context sampling specifications.

The harness should load observations through the model's observation whitelist
and read labels independently:

```python
observations = wm.load_observations(packet_root / row['packet_path'])
label = int(np.load(packet_root / row['label_path'], allow_pickle=False))
```

The observation NPZ contains no labels key. Context coordinate multiplicity,
continuous densities and quadrature weights remain present for the coordinate
model adapter. Neither worker completion order nor file creation order changes
the registered draw order in the final manifest.
