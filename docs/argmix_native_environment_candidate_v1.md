# Current-release ARGMix native environment candidate v1

Status: source-only candidate, not installed or admitted. The unchanged current author release is `c2481248eb0de5fda659d68f59c51e0e27becd20`, preserved under `.runtime/author_sources/argmix_c2481248eb0d`. The user accepted this current release as a separately labelled baseline.

## Author lock conflict

| Evidence | Immutable pin | Native consequence |
| --- | --- | --- |
| Author Pixi lock | Python 3.12.13; PyTorch/libtorch 2.12.0, CUDA 12.9; torchvision 0.27.1 | Python-library linkage supplies Torch 2.12.0 |
| Author Cargo lock | `tch` 0.23.0 and `torch-sys` 0.23.0 | Pinned `torch-sys/build.rs` requires base Torch 2.10.0 |
| Pinned Torch guard | Removes local suffix, then compares with 2.10.0 | `2.12.0` is rejected; `2.10.0+cpu` and `2.10.0+cu129` satisfy only the version check |
| Pinned standalone downloader | Defaults to Torch 2.10.0 CPU; CUDA selector excludes cu129 | Downloading another libtorch does not establish compatibility with Python Torch 2.12.0 in one process |

The supplied Pixi/Cargo pair is not coherent for the stock default Torch extension. No installable exact-author-lock specification is emitted. Any presence of `LIBTORCH_BYPASS_VERSION_CHECK`, including `0`, disables the guard and must be unset. Patching source/Cargo or mixing Python Torch 2.12 with standalone Rust libtorch 2.10 would conceal the conflict.

Archive bytes for pinned `tch` and `torch-sys` were fetched in memory and matched Cargo.lock checksums. The pinned `torch-sys/build.rs` SHA256 is `b33049fab5da259a95facf29ded268bee87137d02d2f3200a541dbb11d9b6c61`; its version/guard/Python path logic is at lines 13, 156-168, 182-190 and 206-232, and standalone selectors at 285-321. The pinned `tch` README independently requires libtorch 2.10.0. Source/lock/archive hashes are recorded in the admission JSON.

## Separate narrow candidate

Use unchanged author model/Rust code and Cargo.lock with Python 3.12.13, official Torch 2.10.0 and matching torchvision 0.25.0. Select separate `cpu` or `cu129` wheel profiles. Set `LIBTORCH_USE_PYTORCH=1` so Rust links to that same Python Torch library. This makes the core version tuple plausible but does not admit dependency resolution, compiler ABI or runtime behavior.

For CUDA 12.9, Python linkage avoids the standalone download selector; it does not bypass the version guard. Root reported NVIDIA driver 595.71.05 on the A100s. Driver compatibility remains preliminary until actual native admission.

All proposed prefixes, source copies, targets, solver products, caches and temporary/log paths live under `.runtime/official_argmix_abi210_v1/<profile>/` inside the repository. CPU/CUDA source copies and build targets are separate. The immutable author tree and prior frozen artifacts are not edited.

## Recipe and admission gates

The new recipe files are in [`configs/official_argmix_native_environment_v1`](../configs/official_argmix_native_environment_v1/):

- `abi-admission.json` records source evidence, the lock conflict, explicit runtime deviations and false admission flags.
- `candidate-core-wheels.json` records official cp312/manylinux_2_28 x86_64 core wheel URLs, published SHA256, index and PEP 658 metadata evidence, and direct dependency requirements. No wheel bytes were downloaded; this is partial metadata, not a complete hash lock.
- `candidate-toolchain.yml` is an unsolved Conda toolchain seed retaining author locked top-level versions where possible. It excludes Torch/libtorch/torchvision.
- `candidate-author-python-requirements.txt` retains author scientific package constraints, including the intersection of its Polars requirements. Resolving the former Conda scientific packages through PyPI is an explicit candidate deviation.
- `containment.env.sh` proposes repository-local cache/tool/interpreter paths without creating directories, installing or building.
- `BUILD_ROUTE.md` gives root's resolution, source-copy, build and qualification gates and the unchanged stock compile bodies with `--locked`.

Root must freeze the complete Conda/Python dependency closure and verify downloaded hashes before installation. Then record actual interpreters, Torch ABI and include/library paths; compile both stock artifacts; inspect loaded libraries/rpaths; and admit a Python-to-Rust TorchScript round trip. GPU and end-to-end qualification follow under root scheduling. A `build_training --no-default-features` build alone does not test Torch FFI.

This candidate is a packaging/runtime deviation for the separately labelled current author release. It makes no exact paper-era reproduction, native pipeline readiness or scientific-performance claim.
