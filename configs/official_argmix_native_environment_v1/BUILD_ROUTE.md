# Current-release ARGMix native ABI candidate

Status: recipe only. No environment has been created, packages installed, author source changed, native builds run, or native runtime admitted by this recipe.

Use the unchanged author release at `c2481248eb0de5fda659d68f59c51e0e27becd20`. The author Pixi lock supplies Torch/libtorch 2.12.0, but its unchanged Cargo lock supplies `tch`/`torch-sys` 0.23.0 whose required base Torch version is 2.10.0. Do not run the original `pixi install` for this separate route. Do not set `LIBTORCH_BYPASS_VERSION_CHECK`, even to `0`.

1. Root selects `cpu` or `cu129` and resolves the complete candidate Conda and Python closure. Freeze package URLs, builds, hashes, dependency metadata, platform requirements and solver provenance before installing. `candidate-toolchain.yml` and `candidate-author-python-requirements.txt` are unsolved inputs. `candidate-core-wheels.json` records only core wheel index/metadata evidence and published hashes, not a complete lock.
2. Keep all solver files, prefixes, downloads, caches, logs, temporary files and outputs inside `/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/<profile>/`. Root owns installation. Use the existing `/home/jovyan/miniconda3/bin/conda` with `scripts/project_env.sh`. Check storage against the repository's 450 GB warning / 500 GB stop limits before bulk writes.
3. Root stages an ordinary per-profile copy of the immutable author tree as `source/`, excluding pre-existing build outputs and environments. It must not be a symlink back to the immutable tree. Retain author source/lock hashes. Build output writes are permitted only in this copy and profile-local targets. The stock CLI discovers `src/rust/ancpath/target/cli/release/build_training`, so preserve that path inside the copy.
4. Activate the root-created candidate prefix and re-source `containment.env.sh`. Confirm `python`, `python3`, `PYO3_PYTHON`, `cargo`, `maturin`, C/C++ compilers and linker resolve to the intended candidate tools. `torch-sys` chooses PATH `python` when `VIRTUAL_ENV` exists, otherwise `python3`; checking only `PYO3_PYTHON` is insufficient.
5. Before any native compile, record actual Python version, `torch.__version__`, `torchvision.__version__`, `torch.version.cuda`, `torch._C._GLIBCXX_USE_CXX11_ABI`, and `torch.utils.cpp_extension.include_paths()` / `library_paths()`. Require the profile's exact local wheel suffix and base Torch 2.10.0 / vision 0.25.0. Verify wheel bytes against their published SHA256 and the completed dependency lock. Verify host glibc against the `manylinux_2_28_x86_64` tags. Fail on inherited bypass/library overrides or Torch paths outside the candidate prefix.

With those gates passed, root may run the stock compile bodies with explicit lock enforcement and separate targets:

```bash
cd "$ARGMIX_NATIVE_SOURCE/src/rust/ancpath"
env -u LIBTORCH -u LIBTORCH_INCLUDE -u LIBTORCH_LIB \
  "$ARGMIX_NATIVE_PREFIX/bin/maturin" develop --release --locked \
  --target-dir "$ARGMIX_NATIVE_WORK/targets/python"

"$ARGMIX_NATIVE_PREFIX/bin/cargo" build --release --locked --no-default-features \
  --manifest-path "$ARGMIX_NATIVE_SOURCE/src/rust/ancpath/Cargo.toml" \
  --bin build_training \
  --target-dir "$ARGMIX_NATIVE_SOURCE/src/rust/ancpath/target/cli"
```

The compiler feature flags and source stay unchanged. The second body matches the author's extraction CLI route; its `--no-default-features` success alone does not validate Torch FFI. The optional deprecated incremental Python backend is outside normal author installation and this candidate.

6. After compile, root records all native artifacts and hashes, checks `ldd`/rpath and `/proc/<pid>/maps`, and admits Python import plus a Python-produced TorchScript module loaded and evaluated by the actual Rust/Torch path. Compare outputs and ensure exactly the candidate Torch libraries are loaded. Successful compilation or matching base versions alone is not runtime admission.
7. For `cu129`, Python linkage (`LIBTORCH_USE_PYTORCH=1`) obtains CUDA 12.9 libraries from the selected Python wheel; it does not use the pinned standalone downloader, which rejects `cu129`. Driver 595.71.05 was reported by root on the A100s and is only preliminary compatibility context. Root owns eligible GPU availability checks, assignment, project launcher/process-title requirements and native CUDA qualification. The containment recipe leaves CUDA hidden until root applies the authorized launcher.
8. Freeze the admitted environment and run the separately labelled current-release end-to-end ARGMix pipeline only after native import, loaded-library and Python-to-Rust TorchScript gates pass. Preserve split/provenance and ledger requirements. The earlier small model bridge does not qualify this native pipeline.

The original author Pixi/Cargo files remain immutable evidence. This route deliberately changes the runtime package tuple and scientific package distribution source. Report it as the current author implementation in a separate ABI candidate, not exact author-lock or paper-era reproduction.
