# Current-author ARGMix ABI210 execution driver v1

This root-owned driver continues H012 with unchanged author source `c2481248eb0de5fda659d68f59c51e0e27becd20`. It uses the separate Python3.12.13 / Torch2.10.0+cu129 / torchvision0.25.0+cu129 runtime candidate and stock Cargo`tch`/`torch-sys`0.23.0. It does not claim exact author Pixi2.12 packaging or paper-era reproduction.

Status at preparation: root completed the frozen84-package Conda toolchain in `.runtime/official_argmix_abi210_v1/cu129/env`. The first wheel-only Python resolver failed because Snakemake's `connection_pool>=0.0.3` has no matching wheel. That failure and its logs remain unchanged. Root subsequently built and froze the single reviewed4,106-byte connection-pool wheel, then completed the123-wheel closure in resolver attemptv3. Attemptv2 failed before invoking pip due to a root logfile-path collision; both failed attempts remain preserved. The frozen artifact manifest keeps its v2 filename. This agent performed source/metadata inspection only; no package installation, source-wheel/native build, GPU work or Git operation.

## Frozen bootstrap and narrow source-wheel exception

The driver verifies H012, all original candidate recipe hashes, the resolved explicit Conda specification, root resolution/bootstrap receipts and every installed Conda name/version/build/URL/MD5/SHA256 record. Before bulk phases it verifies the cached84 Conda archives against their frozen SHA256. It never invokes Conda or duplicates bootstrap. The original86-file source inventory is checked before every phase and after builds.

The source/backend pin is [`approved-connection-pool-source-build-v1.json`](../configs/official_argmix_native_environment_v1/approved-connection-pool-source-build-v1.json), SHA256 `a16b40d410609d9e77d66485c01a67c88d613f895069eb7b1f2a57759066a003`. It fixes the connection-pool0.0.3 sdist URL/SHA256, setup.py/setup.cfg/PKG-INFO hashes, absence of pyproject.toml, and its implicit `setuptools.build_meta:__legacy__` route. Setup declares no install/setup requirements and reads local README metadata.

Actual installed builder pins are setuptools84.0.0 and wheel0.48.0, including full installed source/metadata fingerprints and frozen Conda package records. The structured `prepare_command` verifies those fingerprints and downloads/verifies/freezes the3,795-byte archive before source metadata executes. Root then runs the structured `wheel_build_command` with the pin's local environment: `pip wheel --no-build-isolation --no-deps --no-index`. It builds only this reviewed archive in its dedicated repository-local workspace. No source package is installed at this stage. The actual root-built wheel SHA256 is `1e3d8b819323fe8635d9bc267fa56b8f81521657991d08e6a869f1f0c7500586`, recorded in `built-connection-pool-wheel-v1.json`; the driver verifies this receipt and the exact source/backend pin.

Root retains all source-wheel logs, pins and output bytes, freezes the resulting wheel SHA256, and supplies its direct local file URL/hash to a new wheel-only resolver condition. Broad sdist fallback remains disabled. An additional missing wheel requires a new explicit source/backend decision; this driver does not permit one automatically.

## Root phase commands

Run from `/workspace-SR003.nfs2/ARG-GNN` using the existing project environment. `plan` only reads/verifies pinned files and reports phase readiness. It creates no output or environment.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/prepare_argmix_native_abi210_v1.py --phase plan
```

Once root's new successful resolver has frozen the full artifact manifest, seal every selected wheel without re-running its resolver:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/prepare_argmix_native_abi210_v1.py --phase freeze-python --run-id v2 \
  --resolved-artifacts configs/official_argmix_native_environment_v1/resolved-python-artifacts-v2.json \
  --resolution-receipt experiments/results/argmix_native_python_resolution_v3/completion.json
```

The two paths are overridable to match root's completed producer. It uses the root resolver schema: completion status `complete_python_wheel_closure_frozen_installation_and_build_pending`, exit0, complete hash lock true and artifact-manifest SHA256; manifest `profile=cu129`, `wheel_only=true`, actual Linux Python3.12.13 environment, author/core manifest hashes and artifact records `{name,version,url,sha256,requires_dist,requested}`. Failed/running v1 products are rejected. Only connection-pool0.0.3 may be a local wheel, inside the reviewed source workspace with an identical prebuild pin copy; all other artifacts must be fixed HTTPS wheels.

Freeze writes a URL/SHA256 draft before downloading, verifies every wheel byte/Name/Version/METADATA, verifies both published core-wheel and inspected PEP658 metadata hashes, and saves the completed runtime lock under `.runtime/official_argmix_abi210_v1/cu129/locks/`. Its manifest records the full Conda closure, unchanged Cargo registry checksum/URL closure, solver input/evidence and the reviewed source-wheel exception. No package is installed by this phase.

After reviewing the complete lock, root may explicitly install offline and build:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/prepare_argmix_native_abi210_v1.py --phase install-python --install --run-id v2

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/prepare_argmix_native_abi210_v1.py --phase build --build --run-id v2
```

Offline installation uses only the sealed wheelhouse with `--no-index --no-deps --require-hashes --force-reinstall`. Force reinstall ensures the Python distributions match wheel artifacts even when a same-version Conda distribution was present. Conda's original package records remain provenance; PyPI scientific distributions are an explicit candidate deviation. Verify installed distributions, `pip check`, actual Python/torch/vision versions, Torch C++11 ABI and include/library paths before a native compile.

## Unchanged stock builds and containment

Every stage uses the original containment recipe, removes inherited Torch/Python/linker overrides, explicitly unsets any version-guard bypass, sets the candidate interpreter/tool paths and Conda prefix, and hides CUDA. It never repurposes HOME. All envs, downloads, source copies, Cargo/cache/target directories, TMP and logs remain inside the repository.

Build uses a separate ordinary author-source copy at `cu129/source`, checked against all86 original file hashes. Cargo fetch/build and maturin use `--locked`. The default extension keeps stock Torch and extension-module features. The extraction CLI keeps `--no-default-features` and its stock discovery path `source/src/rust/ancpath/target/cli/release/build_training`. CLI success alone does not qualify Torch FFI. The immutable author source is never built in place or patched.

Root's same-turn shared repository storage measurement/reservation is reused rather than repeating a full NFS scan. Receipts retain its actual measured58,976,223,232-byte value,30GB reservation and450/500GB limits; wheel download bytes are bounded. Root continues to own total project allocation and concurrent activity.

## Native library admission preparation

Successful build hashes the native extension(s) and extraction CLI, records `ldd` and `readelf -d` outputs, and prepares `run_cpu_library_admission.sh` plus a small CPU probe script under that build's receipt directory. The build receipt still says native runtime admission is pending.

Root's prepared command imports the actual stock extension, traces a three-observation handmade TorchScript fixture, loads/evaluates it via stock `ancpath.RustTorchScriptInference`, compares Python/Rust outputs and CPU device, then records `/proc/self/maps`. It requires all loaded libtorch/libc10 paths to belong to the selected Python wheel and records actual native extension/HDF5/C++ library mappings. This is new native FFI/loaded-library admission, not an author-model accuracy test or biological experiment.

Execute the generated command only after build review:

```bash
bash experiments/results/argmix_native_abi210_execution_v1/v2/build/run_cpu_library_admission.sh
```

The probe creates a fresh repository-local admission directory and retains model/vocabulary/checksum/library receipts. It requests no CUDA context. A passed handmade CPU probe still does not admit the full source model, GPU path or end-to-end native biological pipeline; those remain distinct root qualifications.

## Honest receipts and preserved failures

Each fresh phase is recorded under `experiments/results/argmix_native_abi210_execution_v1/<run-id>/<phase>/`. Record driver/input hashes, argv/cwd, stdout/stderr hashes, exit codes, elapsed times, source integrity, exact wheel/Conda/Cargo closure and stage-specific actual installation/build progress. A failed command retains partial directories/logs and a traceback. No prior receipt, resolver output, wheel lock or failed workspace is overwritten, deleted or silently retried. A reviewed new attempt uses a new run ID.
