#!/usr/bin/env python
"""Root-owned H012 phases: freeze wheels, offline install, stock build, admission preparation."""
from __future__ import annotations
import argparse
import datetime as dt
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import tomllib
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zipfile

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
CONFIG = ROOT / "configs/official_argmix_native_environment_v1"
WORK = ROOT / ".runtime/official_argmix_abi210_v1/cu129"
PREFIX = WORK / "env"
SOURCE = WORK / "source"
AUTHOR = ROOT / ".runtime/author_sources/argmix_c2481248eb0d"
PYTHON = PREFIX / "bin/python"
RESULTS = ROOT / "experiments/results/argmix_native_abi210_execution_v1"
INVENTORY = ROOT / ".runtime/tmp/official_argmix_native_v1/author_original_source_inventory.json"
FROZEN = {
    "configs/official_argmix_native_environment_v1/abi-admission.json":
        "36cb7ddab5e38f21ab0808a56bed4f89b833c69d1f49b1d7756159d1fd720556",
    "configs/official_argmix_native_environment_v1/resolved-toolchain-linux-64-v1.txt":
        "a91809e4046038613f1d2bccab297ea6e78e863f6b21613355b94bb8e0a3fd4d",
    "experiments/results/argmix_native_dependency_resolution_v1/completion.json":
        "64bf246769fc9257f580b0ba6e3a66fb40c2c0e52604e945ad90e389f017771f",
    "experiments/results/argmix_native_dependency_resolution_v1/conda_solve.json":
        "5ccf635891f17e62ba2ed316925c8c19490473bcdcb31ff212e39a3cef9d7ef4",
    "experiments/results/argmix_native_toolchain_install_v1/completion.json":
        "e3d661ebf3edb96db39b408793fd3b2725ac7b0dbf2ae99b3c86b83f52eb4a6a",
    "experiments/results/argmix_native_toolchain_install_v1/installed-package-records.json":
        "62469af93d146527f9e64978aae4b12542667b3d2bc9ebdec642dbf39bf11087",
    ".runtime/tmp/official_argmix_native_v1/author_original_source_inventory.json":
        "eaa1fd3f358158085ee647db0ea52aa49572f20f9acff08b6848e266556eb1c4",
    "configs/official_argmix_native_environment_v1/approved-connection-pool-source-build-v1.json":
        "a16b40d410609d9e77d66485c01a67c88d613f895069eb7b1f2a57759066a003",
    "configs/official_argmix_native_environment_v1/built-connection-pool-wheel-v1.json":
        "52038a3132f4811a9579a6a5e6fb810b37c1524c2e10646945fc7443b17447d5",
    "configs/official_argmix_native_environment_v1/resolved-python-artifacts-v2.json":
        "8a97a356d19c568a3d6ae12d883586dac41c56244a0dddb9cac539b058568e44",
}

PREFLIGHT = r'''
import importlib.metadata as md
import json, os, pathlib, shutil, sys
p=pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path(sys.prefix).resolve()!=p or sys.version_info[:3]!=(3,12,13):
    raise RuntimeError("Candidate interpreter/prefix mismatch")
for name in ("python","python3"):
    if pathlib.Path(shutil.which(name)).resolve()!=pathlib.Path(sys.executable).resolve():
        raise RuntimeError("torch-sys PATH interpreter mismatch: "+name)
if "LIBTORCH_BYPASS_VERSION_CHECK" in os.environ:
    raise RuntimeError("Any bypass presence, including zero, is forbidden")
import torch, torchvision
from torch.utils.cpp_extension import include_paths, library_paths
if torch.__version__!="2.10.0+cu129" or torchvision.__version__!="0.25.0+cu129":
    raise RuntimeError("Coherent core tuple mismatch")
if torch.version.cuda!="12.9":
    raise RuntimeError("Unexpected compiled CUDA version")
includes,libs=include_paths(),library_paths()
if any(not pathlib.Path(x).resolve().is_relative_to(p) for x in includes+libs):
    raise RuntimeError("Torch include/library paths outside candidate prefix")
print(json.dumps({"python":sys.version,"executable":sys.executable,
 "torch":torch.__version__,"torchvision":torchvision.__version__,
 "compiled_cuda":torch.version.cuda,"CXX11_ABI":bool(torch._C._GLIBCXX_USE_CXX11_ABI),
 "include_paths":includes,"library_paths":libs,"GPU_context_requested":False}))
'''

ADMISSION_PROBE = r'''#!/usr/bin/env python
"""Small CPU FFI/library probe; handmade fixture, not author-model or biological reproduction."""
import hashlib,json,os,pathlib,sys
if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
    raise RuntimeError("CPU-only admission requires empty CUDA visibility")
if "LIBTORCH_BYPASS_VERSION_CHECK" in os.environ:
    raise RuntimeError("Version-guard bypass is forbidden")
import torch
import ancpath
prefix=pathlib.Path(sys.prefix).resolve()
expected=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/env")
if prefix!=expected or torch.__version__!="2.10.0+cu129":
    raise RuntimeError("Candidate runtime identity mismatch")
if not hasattr(ancpath,"RustTorchScriptInference"):
    raise RuntimeError("Actual default-feature native Torch class missing")
out=pathlib.Path(sys.argv[1]).resolve()
out.relative_to(pathlib.Path("/workspace-SR003.nfs2/ARG-GNN"))
out.mkdir(parents=True,exist_ok=False)
class Probe(torch.nn.Module):
    def forward(self,ids,tmrca,mask):
        slot=(ids[:,0]+tmrca[:,0,0]+mask.sum(1)).remainder(4)
        return torch.nn.functional.one_hot(slot,4).to(torch.float32)*3
tokens=[["A"],["B","A"],["missing"]]
matrices=[[[0,0],[0,0]],[[1,0],[0,0]],[[3,0],[0,0]]]
ids=torch.tensor([[2,0],[3,2],[1,0]],dtype=torch.int64)
tmrca=torch.tensor(matrices,dtype=torch.int64)
mask=ids.ne(0)
module=torch.jit.trace(Probe().eval(),(ids,tmrca,mask))
model=out/"handmade_ffi_probe.pt"
module.save(str(model))
vocab=out/"vocab.json"
vocab.write_text(json.dumps({"<PAD>":0,"<UNK>":1,"A":2,"B":3})+"\n")
native=ancpath.RustTorchScriptInference()
native.load_model(str(model),str(vocab),False)
actual=native.predict_batch(tokens,matrices)
expected_values=module(ids,tmrca,mask).argmax(-1).tolist()
if actual!=expected_values or native.get_device_info()!="Cpu":
    raise RuntimeError("Actual Python-to-Rust TorchScript CPU outputs/device differ")
paths=sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
              if "/" in line and any(s in line for s in ("libtorch","libc10","libhdf5","ancpath","libstdc++"))})
torchlib=(pathlib.Path(torch.__file__).resolve().parent/"lib").resolve()
torchpaths=[pathlib.Path(x).resolve() for x in paths if "libtorch" in x or "libc10" in x]
if not torchpaths or any(not x.is_relative_to(torchlib) for x in torchpaths):
    raise RuntimeError("Loaded Torch libraries are absent or outside the selected Python wheel")
extensions=sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
                   if "ancpath" in line and ".so" in line})
if not extensions:
    raise RuntimeError("No actual native ancpath extension mapping")
receipt={"status":"passed","actual_CPU_Python_to_Rust_TorchScript_probe":True,
 "fixture":"three handmade observations; no targets or biological data",
 "Python_argmax":expected_values,"Rust_argmax":actual,"Rust_device":native.get_device_info(),
 "torch_version":torch.__version__,"ancpath_module":ancpath.__file__,
 "loaded_library_paths":paths,"native_extension_paths":extensions,
 "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
 "vocab_sha256":hashlib.sha256(vocab.read_bytes()).hexdigest(),
 "GPU_context_requested":False,"author_model_numerical_reproduction":False,
 "full_native_pipeline_admitted":False}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
'''


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(value):
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def safe_url(url):
    parts = urlsplit(url)
    if parts.scheme == "file":
        if parts.netloc or parts.query:
            raise ValueError("Only repository-local file artifact URLs are allowed")
        local(unquote(parts.path))
        return urlunsplit(parts._replace(fragment=""))
    if parts.scheme != "https" or parts.username or parts.password or parts.query:
        raise ValueError("Only credential-free fixed HTTPS artifact URLs are allowed")
    return urlunsplit(parts._replace(fragment=""))


def author_unchanged(directory):
    inventory = json.loads(INVENTORY.read_text())
    for name, expected in inventory.items():
        p = directory / name
        if p.is_symlink() or not p.is_file() or sha(p) != expected:
            raise ValueError("Pinned author-file drift: " + str(p))
    return inventory


def validate_inputs():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from the dedicated anogena repository")
    for name, expected in FROZEN.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen input drift: " + name)
    decisions = json.loads((ROOT / "research/experiment_decisions.json").read_text())["decisions"]
    h012 = next((r for r in decisions if r["id"] == "H012"), None)
    if h012 is None:
        raise ValueError("H012 must be registered before this work")
    admission = json.loads((CONFIG / "abi-admission.json").read_text())
    for name, expected in admission["recipe_files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Candidate recipe drift: " + name)
    inventory = author_unchanged(AUTHOR)
    completion_path = ROOT / "experiments/results/argmix_native_toolchain_install_v1/completion.json"
    completion = json.loads(completion_path.read_text())
    if (completion["status"] != "frozen_toolchain_installed_python_closure_and_native_build_pending"
            or completion["exit_code"] != 0 or completion["installed_conda_packages"] != 84
            or completion["recipe_sha256"] != FROZEN[
                "configs/official_argmix_native_environment_v1/resolved-toolchain-linux-64-v1.txt"]):
        raise RuntimeError("Root's frozen toolchain bootstrap has not completed successfully")
    records_path = completion_path.parent / "installed-package-records.json"
    if sha(records_path) != completion["installed_records_sha256"]:
        raise ValueError("Installed Conda record receipt mismatch")
    installed = json.loads(records_path.read_text())
    solved = json.loads((ROOT / "experiments/results/argmix_native_dependency_resolution_v1/conda_solve.json").read_text())
    packages = solved["actions"]["FETCH"]
    expected = {r["name"]: r for r in packages}
    if len(packages) != 84 or len(installed) != 84 or len(expected) != 84:
        raise ValueError("The Conda closure must contain exactly its frozen 84 packages")
    for record in installed:
        bound = expected[record["name"]]
        if any(record[k] != bound[k] for k in ("version", "build", "url", "sha256", "md5")):
            raise ValueError("Installed Conda package differs from the frozen solve: " + record["name"])
        if not re.fullmatch("[0-9a-f]{64}", bound["sha256"]):
            raise ValueError("Conda SHA256 missing from the frozen closure")
    if any(r["name"] in ("pytorch", "libtorch", "torchvision") for r in installed):
        raise ValueError("The bootstrap must not install the conflicting author Torch2.12 stack")
    core = json.loads((CONFIG / "candidate-core-wheels.json").read_text())["profiles"]["cu129"]
    return dict(admission=admission, core=core, conda_packages=packages,
                bootstrap=completion, author_inventory=inventory, H012=h012)


def cargo_closure():
    lock = tomllib.loads((AUTHOR / "src/rust/ancpath/Cargo.lock").read_text())
    records = []
    for package in lock["package"]:
        source = package.get("source")
        if source is None:
            records.append(dict(name=package["name"], version=package["version"],
                                source="unchanged local author tree"))
        elif source == "registry+https://github.com/rust-lang/crates.io-index":
            checksum = package["checksum"]
            if not re.fullmatch("[0-9a-f]{64}", checksum):
                raise ValueError("Cargo registry checksum missing")
            name, version = package["name"], package["version"]
            records.append(dict(name=name, version=version, sha256=checksum,
                                url=f"https://static.crates.io/crates/{name}/{name}-{version}.crate"))
        else:
            raise ValueError("Uncovered non-crates.io Cargo source: " + source)
    return records


def contained_environment():
    env = dict(os.environ, ARGMIX_NATIVE_PROFILE="cu129")
    for name in ("LIBTORCH_BYPASS_VERSION_CHECK", "LIBTORCH", "LIBTORCH_INCLUDE", "LIBTORCH_LIB",
                 "LIBTORCH_STATIC", "LIBTORCH_CXX11_ABI", "TORCH_CUDA_VERSION", "PYTHONHOME",
                 "PYO3_CONFIG_FILE", "PYO3_CROSS", "VIRTUAL_ENV", "CPATH", "C_INCLUDE_PATH",
                 "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "CMAKE_PREFIX_PATH",
                 "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_TRUSTED_HOST",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_REQUIRE_VIRTUALENV"):
        env.pop(name, None)
    command = "source " + shlex.quote(str(CONFIG / "containment.env.sh")) + "; env -0"
    proc = subprocess.run(["bash", "-c", command], env=env, cwd=ROOT,
                          capture_output=True, check=True)
    # Never print/persist the full inherited environment or credentials.
    resolved = {}
    for row in proc.stdout.split(b"\0"):
        if b"=" in row:
            name, value = row.split(b"=", 1)
            resolved[name.decode()] = value.decode()
    resolved.update(CONDA_PREFIX=str(PREFIX), CONDA_DEFAULT_ENV=str(PREFIX),
                    CUDA_VISIBLE_DEVICES="", PIP_DISABLE_PIP_VERSION_CHECK="1")
    for name in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                 "CARGO_HOME", "RUSTUP_HOME", "CONDA_PKGS_DIRS", "CONDA_ENVS_PATH", "PIP_CACHE_DIR",
                 "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "PIXI_HOME", "PIXI_CACHE_DIR",
                 "RATTLER_CACHE_DIR", "HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR",
                 "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "WANDB_DIR",
                 "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR", "PYTHONPYCACHEPREFIX"):
        local(resolved[name]).mkdir(parents=True, exist_ok=True)
    if "LIBTORCH_BYPASS_VERSION_CHECK" in resolved:
        raise ValueError("The version-guard bypass must remain absent")
    return resolved


class Journal:
    def __init__(self, output, phase, run_id, inputs):
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.record = dict(hypothesis_id="H012", kind="argmix_native_abi210_root_execution_v1",
                           phase=phase, run_id=run_id, status="running",
                           started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                           driver_sha256=sha(Path(__file__)), input_sha256=FROZEN,
                           author_commit=inputs["admission"]["author_source"]["commit"],
                           author_files=len(inputs["author_inventory"]), commands=[],
                           GPU_used=False, python_install_attempted=False,
                           python_install_command_succeeded=False, compiled=False,
                           native_extension_build_command_succeeded=False,
                           extraction_CLI_build_command_succeeded=False,
                           native_runtime_admitted=False,
                           full_native_pipeline_admitted=False,
                           exact_author_Pixi_environment=False,
                           label="unchanged current author source; separate ABI210 runtime candidate")
        self.save()

    def save(self):
        write_json(self.output / "receipt.json", self.record)

    def command(self, argv, env, cwd=ROOT, tag=None):
        index = len(self.record["commands"])
        base = self.output / (str(index).zfill(2) + "_" + (tag or "command"))
        out, err = base.with_suffix(".stdout"), base.with_suffix(".stderr")
        item = dict(argv=list(map(str, argv)), cwd=str(cwd), started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        self.record["commands"].append(item)
        self.save()
        start = time.monotonic()
        with out.open("xb") as stdout, err.open("xb") as stderr:
            result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env,
                                    stdout=stdout, stderr=stderr)
        item.update(exit_code=result.returncode, seconds=time.monotonic() - start,
                    stdout=dict(path=str(out.relative_to(ROOT)), sha256=sha(out)),
                    stderr=dict(path=str(err.relative_to(ROOT)), sha256=sha(err)))
        self.save()
        if result.returncode:
            raise RuntimeError("Command failed; complete logs retained: " + str(base))
        return out

    def finish(self, status, **fields):
        self.record.update(status=status, **fields, seconds=time.monotonic() - self.started,
                           finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        self.save()


def verify_conda_archives(inputs):
    evidence = []
    for row in inputs["conda_packages"]:
        archive = WORK / "cache/conda-pkgs" / row["fn"]
        if not archive.is_file() or sha(archive) != row["sha256"]:
            raise ValueError("Frozen Conda archive missing/SHA256 mismatch: " + str(archive))
        evidence.append(dict(name=row["name"], url=row["url"], sha256=row["sha256"],
                             bytes=archive.stat().st_size))
    return evidence


def storage_admission(inputs):
    guard = inputs["bootstrap"]["shared_storage_guard"]
    # Root's same-turn complete repository measurement plus its 30GB reservation.
    if guard["bytes"] + guard["headroom_bytes"] >= 450_000_000_000:
        raise RuntimeError("Root shared storage admission leaves insufficient bulk-write headroom")
    return dict(root_shared_guard=guard, candidate_bulk_reservation_bytes=guard["headroom_bytes"],
                repository_warning_bytes=450_000_000_000, repository_stop_bytes=500_000_000_000,
                full_repository_scan_repeated=False)


def wheel_metadata(path):
    with zipfile.ZipFile(path) as z:
        names = [name for name in z.namelist() if len(Path(name).parts) == 2 and name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Require exactly one wheel METADATA record")
        raw = z.read(names[0])
    metadata = BytesParser().parsebytes(raw)
    return dict(name=metadata["Name"], version=metadata["Version"],
                requires_python=metadata.get("Requires-Python"),
                requires_dist=metadata.get_all("Requires-Dist", []),
                metadata_sha256=hashlib.sha256(raw).hexdigest())


def download_wheel(package, target, budget):
    url = safe_url(package["url"])
    # Reuse only exact frozen bytes from preserved earlier project wheelhouses.
    # Failed receipts remain immutable, and a cache hit is not a new HTTP fetch.
    for directory in sorted(WORK.glob("wheelhouse-*")):
        cached = directory / package["filename"]
        if (directory == target.parent or not cached.is_file() or cached.is_symlink()
                or not cached.resolve().is_relative_to(WORK) or sha(cached) != package["sha256"]):
            continue
        count = cached.stat().st_size
        if count > budget:
            raise RuntimeError("Cached wheel exceeds the root candidate byte reservation")
        metadata = wheel_metadata(cached)
        if (normalized(metadata["name"]) != normalized(package["name"])
                or metadata["version"] != package["version"]):
            raise ValueError("Cached wheel identity differs from the resolver")
        if target.exists():
            raise ValueError("Fresh target required for verified cache reuse")
        os.link(cached, target)
        return dict(path=str(target.relative_to(ROOT)), sha256=package["sha256"], bytes=count,
                    final_download_url=None, origin_url=url,
                    verified_cache_source=str(cached.relative_to(ROOT)),
                    reused_frozen_bytes_without_HTTP=True, **metadata)
    partial = target.with_suffix(target.suffix + ".partial")
    h, count = hashlib.sha256(), 0
    if urlsplit(url).scheme == "file":
        source = local(unquote(urlsplit(url).path))
        final_url = url
        stream_source = source.open("rb")
    else:
        stream_source = urlopen(Request(url, headers={"User-Agent": "ARG-GNN-H012-ABI210/1"}), timeout=60)
        final_url = safe_url(stream_source.url)
    with stream_source as response:
        with partial.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > budget:
                    raise RuntimeError("Artifact download exceeds the root candidate byte reservation")
                stream.write(block); h.update(block)
    if h.hexdigest() != package["sha256"]:
        raise ValueError("Downloaded wheel SHA256 differs: " + target.name)
    partial.rename(target)
    metadata = wheel_metadata(target)
    if (normalized(metadata["name"]) != normalized(package["name"])
            or metadata["version"] != package["version"]):
        raise ValueError("Downloaded wheel identity differs from the resolver")
    return dict(path=str(target.relative_to(ROOT)), sha256=h.hexdigest(), bytes=count,
                final_download_url=final_url, **metadata)


def freeze_python(inputs, env, journal, run_id, resolved_artifacts, resolution_receipt):
    locks = WORK / "locks"
    locks.mkdir(exist_ok=True)
    manifest_path = locks / ("python-closure-" + run_id + ".json")
    requirements_path = locks / ("python-closure-" + run_id + ".txt")
    if manifest_path.exists() or requirements_path.exists():
        raise ValueError("Refusing to replace a frozen Python closure")
    wheelhouse = WORK / ("wheelhouse-" + run_id)
    wheelhouse.mkdir(exist_ok=False)
    # Consume root's successful new resolver products; never repeat failed v1
    # or execute an unreviewed sdist from this installer.
    source_manifest = local(resolved_artifacts)
    completion_path = local(resolution_receipt)
    completion = json.loads(completion_path.read_text())
    if (completion.get("status") != "complete_python_wheel_closure_frozen_installation_and_build_pending"
            or completion.get("exit_code") != 0 or completion.get("complete_hash_lock_written") is not True):
        raise ValueError("Root's successful complete wheel closure is required; failed v1 stays preserved")
    if sha(source_manifest) != completion["artifacts_manifest_sha256"]:
        raise ValueError("Root resolved-artifact manifest hash differs from its completion receipt")
    frozen = json.loads(source_manifest.read_text())
    if (frozen.get("profile") != "cu129" or frozen.get("wheel_only") is not True
            or frozen["source_manifest"] != sha(CONFIG / "candidate-author-python-requirements.txt")
            or frozen["core_wheels_manifest"] != sha(CONFIG / "candidate-core-wheels.json")):
        raise ValueError("Resolved closure must preserve author constraints and pinned core wheels")
    solver_env = frozen["solver_environment"]
    if solver_env.get("python_full_version") != "3.12.13" or solver_env.get("sys_platform") != "linux":
        raise ValueError("Root resolver did not use the actual Linux Python3.12.13 candidate")
    requested = journal.output / "root-resolver-input-provenance.json"
    write_json(requested, dict(artifact_manifest=str(source_manifest.relative_to(ROOT)),
                               artifact_manifest_sha256=sha(source_manifest),
                               completion=str(completion_path.relative_to(ROOT)),
                               completion_sha256=sha(completion_path)))
    report_path = journal.output / "root-resolved-artifacts.json"
    report_path.write_bytes(source_manifest.read_bytes())
    selected = []
    approved_source = json.loads((CONFIG / "approved-connection-pool-source-build-v1.json").read_text())
    built_source = json.loads((CONFIG / "built-connection-pool-wheel-v1.json").read_text())
    if (built_source["pin_sha256"] != FROZEN[
            "configs/official_argmix_native_environment_v1/approved-connection-pool-source-build-v1.json"]
            or built_source["source_sha256"] != approved_source["source_artifact"]["sha256"]
            or built_source["backend_pins"] != approved_source["build_backend"]
            or sha(local(built_source["path"])) != built_source["sha256"]):
        raise ValueError("Root reviewed source-wheel receipt/source/backend/bytes mismatch")
    for item in frozen["artifacts"]:
        url = safe_url(item["url"])
        expected = item["sha256"]
        filename = Path(unquote(urlsplit(url).path)).name
        if not re.fullmatch("[0-9a-f]{64}", expected) or not filename.endswith(".whl"):
            raise ValueError("Full runtime closure accepts fixed SHA256 wheels only")
        if normalized(item["name"]) == "connection-pool":
            expected_parent = local(approved_source["workspace"]) / "wheels"
            if (item["version"] != "0.0.3" or urlsplit(url).scheme != "file"
                    or local(unquote(urlsplit(url).path)).parent != expected_parent):
                raise ValueError("connection-pool must use the single reviewed root-built local wheel")
            if (url != built_source["file_url"] or expected != built_source["sha256"]):
                raise ValueError("Resolver connection-pool wheel differs from its frozen build receipt")
            pin_copy = expected_parent.parent / "prebuild-frozen-pin.json"
            if json.loads(pin_copy.read_text()) != approved_source:
                raise ValueError("Root source wheel lacks the frozen source/backend prebuild pin")
        elif urlsplit(url).scheme != "https":
            raise ValueError("No additional local/source-built wheel fallback is approved")
        selected.append(dict(name=item["name"], version=item["version"], url=url,
                             sha256=expected, filename=filename,
                             resolver_requires_dist=item.get("requires_dist", []),
                             requested=item.get("requested", False)))
    by_name = {normalized(row["name"]): row for row in selected}
    if len(by_name) != len(selected):
        raise ValueError("Duplicate Python projects in resolver closure")
    for core in inputs["core"]["wheels"]:
        actual = by_name.get(core["name"])
        if (actual is None or actual["version"] != core["version"]
                or actual["sha256"] != core["sha256_published"]
                or unquote(actual["url"]) != unquote(safe_url(core["url"]))):
            raise ValueError("Resolver did not preserve the pinned official core wheel")
    # Freeze every selected URL/hash before downloading or installing the closure.
    draft_path = journal.output / "resolved-python-url-sha256-draft.json"
    write_json(draft_path, dict(packages=selected, resolver_report_sha256=sha(report_path),
                               installed=False, wheel_bytes_verified=False))
    downloads, total = [], 0
    for package in selected:
        if any(row["filename"] == package["filename"] for row in downloads):
            raise ValueError("Wheel filename collision")
        artifact = download_wheel(package, wheelhouse / package["filename"],
                                  inputs["bootstrap"]["shared_storage_guard"]["headroom_bytes"] - total)
        total += artifact["bytes"]
        downloaded = dict(package, **{k: v for k, v in artifact.items()
                                      if k not in ("name", "version")})
        for core in inputs["core"]["wheels"]:
            if package["name"].lower() == core["name"]:
                if artifact["metadata_sha256"] != core["metadata_sha256_observed"]:
                    raise ValueError("Pinned core wheel METADATA differs from inspected PEP658 metadata")
        downloads.append(downloaded)
        journal.record.update(downloaded_packages=len(downloads), downloaded_wheel_bytes=total)
        journal.save()
        print(json.dumps(dict(phase="freeze-python", packages_verified=len(downloads),
                              packages_total=len(selected), wheel_bytes=total)), flush=True)
    requirements_path.write_text("\n".join(
        row["name"] + "==" + row["version"] + " --hash=sha256:" + row["sha256"]
        for row in sorted(downloads, key=lambda x: normalized(x["name"]))) + "\n")
    lock = dict(schema_version=1, kind="argmix_abi210_full_python_wheel_closure_v1",
                profile="cu129", platform="linux-64", Python="3.12.13",
                core_tuple={"torch": "2.10.0+cu129", "torchvision": "0.25.0+cu129"},
                solver="root completed new wheel-only pip closure; reviewed connection-pool wheel already frozen",
                resolver_report=dict(path=str(report_path.relative_to(ROOT)), sha256=sha(report_path)),
                resolver_input=dict(path=str(requested.relative_to(ROOT)), sha256=sha(requested)),
                conda_package_records=inputs["conda_packages"],
                Cargo_lock_package_closure=cargo_closure(),
                reviewed_source_wheel_exception=approved_source,
                reviewed_source_wheel_build=built_source,
                root_resolution_completion=dict(path=str(completion_path.relative_to(ROOT)),
                                                sha256=sha(completion_path)),
                source_and_recipe_sha256=FROZEN,
                package_count=len(downloads), packages=downloads,
                wheelhouse=str(wheelhouse.relative_to(ROOT)),
                hash_requirements=dict(path=str(requirements_path.relative_to(ROOT)),
                                       sha256=sha(requirements_path)),
                complete_resolver_closure_frozen=True, all_wheel_bytes_SHA256_verified=True,
                installed=False, compiled=False, native_runtime_admitted=False)
    write_json(manifest_path, lock)
    journal.finish("python_closure_frozen_verified_installation_pending",
                   closure=dict(path=str(manifest_path.relative_to(ROOT)), sha256=sha(manifest_path)),
                   wheel_bytes=total, closure_packages=len(downloads), installed=False, compiled=False)
    return manifest_path


def verify_lock(run_id):
    path = WORK / "locks" / ("python-closure-" + run_id + ".json")
    lock = json.loads(path.read_text())
    if (lock.get("profile") != "cu129" or not lock["complete_resolver_closure_frozen"]
            or not lock["all_wheel_bytes_SHA256_verified"]
            or lock["source_and_recipe_sha256"] != FROZEN):
        raise ValueError("A complete verified candidate closure is required")
    requirement = local(lock["hash_requirements"]["path"])
    if sha(requirement) != lock["hash_requirements"]["sha256"]:
        raise ValueError("Frozen hash requirements drift")
    wheelhouse = local(lock["wheelhouse"])
    for package in lock["packages"]:
        archive = local(package["path"])
        if archive.parent != wheelhouse or sha(archive) != package["sha256"]:
            raise ValueError("Frozen wheel bytes missing/changed: " + package["name"])
    return path, lock, requirement, wheelhouse


def installed_inventory(lock, env, journal):
    helper = ("import importlib.metadata as m,json,sys,pathlib;"
              "assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve();"
              "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version "
              "for d in m.distributions()}))")
    stdout = journal.command([PYTHON, "-c", helper, PREFIX], env, tag="installed_python_inventory")
    installed = json.loads(stdout.read_text())
    for row in lock["packages"]:
        name = normalized(row["name"])
        actual = next((v for k, v in installed.items() if normalized(k) == name), None)
        if actual != row["version"]:
            raise ValueError("Installed distribution differs from the wheel lock: " + name)
    return installed


def install_python(inputs, env, journal, run_id):
    path, lock, requirement, wheelhouse = verify_lock(run_id)
    journal.record["closure"] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    journal.save()
    journal.record["python_install_attempted"] = True
    journal.save()
    journal.command([PYTHON, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
                     "--no-deps", "--require-hashes", "--force-reinstall", "-r", requirement],
                    env, tag="offline_hash_install")
    journal.record["python_install_command_succeeded"] = True
    journal.save()
    journal.command([PYTHON, "-m", "pip", "check"], env, tag="pip_check")
    inventory = installed_inventory(lock, env, journal)
    stdout = journal.command([PYTHON, "-c", PREFLIGHT, PREFIX], env, tag="Torch_ABI_preflight")
    preflight = json.loads(stdout.read_text())
    journal.finish("frozen_python_closure_installed_native_build_pending", installed=True,
                   compiled=False, installed_python_distributions=inventory,
                   actual_Torch_ABI_preflight=preflight)


def prepare_admission(output):
    helper = output / "native_torchscript_cpu_probe.py"
    helper.write_text(ADMISSION_PROBE)
    probe_output = WORK / "admission" / output.parent.name
    command = output / "run_cpu_library_admission.sh"
    command.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "cd " + shlex.quote(str(ROOT)) + "\nexport ARGMIX_NATIVE_PROFILE=cu129\n"
        "source " + shlex.quote(str(CONFIG / "containment.env.sh")) + "\n"
        + shlex.quote(str(PYTHON)) + " " + shlex.quote(str(helper)) + " "
        + shlex.quote(str(probe_output)) + "\n")
    return dict(helper=dict(path=str(helper.relative_to(ROOT)), sha256=sha(helper)),
                command=dict(path=str(command.relative_to(ROOT)), sha256=sha(command)),
                probe_output=str(probe_output.relative_to(ROOT)),
                executed=False, native_runtime_admitted=False,
                probe_scope="handmade CPU fixture through actual stock RustTorchScriptInference; not author-model performance")


def build(inputs, env, journal, run_id):
    path, lock, _, _ = verify_lock(run_id)
    installed_inventory(lock, env, journal)
    preflight_out = journal.command([PYTHON, "-c", PREFLIGHT, PREFIX], env, tag="Torch_ABI_preflight")
    preflight = json.loads(preflight_out.read_text())
    if SOURCE.exists():
        author_unchanged(SOURCE)
    else:
        if any(p.is_symlink() for p in AUTHOR.rglob("*")):
            raise ValueError("Author copy cannot contain symlink escapes")
        shutil.copytree(AUTHOR, SOURCE)
        author_unchanged(SOURCE)
    manifest = SOURCE / "src/rust/ancpath/Cargo.toml"
    journal.command([PREFIX / "bin/cargo", "fetch", "--locked", "--manifest-path", manifest],
                    env, tag="locked_Cargo_fetch")
    journal.record["Cargo_lock_package_closure"] = cargo_closure()
    journal.save()
    journal.command([PREFIX / "bin/maturin", "develop", "--release", "--locked",
                     "--target-dir", WORK / "targets/python"],
                    env, cwd=SOURCE / "src/rust/ancpath", tag="stock_default_Torch_extension")
    journal.record["native_extension_build_command_succeeded"] = True
    journal.save()
    journal.command([PREFIX / "bin/cargo", "build", "--release", "--locked", "--no-default-features",
                     "--manifest-path", manifest, "--bin", "build_training",
                     "--target-dir", SOURCE / "src/rust/ancpath/target/cli"],
                    env, tag="stock_extraction_CLI")
    journal.record.update(extraction_CLI_build_command_succeeded=True, compiled=True)
    journal.save()
    author_unchanged(AUTHOR); author_unchanged(SOURCE)
    extension_root = PREFIX / "lib/python3.12/site-packages"
    extensions = sorted(set(extension_root.glob("ancpath*.so")) |
                        set(extension_root.glob("ancpath*/*.so")))
    binary = SOURCE / "src/rust/ancpath/target/cli/release/build_training"
    if not extensions or not binary.is_file():
        raise RuntimeError("Expected stock extension or stock-discovery extraction binary missing")
    artifacts = [*extensions, binary]
    native = []
    for index, artifact in enumerate(artifacts):
        native.append(dict(path=str(artifact.relative_to(ROOT)), sha256=sha(artifact),
                           bytes=artifact.stat().st_size))
        journal.command(["ldd", artifact], env, tag="ldd_" + str(index))
        journal.command(["readelf", "-d", artifact], env, tag="rpath_" + str(index))
    admission = prepare_admission(journal.output)
    journal.finish("stock_native_build_passed_CPU_runtime_admission_pending",
                   closure=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
                   installed=True, compiled=True, actual_Torch_ABI_preflight=preflight,
                   native_artifacts=native, copied_author_files_unchanged=True,
                   immutable_author_files_unchanged=True, admission_commands=admission,
                   CLI_discovery_path_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "freeze-python", "install-python", "build",
                                           "prepare-admission"), default="plan")
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--resolved-artifacts", default="configs/official_argmix_native_environment_v1/resolved-python-artifacts-v2.json")
    parser.add_argument("--resolution-receipt", default="experiments/results/argmix_native_python_resolution_v3/completion.json")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[a-z0-9][a-z0-9_-]{0,40}", args.run_id):
        parser.error("Use a short lowercase artifact run ID")
    if args.install != (args.phase == "install-python") or args.build != (args.phase == "build"):
        parser.error("Only install-python requires --install; only build requires --build")
    inputs = validate_inputs()
    if args.phase == "plan":
        print(json.dumps(dict(status="source_and_frozen_bootstrap_inputs_verified",
                              bootstrap_duplicated=False, conda_packages=84,
                              author_files=len(inputs["author_inventory"]),
                              profile="cu129", installed_by_this_driver=False, built=False,
                              root_phases=["freeze-python", "install-python --install",
                                           "build --build", "prepared CPU admission command"])))
        return 0
    output = RESULTS / args.run_id / args.phase
    journal = Journal(output, args.phase, args.run_id, inputs)
    try:
        journal.record["storage_admission"] = storage_admission(inputs)
        journal.record["verified_Conda_archive_closure"] = verify_conda_archives(inputs)
        journal.save()
        env = contained_environment()
        if args.phase == "freeze-python":
            freeze_python(inputs, env, journal, args.run_id,
                          args.resolved_artifacts, args.resolution_receipt)
        elif args.phase == "install-python":
            install_python(inputs, env, journal, args.run_id)
        elif args.phase == "build":
            build(inputs, env, journal, args.run_id)
        else:
            journal.finish("CPU_admission_commands_prepared_not_executed",
                           admission_commands=prepare_admission(output))
    except Exception as exc:
        journal.finish("failed_partial_preserved", failure_type=type(exc).__name__,
                       failure_message=str(exc), traceback=traceback.format_exc())
        print(json.dumps(dict(status="failed_partial_preserved", receipt=str(
            (output / "receipt.json").relative_to(ROOT)))), flush=True)
        return 1
    print(json.dumps(dict(status=journal.record["status"],
                          receipt=str((output / "receipt.json").relative_to(ROOT)))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
