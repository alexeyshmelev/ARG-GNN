"""Qualify real CPU TensorFlow APIs for the source-native AncestralPaths adapter.

Handmade random arrays only. No biological dataset, targets or author CLI.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Dedicated root and empty CUDA visibility required")
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    output.mkdir(exist_ok=False)
    checkpoint_dir = ROOT / "checkpoints/ancestralpaths_native_runtime_v1"
    checkpoint_dir.mkdir(exist_ok=False)
    receipt = {"hypothesis_id": "H014", "kind": "ancestralpaths_real_tensorflow_CPU_API_qualification",
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "input": "twelve deterministic handmade random feature matrices and balanced synthetic labels",
               "biological_data_used": False, "author_CLI_executed": False,
               "GPU_used": False, "scientific_accuracy_claimed": False}
    start = time.monotonic()
    status = 1
    try:
        import numpy as np
        import tensorflow as tf
        import tskit
        import msprime
        from arggnn.ancestralpaths_native_bridge import build_native_keras_model, native_model_spec
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("Unexpected GPU device in CPU-only qualification")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.keras.utils.set_random_seed(17)
        tf.config.experimental.enable_op_determinism()
        receipt["runtime"] = {"python": platform.python_version(), "executable": os.sys.executable,
                             "packages": {p: importlib.metadata.version(p) for p in
                             ("numpy", "tensorflow-cpu", "keras", "tskit", "msprime", "demes", "pandas")}}
        receipt["native_model_spec"] = native_model_spec()
        model = build_native_keras_model(tf.keras)
        if model.count_params() != 26630 or model.input_shape != (None, 5, 9) or model.output_shape != (None, 6):
            raise RuntimeError("Native architecture contract mismatch")
        x = np.random.default_rng(17).normal(size=(12, 5, 9)).astype(np.float32)
        y = tf.keras.utils.to_categorical(np.tile(np.arange(6), 2), num_classes=6)
        before = [w.numpy().copy() for w in model.trainable_weights]
        initial = model(x, training=False).numpy()
        if not np.isfinite(initial).all() or not np.allclose(initial.sum(axis=-1), 1, atol=1e-6):
            raise RuntimeError("Invalid six-class forward probabilities")
        loss, accuracy = model.train_on_batch(x, y)
        if not np.isfinite([loss, accuracy]).all():
            raise RuntimeError("Invalid synthetic update")
        weight_delta = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(before, model.trainable_weights))
        if weight_delta <= 0:
            raise RuntimeError("No parameter changed in actual training update")
        predictions = model(x, training=False).numpy()
        checkpoint = checkpoint_dir / "native_dense_api_check.keras"
        model.save(checkpoint)
        replay_model = tf.keras.models.load_model(checkpoint)
        replay = replay_model(x, training=False).numpy()
        delta = float(np.abs(predictions - replay).max())
        if delta > 1e-7:
            raise RuntimeError("Checkpoint predictions changed beyond qualification tolerance")
        # Actual pinned tree-library imports and handmade table roundtrip.
        tables = tskit.TableCollection(sequence_length=10)
        child = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        parent = tables.nodes.add_row(time=10)
        tables.edges.add_row(0, 10, parent, child)
        tables.sort()
        ts = tables.tree_sequence()
        if ts.num_trees != 1 or ts.first().parent(child) != parent:
            raise RuntimeError("Pinned tskit handmade tree API failure")
        receipt.update(status="passed", parameters=model.count_params(), forward_shape=list(initial.shape),
                       actual_tensorflow_forward=True, actual_synthetic_optimizer_steps=1,
                       maximum_weight_delta=weight_delta, checkpoint_replay_max_abs_delta=delta,
                       checkpoint={"path": str(checkpoint.relative_to(ROOT)),
                                   "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                       actual_tskit_msprime_import=True,
                       biological_training_performed=False,
                       native_author_numerical_reproduction=False)
        status = 0
    except Exception:
        receipt.update(status="failed", error=traceback.format_exc())
    finally:
        receipt["seconds"] = time.monotonic() - start
        receipt["source_artifacts"] = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in
            ("scripts/qualify_ancestralpaths_runtime.py", "src/arggnn/ancestralpaths_native_bridge.py",
             "configs/ancestralpaths_native_cpu_requirements_v1.txt")}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        (output / "exit_status").write_text(str(status) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(output), "seconds": receipt["seconds"]}))
    raise SystemExit(status)

if __name__ == "__main__":
    main()
