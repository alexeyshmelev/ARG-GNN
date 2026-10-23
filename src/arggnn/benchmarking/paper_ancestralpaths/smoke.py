"""Small synthetic CPU qualification; no benchmark fit, TEST truth or CUDA."""
from __future__ import annotations
import json
from pathlib import Path
import time

from . import metrics
from .model import build_cnn, load_runtime, model_spec
from .. import european_native_models as io


def run(output_directory):
    import numpy as np
    output = io._new_output(output_directory)
    started = time.monotonic()
    tf = load_runtime(Path(__import__("sys").prefix), device="cpu", threads=1)
    tf.keras.utils.set_random_seed(20261024)
    rng = np.random.default_rng(20261024)
    x = rng.uniform(-0.4, 0.7, (3, 1, 5, 9)).astype(np.float32)
    native = build_cnn(tf, layout="native_nchw")
    portable = build_cnn(tf, layout="portable_nhwc")
    weights = [rng.uniform(-0.15, 0.15, w.shape).astype(np.float32) for w in native.weights]
    native.set_weights(weights); portable.set_weights(weights)
    w, b, d, db, o, ob = weights
    conv = np.maximum(np.einsum("nhw,hf->nwf", x[:, 0], w[:, 0, 0]) + b, 0)
    flat = conv.reshape(3, 288)
    hidden = np.maximum(flat @ d + db, 0)
    logits = hidden @ o + ob
    manual = np.exp(logits - logits.max(1, keepdims=True))
    manual /= manual.sum(1, keepdims=True)
    native_supported = True
    native_runtime_error = None
    try:
        native_actual = native(x, training=False).numpy()
    except tf.errors.UnimplementedError as error:
        native_supported = False
        native_runtime_error = str(error)
    actual = portable(x, training=False).numpy()
    adapted = native_actual if native_supported else actual
    err_manual = float(np.max(np.abs(actual - manual)))
    err_portable = float(np.max(np.abs(actual - adapted)))
    io._need(err_manual < 2e-6 and err_portable < 2e-6,
             "Manual convolution/dense/softmax and both layouts must agree")
    intermediate = tf.keras.Model(portable.input, portable.get_layer("author_flatten").output)
    err_flat = float(np.max(np.abs(intermediate(x).numpy() - flat)))
    io._need(err_flat < 2e-6, "Author feature-position-major/filter-minor ordering required")
    order = np.arange(2*32*9, dtype=np.float32).reshape(2, 32, 1, 9)
    keras_flat = tf.keras.layers.Flatten(data_format="channels_first")(order).numpy()
    expected_flat = order.transpose(0, 2, 3, 1).reshape(2, 288)
    io._need(np.array_equal(keras_flat, expected_flat)
        and not np.array_equal(keras_flat, order.reshape(2, 288)),
        "Keras2.3.1 channels-first Flatten semantics required")

    # Independent Keras2 Adam equation: uncorrected moments and bias-corrected lr,
    # with epsilon outside sqrt(v), as recovered from standalone Keras.
    variable = tf.Variable([1., -2.], dtype=tf.float32)
    optimizer = tf.keras.optimizers.legacy.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    reference = np.array([1., -2.], np.float32)
    m = np.zeros(2, np.float32); v = np.zeros(2, np.float32)
    errors = []
    for t, gradient in enumerate(([0.2, -0.3], [0.1, 0.4], [-0.4, 0.05]), 1):
        g = np.asarray(gradient, dtype=np.float32)
        beta1, beta2 = np.float32(.9), np.float32(.999)
        m = beta1*m + (np.float32(1)-beta1)*g
        v = beta2*v + (np.float32(1)-beta2)*g*g
        alpha = np.float32(.001) * np.sqrt(np.float32(1)-beta2**t)/(np.float32(1)-beta1**t)
        reference -= alpha * m / (np.sqrt(v) + np.float32(1e-7))
        optimizer.apply_gradients([(tf.constant(g), variable)])
        errors.append(float(np.max(np.abs(variable.numpy() - reference))))
    io._need(max(errors) < 5e-7 and int(optimizer.iterations.numpy()) == 3,
             "Independent historical Adam update equations must agree")
    y = tf.keras.utils.to_categorical(np.array([0, 1, 5]), 6)
    for _ in range(2):
        result = portable.train_on_batch(x, y)
        io._need(np.isfinite(result).all(), "Finite synthetic train-on-batch required")
    io._need(int(portable.optimizer.iterations.numpy()) == 2, "Two fresh synthetic Adam updates required")
    saved = output / "synthetic_cnn.h5"
    portable.save(saved)
    restored = tf.keras.models.load_model(saved)
    save_error = float(np.max(np.abs(portable(x).numpy() - restored(x).numpy())))
    optimizer_weights_equal = (len(portable.optimizer.get_weights()) == len(restored.optimizer.get_weights())
        and all(np.array_equal(a, b) for a, b in
                zip(portable.optimizer.get_weights(), restored.optimizer.get_weights())))
    io._need(save_error == 0 and int(restored.optimizer.iterations.numpy()) == 2
        and optimizer_weights_equal,
        "Exact checkpoint probabilities, iteration and every optimizer slot required")

    # Asymmetric confusion distinguishes precision columns from recall rows.
    labels = np.asarray([0, 0, 0, 1, 1, 2], np.int64)
    calls = np.asarray([0, 1, 1, 1, 2, 2])
    p = np.eye(6, dtype=np.float32)[calls]
    scored = metrics.endpoints(p, labels)
    native_metrics = scored["native6"]
    io._need(native_metrics["per_class"]["CHG_nonBAA"]["precision"] == 1/3
        and native_metrics["per_class"]["CHG_nonBAA"]["recall"] == 1/2
        and native_metrics["raw_NLL_is_infinite"]
        and native_metrics["source_style_confusion_predicted_rows_truth_columns"] ==
            np.asarray(native_metrics["confusion"]).T.tolist(),
        "Confusion orientation, precision denominator and zero-probability NLL required")
    # Founder collapse must sum probabilities BEFORE argmax.
    collapse = metrics.endpoints(np.asarray([[.3, .4, .05, .05, .19, .01]]), np.asarray([4]))
    io._need(collapse["native6"]["accuracy"] == 0 and collapse["derived_C4"]["accuracy"] == 1,
             "Derived C4 must collapse probabilities before making calls")
    summary = metrics.parent_summary([.8, .9, .7, .85, .75])
    io._need(abs(summary["sample_SD"] - np.std([.8,.9,.7,.85,.75],ddof=1)) < 1e-15,
             "Equal-parent sample SD must use ddof1")
    receipt = dict(schema="paper_ancestralpaths_synthetic_cpu_smoke_v1", status="PASS",
        device="cpu", tensorflow=tf.__version__, keras="2.15.0", threads=1,
        synthetic_only=True, scientific_training_or_prediction=False,
        test_truth_read=False, original_checkpoint_read=False,
        parameters=portable.count_params(), model_spec=model_spec(),
        qualified_layout="portable_nhwc", native_nchw_cpu_supported=native_supported,
        native_nchw_runtime_error=native_runtime_error,
        manual_forward_max_abs_error=err_manual, native_portable_max_abs_error=err_portable if native_supported else None,
        flatten_manual_max_abs_error=err_flat, flatten_wrong_reshape_rejected=True,
        adam_three_step_max_abs_errors=errors, synthetic_model_updates=2,
        checkpoint_probability_max_abs_error=save_error,
        checkpoint_optimizer_iterations=int(restored.optimizer.iterations.numpy()),
        every_optimizer_slot_roundtrip_equal=optimizer_weights_equal,
        asymmetric_precision_and_recall_checked=True, C4_probability_sum_checked=True,
        sample_SD_ddof1_checked=True, elapsed_seconds=time.monotonic()-started,
        checkpoint_ref=io._ref(saved))
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return receipt
