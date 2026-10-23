"""Readable transcription of recovered author CNN; no source execution loader."""
from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import sys

from .. import european_native_models as io

AUTHOR = dict(
    path="research/reviews/author_repository_deep_audit_20261023_v1/ancestralpaths/history_source/previous_training.py",
    sha256="fc90bb333e3589f3e4f76098dfc591222de1e40c7382a3f367e876270393b4eb")
AUTHOR_COMMIT = "e6f5d9c6be628007c662e1c71a07358cf350efaf"
FLATTEN_SOURCE = "research/reviews/author_repository_file_agreement_20261023_v2/ancestralpaths/keras_2_3_1_core.py"
WEIGHT_SHAPES = [[5, 1, 1, 32], [32], [288, 64], [64], [64, 6], [6]]


def load_runtime(prefix, *, device, threads):
    """Select an explicitly provisioned TF2.15 runtime; never install or fall back."""
    io._need(device in ("cpu", "gpu") and type(threads) is int and threads > 0
             and Path(sys.prefix).resolve() == io._path(prefix),
             "Explicit project runtime, CPU/GPU device and thread count required")
    io._need(importlib.metadata.version("keras") == "2.15.0",
             "Qualified Keras2.15.0 required")
    if device == "cpu":
        io._need(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                 and importlib.metadata.version("tensorflow-cpu") == "2.15.1",
                 "Explicit zero-CUDA TFCPU2.15.1 runtime required")
    else:
        io._need(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"),
                 "ROOT must bind the admitted physical GPU before runtime import")
    tf = importlib.import_module("tensorflow")
    io._need(tf.__version__ == "2.15.1", "Qualified TF2.15.1 required")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    gpus = tf.config.list_physical_devices("GPU")
    io._need((not gpus) if device == "cpu" else len(gpus) == 1,
             "Runtime must expose exactly the admitted device condition")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    return tf


def build_cnn(tf, *, layout="native_nchw"):
    """Original NCHW CNN and explicit, weight-equivalent NHWC CPU option.

    Original Keras2.3.1 Flatten moves channels to the final axis before flatten.
    The portable option performs that permutation before Conv2D and produces
    the same [N,1,9,32] flatten order. No historical checkpoint is read.
    """
    io._checked(AUTHOR)
    io._need(layout in ("native_nchw", "portable_nhwc"), "Declared CNN layout required")
    layers = tf.keras.layers
    inputs = layers.Input(shape=(1, 5, 9), dtype="float32", name="native_nchw_input")
    kwargs = dict(filters=32, kernel_size=(5, 1), strides=(1, 1), padding="valid",
                  activation="relu", use_bias=True, kernel_initializer="glorot_uniform",
                  bias_initializer="zeros", name="author_conv")
    if layout == "native_nchw":
        x = layers.Conv2D(data_format="channels_first", **kwargs)(inputs)
        x = layers.Flatten(data_format="channels_first", name="author_flatten")(x)
    else:
        x = layers.Permute((2, 3, 1), name="explicit_nchw_to_nhwc")(inputs)
        x = layers.Conv2D(data_format="channels_last", **kwargs)(x)
        x = layers.Flatten(data_format="channels_last", name="author_flatten")(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="glorot_uniform",
                     bias_initializer="zeros", name="author_dense64")(x)
    outputs = layers.Dense(6, activation="softmax", kernel_initializer="glorot_uniform",
                           bias_initializer="zeros", name="author_dense6")(x)
    network = tf.keras.Model(inputs, outputs, name="ancestralpaths_historical_cnn")
    # Legacy Adam retains Keras2 bias correction and epsilon placement.
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
        decay=0.0, amsgrad=False)
    network.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    io._need(network.count_params() == 19078 and network.output_shape == (None, 6)
             and [list(w.shape) for w in network.weights] == WEIGHT_SHAPES
             and int(network.optimizer.iterations.numpy()) == 0,
             "Original CNN shapes, 19078 parameters and fresh optimizer required")
    return network


def model_spec():
    return dict(author_training_ref=AUTHOR, author_commit=AUTHOR_COMMIT,
        flatten_source_ref=io._ref(io.ROOT / FLATTEN_SOURCE),
        source_lines=[23, 199, 203, 205, 206, 207, 208, 210, 211, 213],
        input_shape=[1, 5, 9], convolution_output_nchw=[32, 1, 9],
        flattened_index="32 * feature_position + filter",
        flatten_semantics="NCHW -> NHWC -> row-major flatten", parameters=19078,
        weight_shapes=WEIGHT_SHAPES, loss="categorical_crossentropy",
        optimizer=dict(name="legacy.Adam", learning_rate=0.001, beta_1=0.9,
                       beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False),
        initialization="Glorot uniform kernels; zero biases; fresh declared seed",
        batch_size=30, epochs=5, shuffle=True, optimizer_updates=8335,
        selection="final_epoch5", validation_selection=False,
        tensorflow="2.15.1", keras="2.15.0",
        exact_historical_runtime_and_random_stream_reproduced=False,
        old_checkpoint_initialization=False,
        cpu_layout="explicit NCHW-to-NHWC before equivalent Conv2D; author flatten order",
        historical_checkpoint_format="HDF5",
        paper_count_training_is_distinct_from_author_812500_row_sampler=True)
