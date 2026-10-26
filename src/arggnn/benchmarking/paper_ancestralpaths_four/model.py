"""Fresh four-output CNN preserving the recovered author's layer construction."""
from __future__ import annotations

from .. import european_native_models as io
from ..paper_ancestralpaths.model import AUTHOR, AUTHOR_COMMIT, FLATTEN_SOURCE, load_runtime
from . import CONDITION, PATH_NAMES

WEIGHT_SHAPES = [[5, 1, 1, 32], [32], [256, 64], [64], [64, 4], [4]]
PARAMETERS = 16900


def build_cnn(tf, *, layout="native_nchw"):
    """Build new weights and optimizer with [1,5,8] input and four outputs.

    Supplement 3.1 changes the distribution width and number of classes. The
    recovered author convolution, hidden width, flatten order and optimizer are
    retained. No six-output checkpoint or optimizer state is read or transferred.
    """
    io._checked(AUTHOR)
    io._need(layout in ("native_nchw", "portable_nhwc"), "Declared CNN layout required")
    layers = tf.keras.layers
    inputs = layers.Input(shape=(1, 5, 8), dtype="float32", name="native_nchw_input")
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
    outputs = layers.Dense(4, activation="softmax", kernel_initializer="glorot_uniform",
                           bias_initializer="zeros", name="author_dense4")(x)
    network = tf.keras.Model(inputs, outputs, name=CONDITION)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
        decay=0.0, amsgrad=False)
    network.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    io._need(network.count_params() == PARAMETERS and network.output_shape == (None, 4)
             and [list(w.shape) for w in network.weights] == WEIGHT_SHAPES
             and int(network.optimizer.iterations.numpy()) == 0,
             "Four-output CNN shapes, 16900 parameters and fresh optimizer required")
    return network


def expected_optimizer_updates(training_rows):
    io._need(type(training_rows) is int and training_rows > 0, "Positive training row count required")
    return 5 * ((training_rows + 29) // 30)


def model_spec():
    return dict(
        condition=CONDITION, author_training_ref=AUTHOR, author_commit=AUTHOR_COMMIT,
        author_source_lines=[205, 206, 207, 208, 210, 211],
        paper_section="Supplementary Note 3.1, printed pp. 43-44",
        flatten_source_ref=io._ref(io.ROOT / FLATTEN_SOURCE),
        input_shape=[1, 5, 8], convolution_output_nchw=[32, 1, 8],
        flattened_index="32 * feature_position + filter",
        flatten_semantics="NCHW -> NHWC -> row-major flatten",
        parameters=PARAMETERS, weight_shapes=WEIGHT_SHAPES,
        class_names=list(PATH_NAMES), loss="categorical_crossentropy",
        optimizer=dict(name="legacy.Adam", learning_rate=0.001, beta_1=0.9,
                       beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False),
        batch_size=30, epochs=5, shuffle=True, selection="final_epoch5",
        validation_selection=False, old_checkpoint_initialization=False,
        initialization="fresh Glorot uniform kernels and zero biases",
        probability_collapsing=False, six_class_weights_reused=False,
        tensorflow="2.15.1", keras="2.15.0",
        exact_historical_runtime_and_random_stream_reproduced=False,
        published_four_path_training_counts_specified=False,
        four_path_architecture_is_paper_guided_source_reconstruction=True)
