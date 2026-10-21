"""Train a CNN on a small dataset · learner-only exercises (3/3 contracts ready)"""

# Exercise: Apply the trained model to one \(10\times10\) image. Predict the intermediate activation shape and total parameter count before executing, then verify both assertions.

# Inputs:
# - The trained convolutional classifier and one image with shape (1, 10, 10, channels).

# Outputs:
# - The intermediate activation shape for that image, the trainable parameter count, and the prediction shape.

# Invariants to check:
# - The observed activation dimensions follow the convolution and pooling sequence; parameter count is independent of input batch size.

# Hints:
# - Trace spatial dimensions through each convolution and pooling operation.
# - Count model parameter leaves once; do not count activations as parameters.

import jax
import jax.numpy as jnp
from flax import nnx

def inspect_single_image(model, image):
    # TODO: Run one image, capture the requested intermediate shape, and count trainable parameter scalars.
    raise NotImplementedError




# Read a confusion matrix as conditional rates (Transfer):
# Normalize each confusion-matrix row by its true-class count. Explain the diagonal and verify that weighting class recalls by class counts reproduces overall accuracy. Then remove a class in a constructed case and report its recall as undefined.

# Inputs:
# - Normalize each confusion-matrix row by its true-class count. Explain the diagonal and verify that weighting class recalls by class counts reproduces overall accuracy. Then remove a class in a constructed case and report its recall as undefined.

# Outputs:
# - Per-class recalls, undefined absent-class recall, and reconstructed accuracy.

# Invariants to check:
# - Rows are conditioned on true labels; classes with zero support are not assigned a numeric recall.

# Hints:
# - Keep the integer counts; a normalized row without a count hides how much evidence supports it.

import jax.numpy as jnp

def practice_read_a_confusion_matrix_as_conditional_rates(confusion, class_counts):
    """Implement the read a confusion matrix as conditional rates check."""
    # TODO: Divide each true-class row by its support, mask zero-support recall, and reconstruct accuracy by support weighting.
    raise NotImplementedError




# Catch a channel-order mistake (Intermediate):
# Transpose NHWC images to NCHW and show why this convolution rejects them. Repair the layout and compare predictions.

# Inputs:
# - Transpose NHWC images to NCHW and show why this convolution rejects them. Repair the layout and compare predictions.

# Outputs:
# - The failure case and repaired NHWC output/prediction comparison.

# Invariants to check:
# - The input batch follows the convolution module’s declared spatial/channel axis order.

# Hints:
# - The final axis is the input-channel dimension for NNX Conv.

import jax.numpy as jnp

def practice_catch_a_channel_order_mistake(model, images):
    """Implement the catch a channel-order mistake check."""
    # TODO: Transpose the declared NHWC batch to NCHW to reproduce the shape rejection, then restore NHWC and compare predictions.
    raise NotImplementedError

