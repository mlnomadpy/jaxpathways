"""Move models between JAX, Keras, TensorFlow and PyTorch · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a batch with four rows and preserve the same parameters. Verify each row independently using a feature-by-feature sum.

# Inputs:
# - A dense layer weight/bias pair, the source preprocessing rule, and one or more raw input rows.

# Outputs:
# - JAX-compatible predictions for each input row plus a per-feature independent arithmetic check.

# Invariants to check:
# - Weight layout, input normalization, bias, activation, and batch axis are all explicit at the framework boundary.

# Hints:
# - Use an asymmetric weight matrix and inputs so a transpose or scale mismatch is visible.
# - Compare the same raw values through both paths before testing a new batch.

import numpy as np
import jax
import jax.numpy as jnp

def map_dense_inference(weights, bias, inputs, input_transform):
    """JAX-compatible predictions for each input row plus a per-feature independent arithmetic check."""
    # TODO: Keep raw input normalization and dense weight layout explicit, then compare each output row with a NumPy reference.
    raise NotImplementedError




# Detect a preprocessing mismatch (Transfer):
# Suppose the source model expects inputs divided by \(255\), while the target receives raw pixels. Demonstrate why correct weight transfer still fails.

# Inputs:
# - Suppose the source model expects inputs divided by \(255\), while the target receives raw pixels. Demonstrate why correct weight transfer still fails.

# Outputs:
# - A pair of prediction paths using identical transferred weights.

# Invariants to check:
# - Changing only input normalization changes the model input seen by the dense layer.

# Hints:
# - Compute both predictions from the same raw input: once with the source normalization and once without it. Matching weights cannot cancel an input-scale mismatch.
# - Transform the raw pixel row before inference and compare it with the target path.

import numpy as np
import jax
import jax.numpy as jnp

def practice_detect_a_preprocessing_mismatch(source_model, target_model, raw_pixels, source_scale):
    """Complete the detect a preprocessing mismatch check."""
    # TODO: ('Apply source scaling and target raw-input paths to the same pixels and compare their dense outputs.', 'Two predictions and a mismatch attributable to preprocessing while weights remain fixed.', 'The only changed factor is input normalization; transferred parameters are identical.')
    raise NotImplementedError




# Map a second layer and locate an omitted activation (Transfer / diagnosis):
# Extend the dense mapping with ReLU and a second dense layer. Use independently computed final scores to catch a converter that forgets ReLU. Then show why an all-positive probe would be a weak test.

# Inputs:
# - Extend the dense mapping with ReLU and a second dense layer. Use independently computed final scores to catch a converter that forgets ReLU. Then show why an all-positive probe would be a weak test.

# Outputs:
# - Independent scores for a two-layer dense mapping with an intermediate activation.

# Invariants to check:
# - A negative hidden preactivation exercises the ReLU boundary in the parity check.

# Hints:
# - Compute both first-layer rows by hand, replace negative entries by zero, and apply the new output weights.
# - Compare the first layer, activation, and second layer in sequence.

import numpy as np
import jax
import jax.numpy as jnp

def practice_map_a_second_layer_and_locate_an_omitted_activation(weight1, bias1, weight2, bias2, inputs):
    """Complete the map a second layer and locate an omitted activation check."""
    # TODO: ('Implement both dense layers with the intermediate ReLU and compare against independent feature-by-feature scores on a probe containing a negative hidden activation.', 'Independent two-layer scores and a converter failure when the activation is omitted.', 'The comparison includes a probe where ReLU changes the intermediate representation.')
    raise NotImplementedError

