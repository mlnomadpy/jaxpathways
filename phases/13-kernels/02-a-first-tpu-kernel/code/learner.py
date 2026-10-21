"""A first TPU kernel · learner-only exercises (3/3 contracts ready)"""

# Exercise: Change the activation shape to (17,257), freeze a NumPy random seed and compare two target-compatible block choices in interpretation. Keep the bias values distinct by column.

# Inputs:
# - A frozen-seed random matrix shaped (17,257), distinct per-column bias values, and two target-compatible tile choices.

# Outputs:
# - Interpreted outputs, exact shapes, and NumPy agreement for each tile.

# Invariants to check:
# - Both candidates consume identical inputs and bias values.
# - The final partial tiles do not change output shape or values.

# Hints:
# - Generate the fixture once from a local seeded generator.
# - Compare block choices with the same reference and tolerance.

import numpy as np

def compare_activation_tiles(interpreter, seed, block_choices):
    # TODO: build one seeded (17,257) fixture with distinct column biases.
    raise NotImplementedError("Compare target-compatible tiles on fixed inputs")



# Reject hidden bias-axis broadcasting (Practice):
# Give the wrapper a matrix-shaped bias and then a wrong-length vector. Verify both fail before launch.

# Inputs:
# - The lesson wrapper and bias vectors with matrix-shaped and wrong-length variants.

# Outputs:
# - Two captured validation failures before launch.

# Invariants to check:
# - Bias rank and length exactly match the column axis.
# - No implicit broadcasting is accepted by this kernel interface.

# Hints:
# - Test rank first, then the expected column length.
# - Assert the launch callback was not reached on invalid inputs.

def reject_invalid_bias(wrapper, matrix, bias_matrix, short_bias):
    # TODO: exercise both malformed cases and confirm validation precedes launch.
    raise NotImplementedError("Check rank and length for the per-column bias")



# Check a permutation invariant (Challenge):
# Permute columns and bias entries together, then undo the permutation. Verify the output is unchanged.

# Inputs:
# - A matrix, distinct per-column bias, and a nontrivial column permutation.

# Outputs:
# - Original output, inverse-permuted output, and equality check.

# Invariants to check:
# - Matrix columns and bias entries move together.
# - Undoing the permutation restores the original output.

# Hints:
# - Apply the same permutation to columns and bias.
# - Invert the permutation only on the output column axis.

import numpy as np

def check_column_permutation(kernel, matrix, bias, permutation):
    # TODO: compare the original output with the restored permuted output.
    raise NotImplementedError("Test equivariance under joint column and bias permutation")
