"""Design an input pipeline · learner-only exercises (3/3 contracts ready)"""

# Exercise: Pad the final three-label batch to four elements with zero. Show that an unmasked mean differs, then derive the correct masked mean.

# Inputs:
# - The final three labels from the lesson’s last batch.

# Outputs:
# - Four padded label slots, a validity mask, the ordinary padded mean, and the masked mean.

# Invariants to check:
# - The padded unmasked mean differs from the mean of the three real labels; the masked mean equals that real-label mean.

# Hints:
# - Build the mask from the number of real labels, not from label values.
# - Divide the masked sum by the count of valid elements.

import numpy as np

def compare_padded_and_masked_mean(labels):
    # TODO: Pad three labels to four with zero and compare the two means using a validity mask.
    raise NotImplementedError




# Fingerprint meaning, not just a row count (Transfer):
# Reorder features and labels together, then change one label without changing any shape. Verify that both changes alter the dataset fingerprint while copying unchanged arrays preserves it. Explain why a new iterator order differs from a changed source dataset.

# Inputs:
# - Reorder features and labels together, then change one label without changing any shape. Verify that both changes alter the dataset fingerprint while copying unchanged arrays preserves it. Explain why a new iterator order differs from a changed source dataset.

# Outputs:
# - Three fingerprint comparisons for reordered data, changed labels, and copied arrays.

# Invariants to check:
# - Changes to row order or values alter the fingerprint while exact copies preserve it.

# Hints:
# - The fingerprint includes dtype, shape and array bytes.

import numpy as np
import hashlib

def practice_fingerprint_meaning_not_just_a_row_count(features, labels):
    """Implement the fingerprint meaning, not just a row count check."""
    # TODO: Hash canonical feature and label bytes plus shape/dtype metadata; compare reorder, label-edit, and unchanged-copy cases.
    raise NotImplementedError




# Catch a feature/label shuffle mismatch (Challenge):
# Reverse the target order inside a batch but keep features unchanged. Reject this batch with an independent alignment check, then repair shared indexing.

# Inputs:
# - Reverse the target order inside a batch but keep features unchanged. Reject this batch with an independent alignment check, then repair shared indexing.

# Outputs:
# - An accepted jointly shuffled batch and a rejected split-shuffled batch.

# Invariants to check:
# - Features and labels carry the same row IDs after batching.

# Hints:
# - The known relation \(y=2x-1\) provides a stronger oracle than a falling training loss.

import numpy as np
import hashlib

def practice_catch_a_feature_label_shuffle_mismatch(features, labels, order):
    """Implement the catch a feature/label shuffle mismatch check."""
    # TODO: Apply one shared permutation to aligned arrays, then deliberately reverse only labels and reject the mismatched rows.
    raise NotImplementedError

