"""Bootstrap .venv and verify a TPU runtime receipt · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute `closed_form_sum_of_squares(4)` and `closed_form_sum_of_squares(8)`, verify that `closed_form_sum_of_squares(4) == 10416.0`, and assert that `receipt['verified']` is `True`.

# Inputs:
# - The closed-form sum helper and a runtime receipt produced by the active CPU or TPU backend.

# Outputs:
# - Theoretical sums for the requested four- and eight-chip mappings, plus the receipt verification flag, observed backend, and actual device counts.

# Invariants to check:
# - The four-chip theoretical total matches the exercise value and the active receipt reports its actual backend and counts. A CPU receipt may pass this local exercise but does not qualify as physical TPU evidence.

# Hints:
# - Evaluate the helper for both requested chip counts.
# - Keep theoretical chip-mapping totals separate from observed device arithmetic; read verified status, backend, and device counts from the receipt.

def audit_runtime_sum_receipt(receipt, closed_form_sum_of_squares):
    # TODO: check both theoretical totals and report the active receipt fields as observed.
    raise NotImplementedError



# Detect a silent fallback from 4 TPU chips to 1 CPU device (Foundations):
# Write a check `is_four_chip_tpu_receipt(r)` that returns `True` only when `r['backend'] == 'tpu'`, `r['local_device_count'] == 4`, and `r['observed_total'] == 10416.0`.

# Inputs:
# - A candidate runtime receipt mapping with backend, local device count, and observed total fields.

# Outputs:
# - A Boolean predicate for the exact four-chip TPU receipt contract.

# Invariants to check:
# - All required fields must match; missing fields or CPU backend return false.

# Hints:
# - Validate backend, count, and total independently.
# - Use a tolerance only for the floating representation of the exact integer total.

import jax.numpy as jnp
import numpy as np

def is_four_chip_tpu_receipt(receipt):
    # TODO: return whether all required four-chip TPU receipt fields are valid.
    raise NotImplementedError



# Estimate hourly billing savings from immediate teardown (Transfer / diagnosis):
# If a 4-chip TPU slice costs \(4.80\) USD/hour and verification plus experiment takes \(0.5\) hours, compute how much you save by deleting the VM immediately instead of keeping it running for \(16\) total hours, including the active time.

# Inputs:
# - Hourly slice cost, active verification duration of 0.5 hours, and total elapsed VM time of 16 hours.

# Outputs:
# - Avoided idle duration and the corresponding teardown savings.

# Invariants to check:
# - Total elapsed time includes the active work, so idle time is total time minus active time; immediate teardown and the overnight choice share the active-time cost.

# Hints:
# - Subtract active hours from total hours to get avoidable idle hours.
# - Multiply only those idle hours by the hourly rate; 16 idle hours would be $76.80, while 16 total hours including 0.5 active hour yields $74.40.

def idle_teardown_savings(hourly_cost, active_hours, total_hours):
    # TODO: compute avoidable idle hours and savings when total_hours includes active work.
    raise NotImplementedError
