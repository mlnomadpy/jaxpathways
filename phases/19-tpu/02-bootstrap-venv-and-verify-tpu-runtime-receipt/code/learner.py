"""Bootstrap .venv and verify a TPU runtime receipt · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute the 4-device and 8-device oracle values. On the available runtime, assert arithmetic_ok; assert requested_target_ready only when observed_backend is tpu and device_count is exactly 4.

# Inputs:
# - The closed-form sum helper and the active CPU/TPU receipt with expected and observed backend/device counts.

# Outputs:
# - Calculated four-/eight-device oracle values, actual arithmetic result and requested-target readiness.

# Invariants to check:
# - arithmetic_ok is separate from expected_backend_ok and expected_topology_ok.
# - requested_target_ready requires the requested TPU backend and exact expected count; a CPU rehearsal does not qualify TPU hardware.

# Hints:
# - Use observed_backend and device_count from the actual receipt.
# - Test a schema-only wrong-count control separately from real hardware observations.

def audit_runtime_sum_receipt(receipt, closed_form_sum_of_squares):
    # TODO: compare both theoretical totals and assert readiness matches the requested backend/count contract.
    raise NotImplementedError




# Detect a silent fallback from 4 TPU chips to 1 CPU device (Foundations):
# Write `is_four_chip_tpu_receipt(r)` using the real receipt schema: return True only when `r['observed_backend'] == 'tpu'`, `r['local_device_count'] == 4`, `r['device_count'] == 4`, and `r['observed_total']` is 10416.0. Missing keys should return False.

# Inputs:
# - A candidate runtime receipt mapping with backend, local device count, and observed total fields.

# Outputs:
# - A Boolean predicate for the exact four-chip TPU receipt contract.

# Invariants to check:
# - Use observed_backend from the emitted receipt.
# - Missing keys, CPU backend, wrong local/global device count, or wrong synchronized total return False.

# Hints:
# - Validate backend, count, and total independently.
# - Use a tolerance only for the floating representation of the exact integer total.

import numpy as np

def is_four_chip_tpu_receipt(receipt):
    try:
        # Use the schema returned by build_runtime_receipt.
        return bool(
            receipt["observed_backend"] == "tpu"
            and receipt["local_device_count"] == 4
            and receipt["device_count"] == 4
            and np.isclose(receipt["observed_total"], 10416.0)
        )
    except (KeyError, TypeError):
        return False



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
