"""Compiled loops with lax.scan · learner-only exercises (3/3 contracts ready)"""

# Exercise: Make the decay an explicit scalar argument. Differentiate the final value with respect to decay and compare at \(0.5\) with the analytic derivative \(2.75\).

# Inputs:
# - The initial scalar state, increments, and explicit decay scalar from the exercise.

# Outputs:
# - The final scan value, per-step history, autodiff derivative with respect to decay, and comparison with the analytic value at 0.5.

# Invariants to check:
# - The final value is differentiated through the scan; the derivative check uses decay 0.5 and the stated analytic reference.

# Hints:
# - Pass decay as an explicit scan-transition input.
# - Differentiate the function returning only the final carry, then compare at the requested decay.

import jax
import jax.numpy as jnp

def check_scan_decay_derivative(initial, increments, decay):
    # TODO: Return the scan history and final carry, then differentiate the final carry with respect to decay.
    raise NotImplementedError




# Record a different output (Practice):
# Carry the state, but record its squared value. Confirm that final carry still means state.

# Inputs:
# - An initial scalar carry and a sequence of increments.

# Outputs:
# - The final state and the squared state recorded at every step.

# Invariants to check:
# - The carry remains unsquared between transitions.
# - Recorded output has the same leading length as increments.

# Hints:
# - Compute the next state once in the transition.
# - Return that state as carry and its square as the scan output.

import jax
import jax.numpy as jnp

def state_and_energy(initial: jax.Array, increments: jax.Array, decay: float = 0.5) -> tuple[jax.Array, jax.Array]:
    """Scan a scalar state while recording its squared energy."""
    # TODO: define the transition and return lax.scan outputs.
    raise NotImplementedError




# Diagnose a growing carry (Challenge):
# Try appending each input to the carry with concatenate. Explain why scan rejects it and replace it with stacked output.

# Inputs:
# - A fixed-shape scan carry and a sequence whose values are to be recorded.

# Outputs:
# - A scan result whose carry shape stays fixed while all per-step values are stacked in output.

# Invariants to check:
# - Carry shape and dtype agree on every iteration.
# - History length equals the sequence length, including empty sequences.

# Hints:
# - A growing array belongs in the scan output, not in carry.
# - Keep the carry fixed and return each new value as the output.

import jax
import jax.numpy as jnp

def collect_scan(initial: jax.Array, values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Keep a fixed carry and collect each input in the scan history."""
    # TODO: implement a fixed-type carry transition with per-step output.
    raise NotImplementedError

