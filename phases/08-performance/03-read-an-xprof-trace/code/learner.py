"""Read an XProf trace · learner-only exercises (2/2 contracts ready)"""

# Exercise: Convert every model interval to milliseconds and summarize its minimum, median and maximum. Keep the individual samples and explain why four observations is distinct from a tail-latency service guarantee.

# Inputs:
# - Named nested trace intervals in a declared time unit.

# Outputs:
# - Per-model min/median/max and non-double-counted interval summaries.

# Invariants to check:
# - Nested event durations are not summed as independent wall time; small sample count is not a tail guarantee.

# Hints:
# - Convert units before summarizing.
# - Compute interval unions when events overlap.

import gzip
import json
from pathlib import Path

def summarize_trace_intervals(intervals):
    """Per-model min/median/max and non-double-counted interval summaries."""
    # TODO: Parse interval units, identify nesting, and summarize model timings without double-counting overlaps.
    raise NotImplementedError




# Compute a union instead of summing overlap (Transfer):
# For hypothetical intervals [0,10], [2,7] and [8,12] microseconds, calculate their union duration and compare with their summed durations. Explain why this matters for nested host events.

# Inputs:
# - For hypothetical intervals [0,10], [2,7] and [8,12] microseconds, calculate their union duration and compare with their summed durations. Explain why this matters for nested host events.

# Outputs:
# - Union duration and naive sum in microseconds.

# Invariants to check:
# - Overlapping time is counted once in wall-clock union duration.

# Hints:
# - Sort endpoints and merge intervals that intersect.

import gzip
import json
from pathlib import Path

def practice_compute_a_union_instead_of_summing_overlap(intervals):
    """Implement the compute a union instead of summing overlap check."""
    # TODO: Sort the three intervals, merge overlaps, and compare union duration with the sum of individual durations.
    raise NotImplementedError

