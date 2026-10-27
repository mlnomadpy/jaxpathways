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




# Account for the actual nested timeline (Transfer):
# Use the retained 2 ms condition: learner_step [122,2787] us, input_wait [125,2645] us, model_and_wait [2668,2787] us. Compute the union of the child intervals, the outer remainder, and the naive sum that adds the parent to both children. Explain which interval is the measured host-path bottleneck and what this trace cannot establish.

# Inputs:
# - Read phases/08-performance/03-read-an-xprof-trace/data/retained_cpu_timeline_p2_81.json.

# Outputs:
# - Child interval union, enclosing-step remainder, naive nested sum, and bounded critical-path interpretation.

# Invariants to check:
# - Intervals use microseconds; timestamps plus duration equal end time; nested parent is not independent work.

# Hints:
# - The children in the 2 ms condition are ordered and disjoint. The parent contains both.

import json
from pathlib import Path

def interpret_retained_cpu_timeline(path):
    data=json.loads(Path(path).read_text())
    events={e['name']:e for e in data['events'] if e['condition']=='2 ms injected wait'}
    # TODO: Verify start + duration = end; calculate child union and remainder;
    # then demonstrate the incorrect parent + child sum as a negative control.
    raise NotImplementedError

