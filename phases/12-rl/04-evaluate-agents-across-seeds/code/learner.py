"""Evaluate agents across seeds · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute the paired held-out improvement over the uniform baseline for each trained agent. Report all \(5\) differences and their mean, keeping the training run as the unit.

# Inputs:
# - A vector of five held-out means, one per trained policy.
# - The baseline mean evaluated on the same streams.

# Outputs:
# - The five paired per-agent improvements and their mean.

# Invariants to check:
# - Keep one result per training run.
# - The baseline uses the same held-out streams for every agent.
# - Do not treat episodes as independent training runs.

# Hints:
# - Subtract the shared baseline from each agent mean before averaging.
# - Preserve the per-agent differences for inspection.

import numpy as np

def paired_improvements(agent_means, baseline_mean):
    # TODO: compute one difference for each independently trained policy
    differences = ...
    # TODO: return the individual differences and their average
    raise NotImplementedError




# Expose an unequal-batch averaging bug (Transfer / diagnosis):
# One evaluation shard contains one episode with return \(1\); another contains nine episodes with return \(0\). Compare the unweighted mean of shard means with the true episode mean.

# Inputs:
# - Two evaluation shards with means 1 and 0.
# - Episode counts 1 and 9.

# Outputs:
# - An unweighted average of shard means and the episode-weighted mean.

# Invariants to check:
# - Each episode has equal weight in the episode-level mean.
# - Unequal shard sizes change the result if shard means are averaged equally.

# Hints:
# - Use the episode counts as weights.
# - Keep the two summaries named to expose the aggregation choice.

import numpy as np

def compare_shard_averages():
    means = np.array([1., 0.])
    counts = np.array([1, 9])
    # TODO: compute an unweighted and episode-weighted summary
    raise NotImplementedError




# Distinguish spread from standard error (Transfer / diagnosis):
# For agent means \((0.8,0.9,1.0)\), compute sample standard deviation and the naive standard error assuming independent agent measurements. Explain why neither automatically describes our shared-stream protocol.

# Inputs:
# - Three trained-agent means: 0.8, 0.9, and 1.0.
# - A sample-standard-deviation calculation and the number of agents.

# Outputs:
# - Across-agent spread and a naive standard-error calculation.

# Invariants to check:
# - Use sample standard deviation with ddof=1.
# - Label the unit of observation as a trained agent.

# Hints:
# - The standard error divides spread by the square root of the independent-agent count.
# - Explain the limits of that calculation under shared evaluation streams.

import numpy as np

def summarize_agent_spread():
    agent_means = np.array([.8, .9, 1.])
    # TODO: calculate sample spread and the naive standard error
    raise NotImplementedError

