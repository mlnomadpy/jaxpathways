"""Copy to my_workflow.py. Implement the contracts in README.md, one stage at a time."""


def resolve_config(value):
    # Stage 1: require the exact schema; reject bool-as-int, nonfinite rates,
    # duplicate loggers, invalid cadence, unsupported backend/precision.
    # Copy the input, sort logger names and return the resolved configuration.
    raise NotImplementedError('Implement configuration validation')


def validate_history(rows, config):
    # Stage 2: derive expected logged steps, including zero and final step.
    # Require exact columns, ordered unique steps, finite nonnegative values.
    # At step zero gradient_norm and update_ms must be None, not invented zeros.
    raise NotImplementedError('Implement the metric observation contract')


def paired_summary(records, expected_seeds):
    # Stage 3: require exactly one baseline and candidate per declared seed.
    # Compute candidate-minus-baseline validation MSE for each paired seed.
    # Return seeds, differences, count, mean_difference, sample_sd_difference,
    # baseline_mean, candidate_mean. Sample SD divides by n-1.
    raise NotImplementedError('Implement complete paired comparison')
