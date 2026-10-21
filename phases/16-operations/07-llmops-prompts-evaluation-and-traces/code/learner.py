"""LLMOps: version prompts, evaluate behavior and trace requests · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a response with an invented citation and require the evaluator to fail it.

# Inputs:
# - The lesson’s answerable case, its allowed citations, and a response containing an invented citation.

# Outputs:
# - The evaluator result showing that the unsupported citation fails the case.

# Invariants to check:
# - A citation absent from the case’s allowed evidence cannot pass the citation check.

# Hints:
# - Check citation support against the case evidence, not whether the citation string is well formed.
# - Keep the test response otherwise valid so the invented citation is the failure cause.

def reject_invented_citation(case, response, allowed_citations):
    # TODO: Evaluate the response and confirm its unsupported citation causes a failure.
    raise NotImplementedError




# Check abstention structure (transfer):
# Try a response that claims to abstain but still returns an unsupported answer.

# Inputs:
# - Try a response that claims to abstain but still returns an unsupported answer.

# Outputs:
# - A failed evaluator result for abstention containing an unsupported answer.

# Invariants to check:
# - The case remains in results with a structured failure reason.

# Hints:
# - The fixture contract requires both empty answer text and no citations.
# - Validate response form and evidence support independently.

import os
import tempfile
from pathlib import Path
import mlflow
from mlflow import MlflowClient

def practice_check_abstention_structure(response, allowed_evidence):
    """Complete the check abstention structure check."""
    # TODO: ('Parse an abstaining response that still contains an unsupported claim and fail the structured response check.', 'A rejected malformed abstention with its case ID retained.', 'An abstention response contains no answer claim unsupported by the evidence.')
    raise NotImplementedError




# Keep failed work in the evaluation inventory (Transfer):
# Replace the answer for the first answerable case with an empty abstention. Keep the case ID and show how the complete-suite score changes.

# Inputs:
# - Replace the answer for the first answerable case with an empty abstention. Keep the case ID and show how the complete-suite score changes.

# Outputs:
# - Per-case result inventory and complete-suite score.

# Invariants to check:
# - The failed case ID remains in the denominator after its answer is replaced.

# Hints:
# - Use the contract-aware replies as the starting point. An empty abstention is correct for the unsupported case but wrong for the selected answerable case.
# - Do not drop failed or abstained responses.

import os
import tempfile
from pathlib import Path
import mlflow
from mlflow import MlflowClient

def practice_keep_failed_work_in_the_evaluation_inventory(cases, responses):
    """Complete the keep failed work in the evaluation inventory check."""
    # TODO: ('Replace the first answerable case output with an empty abstention while retaining its ID, then recompute full-suite score.', 'Complete case inventory and changed full-suite score.', 'Failed and abstained cases remain in the denominator and cannot be dropped.')
    raise NotImplementedError

