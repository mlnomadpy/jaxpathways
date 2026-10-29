"""LLMOps: version prompts, evaluate behavior and trace requests · learner-only exercises (3/3 contracts ready)"""

# Exercise: Inspect case-sensitive-refund-77 in the baseline trace. Identify the correct retrieval evidence and the prompt behavior that caused the answer to fail, then change only the prompt policy and compare on all five IDs.

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

def practice_check_abstention_structure(case_id, response, supported):
    # Return (case ID, pass flag, failure reason). Keep the ID even when invalid.
    # TODO: an unsupported case passes only when response is an empty abstention.
    raise NotImplementedError

result=practice_check_abstention_structure('unsupported-tax-2',
    {'response':'Tax is 15 percent','citations':[],'abstain':True},False)
print(result)



# Keep failed work in the evaluation inventory (Transfer):
# Replace the answer for the first answerable case with an empty abstention. Keep the case ID and show how the complete-suite score changes.

# Inputs:
# - Replace the answer for the first answerable case with an empty abstention. Keep the case ID and show how the complete-suite score changes.

# Outputs:
# - Per-case result inventory and complete-suite score.

# Invariants to check:
# - The failed case ID remains in the denominator after its answer is replaced.

# Hints:
# - Start with the grounded candidate responses. An empty abstention is correct for the unsupported case but wrong for the selected answerable case.
# - Do not drop failed or abstained responses.

def practice_keep_failed_work_in_the_evaluation_inventory(cases, responses):
    # Replace the first supported case with an empty abstention, retain its ID,
    # then call score(cases, responses) so the original denominator remains.
    # TODO: preserve all case IDs and return the updated result inventory.
    raise NotImplementedError
