"""Preflight and independently check the saved-decoder sequence-PPO bridge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RECIPE_REL = "projects/posttraining-audit/transformer_lineage/recipe.json"
FIXTURE_REL = "projects/posttraining-audit/transformer_lineage/fixture.json"
REQUIREMENTS_REL = "projects/posttraining-audit/transformer_lineage/requirements.txt"
SCHEMA_REL = "projects/posttraining-audit/transformer_lineage/evidence-schema.json"
README_REL = "projects/posttraining-audit/transformer_lineage/README.md"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def require_external_temp_output(path: Path, label: str) -> Path:
    """Require fresh evidence under a platform temp root, outside the checkout."""
    resolved = Path(path).resolve()
    temp_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/private/tmp").resolve(),
    }
    if resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"{label} output must be outside the source checkout")
    if not any(resolved.is_relative_to(root) for root in temp_roots):
        raise ValueError(f"{label} output must be under a system temporary directory")
    return resolved


def require_bound_bytes(relative: str, raw: bytes, expected: str, approval_files: dict):
    actual = sha256_bytes(raw)
    if actual != expected or approval_files.get(relative) != expected:
        raise ValueError(f"approval byte binding failed: {relative}")
    return actual


def preflight(
    approval_path: Path,
    expected_approval_sha256: str,
    *,
    run: bool,
    learner_verification: bool = False,
    implementation_path: Path | None = None,
):
    """Verify every authority/source byte using stdlib before candidate/JAX import."""
    approval_path = Path(approval_path).resolve()
    if not approval_path.is_file():
        raise ValueError("approval file does not exist")
    approval_bytes = approval_path.read_bytes()
    approval_sha = sha256_bytes(approval_bytes)
    if approval_sha != expected_approval_sha256:
        raise ValueError("approval bytes do not match caller-supplied expected SHA-256")
    approval = json.loads(approval_bytes)
    if approval.get("schemaVersion") != 1 or not isinstance(
        approval.get("filesSha256"), dict
    ):
        raise ValueError("approval schema is invalid")
    files = approval["filesSha256"]

    recipe_path = REPO_ROOT / RECIPE_REL
    fixture_path = REPO_ROOT / FIXTURE_REL
    recipe_bytes = recipe_path.read_bytes()
    fixture_bytes = fixture_path.read_bytes()
    recipe_sha = sha256_bytes(recipe_bytes)
    fixture_sha = sha256_bytes(fixture_bytes)
    recipe = json.loads(recipe_bytes)
    fixture = json.loads(fixture_bytes)
    if recipe.get("fixture", {}).get("sha256") != fixture_sha:
        raise ValueError("recipe does not bind the exact fixture bytes")

    required_files = {RECIPE_REL: recipe_sha, FIXTURE_REL: fixture_sha}
    for relative, digest in recipe["parent"]["packageFilesSha256"].items():
        path = f"projects/text-harness/foundations/reference/run/inference-package/{relative}"
        required_files[path] = digest
    for relative, digest in recipe["parent"]["foundationSourceSha256"].items():
        path = f"projects/text-harness/foundations/{relative}"
        required_files[path] = digest
    for relative, expected in required_files.items():
        path = REPO_ROOT / relative
        require_bound_bytes(relative, path.read_bytes(), expected, files)

    # Test the same rejection path with changed, still-parseable JSON bytes.
    mutation_controls = []
    for relative in (RECIPE_REL, FIXTURE_REL):
        raw = (REPO_ROOT / relative).read_bytes()
        changed = raw[:-1] + (b" " if raw.endswith(b"\n") else b"\n")
        try:
            require_bound_bytes(relative, changed, required_files[relative], files)
        except ValueError:
            mutation_controls.append(relative)
        else:
            raise AssertionError(
                f"valid-JSON source mutation was not rejected: {relative}"
            )

    parent_package = REPO_ROOT / recipe["parent"]["packagePath"]
    package_manifest = read_json(parent_package / "package.json")
    if package_manifest.get("processor_sha256") != recipe["processor"]["sha256"]:
        raise ValueError("package manifest processor identity differs from recipe")
    if (
        package_manifest.get("identity", {}).get("code_sha256")
        != recipe["parent"]["foundationSourceSha256"]
    ):
        raise ValueError(
            "parent package foundation source identity differs from recipe"
        )
    if (
        package_manifest.get("params_sha256")
        != recipe["parent"]["packageFilesSha256"]["params.npz"]
    ):
        raise ValueError("parent package parameter identity differs from recipe")

    if run:
        if (
            approval.get("kind") != "sol-transformer-lineage-run-approval"
            or approval.get("runAuthorized") is not True
        ):
            raise ValueError(
                "this approval authorizes source review only, not the optimizer run"
            )
        if approval.get("sourceReviewRequired") is True or approval.get(
            "status"
        ) not in ("run-approved", "approved"):
            raise ValueError("run approval is not in its final authorized state")
        if implementation_path is None or not Path(implementation_path).is_file():
            raise ValueError("candidate implementation is unavailable")
        candidate_sha = sha256_file(implementation_path)
        try:
            candidate_rel = (
                Path(implementation_path).resolve().relative_to(REPO_ROOT).as_posix()
            )
        except ValueError as exc:
            raise ValueError(
                "run candidate must be inside the source-reviewed project tree"
            ) from exc
        expected_candidate_rel = (
            "projects/posttraining-audit/transformer_lineage/solution.py"
        )
        if candidate_rel != expected_candidate_rel:
            raise ValueError(
                "campaign approval authorizes only the pinned solution.py candidate"
            )
        if files.get(candidate_rel) != candidate_sha:
            raise ValueError("run approval does not bind the exact candidate source")
        for relative in (
            REQUIREMENTS_REL,
            SCHEMA_REL,
            README_REL,
            "projects/posttraining-audit/transformer_lineage/check.py",
            "projects/posttraining-audit/transformer_lineage/solution.py",
            "projects/posttraining-audit/transformer_lineage/starter.py",
        ):
            if (
                relative not in files
                or sha256_file(REPO_ROOT / relative) != files[relative]
            ):
                raise ValueError(
                    f"run approval does not bind exact executable/source file: {relative}"
                )
        frozen = approval.get("frozenRecipe")
        training = recipe["training"]
        floors = recipe["evaluationFloors"]
        expected_frozen = {
            "batchSize": training["batchSize"],
            "innerEpochsPerOuterUpdate": training["innerEpochs"],
            "primaryOuterUpdates": recipe["resumeVerification"]["lanes"][0]["updates"],
            "replayBeforeRestoreOuterUpdates": recipe["resumeVerification"]["lanes"][1][
                "updatesBeforeFreshProcessRestore"
            ],
            "replayAfterRestoreOuterUpdates": recipe["resumeVerification"]["lanes"][1][
                "updatesAfterRestore"
            ],
            "totalActorOuterUpdateExecutions": recipe["resumeVerification"][
                "totalActorUpdateExecutions"
            ],
            "totalActorOptimizerSteps": recipe["resumeVerification"][
                "totalActorUpdateExecutions"
            ]
            * training["innerEpochs"],
            "totalCriticOptimizerSteps": recipe["resumeVerification"][
                "totalActorUpdateExecutions"
            ]
            * training["innerEpochs"],
            "totalTrainingRolloutEpisodesAcrossLanes": recipe["resumeVerification"][
                "totalActorUpdateExecutions"
            ]
            * training["batchSize"],
            "rewardHeadFits": recipe["resumeVerification"]["rewardHeadFits"],
            "rewardHeadOptimizerSteps": training["rewardFitUpdates"],
            "maximumRestrictedVisitedContextKlPerValidAction": floors[
                "maximumExactRestrictedCategoricalKlPerValidAction"
            ],
            "actionIds": recipe["actions"]["restrictedVocabularyIds"],
            "qualityScoredLane": recipe["resumeVerification"]["lanes"][0]["id"],
        }
        if not isinstance(frozen, dict) or any(
            frozen.get(key) != value for key, value in expected_frozen.items()
        ):
            raise ValueError(
                "run approval structured frozenRecipe does not match the exact recipe and 16-update budget"
            )

    if learner_verification:
        if (
            approval.get("kind")
            != "sol-transformer-lineage-learner-verification-approval"
        ):
            raise ValueError("this is not a learner-verification approval")
        if (
            approval.get("learnerVerificationAuthorized") is not True
            or approval.get("status") != "mechanics-approved"
            or approval.get("runAuthorized") is not False
        ):
            raise ValueError(
                "learner approval must explicitly authorize mechanics, use mechanics-approved status, and keep campaign run authorization false"
            )
        budget = approval.get("learnerVerification")
        expected_budget = {
            "outerUpdateExecutionsPerCandidate": 1,
            "innerEpochsPerOuterUpdate": int(recipe["training"]["innerEpochs"]),
            "rewardHeadFitsPerCandidate": 1,
            "rewardHeadOptimizerStepsPerCandidate": 4,
            "batchSize": int(recipe["training"]["batchSize"]),
            "qualityScored": False,
            "heldoutSelection": False,
            "cloudOrCorpus": False,
        }
        if not isinstance(budget, dict) or any(
            budget.get(key) != value for key, value in expected_budget.items()
        ):
            raise ValueError(
                "learner-verification approval budget does not match the bounded one-update mechanics check"
            )
        if implementation_path is None or not Path(implementation_path).is_file():
            raise ValueError("learner candidate is unavailable")
        for relative in (
            REQUIREMENTS_REL,
            SCHEMA_REL,
            README_REL,
            "projects/posttraining-audit/transformer_lineage/check.py",
            "projects/posttraining-audit/transformer_lineage/solution.py",
            "projects/posttraining-audit/transformer_lineage/starter.py",
        ):
            if (
                relative not in files
                or sha256_file(REPO_ROOT / relative) != files[relative]
            ):
                raise ValueError(
                    f"learner-verification approval does not bind exact checker/source file: {relative}"
                )

    return {
        "approval_sha256": approval_sha,
        "approval_kind": approval.get("kind"),
        "approval_status": approval.get("status"),
        "run_authorized": approval.get("runAuthorized") is True,
        "recipe_sha256": recipe_sha,
        "fixture_sha256": fixture_sha,
        "parent_package_files_sha256": recipe["parent"]["packageFilesSha256"],
        "parent_foundation_source_sha256": recipe["parent"]["foundationSourceSha256"],
        "parent_identity": package_manifest["identity"],
        "mutation_controls_rejected": mutation_controls,
        "recipe": recipe,
        "fixture": fixture,
    }


def _load_candidate(path: Path, project_root: Path):
    os.environ["TRANSFORMER_LINEAGE_PROJECT_ROOT"] = str(project_root)
    spec = importlib.util.spec_from_file_location("lineage_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load selected candidate module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_close(actual, expected, *, atol=1e-6, label="numeric result"):
    import numpy as np

    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=atol):
        raise AssertionError(
            f"{label}: actual={np.asarray(actual)} expected={np.asarray(expected)}"
        )


def _independent_scalar_contracts(candidate):
    """Small hand oracles grade candidate math without source spelling checks."""
    import numpy as np

    eos_records = {
        "target_role_mask": np.asarray([[1, 1]], dtype=bool),
        "terminated": np.asarray([[0, 1]], dtype=bool),
        "truncated": np.asarray([[0, 0]], dtype=bool),
        "rewards": np.asarray([[0.0, 2.0]], dtype=np.float32),
        "old_values": np.asarray([[0.4, 0.6]], dtype=np.float32),
        "post_action_values": np.asarray([[0.6, 0.7]], dtype=np.float32),
    }
    got = candidate.generalized_advantage_estimate(eos_records, gamma=0.9, lam=0.8)
    _assert_close(got["deltas"], [[0.14, 1.4]], label="EOS deltas")
    _assert_close(got["advantages"], [[1.148, 1.4]], label="EOS GAE")
    _assert_close(got["value_targets"], [[1.548, 2.0]], label="EOS value targets")

    timeout_records = {
        "target_role_mask": np.asarray([[1, 1, 1]], dtype=bool),
        "terminated": np.asarray([[0, 0, 0]], dtype=bool),
        "truncated": np.asarray([[0, 0, 1]], dtype=bool),
        "rewards": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        "old_values": np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32),
        "post_action_values": np.asarray([[0.3, 0.5, 0.8]], dtype=np.float32),
    }
    got = candidate.generalized_advantage_estimate(timeout_records, gamma=0.9, lam=0.8)
    _assert_close(got["deltas"], [[0.07, 0.15, 1.22]], label="timeout deltas")
    _assert_close(got["advantages"], [[0.810448, 1.0284, 1.22]], label="timeout GAE")
    if np.asarray(got["advantages"])[0, 2] == 0:
        raise AssertionError("timeout terminal reward was lost")

    ratios = np.asarray([1.4, 0.6], dtype=np.float32)
    advantages = np.asarray([2.0, -2.0], dtype=np.float32)
    loss = candidate.ppo_policy_loss(
        np.log(ratios), np.zeros(2), advantages, np.ones(2), clip=0.2
    )
    _assert_close(loss, -0.4, label="signed PPO clipping")

    new_values = np.asarray([[2.0, -1.0]], dtype=np.float32)
    old_values = np.asarray([[0.0, 1.0]], dtype=np.float32)
    targets = np.asarray([[1.0, 3.0]], dtype=np.float32)
    mask = np.asarray([[1.0, 1.0]], dtype=np.float32)
    # Hand calculation: clipped values [0.2, 0.8], squared-error maxima [1, 16].
    value_loss = candidate.clipped_value_loss(
        new_values, old_values, targets, mask, clip=0.2
    )
    _assert_close(value_loss, 4.25, label="raw-dominant clipped critic loss")
    clipped_dominant = candidate.clipped_value_loss(
        np.asarray([[2.0]], np.float32),
        np.asarray([[0.0]], np.float32),
        np.asarray([[3.0]], np.float32),
        np.asarray([[1.0]], np.float32),
        clip=0.2,
    )
    _assert_close(clipped_dominant, 3.92, label="clipped-dominant critic loss")

    # A padding value with arbitrary magnitude must not change either loss.
    padded = candidate.clipped_value_loss(
        np.asarray([[2.0, -1.0, 999.0]], np.float32),
        np.asarray([[0.0, 1.0, -999.0]], np.float32),
        np.asarray([[1.0, 3.0, 1e6]], np.float32),
        np.asarray([[1.0, 1.0, 0.0]], np.float32),
        clip=0.2,
    )
    _assert_close(padded, value_loss, label="masked padding critic denominator")


def _tree_arrays_equal(jax, left, right):
    import numpy as np

    a, adef = jax.tree.flatten(left)
    b, bdef = jax.tree.flatten(right)
    if adef != bdef or len(a) != len(b):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def _rollout_boundary_contracts(candidate, parent, state, reference, reward_params):
    """Exercise real decoder prefixes for sampled EOS and truncation boundaries."""
    import numpy as np
    import jax

    eos = candidate.rollout(
        parent,
        state["actor_params"],
        state["critic_params"],
        reference,
        reward_params,
        [0],
        jax.random.PRNGKey(801),
        forced_action_sequences=[[383, 4, 2]],
    )
    timeout = candidate.rollout(
        parent,
        state["actor_params"],
        state["critic_params"],
        reference,
        reward_params,
        [1],
        jax.random.PRNGKey(802),
        forced_action_sequences=[[381, 4, 383, 4]],
    )
    if eos["sampling_mode"] != "forced-contract-probe" or not eos["terminated"][0, 2]:
        raise AssertionError(
            "real decoder EOS probe did not terminate at its sampled EOS action"
        )
    if not np.array_equal(eos["target_role_mask"], [[True, True, True, False]]):
        raise AssertionError(
            "EOS probe target-role or post-EOS padding mask is misaligned"
        )
    if not np.array_equal(eos["input_role_mask"], [[False, True, True, False]]):
        raise AssertionError("EOS probe input-role mask does not shift action targets")
    if (
        eos["terminal_reward"][0, :2].any()
        or eos["terminal_reward"][0, 2] != eos["sequence_scores"][0]
    ):
        raise AssertionError(
            "EOS sequence reward is not placed only on its final valid action"
        )
    if eos["post_action_values"][0, 2] != 0.0:
        raise AssertionError("terminal EOS did not use zero value bootstrap")
    if (
        timeout["sampling_mode"] != "forced-contract-probe"
        or not timeout["truncated"][0, 3]
    ):
        raise AssertionError(
            "real decoder timeout probe did not truncate at the declared limit"
        )
    if not timeout["target_role_mask"].all() or timeout["padding_mask"].any():
        raise AssertionError(
            "timeout probe incorrectly masked a valid partial-response action"
        )
    if abs(float(timeout["post_action_values"][0, 3])) <= 1e-7:
        raise AssertionError(
            "timeout probe lacks the required nonzero post-action critic bootstrap"
        )
    if (
        timeout["terminal_reward"][0, :3].any()
        or timeout["terminal_reward"][0, 3] != timeout["sequence_scores"][0]
    ):
        raise AssertionError(
            "truncated sequence reward is not placed at its final valid action"
        )
    return {"eos": eos, "timeout": timeout}


def _verify_raw_update_kl(update, recipe):
    """Independently aggregate saved conditional KL over valid action targets."""
    import numpy as np

    mask = np.asarray(update["target_role_mask"], dtype=bool)
    per_context = np.asarray(
        update["exact_restricted_conditional_kl"], dtype=np.float64
    )
    old_p = np.asarray(update["old_restricted_probabilities"], dtype=np.float64)
    parent_p = np.asarray(update["parent_restricted_probabilities"], dtype=np.float64)
    support = np.asarray(update["eligible_action_mask"], dtype=bool)
    if per_context.shape != mask.shape:
        raise AssertionError(
            "saved conditional KL array shape differs from action mask"
        )
    if (
        old_p.shape != (*mask.shape, 4)
        or parent_p.shape != old_p.shape
        or support.shape != old_p.shape
    ):
        raise AssertionError(
            "saved restricted probabilities or action support have incorrect shape"
        )
    if (
        np.any(support[~mask])
        or np.any(old_p[~mask] != 0)
        or np.any(parent_p[~mask] != 0)
    ):
        raise AssertionError(
            "padding contains probability mass or eligible action support"
        )
    if not np.array_equal(old_p > 0, support) or not np.array_equal(
        parent_p > 0, support
    ):
        raise AssertionError(
            "old and reference distributions do not use the same exact support"
        )
    if np.any(np.abs(old_p[mask].sum(axis=-1) - 1.0) > 1e-6) or np.any(
        np.abs(parent_p[mask].sum(axis=-1) - 1.0) > 1e-6
    ):
        raise AssertionError("saved restricted action probabilities are not normalized")
    independently_computed = np.zeros_like(per_context)
    for row, position in zip(*np.nonzero(mask)):
        p = old_p[row, position, support[row, position]]
        q = parent_p[row, position, support[row, position]]
        independently_computed[row, position] = np.sum(p * (np.log(p) - np.log(q)))
    if not np.allclose(per_context, independently_computed, atol=1e-6, rtol=1e-6):
        raise AssertionError(
            "saved conditional KL does not reconstruct from raw restricted probabilities"
        )
    actions = np.asarray(update["actions"], dtype=np.int64)
    old_logps = np.asarray(update["old_logps"], dtype=np.float64)
    parent_logps = np.asarray(update["parent_logps"], dtype=np.float64)
    sampled_kl = np.asarray(update["sampled_kl"], dtype=np.float64)
    for row, position in zip(*np.nonzero(mask)):
        action_index = {383: 0, 381: 1, 4: 2, 2: 3}[int(actions[row, position])]
        if not np.isclose(
            old_logps[row, position],
            np.log(old_p[row, position, action_index]),
            atol=1e-6,
        ):
            raise AssertionError(
                "selected old log probability does not reconstruct from raw probabilities"
            )
        if not np.isclose(
            parent_logps[row, position],
            np.log(parent_p[row, position, action_index]),
            atol=1e-6,
        ):
            raise AssertionError(
                "selected parent log probability does not reconstruct from raw probabilities"
            )
        if not np.isclose(
            sampled_kl[row, position],
            old_logps[row, position] - parent_logps[row, position],
            atol=1e-6,
        ):
            raise AssertionError(
                "sampled KL component does not reconstruct from selected log probabilities"
            )
    if np.any(per_context[~mask] != 0.0) or np.any(per_context[mask] < -1e-6):
        raise AssertionError(
            "saved conditional KL has padding exposure or negative values"
        )
    denom = max(int(mask.sum()), 1)
    return float(per_context.sum() / denom)


def _evaluation_smoke_worker_source():
    """Return a tiny fresh-process worker that reloads a zero-step checkpoint."""
    return r"""import hashlib, importlib.util, json, os, sys
from pathlib import Path
import jax
import jax.numpy as jnp

candidate_path, project_root, repo_root, checkpoint, identity_json = sys.argv[1:]
candidate_path, project_root, repo_root, checkpoint = map(
    Path, (candidate_path, project_root, repo_root, checkpoint)
)
identity = json.loads(identity_json)
if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != identity["candidate_source_sha256"]:
    raise RuntimeError("candidate bytes changed before fresh-process smoke import")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["TRANSFORMER_LINEAGE_PROJECT_ROOT"] = str(project_root)
spec = importlib.util.spec_from_file_location("evaluation_smoke_candidate", candidate_path)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load evaluation smoke candidate")
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
parent = candidate.load_verified_parent(repo_root)
candidate.verify_prompt_and_action_ids(parent)
state, reference, _, _ = candidate.initialize(parent)
zero_reward = {"weight": jnp.zeros((3, len(candidate.ACTION_IDS)), dtype=jnp.float32)}
def fresh_template():
    template, template_reference, _, _ = candidate.initialize(parent)
    template["reward_params"] = zero_reward
    template["source_identity"] = identity
    return template, template_reference

primary_template, primary_reference = fresh_template()
primary_restored = candidate.restore_state(checkpoint, primary_template)
replay_template, replay_reference = fresh_template()
replay_restored = candidate.restore_state(checkpoint, replay_template)
state_keys = ("actor_params", "critic_params", "actor_opt_state", "critic_opt_state", "reward_params")
primary_hashes = {key: candidate._tree_sha256(primary_restored[key]) for key in state_keys}
replay_hashes = {key: candidate._tree_sha256(replay_restored[key]) for key in state_keys}
if primary_hashes != replay_hashes:
    raise RuntimeError("two cold restores of the initial checkpoint differ")
for restored in (primary_restored, replay_restored):
    if (restored["update_clock"], restored["inner_epoch_clock"], restored["prompt_cursor"]) != (0, 0, 0):
        raise RuntimeError("initial checkpoint did not restore at zero clocks")
    if restored["source_identity"] != identity:
        raise RuntimeError("initial checkpoint source identity changed on restore")
evaluation = candidate.evaluate(
    parent,
    primary_restored["actor_params"],
    primary_reference,
    primary_restored["critic_params"],
    primary_restored["reward_params"],
)
checkpoint_hashes = {
    name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
    for name in ("state.json", "state.npz")
}
print(json.dumps({
    "evaluation": evaluation,
    "actor_sha256": candidate._tree_sha256(primary_restored["actor_params"]),
    "critic_sha256": candidate._tree_sha256(primary_restored["critic_params"]),
    "reward_sha256": candidate._tree_sha256(primary_restored["reward_params"]),
    "reference_sha256": candidate._tree_sha256(reference),
    "checkpoint_source_identity": primary_restored["source_identity"],
    "primary_restore_hashes": primary_hashes,
    "replay_restore_hashes": replay_hashes,
    "initial_checkpoint_file_hashes": checkpoint_hashes,
    "fresh_process_pid": os.getpid(),
}, sort_keys=True))
"""


def _validate_evaluation_smoke(evaluation, fixture, recipe):
    """Check complete paired exposures and reconstruct each row's conditional KL."""
    import numpy as np

    seeds = recipe["evaluationFloors"]["matchedSamplingSeeds"]
    for split in ("dev", "heldout"):
        result = evaluation[split]
        expected_ids = fixture["matchedEvaluationPromptIds"][split]
        expected_pairs = [
            (seed, prompt_id) for seed in seeds for prompt_id in expected_ids
        ]
        count = len(expected_pairs)
        child_rows = result["child_rollouts"]
        parent_rows = result["matched_parent_rollouts"]
        if (
            result["matched_prompt_ids"] != expected_ids
            or result["matched_sampling_seeds"] != seeds
            or result["unique_matched_episode_count"] != count
            or result["sequence_reward_denominator"] != count
            or len(child_rows) != count
            or len(parent_rows) != count
            or len(result["exact_restricted_conditional_kl_per_sequence"]) != count
        ):
            raise AssertionError(f"{split} evaluation has incomplete matched exposures")
        child_keys, parent_keys = [], []
        child_rewards, parent_rewards, row_kl = [], [], []
        for index, (_, prompt_id) in enumerate(expected_pairs):
            child, baseline = child_rows[index], parent_rows[index]
            if child["prompt_ids"] != [prompt_id] or baseline["prompt_ids"] != [
                prompt_id
            ]:
                raise AssertionError(
                    f"{split} evaluation prompt exposure order changed"
                )
            if child["sampling_key"] != baseline["sampling_key"]:
                raise AssertionError(
                    f"{split} child/parent sampling keys are not matched"
                )
            child_keys.append(tuple(child["sampling_key"]))
            parent_keys.append(tuple(baseline["sampling_key"]))
            if child["reference_sha256"] != baseline["behavior_policy_sha256"]:
                raise AssertionError(
                    f"{split} baseline actor is not the frozen reference"
                )
            if child["reference_sha256"] != baseline["reference_sha256"]:
                raise AssertionError(f"{split} baseline reference identity changed")
            _verify_raw_update_kl(child, recipe)
            _verify_raw_update_kl(baseline, recipe)
            mask = np.asarray(child["target_role_mask"], dtype=bool)
            exact = np.asarray(
                child["exact_restricted_conditional_kl"], dtype=np.float64
            )
            baseline_mask = np.asarray(baseline["target_role_mask"], dtype=bool)
            if not np.array_equal(mask, baseline_mask):
                raise AssertionError(f"{split} matched sequence lengths differ")
            per_sequence_kl = float(exact[mask].sum())
            if not math.isfinite(per_sequence_kl) or per_sequence_kl < -1e-7:
                raise AssertionError(f"{split} per-row conditional KL is not finite")
            if not math.isclose(
                per_sequence_kl,
                float(result["exact_restricted_conditional_kl_per_sequence"][index]),
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    f"{split} per-row conditional KL does not reconstruct"
                )
            row_kl.append(per_sequence_kl)
            child_rewards.append(float(child["sequence_scores"][0]))
            parent_rewards.append(float(baseline["sequence_scores"][0]))
            if not np.array_equal(child["actions"], baseline["actions"]):
                raise AssertionError(
                    f"{split} zero-update child and parent actions differ"
                )
        if len(set(child_keys)) != count or child_keys != parent_keys:
            raise AssertionError(
                f"{split} evaluation reused or mismatched exposure keys"
            )
        if not np.all(np.isfinite(child_rewards + parent_rewards + row_kl)):
            raise AssertionError(f"{split} evaluation contains a non-finite row metric")
        if not math.isclose(
            float(np.mean(child_rewards)),
            float(result["mean_child_reward_per_sequence"]),
            abs_tol=1e-6,
        ) or not math.isclose(
            float(np.mean(parent_rewards)),
            float(result["mean_parent_reward_per_sequence"]),
            abs_tol=1e-6,
        ):
            raise AssertionError(
                f"{split} row rewards do not reconstruct evaluation means"
            )
        valid_actions = sum(
            int(np.asarray(row["target_role_mask"], dtype=bool).sum())
            for row in child_rows
        )
        if valid_actions != result["valid_action_denominator"]:
            raise AssertionError(
                f"{split} valid-action denominator is not reconstructible"
            )
        expected_mean_kl = sum(row_kl) / valid_actions
        if not math.isclose(
            expected_mean_kl,
            float(
                result[
                    "mean_exact_restricted_conditional_kl_per_valid_action_at_child_visited_contexts"
                ]
            ),
            abs_tol=1e-6,
        ):
            raise AssertionError(f"{split} conditional KL mean does not reconstruct")
    return {
        split: {
            "matched_episode_count": evaluation[split]["unique_matched_episode_count"],
            "valid_action_denominator": evaluation[split]["valid_action_denominator"],
            "per_row_kl_reconstructed": len(
                evaluation[split]["exact_restricted_conditional_kl_per_sequence"]
            ),
        }
        for split in ("dev", "heldout")
    }


def _fresh_process_evaluation_smoke(candidate, candidate_path, report):
    """Cold-load an initialized zero-step checkpoint and exercise evaluate()."""
    import jax
    import jax.numpy as jnp

    parent = candidate.load_verified_parent(REPO_ROOT)
    candidate.verify_prompt_and_action_ids(parent)
    state, _, _, _ = candidate.initialize(parent)
    reward_params, metrics = candidate.fit_pairwise_reward_head(parent, max_updates=0)
    if metrics.get("fit_updates") != 0:
        raise AssertionError(
            "evaluation smoke reward head must use zero optimizer steps"
        )
    state["reward_params"] = reward_params
    candidate_sha = sha256_file(candidate_path)
    state["source_identity"] = {
        "recipe_sha256": report["recipe_sha256"],
        "fixture_sha256": report["fixture_sha256"],
        "approval_sha256": report["approval_sha256"],
        "candidate_source_sha256": candidate_sha,
        "reward_head_sha256": candidate._tree_sha256(reward_params),
    }
    with tempfile.TemporaryDirectory(
        prefix="lineage-evaluation-smoke-", dir=tempfile.gettempdir()
    ) as temp_name:
        folder = Path(temp_name)
        checkpoint = folder / "zero-step-checkpoint"
        checkpoint_metadata = candidate.save_state(checkpoint, state)
        primary_restore, primary_reference, primary_hashes, primary_files = (
            _restore_fitted_initial_state(
                candidate,
                parent,
                checkpoint,
                checkpoint_metadata,
                state["source_identity"],
            )
        )
        replay_restore, replay_reference, replay_hashes, replay_files = (
            _restore_fitted_initial_state(
                candidate,
                parent,
                checkpoint,
                checkpoint_metadata,
                state["source_identity"],
            )
        )
        if (
            primary_hashes != replay_hashes
            or primary_files != replay_files
            or candidate._tree_sha256(primary_restore["reward_params"])
            != candidate._tree_sha256(reward_params)
            or candidate._tree_sha256(replay_restore["reward_params"])
            != candidate._tree_sha256(reward_params)
            or candidate._tree_sha256(primary_reference)
            != candidate._tree_sha256(replay_reference)
        ):
            raise AssertionError(
                "same immutable checkpoint did not cold-load for both initial lanes"
            )
        identity = json.dumps(state["source_identity"], sort_keys=True)
        worker = folder / "evaluation-smoke-worker.py"
        worker.write_text(_evaluation_smoke_worker_source())
        env = dict(os.environ)
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["TRANSFORMER_LINEAGE_PROJECT_ROOT"] = str(HERE)
        process = subprocess.run(
            [
                sys.executable,
                str(worker),
                str(Path(candidate_path).resolve()),
                str(HERE),
                str(REPO_ROOT),
                str(checkpoint),
                identity,
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=600,
        )
        if process.returncode != 0:
            raise AssertionError(
                "fresh-process reloaded evaluation smoke failed: "
                + process.stderr[-3000:]
            )
        try:
            worker_result = json.loads(process.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise AssertionError(
                "evaluation smoke worker returned no JSON result"
            ) from exc
        if worker_result["checkpoint_source_identity"] != state["source_identity"]:
            raise AssertionError("evaluation smoke checkpoint source identity changed")
        if (
            worker_result["reward_sha256"]
            != checkpoint_metadata["reward_params_sha256"]
        ):
            raise AssertionError("evaluation smoke reward head was not cold-loaded")
        expected_tree_hashes = {
            "actor_params": checkpoint_metadata["actor_params_sha256"],
            "critic_params": checkpoint_metadata["critic_params_sha256"],
            "actor_opt_state": checkpoint_metadata["actor_opt_state_sha256"],
            "critic_opt_state": checkpoint_metadata["critic_opt_state_sha256"],
            "reward_params": checkpoint_metadata["reward_params_sha256"],
        }
        if (
            worker_result["primary_restore_hashes"] != expected_tree_hashes
            or worker_result["replay_restore_hashes"] != expected_tree_hashes
        ):
            raise AssertionError(
                "two fresh templates did not cold-load identical full-state trees"
            )
        expected_checkpoint_hashes = {
            name: sha256_file(checkpoint / name) for name in ("state.json", "state.npz")
        }
        if (
            worker_result["initial_checkpoint_file_hashes"]
            != expected_checkpoint_hashes
        ):
            raise AssertionError(
                "initial smoke checkpoint bytes changed during cold restores"
            )
        if worker_result["actor_sha256"] != worker_result["reference_sha256"]:
            raise AssertionError(
                "evaluation smoke actor did not remain at parent initialization"
            )
        exposure = _validate_evaluation_smoke(
            worker_result["evaluation"], report["fixture"], report["recipe"]
        )
        return {
            "fresh_process_reloaded_checkpoint": True,
            "zero_step_reward_fit_updates": metrics["fit_updates"],
            "candidate_source_sha256": candidate_sha,
            "checkpoint_reward_head_sha256": checkpoint_metadata[
                "reward_params_sha256"
            ],
            "primary_and_replay_restore_hashes": expected_tree_hashes,
            "same_initial_checkpoint_file_hashes": expected_checkpoint_hashes,
            "campaign_restore_helper_positive": True,
            "child_process_pid": worker_result["fresh_process_pid"],
            "exposures": exposure,
        }


def _restore_fitted_initial_state(
    candidate, parent, checkpoint, checkpoint_metadata, expected_identity
):
    """Initialize a fresh objective template and restore the immutable fit checkpoint."""
    import jax.numpy as jnp

    checkpoint = Path(checkpoint)
    file_hashes_before = {
        name: sha256_file(checkpoint / name) for name in ("state.json", "state.npz")
    }
    template, reference, _, _ = candidate.initialize(parent)
    template["reward_params"] = {
        "weight": jnp.zeros((3, len(candidate.ACTION_IDS)), dtype=jnp.float32)
    }
    template["source_identity"] = dict(expected_identity)
    restored = candidate.restore_state(checkpoint, template)
    expected_hashes = {
        "actor_params": checkpoint_metadata["actor_params_sha256"],
        "critic_params": checkpoint_metadata["critic_params_sha256"],
        "actor_opt_state": checkpoint_metadata["actor_opt_state_sha256"],
        "critic_opt_state": checkpoint_metadata["critic_opt_state_sha256"],
        "reward_params": checkpoint_metadata["reward_params_sha256"],
    }
    actual_hashes = {
        name: candidate._tree_sha256(restored[name]) for name in expected_hashes
    }
    if actual_hashes != expected_hashes:
        raise AssertionError(
            "cold-restored initial state differs from saved full state"
        )
    if restored["source_identity"] != expected_identity:
        raise AssertionError("cold-restored initial source/reward identity changed")
    if (
        restored["update_clock"] != 0
        or restored["inner_epoch_clock"] != 0
        or restored["prompt_cursor"] != 0
        or restored["rng_key"] != checkpoint_metadata["rng_key"]
    ):
        raise AssertionError("cold-restored initial clocks, cursor, or RNG changed")
    if candidate._tree_sha256(reference) != parent["params_sha256"]:
        raise AssertionError("cold-restored lane reference differs from frozen parent")
    file_hashes_after = {
        name: sha256_file(checkpoint / name) for name in ("state.json", "state.npz")
    }
    if file_hashes_after != file_hashes_before:
        raise AssertionError(
            "immutable fitted initial checkpoint changed during restore"
        )
    return restored, reference, actual_hashes, file_hashes_after


def _semantic_no_update_contracts(candidate, repo_root, report):
    """Exercise all learner APIs on frozen inputs without any optimizer update."""
    import numpy as np
    import jax

    poisoned = {name: sys.modules.get(name) for name in ("train", "model")}
    sys.modules["train"] = types.ModuleType("train")
    sys.modules["model"] = types.ModuleType("model")
    try:
        parent = candidate.load_verified_parent(repo_root)
    finally:
        for name, previous in poisoned.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    candidate.verify_prompt_and_action_ids(parent)
    expected_foundation = repo_root / "projects/text-harness/foundations"
    if (
        Path(parent["parent_train_module"].__file__).resolve()
        != (expected_foundation / "train.py").resolve()
    ):
        raise AssertionError("parent loader did not bind the pinned train.py bytes")
    if (
        Path(parent["model_module"].__file__).resolve()
        != (expected_foundation / "model.py").resolve()
    ):
        raise AssertionError("parent loader did not bind the pinned model.py bytes")
    state, reference, _, _ = candidate.initialize(parent)
    if candidate._tree_sha256(state["actor_params"]) != parent["params_sha256"]:
        raise AssertionError(
            "semantic candidate did not copy the verified parent actor"
        )
    reward_params, reward_metrics = candidate.fit_pairwise_reward_head(
        parent, max_updates=0
    )
    weight = np.asarray(reward_params["weight"])
    if weight.shape != (3, 4) or np.any(weight != 0):
        raise AssertionError(
            "zero-update reward interface changed its declared initial head"
        )
    if reward_metrics["fit_updates"] != 0 or reward_metrics["pair_count"] != 4:
        raise AssertionError(
            "zero-update reward interface did not report the fixed training-pair contract"
        )
    if not math.isclose(
        reward_metrics["initial_pairwise_loss"], math.log(2.0), abs_tol=1e-6
    ):
        raise AssertionError(
            "zero-update reward initial loss differs from the independent log(2) oracle"
        )
    if not math.isclose(
        reward_metrics["final_pairwise_loss"], math.log(2.0), abs_tol=1e-6
    ):
        raise AssertionError(
            "zero-update reward interface unexpectedly optimized the head"
        )

    eos = candidate.rollout(
        parent,
        state["actor_params"],
        state["critic_params"],
        reference,
        reward_params,
        [report["fixture"]["trainingPromptIds"][0]],
        jax.random.PRNGKey(801),
        forced_action_sequences=[[383, 4, 2]],
    )
    timeout = candidate.rollout(
        parent,
        state["actor_params"],
        state["critic_params"],
        reference,
        reward_params,
        [report["fixture"]["trainingPromptIds"][1]],
        jax.random.PRNGKey(802),
        forced_action_sequences=[[381, 4, 383, 4]],
    )
    if not np.array_equal(eos["actions"], [[383, 4, 2, 0]]) or not np.array_equal(
        eos["target_role_mask"], [[True, True, True, False]]
    ):
        raise AssertionError(
            "semantic candidate EOS action targets or post-EOS padding are misaligned"
        )
    if not np.array_equal(eos["input_role_mask"], [[False, True, True, False]]):
        raise AssertionError(
            "semantic candidate input roles are not shifted from action targets"
        )
    if eos["post_action_values"][0, 2] != 0 or np.any(eos["terminal_reward"] != 0):
        raise AssertionError(
            "zero-reward EOS rollout did not retain terminal bootstrap and reward contracts"
        )
    if (
        not np.array_equal(timeout["actions"], [[381, 4, 383, 4]])
        or not timeout["truncated"][0, 3]
    ):
        raise AssertionError(
            "semantic candidate did not preserve the full valid timeout response"
        )
    if abs(float(timeout["post_action_values"][0, 3])) <= 1e-7:
        raise AssertionError(
            "semantic candidate timeout lacks a nonzero learned post-action value"
        )
    _verify_raw_update_kl(eos, report["recipe"])
    _verify_raw_update_kl(timeout, report["recipe"])

    before_actor = candidate._tree_sha256(state["actor_params"])
    before_critic = candidate._tree_sha256(state["critic_params"])
    state["reward_params"] = reward_params
    state["source_identity"] = {
        "recipe_sha256": report["recipe_sha256"],
        "fixture_sha256": report["fixture_sha256"],
        "approval_sha256": report["approval_sha256"],
        "candidate_source_sha256": sha256_file(candidate.__file__),
        "reward_head_sha256": candidate._tree_sha256(reward_params),
    }
    state_after, history = candidate.train(
        parent, state, reference, reward_params, max_updates=0
    )
    if (
        history
        or state_after["update_clock"] != 0
        or state_after["inner_epoch_clock"] != 0
    ):
        raise AssertionError(
            "zero-update learner train call executed an optimizer update"
        )
    if (
        candidate._tree_sha256(state_after["actor_params"]) != before_actor
        or candidate._tree_sha256(state_after["critic_params"]) != before_critic
    ):
        raise AssertionError(
            "zero-update learner train call changed actor or critic parameters"
        )
    if candidate._tree_sha256(state_after["reward_params"]) != candidate._tree_sha256(
        reward_params
    ):
        raise AssertionError(
            "zero-update learner train call changed the frozen reward head"
        )
    restore_controls = _restore_negative_controls(candidate, state_after)
    return {
        "parent_parameter_tree_sha256": parent["params_sha256"],
        "zero_update_reward_metrics": reward_metrics,
        "forced_eos_rollout": candidate._serializable_rollout(eos),
        "forced_timeout_rollout": candidate._serializable_rollout(timeout),
        "actor_parameter_tree_unchanged": True,
        "critic_parameter_tree_unchanged": True,
        "optimizer_updates": 0,
        "restore_mutation_controls_rejected": restore_controls,
    }


def _restore_negative_controls(candidate, state):
    """Prove full-state restore accepts an exact checkpoint and rejects bad edits."""
    import hashlib
    import shutil
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory(
        prefix="lineage-state-check-", dir=tempfile.gettempdir()
    ) as temp_name:
        root = Path(temp_name)
        base = root / "valid"
        candidate.save_state(base, state)
        restored = candidate.restore_state(base, state)
        for key in (
            "actor_params",
            "critic_params",
            "actor_opt_state",
            "critic_opt_state",
            "reward_params",
        ):
            if candidate._tree_sha256(restored[key]) != candidate._tree_sha256(
                state[key]
            ):
                raise AssertionError(f"positive checkpoint round trip changed {key}")
        for key in (
            "rng_key",
            "prompt_cursor",
            "update_clock",
            "inner_epoch_clock",
            "source_identity",
        ):
            if restored[key] != state[key]:
                raise AssertionError(f"positive checkpoint round trip changed {key}")

        def clone_case(name):
            folder = root / name
            shutil.copytree(base, folder)
            metadata_path = folder / "state.json"
            metadata = json.loads(metadata_path.read_text())
            arrays_path = folder / "state.npz"
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key]) for key in archive.files}
            return folder, metadata, arrays

        def save_case(folder, metadata, arrays):
            np.savez(folder / "state.npz", **arrays)
            metadata["state_npz_sha256"] = hashlib.sha256(
                (folder / "state.npz").read_bytes()
            ).hexdigest()
            (folder / "state.json").write_text(
                json.dumps(metadata, sort_keys=True, indent=2)
            )

        controls = {}
        folder, metadata, arrays = clone_case("changed-optimizer-moment")
        moment_key = next(key for key in arrays if key.startswith("actor_opt_state_"))
        arrays[moment_key] = arrays[moment_key].copy()
        arrays[moment_key].flat[0] += 1
        save_case(folder, metadata, arrays)
        controls["changed_optimizer_moment"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("missing-state-leaf")
        arrays.pop(next(iter(arrays)))
        save_case(folder, metadata, arrays)
        controls["missing_state_leaf"] = _restore_must_reject(candidate, folder, state)

        folder, metadata, arrays = clone_case("nonfinite-parameter")
        parameter_key = next(
            key
            for key in arrays
            if key.startswith("actor_params_")
            and np.issubdtype(arrays[key].dtype, np.floating)
        )
        arrays[parameter_key] = arrays[parameter_key].copy()
        arrays[parameter_key].flat[0] = np.nan
        save_case(folder, metadata, arrays)
        controls["nonfinite_parameter"] = _restore_must_reject(candidate, folder, state)

        folder, metadata, arrays = clone_case("negative-update-clock")
        metadata["update_clock"] = -1
        save_case(folder, metadata, arrays)
        controls["negative_update_clock"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("wrong-critic-hash")
        metadata["critic_params_sha256"] = "0" * 64
        save_case(folder, metadata, arrays)
        controls["critic_parameter_hash"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("out-of-range-cursor")
        metadata["prompt_cursor"] = len(candidate.FIXTURE["trainingPromptIds"])
        save_case(folder, metadata, arrays)
        controls["out_of_range_prompt_cursor"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("inconsistent-cursor")
        metadata["prompt_cursor"] = (int(metadata["prompt_cursor"]) + 1) % len(
            candidate.FIXTURE["trainingPromptIds"]
        )
        save_case(folder, metadata, arrays)
        controls["inconsistent_prompt_cursor"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("wrong-reward-identity")
        metadata["source_identity"]["reward_head_sha256"] = "0" * 64
        save_case(folder, metadata, arrays)
        controls["reward_head_identity"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("tampered-reward-head-values")
        reward_key = next(key for key in arrays if key.startswith("reward_params_"))
        arrays[reward_key] = arrays[reward_key].copy()
        arrays[reward_key].flat[0] += 0.25
        save_case(folder, metadata, arrays)
        controls["tampered_reward_head_values"] = _restore_must_reject(
            candidate, folder, state
        )

        folder, metadata, arrays = clone_case("missing-source-identity")
        metadata["source_identity"].pop("candidate_source_sha256")
        save_case(folder, metadata, arrays)
        controls["missing_source_identity_field"] = _restore_must_reject(
            candidate, folder, state
        )
        return controls


def _restore_must_reject(candidate, folder, template):
    try:
        candidate.restore_state(folder, template)
    except (ValueError, KeyError, AssertionError):
        return True
    raise AssertionError(f"checkpoint restore accepted mutated case {folder.name}")


def _verify_update_math_independently(update, recipe):
    """Rebuild GAE and pre-update PPO/critic losses using NumPy scalar math."""
    import numpy as np

    mask = np.asarray(update["target_role_mask"], dtype=bool)
    rewards = np.asarray(update["rewards"], dtype=np.float64)
    old_values = np.asarray(update["old_values"], dtype=np.float64)
    post_values = np.asarray(update["post_action_values"], dtype=np.float64)
    terminal = np.asarray(update["terminated"], dtype=bool)
    stopped = terminal | np.asarray(update["truncated"], dtype=bool)
    gamma = float(recipe["training"]["gamma"])
    lam = float(recipe["training"]["gaeLambda"])
    deltas = mask * (rewards + gamma * (~terminal) * post_values - old_values)
    advantages = np.zeros_like(deltas)
    carry = np.zeros((deltas.shape[0],), dtype=np.float64)
    for position in range(deltas.shape[1] - 1, -1, -1):
        carry = deltas[:, position] + gamma * lam * (~stopped[:, position]) * carry
        carry = np.where(mask[:, position], carry, 0.0)
        advantages[:, position] = carry
    targets = old_values + advantages
    for name, expected in (
        ("deltas", deltas),
        ("advantages", advantages),
        ("value_targets", targets),
    ):
        if not np.allclose(np.asarray(update[name]), expected, atol=1e-6, rtol=1e-6):
            raise AssertionError(f"raw rollout does not reconstruct independent {name}")
    denominator = max(int(mask.sum()), 1)
    clip = float(recipe["training"]["policyClip"])
    value_clip = float(recipe["training"]["valueClip"])
    metrics = update["epoch_metrics"]
    pre_logps = update["pre_update_logps_per_epoch"]
    pre_values = update["pre_update_values_per_epoch"]
    if len(metrics) != int(recipe["training"]["innerEpochs"]):
        raise AssertionError("learner did not record both inner-epoch loss inputs")
    for epoch, row in enumerate(metrics):
        logps = np.asarray(pre_logps[epoch], dtype=np.float64)
        ratios = np.exp(logps - np.asarray(update["old_logps"], dtype=np.float64))
        surrogate = np.minimum(
            ratios * advantages, np.clip(ratios, 1 - clip, 1 + clip) * advantages
        )
        policy_loss = -float((surrogate * mask).sum() / denominator)
        values = np.asarray(pre_values[epoch], dtype=np.float64)
        clipped_values = old_values + np.clip(
            values - old_values, -value_clip, value_clip
        )
        value_errors = np.maximum(
            (values - targets) ** 2, (clipped_values - targets) ** 2
        )
        critic_loss = 0.5 * float((value_errors * mask).sum() / denominator)
        if not math.isclose(
            float(row["policy_loss"]), policy_loss, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise AssertionError(
                f"inner epoch {epoch} policy loss differs from independent raw reconstruction"
            )
        if not math.isclose(
            float(row["critic_loss"]), critic_loss, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise AssertionError(
                f"inner epoch {epoch} critic loss differs from independent raw reconstruction"
            )
        if int(row["valid_action_denominator"]) != denominator:
            raise AssertionError(
                "policy/critic loss reported a different valid-action denominator"
            )
    return {
        "valid_action_denominator": denominator,
        "independent_gae": True,
        "independent_inner_epoch_losses": True,
    }


def _independent_parent_prompt_feature(parent, fixture, recipe, prompt_id):
    """Compute restricted frozen-parent prompt log probabilities without candidate helpers."""
    import numpy as np
    import jax.numpy as jnp

    prompt = next(
        item for item in fixture["prompts"] if int(item["id"]) == int(prompt_id)
    )
    tokens = jnp.asarray([[1, *map(int, prompt["tokenIds"])]], dtype=jnp.int32)
    segments = jnp.ones_like(tokens, dtype=jnp.int32)
    logits = np.asarray(
        parent["model_module"].logits(
            parent["params"],
            tokens,
            parent["config"],
            valid_mask=jnp.ones_like(tokens, dtype=bool),
            segment_ids=segments,
        )[0, -1],
        dtype=np.float64,
    )
    action_ids = recipe["actions"]["restrictedVocabularyIds"][:3]
    selected = logits[np.asarray(action_ids, dtype=np.int64)]
    selected -= np.max(selected)
    log_norm = np.log(np.exp(selected).sum())
    return np.asarray([*(selected - log_norm), 0.0], dtype=np.float64)


def _verify_reward_fit_independently(
    candidate, parent, report, reward_params, reward_metrics
):
    import numpy as np

    weight = np.asarray(reward_params["weight"], dtype=np.float64)
    fixture, recipe = report["fixture"], report["recipe"]
    if (
        weight.shape != (3, 4)
        or not np.isfinite(weight).all()
        or not np.linalg.norm(weight) > 0
    ):
        raise AssertionError(
            "bounded reward fit did not produce a finite learned interaction head"
        )
    signed_margins = []
    for pair in fixture["preferencePairs"]:
        context = _independent_parent_prompt_feature(
            parent, fixture, recipe, pair["promptId"]
        )

        def score(actions):
            counts = np.zeros(3, dtype=np.float64)
            for token in actions:
                if int(token) == int(
                    recipe["actions"]["singleTokenSpellings"]["eos"]["id"]
                ):
                    break
                counts[{383: 0, 381: 1, 4: 2}[int(token)]] += 1.0
            return float(np.outer(counts, context).reshape(-1) @ weight.reshape(-1))

        label_sign = 2 * int(pair["preferenceLabel"]) - 1
        signed_margins.append(
            label_sign
            * (
                score(pair["chosenActionSequence"])
                - score(pair["rejectedActionSequence"])
            )
        )
    margins = np.asarray(signed_margins)
    independently_computed_loss = float(
        np.mean(np.logaddexp(0.0, -margins)) + 1e-3 * np.mean(weight**2)
    )
    if not np.allclose(
        reward_metrics["training_pair_margins"], margins, atol=1e-5, rtol=1e-5
    ):
        raise AssertionError(
            "reward-fit training margins do not reconstruct from raw head and frozen parent features"
        )
    if not math.isclose(
        float(reward_metrics["final_pairwise_loss"]),
        independently_computed_loss,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "reward-fit loss does not reconstruct from independent fixed-pair math"
        )
    if float(reward_metrics["final_pairwise_loss"]) >= float(
        reward_metrics["initial_pairwise_loss"]
    ):
        raise AssertionError(
            "bounded reward fit did not reduce the fixed training-pair objective"
        )
    if int(reward_metrics["fit_updates"]) != 4 or int(
        reward_metrics["pair_count"]
    ) != len(fixture["preferencePairs"]):
        raise AssertionError(
            "bounded learner check exceeded or misreported its reward-fit budget"
        )
    # Rebuild every frozen pair feature and reproduce all four Adam steps without
    # calling the candidate's reward feature, loss, or optimizer helper.
    import jax
    import jax.numpy as jnp
    import optax

    rows = []
    for pair in fixture["preferencePairs"]:
        context = _independent_parent_prompt_feature(
            parent, fixture, recipe, pair["promptId"]
        )

        def action_counts(actions):
            counts = np.zeros(3, dtype=np.float64)
            for token in actions:
                if int(token) == int(
                    recipe["actions"]["singleTokenSpellings"]["eos"]["id"]
                ):
                    break
                counts[{383: 0, 381: 1, 4: 2}[int(token)]] += 1.0
            return counts

        chosen_feature = np.outer(
            action_counts(pair["chosenActionSequence"]), context
        ).reshape(-1)
        rejected_feature = np.outer(
            action_counts(pair["rejectedActionSequence"]), context
        ).reshape(-1)
        rows.append((chosen_feature, rejected_feature, int(pair["preferenceLabel"])))
    chosen = jnp.asarray(np.stack([row[0] for row in rows]), dtype=jnp.float32)
    rejected = jnp.asarray(np.stack([row[1] for row in rows]), dtype=jnp.float32)
    labels = jnp.asarray([row[2] for row in rows], dtype=jnp.float32)

    def independent_loss(weight_vector):
        signed = (2.0 * labels - 1.0) * ((chosen - rejected) @ weight_vector)
        return jnp.mean(jax.nn.softplus(-signed)) + 1e-3 * jnp.mean(weight_vector**2)

    expected_weight = jnp.zeros((12,), dtype=jnp.float32)
    reward_optimizer = optax.adam(float(recipe["training"]["rewardLearningRate"]))
    expected_opt_state = reward_optimizer.init(expected_weight)
    expected_trace = [
        np.asarray(expected_weight, dtype=np.float32).reshape((3, 4)).tolist()
    ]
    for _ in range(4):
        _, gradient = jax.value_and_grad(independent_loss)(expected_weight)
        updates, expected_opt_state = reward_optimizer.update(
            gradient, expected_opt_state, expected_weight
        )
        expected_weight = optax.apply_updates(expected_weight, updates)
        expected_trace.append(
            np.asarray(expected_weight, dtype=np.float32).reshape((3, 4)).tolist()
        )
    expected_weight = np.asarray(expected_weight, dtype=np.float64).reshape((3, 4))
    if not np.allclose(weight, expected_weight, atol=2e-6, rtol=2e-5):
        raise AssertionError(
            "reward head does not match the independently reconstructed four-step Adam fit"
        )
    trace = np.asarray(reward_metrics.get("fit_trace", []), dtype=np.float64)
    if trace.shape != (5, 3, 4) or not np.allclose(
        trace, np.asarray(expected_trace), atol=2e-6, rtol=2e-5
    ):
        raise AssertionError(
            "reward fit trace does not match independently reconstructed initial state and four Adam steps"
        )
    return {
        "signed_margins": margins.tolist(),
        "independent_final_pairwise_loss": independently_computed_loss,
        "independent_four_step_adam_weight_match": True,
        "independent_full_reward_step_trace_match": True,
    }


def _independent_selected_logps(parent, params, records, recipe):
    """Recompute restricted selected logps directly from the frozen decoder API."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    action_ids = tuple(
        int(value) for value in recipe["actions"]["restrictedVocabularyIds"]
    )
    eos_index = action_ids.index(
        int(recipe["actions"]["singleTokenSpellings"]["eos"]["id"])
    )
    mask = np.asarray(records["target_role_mask"], dtype=bool)
    actions = np.asarray(records["actions"], dtype=np.int64)
    rows = []
    for row in range(mask.shape[0]):
        values = []
        for position in range(mask.shape[1]):
            if not mask[row, position]:
                values.append(jnp.asarray(0.0, dtype=jnp.float32))
                continue
            prefix = records["prefix_token_ids"][row][position]
            tokens = jnp.asarray(prefix, dtype=jnp.int32)[None, :]
            segments = jnp.ones_like(tokens, dtype=jnp.int32)
            logits = parent["model_module"].logits(
                params,
                tokens,
                parent["config"],
                valid_mask=jnp.ones_like(tokens, dtype=bool),
                segment_ids=segments,
            )[0, -1, jnp.asarray(action_ids)]
            logits = logits.astype(jnp.float32)
            eos_eligible = position >= int(recipe["actions"]["minimumNonEosActions"])
            logits = logits.at[eos_index].set(
                jnp.where(eos_eligible, logits[eos_index], -1.0e9)
            )
            log_probs = jax.nn.log_softmax(logits)
            if not eos_eligible:
                log_probs = log_probs.at[eos_index].set(-jnp.inf)
            action_index = action_ids.index(int(actions[row, position]))
            values.append(log_probs[action_index])
        rows.append(jnp.stack(values))
    return jnp.stack(rows)


def _verify_first_ppo_update_independently(
    candidate,
    parent,
    initial_state,
    reference,
    reward_params,
    update,
    trained_state,
    recipe,
):
    """Replay both first-update Adam steps from raw arrays and independent objectives."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    mask_np = np.asarray(update["target_role_mask"], dtype=np.float32)
    mask = jnp.asarray(mask_np)
    old_logps = jnp.asarray(update["old_logps"], dtype=jnp.float32)
    old_values = jnp.asarray(update["old_values"], dtype=jnp.float32)
    targets = jnp.asarray(update["value_targets"], dtype=jnp.float32)
    advantages = jax.lax.stop_gradient(
        jnp.asarray(update["advantages"], dtype=jnp.float32)
    )
    denominator = max(float(mask_np.sum()), 1.0)
    features = np.asarray(update["parent_context_features"], dtype=np.float32)
    # Independently confirm the raw rollout behavior probabilities and critic inputs.
    parent_logps = _independent_selected_logps(parent, parent["params"], update, recipe)
    if not np.allclose(
        np.asarray(update["old_logps"])[mask_np.astype(bool)],
        parent_logps[mask_np.astype(bool)],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "first rollout selected logps are not the frozen parent's actual decoder probabilities"
        )
    expected_features = np.zeros_like(features)
    expected_probabilities = np.zeros_like(
        np.asarray(update["old_restricted_probabilities"], dtype=np.float32)
    )
    actions = np.asarray(update["actions"], dtype=np.int64)
    fixture = read_json(HERE / "fixture.json")
    prompt_map = {int(row["id"]): row for row in fixture["prompts"]}
    max_actions = int(recipe["actions"]["maximumResponseActions"])
    min_non_eos = int(recipe["actions"]["minimumNonEosActions"])
    eos_id = int(recipe["actions"]["singleTokenSpellings"]["eos"]["id"])
    expected_masks = np.asarray(update["target_role_mask"], dtype=bool)
    if not np.array_equal(
        np.asarray(update["padding_mask"], dtype=bool), ~expected_masks
    ):
        raise AssertionError(
            "rollout padding mask is not the inverse of valid target actions"
        )
    expected_input_roles = np.zeros_like(expected_masks)
    for row, prompt_id in enumerate(update["prompt_ids"]):
        action_positions = np.flatnonzero(expected_masks[row])
        if action_positions.size == 0 or not np.array_equal(
            action_positions, np.arange(action_positions.size)
        ):
            raise AssertionError(
                "response action targets must be a nonempty contiguous prefix"
            )
        prompt_tokens = [1, *map(int, prompt_map[int(prompt_id)]["tokenIds"])]
        for position in action_positions:
            expected_prefix = [*prompt_tokens, *map(int, actions[row, :position])]
            if (
                list(map(int, update["prefix_token_ids"][row][position]))
                != expected_prefix
            ):
                raise AssertionError(
                    "rollout logits were not computed from the exact pre-action prefix"
                )
            expected_input_roles[row, position] = position > 0
        row_actions = actions[row, action_positions]
        eos_positions = np.flatnonzero(row_actions == eos_id)
        if eos_positions.size:
            eos_position = int(eos_positions[0])
            if eos_position < min_non_eos or eos_position != len(row_actions) - 1:
                raise AssertionError(
                    "sampled EOS eligibility or post-EOS padding is invalid"
                )
            if not np.asarray(update["terminated"], dtype=bool)[row, eos_position]:
                raise AssertionError("EOS action is missing its terminal marker")
            if np.asarray(update["truncated"], dtype=bool)[row].any():
                raise AssertionError("EOS response was also marked as truncated")
        else:
            if (
                len(row_actions) != max_actions
                or not np.asarray(update["truncated"], dtype=bool)[
                    row, len(row_actions) - 1
                ]
            ):
                raise AssertionError(
                    "non-EOS response did not terminate at the declared max-action timeout"
                )
        if np.asarray(update["truncated"], dtype=bool)[row].sum() != int(
            not bool(eos_positions.size)
        ):
            raise AssertionError(
                "timeout marker does not match the final non-EOS action"
            )
        if np.asarray(update["terminated"], dtype=bool)[row].sum() != int(
            bool(eos_positions.size)
        ):
            raise AssertionError(
                "rollout terminal markers do not match sampled EOS actions"
            )
    if not np.array_equal(
        np.asarray(update["input_role_mask"], dtype=bool), expected_input_roles
    ):
        raise AssertionError(
            "input-role mask is not the one-position shift of action targets"
        )
    for row, position in zip(*np.nonzero(mask_np)):
        # Features are restricted parent log probabilities at the exact pre-action prefix.
        one = {
            "target_role_mask": np.asarray([[True]]),
            "actions": np.asarray([[actions[row, position]]]),
            "prefix_token_ids": [[update["prefix_token_ids"][row][position]]],
        }
        # Compute all four restricted logits, not just the sampled action.
        prefix = one["prefix_token_ids"][0][0]
        tokens = jnp.asarray(prefix, dtype=jnp.int32)[None, :]
        segments = jnp.ones_like(tokens, dtype=jnp.int32)
        logits = (
            parent["model_module"]
            .logits(
                parent["params"],
                tokens,
                parent["config"],
                valid_mask=jnp.ones_like(tokens, dtype=bool),
                segment_ids=segments,
            )[0, -1, jnp.asarray(recipe["actions"]["restrictedVocabularyIds"])]
            .astype(jnp.float32)
        )
        eligible = position >= int(recipe["actions"]["minimumNonEosActions"])
        logits = logits.at[3].set(jnp.where(eligible, logits[3], -1.0e9))
        log_probs = jax.nn.log_softmax(logits)
        if not eligible:
            log_probs = log_probs.at[3].set(-jnp.inf)
        expected_features[row, position] = np.asarray(
            jnp.where(jnp.isfinite(log_probs), log_probs, 0.0)
        )
        expected_probabilities[row, position] = np.asarray(
            jnp.where(jnp.isfinite(log_probs), jnp.exp(log_probs), 0.0)
        )
    if not np.allclose(
        features[mask_np.astype(bool)],
        expected_features[mask_np.astype(bool)],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "critic training features do not reconstruct from frozen parent decoder contexts"
        )
    valid_mask = mask_np.astype(bool)
    if not np.allclose(
        np.asarray(update["old_restricted_probabilities"])[valid_mask],
        expected_probabilities[valid_mask],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "old restricted action probabilities do not reconstruct from the frozen parent"
        )
    if not np.allclose(
        np.asarray(update["parent_restricted_probabilities"])[valid_mask],
        expected_probabilities[valid_mask],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "reference restricted action probabilities do not reconstruct from the frozen parent"
        )
    if not np.allclose(
        np.asarray(update["parent_logps"])[valid_mask],
        parent_logps[valid_mask],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "first rollout reference selected logps differ from the frozen parent"
        )
    initial_values = np.einsum(
        "bta,a->bt",
        expected_features,
        np.asarray(initial_state["critic_params"]["weight"]),
    ) + float(initial_state["critic_params"]["bias"])
    if not np.allclose(
        np.asarray(update["old_values"])[mask_np.astype(bool)],
        initial_values[mask_np.astype(bool)],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "first rollout old values do not reconstruct from the initial critic and frozen features"
        )
    expected_post_values = np.zeros_like(
        np.asarray(update["post_action_values"], dtype=np.float64)
    )
    terminal = np.asarray(update["terminated"], dtype=bool)
    critic_weight = np.asarray(
        initial_state["critic_params"]["weight"], dtype=np.float64
    )
    critic_bias = float(initial_state["critic_params"]["bias"])
    for row, position in zip(*np.nonzero(mask_np)):
        if terminal[row, position]:
            continue
        next_prefix = [
            *map(int, update["prefix_token_ids"][row][position]),
            int(actions[row, position]),
        ]
        tokens = jnp.asarray(next_prefix, dtype=jnp.int32)[None, :]
        segments = jnp.ones_like(tokens, dtype=jnp.int32)
        next_logits = (
            parent["model_module"]
            .logits(
                parent["params"],
                tokens,
                parent["config"],
                valid_mask=jnp.ones_like(tokens, dtype=bool),
                segment_ids=segments,
            )[0, -1, jnp.asarray(recipe["actions"]["restrictedVocabularyIds"])]
            .astype(jnp.float32)
        )
        next_eligible = position + 1 >= min_non_eos
        next_logits = next_logits.at[3].set(
            jnp.where(next_eligible, next_logits[3], -1.0e9)
        )
        next_log_probs = jax.nn.log_softmax(next_logits)
        if not next_eligible:
            next_log_probs = next_log_probs.at[3].set(-jnp.inf)
        next_features = np.asarray(
            jnp.where(jnp.isfinite(next_log_probs), next_log_probs, 0.0),
            dtype=np.float64,
        )
        expected_post_values[row, position] = float(
            next_features @ critic_weight + critic_bias
        )
    if not np.allclose(
        np.asarray(update["post_action_values"], dtype=np.float64),
        expected_post_values,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "post-action EOS/timeout critic bootstrap values do not reconstruct from the frozen parent context"
        )

    actor = jax.tree.map(lambda value: jnp.array(value), initial_state["actor_params"])
    critic = jax.tree.map(
        lambda value: jnp.array(value), initial_state["critic_params"]
    )
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(float(recipe["training"]["gradientClipGlobalNorm"])),
        optax.adam(float(recipe["training"]["actorLearningRate"])),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(float(recipe["training"]["gradientClipGlobalNorm"])),
        optax.adam(float(recipe["training"]["criticLearningRate"])),
    )
    actor_opt_state, critic_opt_state = actor_optimizer.init(
        actor
    ), critic_optimizer.init(critic)
    for epoch in range(int(recipe["training"]["innerEpochs"])):
        pre_logs = np.asarray(
            _independent_selected_logps(parent, actor, update, recipe), dtype=np.float64
        )
        if not np.allclose(
            pre_logs,
            np.asarray(update["pre_update_logps_per_epoch"][epoch], dtype=np.float64),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise AssertionError(
                "recorded pre-update actor logps differ from the independent decoder replay"
            )
        pre_values = np.einsum(
            "bta,a->bt", expected_features, np.asarray(critic["weight"])
        ) + float(critic["bias"])
        if not np.allclose(
            pre_values,
            np.asarray(update["pre_update_values_per_epoch"][epoch], dtype=np.float64),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise AssertionError(
                "recorded pre-update critic values differ from the independent value head"
            )

        def actor_objective(params):
            new_logps = jnp.asarray(
                _independent_selected_logps(parent, params, update, recipe)
            )
            ratio = jnp.exp(new_logps - old_logps)
            raw = ratio * advantages
            clipped = (
                jnp.clip(
                    ratio,
                    1.0 - float(recipe["training"]["policyClip"]),
                    1.0 + float(recipe["training"]["policyClip"]),
                )
                * advantages
            )
            return -jnp.sum(mask * jnp.minimum(raw, clipped)) / denominator

        _, actor_grads = jax.value_and_grad(actor_objective)(actor)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state, actor
        )
        actor = optax.apply_updates(actor, actor_updates)
        expected_new_logps = np.asarray(
            _independent_selected_logps(parent, actor, update, recipe), dtype=np.float64
        )
        if not np.allclose(
            expected_new_logps,
            np.asarray(update["new_logps_per_epoch"][epoch], dtype=np.float64),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise AssertionError(
                "recorded post-update actor logps differ from the independent decoder replay"
            )

        def critic_objective(params):
            values = (
                jnp.einsum(
                    "bta,a->bt", jnp.asarray(expected_features), params["weight"]
                )
                + params["bias"]
            )
            clipped_values = old_values + jnp.clip(
                values - old_values,
                -float(recipe["training"]["valueClip"]),
                float(recipe["training"]["valueClip"]),
            )
            raw_error = jnp.square(values - targets)
            clipped_error = jnp.square(clipped_values - targets)
            return (
                0.5
                * jnp.sum(mask * jnp.maximum(raw_error, clipped_error))
                / denominator
            )

        _, critic_grads = jax.value_and_grad(critic_objective)(critic)
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state, critic
        )
        critic = optax.apply_updates(critic, critic_updates)
    if not _tree_arrays_equal(jax, actor, trained_state["actor_params"]):
        raise AssertionError(
            "actor parameters do not match independently reconstructed first-update PPO gradients and Adam moments"
        )
    if not _tree_arrays_equal(jax, critic, trained_state["critic_params"]):
        raise AssertionError(
            "critic parameters do not match independently reconstructed first-update value gradients and Adam moments"
        )
    if not _tree_arrays_equal(jax, actor_opt_state, trained_state["actor_opt_state"]):
        raise AssertionError(
            "actor Adam moments do not match the independent two-epoch first-update oracle"
        )
    if not _tree_arrays_equal(jax, critic_opt_state, trained_state["critic_opt_state"]):
        raise AssertionError(
            "critic Adam moments do not match the independent two-epoch first-update oracle"
        )
    return {
        "actor_two_epoch_adam_match": True,
        "critic_two_epoch_adam_match": True,
        "old_logps_parent_reconstructed": True,
        "old_values_features_reconstructed": True,
    }


def _run_learner_check(candidate, report, candidate_sha256, approval_sha256, output):
    """Run the separate one-update candidate mechanics check; never score quality."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    recipe, fixture = report["recipe"], report["fixture"]
    parent = candidate.load_verified_parent(REPO_ROOT)
    candidate.verify_prompt_and_action_ids(parent)
    state, reference, _, _ = candidate.initialize(parent)
    if candidate._tree_sha256(state["actor_params"]) != parent["params_sha256"]:
        raise AssertionError(
            "learner candidate did not initialize its actor from the verified parent"
        )
    actor_before = candidate._tree_sha256(state["actor_params"])
    critic_before = candidate._tree_sha256(state["critic_params"])
    zero_head = {"weight": jnp.zeros((3, 4), dtype=jnp.float32)}
    reward_params, reward_metrics = candidate.fit_pairwise_reward_head(
        parent, max_updates=4
    )
    reward_check = _verify_reward_fit_independently(
        candidate, parent, report, reward_params, reward_metrics
    )
    if candidate._tree_sha256(reward_params) == candidate._tree_sha256(zero_head):
        raise AssertionError("bounded reward fit returned its unchanged initial head")
    state["reward_params"] = reward_params
    state["source_identity"] = {
        "recipe_sha256": report["recipe_sha256"],
        "fixture_sha256": report["fixture_sha256"],
        "approval_sha256": approval_sha256,
        "candidate_source_sha256": candidate_sha256,
        "reward_head_sha256": candidate._tree_sha256(reward_params),
    }
    initial_state = jax.tree.map(
        lambda value: jnp.array(value) if hasattr(value, "shape") else value,
        state,
    )
    state, history = candidate.train(
        parent, state, reference, reward_params, max_updates=1
    )
    if (
        len(history) != 1
        or state["update_clock"] != 1
        or state["inner_epoch_clock"] != 2
    ):
        raise AssertionError(
            "learner candidate did not use exactly one outer update and both frozen-target epochs"
        )
    update = history[0]
    independent_update = _verify_first_ppo_update_independently(
        candidate,
        parent,
        initial_state,
        reference,
        reward_params,
        update,
        state,
        recipe,
    )
    if len(update["prompt_ids"]) != int(recipe["training"]["batchSize"]):
        raise AssertionError("learner candidate changed the fixed one-update batch")
    if (
        update["prompt_ids"]
        != fixture["trainingPromptIds"][: int(recipe["training"]["batchSize"])]
    ):
        raise AssertionError(
            "first learner update did not use the declared initial prompt order"
        )
    if update["next_prompt_ids"] == update["prompt_ids"]:
        raise AssertionError("batch cursor did not advance to distinct next prompt IDs")
    _verify_raw_update_kl(update, recipe)
    math_check = _verify_update_math_independently(update, recipe)
    action_mask = np.asarray(update["target_role_mask"], dtype=bool)
    expected_rewards = np.where(
        action_mask,
        np.asarray(update["terminal_reward"], dtype=np.float64)
        - float(recipe["training"]["klCoefficient"])
        * np.asarray(update["sampled_kl"], dtype=np.float64),
        0.0,
    )
    if not np.allclose(
        np.asarray(update["rewards"], dtype=np.float64),
        expected_rewards,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "first rollout rewards do not reconstruct from terminal reward and sampled KL shaping"
        )
    if not np.array_equal(
        np.asarray(update["input_role_mask"])[:, 0],
        np.zeros(len(action_mask), dtype=bool),
    ):
        raise AssertionError(
            "first action target was lost to an unshifted input-role mask"
        )
    if np.any(np.asarray(update["input_role_mask"], dtype=bool) & ~action_mask):
        raise AssertionError("padding received a response-role loss mask")
    if not np.allclose(
        np.asarray(update["old_logps"])[action_mask],
        np.asarray(update["parent_logps"])[action_mask],
        atol=1e-6,
    ):
        raise AssertionError(
            "first learner rollout behavior policy differs from its frozen parent"
        )
    if candidate._tree_sha256(state["actor_params"]) == actor_before:
        raise AssertionError("learner actor did not update any real decoder parameters")
    if candidate._tree_sha256(state["critic_params"]) == critic_before:
        raise AssertionError("learner critic did not update its parameters")
    if candidate._tree_sha256(state["reward_params"]) != candidate._tree_sha256(
        reward_params
    ):
        raise AssertionError("learner changed the frozen reward head during PPO")
    after_logs = np.asarray(update["new_logps_per_epoch"], dtype=np.float64)
    before_logs = np.asarray(update["pre_update_logps_per_epoch"], dtype=np.float64)
    if after_logs.shape != before_logs.shape or not np.isfinite(after_logs).all():
        raise AssertionError(
            "learner did not retain finite selected log probabilities at both epochs"
        )
    if not np.any(
        np.abs(after_logs[1][action_mask] - before_logs[0][action_mask]) > 1e-8
    ):
        raise AssertionError(
            "PPO actor update did not change selected action probabilities"
        )

    # Verify that saved sequence reward uses the fitted interaction head and the raw first-prefix feature.
    weight = np.asarray(reward_params["weight"], dtype=np.float64)
    actions = np.asarray(update["actions"], dtype=np.int64)
    features = np.asarray(update["parent_context_features"], dtype=np.float64)
    scores = np.asarray(update["sequence_scores"], dtype=np.float64)
    for row in range(len(actions)):
        counts = np.zeros(3, dtype=np.float64)
        for action in actions[row][action_mask[row]]:
            if int(action) == int(candidate.EOS):
                break
            counts[{383: 0, 381: 1, 4: 2}[int(action)]] += 1.0
        reconstructed = float(
            np.outer(counts, features[row, 0]).reshape(-1) @ weight.reshape(-1)
        )
        if not math.isclose(reconstructed, scores[row], rel_tol=1e-5, abs_tol=1e-6):
            raise AssertionError(
                "terminal reward score does not reconstruct from the frozen interaction head"
            )
        valid_positions = np.flatnonzero(action_mask[row])
        expected_terminal = np.zeros(actions.shape[1], dtype=np.float64)
        expected_terminal[valid_positions[-1]] = reconstructed
        if not np.allclose(
            np.asarray(update["terminal_reward"])[row], expected_terminal, atol=1e-6
        ):
            raise AssertionError(
                "terminal reward is not placed only on the final valid action"
            )

    evidence = {
        "schemaVersion": 1,
        "kind": "mechanics-only learner verification",
        "qualityScored": False,
        "heldoutSelection": False,
        "optimizerBudget": {
            "outer_updates": 1,
            "inner_epochs": 2,
            "reward_fit_updates": 4,
        },
        "identity": {
            "approval_sha256": approval_sha256,
            "candidate_source_sha256": candidate_sha256,
            "recipe_sha256": report["recipe_sha256"],
            "fixture_sha256": report["fixture_sha256"],
            "parent_package_files_sha256": report["parent_package_files_sha256"],
            "parent_foundation_source_sha256": report[
                "parent_foundation_source_sha256"
            ],
            "parent_identity": report["parent_identity"],
            "parent_parameter_tree_sha256": parent["params_sha256"],
            "child_parameter_tree_sha256": candidate._tree_sha256(
                state["actor_params"]
            ),
            "child_critic_tree_sha256": candidate._tree_sha256(state["critic_params"]),
            "reward_head_sha256": candidate._tree_sha256(state["reward_params"]),
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "backend": jax.default_backend(),
        },
        "initial_actor_was_parent": actor_before == parent["params_sha256"],
        "actor_changed": candidate._tree_sha256(state["actor_params"]) != actor_before,
        "critic_changed": candidate._tree_sha256(state["critic_params"])
        != critic_before,
        "reward_fit": {
            "metrics": reward_metrics,
            "independent": reward_check,
            "weight": np.asarray(reward_params["weight"]).tolist(),
        },
        "update": update,
        "independent_update_math": math_check,
        "independent_first_ppo_update": independent_update,
        "final_clocks": {
            "update": state["update_clock"],
            "inner_epoch": state["inner_epoch_clock"],
            "prompt_cursor": state["prompt_cursor"],
        },
        "limitations": [
            "one synthetic mechanics update only",
            "no heldout quality evaluation",
            "not the approved 8+4+4 campaign",
            "no language quality or human preference claim",
        ],
    }
    (output / "learner-verification-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2)
    )
    candidate.save_state(output / "final-state", state)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("solution", "starter"), default="solution"
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--stage",
        choices=("preflight", "semantic", "learner-check", "all", "resume-child"),
        default="all",
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--candidate-sha256",
        help="Required for a separately reviewed external candidate",
    )
    args = parser.parse_args()

    if args.candidate is not None and args.implementation != "solution":
        parser.error("choose either --candidate or --implementation")
    if args.candidate is not None:
        candidate_path = args.candidate.resolve()
    else:
        candidate_path = HERE / f"{args.implementation}.py"
    learner_candidate_sha = None
    if args.stage == "learner-check":
        if args.candidate is None or not args.candidate_sha256:
            parser.error(
                "learner-check requires --candidate and its exact --candidate-sha256"
            )
        learner_candidate_sha = sha256_file(candidate_path)
        if learner_candidate_sha != args.candidate_sha256:
            raise ValueError(
                "learner candidate bytes differ from caller-supplied SHA-256"
            )
    is_run = args.stage in ("all", "resume-child")
    report = preflight(
        args.approval,
        args.expected_approval_sha256,
        run=is_run,
        learner_verification=args.stage == "learner-check",
        implementation_path=(
            candidate_path if is_run or args.stage == "learner-check" else None
        ),
    )
    if args.stage == "preflight":
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("recipe", "fixture", "parent_identity")
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    if args.stage == "semantic":
        if args.candidate is None:
            parser.error("semantic stage requires --candidate")
        actual_candidate_sha = sha256_file(candidate_path)
        if args.candidate_sha256 != actual_candidate_sha:
            raise ValueError(
                "semantic candidate requires its exact caller-supplied source SHA-256"
            )
        expected_versions = {
            "jax": "0.9.2",
            "numpy": "2.4.4",
            "optax": "0.2.8",
            "tokenizers": "0.22.2",
        }
        installed_versions = {
            name: importlib.metadata.version(name) for name in expected_versions
        }
        if installed_versions != expected_versions:
            raise ValueError(
                f"runtime versions differ from pinned recipe: {installed_versions}"
            )
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["TRANSFORMER_LINEAGE_PROJECT_ROOT"] = str(HERE)
        candidate = _load_candidate(candidate_path, HERE)
        import jax

        if jax.default_backend() != "cpu":
            raise ValueError("this lineage check is CPU-only")
        _independent_scalar_contracts(candidate)
        contracts = _semantic_no_update_contracts(candidate, REPO_ROOT, report)
        evaluation_smoke = _fresh_process_evaluation_smoke(
            candidate, candidate_path, report
        )
        print(
            json.dumps(
                {
                    "semantic_contracts": "passed",
                    "candidate_source_sha256": actual_candidate_sha,
                    "approval_sha256": report["approval_sha256"],
                    "recipe_sha256": report["recipe_sha256"],
                    "fixture_sha256": report["fixture_sha256"],
                    **contracts,
                    "fresh_process_evaluation_smoke": evaluation_smoke,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.stage == "learner-check":
        if args.output is None:
            parser.error("learner-check requires a unique --output path")
        output = require_external_temp_output(args.output, "learner-check")
        if output.exists():
            raise ValueError(
                "learner-check output already exists; this runner does not overwrite or retry a candidate attempt"
            )
        expected_versions = {
            "jax": "0.9.2",
            "numpy": "2.4.4",
            "optax": "0.2.8",
            "tokenizers": "0.22.2",
        }
        installed_versions = {
            name: importlib.metadata.version(name) for name in expected_versions
        }
        if installed_versions != expected_versions:
            raise ValueError(
                f"runtime versions differ from pinned recipe: {installed_versions}"
            )
        output.mkdir(parents=True, exist_ok=False)
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["TRANSFORMER_LINEAGE_PROJECT_ROOT"] = str(HERE)
        candidate = _load_candidate(candidate_path, HERE)
        import jax

        if jax.default_backend() != "cpu":
            raise ValueError("this learner check is CPU-only")
        _independent_scalar_contracts(candidate)
        evidence = _run_learner_check(
            candidate, report, learner_candidate_sha, report["approval_sha256"], output
        )
        print(
            json.dumps(
                {
                    "learner_check": "passed",
                    "candidate_source_sha256": learner_candidate_sha,
                    "outer_updates": evidence["final_clocks"]["update"],
                    "inner_epochs": evidence["final_clocks"]["inner_epoch"],
                    "reward_fit_updates": evidence["reward_fit"]["metrics"][
                        "fit_updates"
                    ],
                    "evidence": str(output / "learner-verification-evidence.json"),
                    "quality_scored": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.output is None:
        parser.error("--output is required for training stages")
    output = require_external_temp_output(args.output, "run")
    if output.exists():
        raise ValueError(
            "output path already exists; this runner does not overwrite or retry a run"
        )

    if args.candidate is not None:
        actual_candidate_sha = sha256_file(candidate_path)
        if args.candidate_sha256 != actual_candidate_sha:
            raise ValueError(
                "external candidate requires its exact caller-supplied source SHA-256"
            )
        if is_run:
            raise ValueError(
                "run-approved source is fixed; external candidates are semantic-check only"
            )

    # Dependency metadata checks do not import JAX or execute candidate code.
    expected_versions = {
        "jax": "0.9.2",
        "numpy": "2.4.4",
        "optax": "0.2.8",
        "tokenizers": "0.22.2",
    }
    installed_versions = {
        name: importlib.metadata.version(name) for name in expected_versions
    }
    if installed_versions != expected_versions:
        raise ValueError(
            f"runtime versions differ from pinned recipe: {installed_versions}"
        )
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["TRANSFORMER_LINEAGE_PROJECT_ROOT"] = str(HERE)
    candidate = _load_candidate(candidate_path, HERE)

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise ValueError("this lineage check is CPU-only")
    _independent_scalar_contracts(candidate)

    parent = candidate.load_verified_parent(REPO_ROOT)
    candidate.verify_prompt_and_action_ids(parent)
    state, reference_params, actor_optimizer, critic_optimizer = candidate.initialize(
        parent
    )
    if state["update_clock"] != 0 or state["inner_epoch_clock"] != 0:
        raise AssertionError("child optimizer boundary did not reset update clocks")
    if candidate._tree_sha256(state["actor_params"]) != parent["params_sha256"]:
        raise AssertionError(
            "child actor did not begin from the exact parent parameter tree"
        )
    state["source_identity"] = {
        "recipe_sha256": report["recipe_sha256"],
        "fixture_sha256": report["fixture_sha256"],
        "approval_sha256": report["approval_sha256"],
        "candidate_source_sha256": sha256_file(candidate_path),
    }

    # First validate the real decoder conditioning and differentiability, without updates.
    prompt_id = int(report["fixture"]["trainingPromptIds"][0])
    prefix = candidate._full_prefix(prompt_id, ())
    first_logits = candidate._masked_model_logits(
        parent["model_module"], state["actor_params"], prefix, parent["config"]
    )
    initial_dist = candidate._restricted_log_probs(first_logits, False)
    eligible_dist = candidate._restricted_log_probs(
        candidate._masked_model_logits(
            parent["model_module"],
            state["actor_params"],
            candidate._full_prefix(prompt_id, [383, 4]),
            parent["config"],
        ),
        True,
    )
    eos_position = candidate.ACTION_INDEX[candidate.EOS]
    if (
        bool(jnp.isfinite(initial_dist[eos_position]))
        or float(jnp.sum(jnp.exp(initial_dist))) < 0.99999
    ):
        raise AssertionError(
            "EOS was not excluded with zero probability before the two-action boundary"
        )
    if not bool(jnp.isfinite(eligible_dist[eos_position])) or not np.isclose(
        float(jnp.sum(jnp.exp(eligible_dist))), 1.0, atol=1e-6
    ):
        raise AssertionError(
            "EOS eligibility did not use the same normalized restricted support"
        )
    branched_up = candidate._masked_model_logits(
        parent["model_module"],
        state["actor_params"],
        candidate._full_prefix(prompt_id, [383]),
        parent["config"],
    )
    branched_down = candidate._masked_model_logits(
        parent["model_module"],
        state["actor_params"],
        candidate._full_prefix(prompt_id, [381]),
        parent["config"],
    )
    if float(jnp.max(jnp.abs(branched_up - branched_down))) <= 1e-7:
        raise AssertionError(
            "different first actions did not change next Transformer logits"
        )

    def selected_logp(params):
        logits = candidate._masked_model_logits(
            parent["model_module"], params, prefix, parent["config"]
        )
        return candidate._restricted_log_probs(logits, False)[
            candidate.ACTION_INDEX[383]
        ]

    target_path = ("token_embed", 383, 0)
    gradient = jax.grad(selected_logp)(state["actor_params"])
    analytic = float(gradient[target_path[0]][target_path[1], target_path[2]])
    delta = 1e-3
    plus = jax.tree.map(lambda x: jnp.array(x), state["actor_params"])
    minus = jax.tree.map(lambda x: jnp.array(x), state["actor_params"])
    plus[target_path[0]] = (
        plus[target_path[0]].at[target_path[1], target_path[2]].add(delta)
    )
    minus[target_path[0]] = (
        minus[target_path[0]].at[target_path[1], target_path[2]].add(-delta)
    )
    finite_difference = float(
        (selected_logp(plus) - selected_logp(minus)) / (2 * delta)
    )
    if (
        not math.isclose(analytic, finite_difference, rel_tol=2e-2, abs_tol=2e-3)
        or abs(analytic) <= 1e-7
    ):
        raise AssertionError(
            f"decoder parameter finite difference failed: autodiff={analytic}, fd={finite_difference}"
        )

    evaluation_smoke = None
    if args.stage == "all":
        # Catch evaluation composition errors before creating campaign output or
        # executing any campaign optimizer step.
        evaluation_smoke = _fresh_process_evaluation_smoke(
            candidate, candidate_path, report
        )

    output.mkdir(parents=True, exist_ok=False)
    if args.stage == "resume-child":
        if args.resume_checkpoint is None:
            raise ValueError("resume child requires --resume-checkpoint")
        state["reward_params"] = {
            "weight": jnp.zeros((3, len(candidate.ACTION_IDS)), jnp.float32)
        }
        state = candidate.restore_state(args.resume_checkpoint, state)
        expected_identity = {
            "recipe_sha256": report["recipe_sha256"],
            "fixture_sha256": report["fixture_sha256"],
            "approval_sha256": report["approval_sha256"],
            "candidate_source_sha256": sha256_file(candidate_path),
            "reward_head_sha256": candidate._tree_sha256(state["reward_params"]),
        }
        if state["source_identity"] != expected_identity:
            raise ValueError("resume checkpoint lineage identity mismatch")
        reward_params = state["reward_params"]
        state, history = candidate.train(parent, state, reference_params, reward_params)
        evaluation = candidate.evaluate(
            parent,
            state["actor_params"],
            reference_params,
            state["critic_params"],
            reward_params,
        )
        save_dir = output / "final-state"
        candidate.save_state(save_dir, state)
        raw_result = {
            "state_sha256": candidate._tree_sha256(state["actor_params"]),
            "critic_sha256": candidate._tree_sha256(state["critic_params"]),
            "source_identity": state["source_identity"],
            "parent_package_sha256": state["parent_package_sha256"],
            "processor_sha256": state["processor_sha256"],
            "history": history,
            "evaluation": evaluation,
            "update_clock": state["update_clock"],
            "inner_epoch_clock": state["inner_epoch_clock"],
        }
        (output / "resume-result.json").write_text(
            json.dumps(raw_result, sort_keys=True, indent=2)
        )
        print(
            json.dumps(
                {
                    "resume_child": "passed",
                    "updates": state["update_clock"],
                    "actor_sha256": raw_result["state_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.stage != "all":
        raise ValueError("unsupported training stage")

    reward_params, reward_fit_metrics = candidate.fit_pairwise_reward_head(parent)
    state["reward_params"] = reward_params
    state["source_identity"]["reward_head_sha256"] = candidate._tree_sha256(
        reward_params
    )
    fitted_reward_sha256 = candidate._tree_sha256(reward_params)
    initial_state_dir = output / "fitted-initial-state"
    initial_checkpoint_metadata = candidate.save_state(initial_state_dir, state)
    initial_checkpoint_hashes = {
        name: sha256_file(initial_state_dir / name)
        for name in ("state.json", "state.npz")
    }
    if (
        initial_checkpoint_metadata["reward_params_sha256"] != fitted_reward_sha256
        or initial_checkpoint_metadata["source_identity"].get("reward_head_sha256")
        != fitted_reward_sha256
        or initial_checkpoint_metadata["update_clock"] != 0
        or initial_checkpoint_metadata["inner_epoch_clock"] != 0
        or initial_checkpoint_metadata["prompt_cursor"] != 0
    ):
        raise AssertionError(
            "fitted reward checkpoint does not bind the zero-clock initial state"
        )
    (
        primary_state,
        primary_reference,
        primary_initial_hashes,
        primary_checkpoint_hashes,
    ) = _restore_fitted_initial_state(
        candidate,
        parent,
        initial_state_dir,
        initial_checkpoint_metadata,
        state["source_identity"],
    )
    reference_params = primary_reference
    reward_params = primary_state["reward_params"]
    if primary_checkpoint_hashes != initial_checkpoint_hashes:
        raise AssertionError(
            "primary start did not read the immutable initial checkpoint"
        )
    boundary_probes = _rollout_boundary_contracts(
        candidate, parent, primary_state, reference_params, reward_params
    )
    primary_state, primary_first = candidate.train(
        parent, primary_state, reference_params, reward_params, max_updates=4
    )
    split_copy = jax.tree.map(
        lambda x: jnp.array(x) if hasattr(x, "shape") else x, primary_state
    )
    split_dir = output / "primary-split-state"
    candidate.save_state(split_dir, split_copy)
    primary_state, primary_last = candidate.train(
        parent, primary_state, reference_params, reward_params, max_updates=8
    )
    primary_history = primary_first + primary_last
    primary_evaluation = candidate.evaluate(
        parent,
        primary_state["actor_params"],
        reference_params,
        primary_state["critic_params"],
        reward_params,
    )
    for split in ("dev", "heldout"):
        split_result = primary_evaluation[split]
        expected_episodes = len(split_result["matched_prompt_ids"]) * len(
            split_result["matched_sampling_seeds"]
        )
        if split_result["unique_matched_episode_count"] != expected_episodes:
            raise AssertionError(
                f"{split} evaluation did not report the full unique matched exposure"
            )
        child_rows = split_result["child_rollouts"]
        parent_rows = split_result["matched_parent_rollouts"]
        child_keys = [tuple(row["sampling_key"]) for row in child_rows]
        parent_keys = [tuple(row["sampling_key"]) for row in parent_rows]
        if len(set(child_keys)) != expected_episodes or child_keys != parent_keys:
            raise AssertionError(
                f"{split} matched evaluation reused a key or mismatched child and parent keys"
            )
        for row in child_rows:
            _verify_raw_update_kl(row, report["recipe"])
        for row in parent_rows:
            _verify_raw_update_kl(row, report["recipe"])

    replay_state, replay_reference, replay_initial_hashes, replay_checkpoint_hashes = (
        _restore_fitted_initial_state(
            candidate,
            parent,
            initial_state_dir,
            initial_checkpoint_metadata,
            state["source_identity"],
        )
    )
    if (
        replay_checkpoint_hashes != initial_checkpoint_hashes
        or replay_initial_hashes != primary_initial_hashes
        or candidate._tree_sha256(primary_state["reward_params"])
        != fitted_reward_sha256
        or candidate._tree_sha256(replay_state["reward_params"]) != fitted_reward_sha256
    ):
        raise AssertionError(
            "primary and replay did not cold-load the same fitted reward checkpoint"
        )
    replay_state, replay_first = candidate.train(
        parent,
        replay_state,
        replay_reference,
        replay_state["reward_params"],
        max_updates=4,
    )
    if primary_first[-1]["next_prompt_ids"] != replay_first[-1]["next_prompt_ids"]:
        raise AssertionError("next prompt IDs differ at the split clock")
    replay_split_dir = output / "replay-split-state"
    candidate.save_state(replay_split_dir, replay_state)
    if not _tree_arrays_equal(
        jax, split_copy["actor_params"], replay_state["actor_params"]
    ):
        raise AssertionError("primary and replay actor states differ at split clock")
    if not _tree_arrays_equal(
        jax, split_copy["critic_params"], replay_state["critic_params"]
    ):
        raise AssertionError("primary and replay critic states differ at split clock")
    if not _tree_arrays_equal(
        jax, split_copy["actor_opt_state"], replay_state["actor_opt_state"]
    ):
        raise AssertionError(
            "primary and replay actor Adam moments differ at split clock"
        )
    if not _tree_arrays_equal(
        jax, split_copy["critic_opt_state"], replay_state["critic_opt_state"]
    ):
        raise AssertionError(
            "primary and replay critic Adam moments differ at split clock"
        )
    if any(
        split_copy[k] != replay_state[k]
        for k in ("rng_key", "prompt_cursor", "update_clock", "inner_epoch_clock")
    ):
        raise AssertionError(
            "primary and replay key/cursor/clocks differ at split clock"
        )

    # Spawn a new interpreter for the replay lane, restoring the complete split state.
    child_output = output / "fresh-process-resume"
    child_command = [
        sys.executable,
        str(HERE / "check.py"),
        "--implementation",
        args.implementation,
        "--stage",
        "resume-child",
        "--approval",
        str(args.approval.resolve()),
        "--expected-approval-sha256",
        args.expected_approval_sha256,
        "--output",
        str(child_output),
        "--resume-checkpoint",
        str(replay_split_dir),
    ]
    if args.candidate is not None:
        child_command[child_command.index("--implementation") + 1] = "solution"
        child_command.extend(
            [
                "--candidate",
                str(candidate_path),
                "--candidate-sha256",
                sha256_file(candidate_path),
            ]
        )
    child_run = subprocess.run(
        child_command, capture_output=True, text=True, check=False
    )
    (output / "resume-child.stdout.txt").write_text(child_run.stdout)
    (output / "resume-child.stderr.txt").write_text(child_run.stderr)
    if child_run.returncode != 0:
        raise RuntimeError(f"fresh-process replay failed: {child_run.stderr[-2000:]}")
    child_result = read_json(child_output / "resume-result.json")
    if child_result["source_identity"] != primary_state["source_identity"]:
        raise AssertionError(
            "fresh-process child recorded different recipe, fixture, approval or candidate hashes"
        )
    if (
        child_result["parent_package_sha256"] != primary_state["parent_package_sha256"]
        or child_result["processor_sha256"] != primary_state["processor_sha256"]
    ):
        raise AssertionError(
            "fresh-process child identity differs from the frozen parent package"
        )
    replay_last = child_result["history"]
    replay_history = replay_first + replay_last
    replay_final = candidate.restore_state(child_output / "final-state", replay_state)
    if not _tree_arrays_equal(
        jax, primary_state["actor_params"], replay_final["actor_params"]
    ):
        raise AssertionError(
            "primary and resumed actor parameters differ at final clock"
        )
    if not _tree_arrays_equal(
        jax, primary_state["critic_params"], replay_final["critic_params"]
    ):
        raise AssertionError(
            "primary and resumed critic parameters differ at final clock"
        )
    if not _tree_arrays_equal(
        jax, primary_state["actor_opt_state"], replay_final["actor_opt_state"]
    ):
        raise AssertionError(
            "primary and resumed actor Adam moments differ at final clock"
        )
    if not _tree_arrays_equal(
        jax, primary_state["critic_opt_state"], replay_final["critic_opt_state"]
    ):
        raise AssertionError(
            "primary and resumed critic Adam moments differ at final clock"
        )
    if any(
        primary_state[k] != replay_final[k]
        for k in ("rng_key", "prompt_cursor", "update_clock", "inner_epoch_clock")
    ):
        raise AssertionError(
            "primary and resumed key/cursor/clocks differ at final clock"
        )
    for index, (primary_row, replay_row) in enumerate(
        zip(primary_history, replay_history)
    ):
        for key in (
            "prompt_ids",
            "next_prompt_ids",
            "actions",
            "target_role_mask",
            "input_role_mask",
            "terminated",
            "truncated",
            "old_logps",
            "parent_logps",
            "old_restricted_probabilities",
            "parent_restricted_probabilities",
            "eligible_action_mask",
            "sampled_kl",
            "exact_restricted_conditional_kl",
            "old_values",
            "post_action_values",
            "rewards",
            "deltas",
            "advantages",
            "value_targets",
            "new_logps_per_epoch",
            "epoch_metrics",
        ):
            if primary_row[key] != replay_row[key]:
                raise AssertionError(
                    f"resumed rollout differs at update {index}, field {key}"
                )
        _verify_raw_update_kl(primary_row, report["recipe"])

    if not _tree_arrays_equal(
        jax, primary_state["reward_params"], replay_final["reward_params"]
    ):
        raise AssertionError(
            "frozen reward head differs across uninterrupted and replay lanes"
        )
    primary_final_reward_sha256 = candidate._tree_sha256(primary_state["reward_params"])
    replay_final_reward_sha256 = candidate._tree_sha256(replay_final["reward_params"])
    if (
        primary_final_reward_sha256 != fitted_reward_sha256
        or replay_final_reward_sha256 != fitted_reward_sha256
        or candidate._tree_sha256(reward_params) != fitted_reward_sha256
    ):
        raise AssertionError("frozen reward head changed after cold start")
    if candidate._tree_sha256(reference_params) != parent["params_sha256"]:
        raise AssertionError("frozen parent reference changed during actor training")
    first_update_old_parent = primary_history[0]
    first_mask = np.asarray(first_update_old_parent["target_role_mask"], dtype=bool)
    if not np.allclose(
        np.asarray(first_update_old_parent["old_logps"])[first_mask],
        np.asarray(first_update_old_parent["parent_logps"])[first_mask],
        atol=1e-6,
    ):
        raise AssertionError(
            "first rollout behavior log probabilities do not match the frozen parent"
        )
    second_epoch_delta = []
    for row in primary_history:
        mask = np.asarray(row["target_role_mask"], dtype=bool)
        if len(row["new_logps_per_epoch"]) >= 2:
            difference = np.asarray(row["new_logps_per_epoch"][1]) - np.asarray(
                row["new_logps_per_epoch"][0]
            )
            second_epoch_delta.extend(np.abs(difference[mask]).tolist())
    if not second_epoch_delta or max(second_epoch_delta) <= 1e-8:
        raise AssertionError(
            "inner PPO epochs did not change selected actor log probabilities"
        )

    # Freeze all heldout quality decisions to the primary lane only.
    parent_after = {
        "package_files": {
            name: sha256_file(
                REPO_ROOT / report["recipe"]["parent"]["packagePath"] / Path(name)
            )
            for name in report["recipe"]["parent"]["packageFilesSha256"]
        },
        "foundation_sources": {
            name: sha256_file(
                REPO_ROOT / "projects/text-harness/foundations" / Path(name)
            )
            for name in report["recipe"]["parent"]["foundationSourceSha256"]
        },
    }
    if (
        parent_after["package_files"]
        != report["recipe"]["parent"]["packageFilesSha256"]
    ):
        raise AssertionError("frozen parent package bytes changed during the run")
    if (
        parent_after["foundation_sources"]
        != report["recipe"]["parent"]["foundationSourceSha256"]
    ):
        raise AssertionError("frozen foundation source bytes changed during the run")
    evidence = {
        "schemaVersion": 1,
        "identity": {
            "recipe_sha256": report["recipe_sha256"],
            "fixture_sha256": report["fixture_sha256"],
            "approval_sha256": report["approval_sha256"],
            "candidate_source_sha256": sha256_file(candidate_path),
            "parent_package_files_sha256": report["parent_package_files_sha256"],
            "parent_foundation_source_sha256": report[
                "parent_foundation_source_sha256"
            ],
            "parent_identity": report["parent_identity"],
            "parent_parameter_tree_sha256": parent["params_sha256"],
            "child_parameter_tree_sha256": candidate._tree_sha256(
                primary_state["actor_params"]
            ),
            "processor_sha256": parent["processor_sha256"],
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "optax_version": importlib.metadata.version("optax"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "backend": jax.default_backend(),
        },
        "optimizerBoundary": {
            "actor_started_as_parent_copy": True,
            "actor_optimizer_reset": True,
            "critic_optimizer_reset": True,
            "parent_optimizer_moments_reused": False,
            "parent_source_hashes_after_run": parent_after,
        },
        "budget": {
            "reward_head_fits": 1,
            "primary_updates": len(primary_history),
            "replay_updates": len(replay_history),
            "actor_update_executions": len(primary_history) + len(replay_history),
            "inner_epochs": int(report["recipe"]["training"]["innerEpochs"]),
        },
        "reward_fit": reward_fit_metrics,
        "reward_head_lifecycle": {
            "reward_fit_count": 1,
            "fit_tree_sha256": fitted_reward_sha256,
            "initial_checkpoint_path": "fitted-initial-state",
            "initial_checkpoint_tree_hashes": {
                "actor_params": initial_checkpoint_metadata["actor_params_sha256"],
                "critic_params": initial_checkpoint_metadata["critic_params_sha256"],
                "actor_opt_state": initial_checkpoint_metadata[
                    "actor_opt_state_sha256"
                ],
                "critic_opt_state": initial_checkpoint_metadata[
                    "critic_opt_state_sha256"
                ],
                "reward_params": initial_checkpoint_metadata["reward_params_sha256"],
            },
            "initial_checkpoint_file_sha256": initial_checkpoint_hashes,
            "initial_clock_cursor": {
                "update": initial_checkpoint_metadata["update_clock"],
                "inner_epoch": initial_checkpoint_metadata["inner_epoch_clock"],
                "prompt_cursor": initial_checkpoint_metadata["prompt_cursor"],
            },
            "primary_cold_restore_tree_hashes": primary_initial_hashes,
            "replay_cold_restore_tree_hashes": replay_initial_hashes,
            "primary_checkpoint_hashes_after_restore": primary_checkpoint_hashes,
            "replay_checkpoint_hashes_after_restore": replay_checkpoint_hashes,
            "primary_final_tree_sha256": primary_final_reward_sha256,
            "replay_final_tree_sha256": replay_final_reward_sha256,
            "fresh_process_final_tree_sha256": replay_final_reward_sha256,
        },
        "pre_optimizer_evaluation_smoke": evaluation_smoke,
        "primary": {"updates": primary_history, "evaluation": primary_evaluation},
        "resumeReplay": {
            "updates": replay_history,
            "fresh_process_stdout": child_run.stdout,
            "split_clock_equal": True,
            "final_clock_equal": True,
            "quality_scored": False,
        },
        "boundaryProbes": {
            "eos": candidate._serializable_rollout(boundary_probes["eos"]),
            "timeout": candidate._serializable_rollout(boundary_probes["timeout"]),
        },
        "quality_gate": {
            "primary_only": True,
            "passed": primary_evaluation["quality_gate_passed"],
            "failure_reasons": primary_evaluation["failure_reasons"],
        },
        "limitations": [
            "synthetic preference fixture only",
            "restricted four-action categorical distributions",
            "conditional KL at sampled visited contexts, not full-sequence KL",
            "no human preference, alignment, natural-language quality, or model-release claim",
        ],
    }
    (output / "lineage-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2)
    )
    final_state_dir = output / "primary-final-state"
    candidate.save_state(final_state_dir, primary_state)
    print(
        json.dumps(
            {
                "result": (
                    "PASS"
                    if primary_evaluation["quality_gate_passed"]
                    else "QUALITY_GATE_FAILED"
                ),
                "primary_updates": len(primary_history),
                "replay_updates": len(replay_history),
                "actor_update_executions": len(primary_history) + len(replay_history),
                "evidence": str(output / "lineage-evidence.json"),
                "quality_failures": primary_evaluation["failure_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
