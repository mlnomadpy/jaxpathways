#!/usr/bin/env python3
"""Verify the public numeric packet with Python's standard library only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
CONTRACT_SHA256 = "8e9abc0cce93d20d02f086c7cf5334966a8088ad6e36e27d5aca6e5589850ab9"
SHIPPING_APPROVAL_SHA256 = "28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b"
RUN_APPROVAL_SHA256 = "a47bc625a1a372273f20ddace82d0c3b3ce0d4d5f1336cb448614ace9662eaf9"
REVIEW_SHA256 = "78c1f71266ba7e05bc16010e34700aec6c4997e10781609353eda4713fe43b75"
RAW_SHA256 = "966fe2d13eb22508e21080f7e8fa007dc51d04e197ba01e00d575ec13187f10e"
RECIPE_SHA256 = "8411afc4412c59973d56b2cceae530dd642a5aa1c9e6f57a5dbb7f6ba110bc57"
FIXTURE_SHA256 = "6a47dbe54012ed3013753ca8089fbcd51e57b3d0cbd61ffdb50a6877e0a534ff"
RESULT_RECEIPT_SHA256 = "6a45955dac4e08c173fec3475c3af977c40f88142461c7ce76dcc53f4ad74a3d"
AS_RUN_STARTER_SHA256 = "b7fc9060123b474de92ac04a5480ef818ab296bc216351ed2f7701474741b03d"
SHIPPING_STARTER_SHA256 = "8196af0f989c382ab5c4833102f11a2d1fe2263614e35e8ac1be8ab1ce501d32"
KL_CEILING = 0.8
EXPECTED_PUBLIC_SHA256 = {
    "evaluation.json": "adedff08747a25ffcd0535b51466d68cfc66c4b1d911a755d1d0a245182cecea",
    "training-trace.csv": "a6076e135cc070b0ae2b848c1a433a2bac2a2f57a795a1a28d97c88e5b63a451",
    "boundary-probes.json": "0521d449442b1a29a70df5a3b37af869dd9b3a98640afd84d46ec5089d5a0e34",
}
EXPECTED_AS_RUN_FILES = {
    "projects/posttraining-audit/transformer_lineage/README.md": "2d0f9db1a2ec5c3eec8845c7b8267e6cd3044a97d66ecfc37c7eef97d9ec7c98",
    "projects/posttraining-audit/transformer_lineage/check.py": "12f6709ef0683067ee10d871891bb4d67736c6d5039d5a2809d2dfa92115a0e3",
    "projects/posttraining-audit/transformer_lineage/evidence-schema.json": "59f5ba83b7129b531fb790c0ac62bae22b70a297b3f35caacd3f4e314418e798",
    "projects/posttraining-audit/transformer_lineage/fixture.json": "6a47dbe54012ed3013753ca8089fbcd51e57b3d0cbd61ffdb50a6877e0a534ff",
    "projects/posttraining-audit/transformer_lineage/recipe.json": "8411afc4412c59973d56b2cceae530dd642a5aa1c9e6f57a5dbb7f6ba110bc57",
    "projects/posttraining-audit/transformer_lineage/requirements.txt": "14c6bdedf000ba836f74beedc4c7aa50bc818939d30b75030a2c47a0cf6df537",
    "projects/posttraining-audit/transformer_lineage/solution.py": "df56c036a57e8ad1c7be61341290422e7dfa47ea3326b3c82e5cecad5806b569",
    "projects/posttraining-audit/transformer_lineage/starter.py": AS_RUN_STARTER_SHA256,
    "projects/text-harness/foundations/config.py": "5c05801a7ec1a15b3978c87c77ac4e64d1b056c60487cbcff3cdf72475c8bb64",
    "projects/text-harness/foundations/data.py": "ff8ff698090866d46bc1f2b3540393b9a16db2121ad824110bd2ba53db645bd8",
    "projects/text-harness/foundations/model.py": "94a88ae7fd6a251b5b190b4631496e04bb71a31c1bfe418511fdd37ffef034c9",
    "projects/text-harness/foundations/reference/run/inference-package/package.json": "02b92a9f56325e5a38be2c40bf263d2ca91db6122ba84dd4df266339457bf278",
    "projects/text-harness/foundations/reference/run/inference-package/params.npz": "271cb6457de5da3f727c343c64b7bd350c00e503ea7ac847eecf51fbd5558802",
    "projects/text-harness/foundations/reference/run/inference-package/tokenizer.json": "c92b534443cc6a1eacd5e78406b2c42e59553df704a07102a03c639d3694f586",
    "projects/text-harness/foundations/train.py": "ee9f3bc73505ce8e00fe9680589d97805593e5c8c035cba4709b5f334eb27a2b",
}
TRACE_FIELDS = [
    "update",
    "valid_action_count",
    "eos_count",
    "timeout_count",
    "padding_count",
    "signed_policy_loss_epoch_0_per_valid_action",
    "signed_policy_loss_epoch_1_per_valid_action",
    "clipped_critic_loss_epoch_0_per_valid_action",
    "clipped_critic_loss_epoch_1_per_valid_action",
    "mean_sampled_kl_per_valid_action",
    "mean_exact_restricted_conditional_kl_per_valid_action",
    "reward_fit_steps",
    "reward_fit_initial_pairwise_loss",
    "reward_fit_final_pairwise_loss",
    "split_state_json_sha256",
    "split_state_npz_sha256",
    "split_primary_replay_bytes_equal",
    "final_state_json_sha256",
    "final_state_npz_sha256",
    "final_primary_replay_bytes_equal",
]
PAYLOAD_NAMES = {
    "evaluation.json",
    "training-trace.csv",
    "boundary-probes.json",
    "figure.svg",
    "build_numeric_packet.py",
    "check_numeric_packet.py",
}
EVALUATION_TOP_KEYS = {
    "schemaVersion",
    "kind",
    "scope",
    "klScope",
    "splits",
    "qualityGate",
    "historyDisclosure",
}
EPISODE_KEYS = {
    "promptId",
    "samplingSeed",
    "childActionIds",
    "parentActionIds",
    "childFittedRewardScore",
    "parentFittedRewardScore",
    "validActionDenominator",
    "sampledKlSum",
    "exactRestrictedConditionalKlSumAtChildVisitedContexts",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, label, tol=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise ValueError(f"{label} does not reconstruct: {actual} != {expected}")


def load_builder():
    path = HERE / "build_numeric_packet.py"
    spec = importlib.util.spec_from_file_location("numeric_packet_builder", path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load deterministic figure renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_evaluation(evaluation, repo_root):
    if set(evaluation) != EVALUATION_TOP_KEYS:
        raise ValueError("evaluation contains omitted, extra, or disallowed fields")
    if evaluation["schemaVersion"] != 1 or evaluation["kind"] != "transformer-lineage-v5-public-evaluation":
        raise ValueError("evaluation schema or kind is unsupported")
    if "full-vocabulary" not in evaluation["klScope"] or "not" not in evaluation["klScope"]:
        raise ValueError("evaluation must state restricted KL scope and limits")
    if set(evaluation["splits"]) != {"dev", "heldout"}:
        raise ValueError("evaluation split inventory differs")
    recipe_path = repo_root / "projects/posttraining-audit/transformer_lineage/recipe.json"
    fixture_path = repo_root / "projects/posttraining-audit/transformer_lineage/fixture.json"
    recipe_raw = recipe_path.read_bytes()
    fixture_raw = fixture_path.read_bytes()
    if digest(recipe_raw) != RECIPE_SHA256 or digest(fixture_raw) != FIXTURE_SHA256:
        raise ValueError("fixed recipe or fixture bytes differ")
    recipe = json.loads(recipe_raw)
    fixture = json.loads(fixture_raw)
    seeds = recipe["evaluationFloors"]["matchedSamplingSeeds"]
    reconstructed = {}
    for split, item in evaluation["splits"].items():
        if set(item) != {"episodes", "summary"}:
            raise ValueError(f"{split} split has extra or missing fields")
        episodes = item["episodes"]
        if len(episodes) != 8 or any(set(row) != EPISODE_KEYS for row in episodes):
            raise ValueError(f"{split} public episode rows are incomplete or contain forbidden fields")
        base_ids = fixture["matchedEvaluationPromptIds"][split]
        expected_ids = base_ids * len(seeds)
        expected_seeds = [seed for seed in seeds for _ in base_ids]
        if [row["promptId"] for row in episodes] != expected_ids:
            raise ValueError(f"{split} prompt schedule differs from the frozen fixture")
        if [row["samplingSeed"] for row in episodes] != expected_seeds:
            raise ValueError(f"{split} seed schedule differs from the frozen recipe")
        min_non_eos = recipe["actions"]["minimumNonEosActions"]
        max_actions = recipe["actions"]["maximumResponseActions"]
        allowed = set(recipe["actions"]["restrictedVocabularyIds"])
        eos_id = recipe["actions"]["singleTokenSpellings"]["eos"]["id"]
        for row in episodes:
            if row["validActionDenominator"] != len(row["childActionIds"]):
                raise ValueError("valid-action denominator differs from child action list")
            if len(row["parentActionIds"]) != row["validActionDenominator"]:
                raise ValueError("matched parent action list has a different length")
            for label in ("child", "parent"):
                actions = row[f"{label}ActionIds"]
                if not (min_non_eos <= len(actions) <= max_actions) or any(action not in allowed for action in actions):
                    raise ValueError(f"{split} {label} actions violate the restricted response grammar")
                eos_positions = [i for i, action in enumerate(actions) if action == eos_id]
                if eos_positions and (len(eos_positions) != 1 or eos_positions[0] != len(actions) - 1 or eos_positions[0] < min_non_eos):
                    raise ValueError(f"{split} {label} EOS eligibility or termination differs")
                if not eos_positions and len(actions) != max_actions:
                    raise ValueError(f"{split} {label} non-EOS response is not a max-length truncation")
            for field in ("sampledKlSum", "exactRestrictedConditionalKlSumAtChildVisitedContexts"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite episode {field}")
        count = len(episodes)
        valid = sum(row["validActionDenominator"] for row in episodes)
        child_reward = sum(row["childFittedRewardScore"] for row in episodes) / count
        parent_reward = sum(row["parentFittedRewardScore"] for row in episodes) / count
        sampled = sum(row["sampledKlSum"] for row in episodes) / valid
        conditional = sum(row["exactRestrictedConditionalKlSumAtChildVisitedContexts"] for row in episodes) / valid
        same = sum(row["childActionIds"] == row["parentActionIds"] for row in episodes)
        summary = item["summary"]
        close(summary["matchedEpisodeCount"], count, f"{split} episode count")
        close(summary["validActionDenominator"], valid, f"{split} valid action denominator")
        close(summary["meanChildFittedRewardPerEpisode"], child_reward, f"{split} child reward")
        close(summary["meanParentFittedRewardPerEpisode"], parent_reward, f"{split} parent reward")
        close(summary["meanSampledKlPerValidAction"], sampled, f"{split} sampled KL")
        close(summary["meanExactRestrictedConditionalKlPerValidActionAtChildVisitedContexts"], conditional, f"{split} restricted KL")
        close(summary["sameChildParentActionSequenceCount"], same, f"{split} action match count")
        diag = summary["rewardHeadDiagnostic"]
        if diag["pairCount"] != 2 or len(diag["signedMargins"]) != 2:
            raise ValueError(f"{split} diagnostic pair count differs")
        accuracy = sum(float(margin) > 0 for margin in diag["signedMargins"]) / 2
        close(diag["positiveMarginAccuracy"], accuracy, f"{split} diagnostic accuracy")
        if diag["diagnosticOnly"] is not True:
            raise ValueError("pairwise evaluation diagnostic is not labeled diagnostic-only")
        reconstructed[split] = {
            "meanChildReward": child_reward,
            "meanParentReward": parent_reward,
            "meanConditionalKlPerAction": conditional,
            "validActionDenominator": valid,
            "sameActionSequenceCount": same,
            "pairwiseAccuracy": accuracy,
        }
    quality = evaluation["qualityGate"]
    held = reconstructed["heldout"]
    reward_pass = held["meanChildReward"] > held["meanParentReward"]
    kl_pass = held["meanConditionalKlPerAction"] <= KL_CEILING
    if quality["status"] != "QUALITY_GATE_FAILED" or quality["overallPassed"] is not False:
        raise ValueError("failed quality status was changed or misrepresented")
    if quality["strictHeldoutRewardFloor"] != "mean child fitted reward > mean matched frozen-parent fitted reward":
        raise ValueError("strict reward floor wording changed")
    if quality["heldoutRewardFloorPassed"] is not reward_pass:
        raise ValueError("heldout reward floor result does not reconstruct")
    close(quality["heldoutRestrictedConditionalKlCeilingPerValidAction"], KL_CEILING, "KL ceiling")
    if quality["heldoutRestrictedConditionalKlCeilingPassed"] is not kl_pass:
        raise ValueError("heldout KL floor result does not reconstruct")
    if reward_pass or not kl_pass:
        raise ValueError("published campaign metrics do not match the accepted failed-floor outcome")
    hist = evaluation["historyDisclosure"]
    if hist["cumulativeCampaignExposure"]["outerUpdateExecutionsAcrossV1AndV5"] != 24 or hist["cumulativeCampaignExposure"]["rewardHeadOptimizerStepsAcrossV1AndV5"] != 240:
        raise ValueError("cumulative campaign exposure disclosure changed")
    if hist["v1"] != "The first attempt completed eight outer updates in process and durably saved only update clock 4. It failed during the first paired development evaluation after one child rollout for seed 20261023 / prompt 4; no matched parent episode, remaining development episode, heldout evaluation, or replay result was produced.":
        raise ValueError("v1 exposure disclosure changed")
    return reconstructed


def validate_trace(raw):
    from io import StringIO

    rows = list(csv.DictReader(StringIO(raw.decode("utf-8"))))
    if len(rows) != 8 or list(rows[0]) != TRACE_FIELDS or [int(row["update"]) for row in rows] != list(range(8)):
        raise ValueError("training trace columns or update range differ")
    valid_total = eos_total = timeout_total = padding_total = 0
    for row in rows:
        valid, eos, timeout, pad = (int(row[key]) for key in ("valid_action_count", "eos_count", "timeout_count", "padding_count"))
        if valid + pad != 12 or eos + timeout > valid:
            raise ValueError("per-update target/padding/termination counts do not fit batch shape")
        valid_total += valid
        eos_total += eos
        timeout_total += timeout
        padding_total += pad
        for field in TRACE_FIELDS[5:11]:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite training trace {field}")
    if (valid_total, eos_total, timeout_total, padding_total) != (89, 13, 11, 7):
        raise ValueError("primary aggregate action counts differ from the accepted independent receipt")
    if rows[0]["reward_fit_steps"] != "120":
        raise ValueError("reward fit step count differs")
    close(rows[0]["reward_fit_initial_pairwise_loss"], 0.6931471824645996, "initial reward fit loss")
    close(rows[0]["reward_fit_final_pairwise_loss"], 0.6878268718719482, "final reward fit loss")
    if rows[3]["split_primary_replay_bytes_equal"] != "true" or rows[7]["final_primary_replay_bytes_equal"] != "true":
        raise ValueError("split/final checkpoint byte equality is missing")
    for row, keys in ((rows[3], ("split_state_json_sha256", "split_state_npz_sha256")), (rows[7], ("final_state_json_sha256", "final_state_npz_sha256"))):
        for key in keys:
            value = row[key]
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"invalid checkpoint digest in {key}")
    return rows


def recompute_gae(probe, gamma, lam):
    n = len(probe["actions"])
    arrays = ("targetRoleMask", "inputRoleMask", "paddingMask", "terminatedMask", "truncatedMask", "terminalRewardByAction", "oldCriticValues", "postActionCriticValues")
    if any(len(probe[name]) != n for name in arrays):
        raise ValueError("boundary probe arrays differ in length")
    mask = probe["targetRoleMask"]
    deltas = [0.0] * n
    advantages = [0.0] * n
    next_adv = 0.0
    for t in range(n - 1, -1, -1):
        if not mask[t]:
            next_adv = 0.0
            continue
        reward = float(probe["terminalRewardByAction"][t])
        value = float(probe["oldCriticValues"][t])
        post = 0.0 if probe["terminatedMask"][t] else float(probe["postActionCriticValues"][t])
        deltas[t] = reward + gamma * post - value
        continues = not probe["terminatedMask"][t] and not probe["truncatedMask"][t]
        advantages[t] = deltas[t] + (gamma * lam * next_adv if continues else 0.0)
        next_adv = advantages[t]
    targets = [float(probe["oldCriticValues"][t]) + advantages[t] if mask[t] else 0.0 for t in range(n)]
    for field, expected in (("deltas", deltas), ("advantages", advantages), ("valueTargets", targets)):
        if len(probe[field]) != n or any(not math.isclose(float(a), b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(probe[field], expected)):
            raise ValueError(f"boundary {field} does not independently reconstruct")
    for i in range(n):
        if probe["paddingMask"][i] == mask[i]:
            raise ValueError("boundary target and padding masks are not complements")
        if not mask[i] and any(float(probe[k][i]) != 0.0 for k in ("terminalRewardByAction", "oldCriticValues", "postActionCriticValues", "deltas", "advantages", "valueTargets")):
            raise ValueError("padding position contains nonzero objective data")
    if probe["inputRoleMask"][0] is not False:
        raise ValueError("probe must expose an unscored first action input")
    if not any(probe["terminatedMask"]) and not any(probe["truncatedMask"]):
        raise ValueError("boundary probe has no stop condition")
    return {"deltas": deltas, "advantages": advantages, "valueTargets": targets}


def validate_boundary(boundary):
    if boundary["schemaVersion"] != 1 or boundary["kind"] != "transformer-lineage-v5-forced-boundary-probes":
        raise ValueError("boundary probe schema differs")
    if "zero optimizer updates" not in boundary["label"] or "not trained model quality" not in boundary["label"]:
        raise ValueError("boundary probes are not labeled as synthetic no-update probes")
    if set(boundary["probes"]) != {"eos", "timeout"}:
        raise ValueError("EOS/timeout probe set differs")
    expected_stop = {"eos": (2, True), "timeout": (4, False)}
    results = {}
    for name, probe in boundary["probes"].items():
        stop_action, terminated = expected_stop[name]
        valid_positions = [i for i, valid in enumerate(probe["targetRoleMask"]) if valid]
        if not valid_positions:
            raise ValueError("boundary probe contains no valid actions")
        last = valid_positions[-1]
        if name == "eos" and (probe["actions"][last] != stop_action or not probe["terminatedMask"][last]):
            raise ValueError("EOS probe does not terminate on sampled EOS")
        if name == "timeout" and (probe["actions"][last] != stop_action or not probe["truncatedMask"][last] or probe["terminatedMask"][last]):
            raise ValueError("timeout probe does not truncate with value bootstrap")
        if name == "timeout" and float(probe["postActionCriticValues"][last]) == 0.0:
            raise ValueError("timeout probe has no post-action critic bootstrap")
        if name == "eos" and float(probe["postActionCriticValues"][last]) != 0.0:
            raise ValueError("EOS probe has a nonzero terminal bootstrap")
        results[name] = recompute_gae(probe, float(boundary["gamma"]), float(boundary["gaeLambda"]))
    return results


def validate_packet(packet_dir: Path, repo_root: Path, shipping_approval_path: Path):
    packet_dir = packet_dir.resolve()
    names = {path.name for path in packet_dir.iterdir() if path.is_file()}
    if names != PAYLOAD_NAMES | {"packet-manifest.json"}:
        raise ValueError("packet directory has omitted or extra files")
    manifest = read_json(packet_dir / "packet-manifest.json")
    if manifest.get("schemaVersion") != 1 or manifest.get("kind") != "transformer-lineage-v5-numeric-teaching-packet":
        raise ValueError("packet manifest schema differs")
    if manifest.get("qualityGateFailed") is not True or manifest.get("qualityStatus") != "QUALITY_GATE_FAILED":
        raise ValueError("manifest cannot claim a quality pass")
    if manifest.get("publicTeachingContractSha256") != CONTRACT_SHA256:
        raise ValueError("public teaching contract identity differs")
    as_run = manifest["historicalAsRun"]
    if as_run.get("runApprovalSha256") != RUN_APPROVAL_SHA256 or as_run.get("sourceReviewReceiptSha256") != REVIEW_SHA256 or as_run.get("rawEvidenceSha256") != RAW_SHA256 or as_run.get("campaignResultReceiptSha256") != RESULT_RECEIPT_SHA256 or as_run.get("recipeSha256") != RECIPE_SHA256 or as_run.get("fixtureSha256") != FIXTURE_SHA256:
        raise ValueError("historical campaign evidence identity differs")
    if as_run.get("filesSha256") != EXPECTED_AS_RUN_FILES or as_run.get("boundFileCount") != 15:
        raise ValueError("historical as-run source inventory differs")
    shipping = manifest["shippingSource"]
    if shipping.get("portableNonRunApprovalSha256") != SHIPPING_APPROVAL_SHA256 or shipping.get("runAuthorized") is not False:
        raise ValueError("shipping source approval is not the exact non-run record")
    approval_bytes = shipping_approval_path.read_bytes()
    if digest(approval_bytes) != SHIPPING_APPROVAL_SHA256:
        raise ValueError("shipping approval bytes differ from the pinned non-run record")
    approval = json.loads(approval_bytes)
    if approval.get("runAuthorized") is not False or approval.get("status") != "source-reviewed-nonrun":
        raise ValueError("shipping approval permits or misstates campaign run authorization")
    shipping_files = shipping.get("filesSha256")
    if shipping_files != approval.get("filesSha256") or shipping.get("boundFileCount") != 15:
        raise ValueError("manifest shipping inventory differs from approved inventory")
    if set(shipping_files) != set(EXPECTED_AS_RUN_FILES):
        raise ValueError("shipping source inventory file set differs")
    changed = [path for path in EXPECTED_AS_RUN_FILES if EXPECTED_AS_RUN_FILES[path] != shipping_files[path]]
    if changed != ["projects/posttraining-audit/transformer_lineage/starter.py"]:
        raise ValueError("shipping inventory does not contain exactly the disclosed starter helper fix")
    if shipping_files[changed[0]] != SHIPPING_STARTER_SHA256 or as_run["filesSha256"][changed[0]] != AS_RUN_STARTER_SHA256:
        raise ValueError("historical and shipping starter hashes differ from the approved repair")
    for relative, expected in shipping_files.items():
        path = repo_root / relative
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"shipping source inventory mismatch: {relative}")
    expected_payloads = {}
    for name in PAYLOAD_NAMES:
        raw = (packet_dir / name).read_bytes()
        if name in EXPECTED_PUBLIC_SHA256 and digest(raw) != EXPECTED_PUBLIC_SHA256[name]:
            raise ValueError(f"accepted public numeric payload differs: {name}")
        entry = manifest["payloads"].get(name)
        if not isinstance(entry, dict) or entry.get("bytes") != len(raw) or entry.get("sha256") != digest(raw):
            raise ValueError(f"packet payload byte/hash mismatch: {name}")
        expected_payloads[name] = entry
    if set(manifest["payloads"]) != PAYLOAD_NAMES or manifest.get("manifestSelfExcluded") is not True:
        raise ValueError("manifest payload allowlist differs or self hash is improperly included")
    evaluation = read_json(packet_dir / "evaluation.json")
    metrics = validate_evaluation(evaluation, repo_root)
    trace_raw = (packet_dir / "training-trace.csv").read_bytes()
    rows = validate_trace(trace_raw)
    boundary = read_json(packet_dir / "boundary-probes.json")
    probe_results = validate_boundary(boundary)
    rendered = load_builder().render_svg(evaluation, rows)
    if rendered != (packet_dir / "figure.svg").read_bytes():
        raise ValueError("figure is not the deterministic render of the public numeric data")
    svg_text = rendered.decode("utf-8")
    if "QUALITY_GATE_FAILED" in svg_text or "quality gate failed" not in svg_text.lower() or "0.000123197" not in svg_text:
        raise ValueError("figure must show failed quality outcome and restricted KL result")
    return {
        "status": "passed",
        "qualityStatus": "QUALITY_GATE_FAILED",
        "shippingFilesVerified": len(shipping_files),
        "asRunFilesHistoricallyBound": len(as_run["filesSha256"]),
        "asRunByteIdenticalShippingFiles": shipping["asRunByteIdenticalFileCount"],
        "changedShippingFile": changed[0],
        "evaluation": metrics,
        "boundaryProbeChecks": sorted(probe_results),
        "trainingUpdates": len(rows),
        "payloads": expected_payloads,
    }


def tamper_tests(packet_dir, repo_root, shipping_approval):
    passed = []
    evaluation = read_json(packet_dir / "evaluation.json")
    bad_schedule = json.loads(json.dumps(evaluation))
    bad_schedule["splits"]["dev"]["episodes"][0]["promptId"] += 1
    expect_evaluation_reject("prompt-schedule-mutation", bad_schedule, repo_root, passed)
    bad_grammar = json.loads(json.dumps(evaluation))
    bad_grammar["splits"]["heldout"]["episodes"][0]["childActionIds"] = [2, 4, 383]
    expect_evaluation_reject("early-eos-action-grammar", bad_grammar, repo_root, passed)
    with tempfile.TemporaryDirectory(prefix="lineage-public-packet-negative-") as temp:
        root = Path(temp)
        # Payload bytes altered without changing manifest hash.
        p = root / "payload"
        shutil.copytree(packet_dir, p)
        (p / "evaluation.json").write_text((p / "evaluation.json").read_text().replace('"meanChildFittedRewardPerEpisode": -0.009211033582687378', '"meanChildFittedRewardPerEpisode": -0.009111033582687378'))
        expect_reject("payload-byte-tamper", p, repo_root, shipping_approval, passed)
        # Rehashed numeric payload still fails independent metric reconstruction.
        p = root / "rehash-metric"
        shutil.copytree(packet_dir, p)
        ev_path = p / "evaluation.json"
        evaluation = read_json(ev_path)
        evaluation["splits"]["heldout"]["episodes"][0]["childFittedRewardScore"] += 0.01
        ev_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        update_payload_hash(p, "evaluation.json")
        expect_reject("rehash-metric-tamper", p, repo_root, shipping_approval, passed)
        # Balanced changes preserve split means and floor outcomes, but not the accepted episode payload.
        p = root / "balanced-rehashed-episode-mutation"
        shutil.copytree(packet_dir, p)
        ev_path = p / "evaluation.json"
        evaluation = read_json(ev_path)
        episodes = evaluation["splits"]["heldout"]["episodes"]
        episodes[0]["childFittedRewardScore"] += 0.01
        episodes[1]["childFittedRewardScore"] -= 0.01
        ev_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        update_payload_hash(p, "evaluation.json")
        expect_reject("balanced-rehashed-episode-mutation", p, repo_root, shipping_approval, passed)
        # A plausible but wrong trace reset drops terminal delta propagation to earlier actions.
        p = root / "gae-reset-mutant"
        shutil.copytree(packet_dir, p)
        boundary_path = p / "boundary-probes.json"
        boundary = read_json(boundary_path)
        probe = boundary["probes"]["eos"]
        wrong = [0.0] * len(probe["actions"])
        carry = 0.0
        for t in range(len(wrong) - 1, -1, -1):
            if not probe["targetRoleMask"][t]:
                carry = 0.0
                continue
            bootstrap = 0.0 if probe["terminatedMask"][t] else probe["postActionCriticValues"][t]
            delta = probe["terminalRewardByAction"][t] + boundary["gamma"] * bootstrap - probe["oldCriticValues"][t]
            continues = not probe["terminatedMask"][t] and not probe["truncatedMask"][t]
            wrong[t] = delta + (boundary["gamma"] * boundary["gaeLambda"] * carry if continues else 0.0)
            carry = wrong[t] if continues else 0.0
        probe["advantages"] = wrong
        probe["valueTargets"] = [float(old) + adv if valid else 0.0 for old, adv, valid in zip(probe["oldCriticValues"], wrong, probe["targetRoleMask"])]
        boundary_path.write_text(json.dumps(boundary, indent=2, sort_keys=True) + "\n")
        update_payload_hash(p, "boundary-probes.json")
        expect_reject("terminal-trace-reset-mutant", p, repo_root, shipping_approval, passed)
        # A forbidden probability/logit payload is rejected even with matching public checksums.
        p = root / "logit-payload"
        shutil.copytree(packet_dir, p)
        ev_path = p / "evaluation.json"
        evaluation = read_json(ev_path)
        evaluation["restrictedLogits"] = [[0.1, 0.2, 0.3, 0.4]]
        ev_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        update_payload_hash(p, "evaluation.json")
        expect_reject("forbidden-logit-payload", p, repo_root, shipping_approval, passed)
        # A misleading quality-pass flag is rejected, including after rehashing.
        p = root / "quality-pass-text"
        shutil.copytree(packet_dir, p)
        ev_path = p / "evaluation.json"
        evaluation = read_json(ev_path)
        evaluation["qualityGate"]["overallPassed"] = True
        ev_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        update_payload_hash(p, "evaluation.json")
        expect_reject("misleading-quality-pass", p, repo_root, shipping_approval, passed)
        # Altering a source pin cannot pass the external shipping approval.
        p = root / "source-manifest"
        shutil.copytree(packet_dir, p)
        manifest_path = p / "packet-manifest.json"
        manifest = read_json(manifest_path)
        manifest["shippingSource"]["filesSha256"]["projects/posttraining-audit/transformer_lineage/starter.py"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        expect_reject("source-inventory-tamper", p, repo_root, shipping_approval, passed)
        # Extra archive dependency/public payload is rejected.
        p = root / "extra-file"
        shutil.copytree(packet_dir, p)
        (p / "parent-logits.json").write_text("{}\n")
        expect_reject("extra-payload", p, repo_root, shipping_approval, passed)
    return passed


def update_payload_hash(packet, name):
    manifest_path = packet / "packet-manifest.json"
    manifest = read_json(manifest_path)
    raw = (packet / name).read_bytes()
    manifest["payloads"][name] = {"bytes": len(raw), "sha256": digest(raw)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def expect_reject(label, packet, repo_root, approval, passed):
    try:
        validate_packet(packet, repo_root, approval)
    except (ValueError, KeyError, json.JSONDecodeError):
        passed.append(label)
        return
    raise AssertionError(f"tamper control was accepted: {label}")


def expect_evaluation_reject(label, evaluation, repo_root, passed):
    try:
        validate_evaluation(evaluation, repo_root)
    except (ValueError, KeyError, json.JSONDecodeError):
        passed.append(label)
        return
    raise AssertionError(f"evaluation mutation was accepted: {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=HERE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--shipping-approval", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    report = validate_packet(args.packet_dir, args.repo_root, args.shipping_approval)
    if args.self_test:
        report["tamperControlsRejected"] = tamper_tests(args.packet_dir, args.repo_root, args.shipping_approval)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
