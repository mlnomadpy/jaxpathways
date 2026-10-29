#!/usr/bin/env python3
"""Build the allowlisted v5 teaching numbers without importing a model runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

CONTRACT_SHA256 = "8e9abc0cce93d20d02f086c7cf5334966a8088ad6e36e27d5aca6e5589850ab9"
RAW_SHA256 = "966fe2d13eb22508e21080f7e8fa007dc51d04e197ba01e00d575ec13187f10e"
REVIEW_SHA256 = "78c1f71266ba7e05bc16010e34700aec6c4997e10781609353eda4713fe43b75"
RESULT_RECEIPT_SHA256 = "6a45955dac4e08c173fec3475c3af977c40f88142461c7ce76dcc53f4ad74a3d"
RUN_APPROVAL_SHA256 = "a47bc625a1a372273f20ddace82d0c3b3ce0d4d5f1336cb448614ace9662eaf9"
SHIPPING_APPROVAL_SHA256 = "28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b"
RECIPE_SHA256 = "8411afc4412c59973d56b2cceae530dd642a5aa1c9e6f57a5dbb7f6ba110bc57"
FIXTURE_SHA256 = "6a47dbe54012ed3013753ca8089fbcd51e57b3d0cbd61ffdb50a6877e0a534ff"
AS_RUN_STARTER_SHA256 = "b7fc9060123b474de92ac04a5480ef818ab296bc216351ed2f7701474741b03d"
SHIPPING_STARTER_SHA256 = "8196af0f989c382ab5c4833102f11a2d1fe2263614e35e8ac1be8ab1ce501d32"
KL_CEILING = 0.8
GAMMA = 0.9
GAE_LAMBDA = 0.8


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flatten(row, key):
    value = row[key]
    if value and isinstance(value[0], list):
        value = value[0]
    return value


def valid_values(row, key):
    actions = flatten(row, "actions")
    mask = flatten(row, "target_role_mask")
    values = flatten(row, key)
    return [float(value) for value, valid in zip(values, mask) if valid]


def evaluation_packet(evidence):
    result = {
        "schemaVersion": 1,
        "kind": "transformer-lineage-v5-public-evaluation",
        "scope": "Synthetic prompts and four restricted actions; per-episode IDs are paired by fixed seed. Scores use the once-fitted synthetic reward head.",
        "klScope": "Exact categorical KL over the eligible four-action subset at sampled child-visited contexts, averaged per valid action. It is not full-vocabulary or full-sequence KL.",
        "splits": {},
    }
    for split in ("dev", "heldout"):
        source = evidence["primary"]["evaluation"][split]
        ids = source["matched_prompt_ids"]
        seeds = source["matched_sampling_seeds"]
        episodes_per_seed = len(ids)
        episodes = []
        for index, (child, parent) in enumerate(
            zip(source["child_rollouts"], source["matched_parent_rollouts"])
        ):
            child_actions = [int(x) for x, ok in zip(flatten(child, "actions"), flatten(child, "target_role_mask")) if ok]
            parent_actions = [int(x) for x, ok in zip(flatten(parent, "actions"), flatten(parent, "target_role_mask")) if ok]
            child_reward = float(child["sequence_scores"][0])
            parent_reward = float(parent["sequence_scores"][0])
            sampled = valid_values(child, "sampled_kl")
            conditional = valid_values(child, "exact_restricted_conditional_kl")
            prompt_id = int(child["prompt_ids"][0])
            seed = int(seeds[index // episodes_per_seed])
            episodes.append(
                {
                    "promptId": prompt_id,
                    "samplingSeed": seed,
                    "childActionIds": child_actions,
                    "parentActionIds": parent_actions,
                    "childFittedRewardScore": child_reward,
                    "parentFittedRewardScore": parent_reward,
                    "validActionDenominator": len(child_actions),
                    "sampledKlSum": sum(sampled),
                    "exactRestrictedConditionalKlSumAtChildVisitedContexts": sum(conditional),
                }
            )
        margins = [float(row["signed_margin"]) for row in source["evaluation_pair_rows"]]
        pair_accuracy = sum(margin > 0.0 for margin in margins) / len(margins)
        total_valid = sum(row["validActionDenominator"] for row in episodes)
        result["splits"][split] = {
            "episodes": episodes,
            "summary": {
                "matchedEpisodeCount": len(episodes),
                "validActionDenominator": total_valid,
                "meanChildFittedRewardPerEpisode": sum(row["childFittedRewardScore"] for row in episodes) / len(episodes),
                "meanParentFittedRewardPerEpisode": sum(row["parentFittedRewardScore"] for row in episodes) / len(episodes),
                "meanSampledKlPerValidAction": sum(row["sampledKlSum"] for row in episodes) / total_valid,
                "meanExactRestrictedConditionalKlPerValidActionAtChildVisitedContexts": sum(row["exactRestrictedConditionalKlSumAtChildVisitedContexts"] for row in episodes) / total_valid,
                "sameChildParentActionSequenceCount": sum(row["childActionIds"] == row["parentActionIds"] for row in episodes),
                "rewardHeadDiagnostic": {
                    "pairCount": len(margins),
                    "signedMargins": margins,
                    "positiveMarginAccuracy": pair_accuracy,
                    "diagnosticOnly": True,
                },
            },
        }
    heldout = result["splits"]["heldout"]["summary"]
    heldout_reward_pass = heldout["meanChildFittedRewardPerEpisode"] > heldout["meanParentFittedRewardPerEpisode"]
    heldout_kl_pass = heldout["meanExactRestrictedConditionalKlPerValidActionAtChildVisitedContexts"] <= KL_CEILING
    result["qualityGate"] = {
        "status": "QUALITY_GATE_FAILED",
        "strictHeldoutRewardFloor": "mean child fitted reward > mean matched frozen-parent fitted reward",
        "heldoutRewardFloorPassed": heldout_reward_pass,
        "heldoutRestrictedConditionalKlCeilingPerValidAction": KL_CEILING,
        "heldoutRestrictedConditionalKlCeilingPassed": heldout_kl_pass,
        "overallPassed": heldout_reward_pass and heldout_kl_pass,
        "failureReason": "Heldout child reward equals matched parent reward, so the strict improvement floor fails.",
    }
    result["historyDisclosure"] = {
        "v1": "The first attempt completed eight outer updates in process and durably saved only update clock 4. It failed during the first paired development evaluation after one child rollout for seed 20261023 / prompt 4; no matched parent episode, remaining development episode, heldout evaluation, or replay result was produced.",
        "v5": "The second attempt completed eight primary updates and an eight-update split/restore replay lane; only the uninterrupted primary lane was quality scored.",
        "cumulativeCampaignExposure": {
            "outerUpdateExecutionsAcrossV1AndV5": 24,
            "rewardHeadOptimizerStepsAcrossV1AndV5": 240,
        },
        "mechanicsChecks": "Five candidate-agnostic invocations under the historical pre-shipping checker formed a separate bounded mechanics exercise. They are not quality evaluations or part of campaign update totals. The new clean-archive shipping qualification remains pending and is not counted as executed.",
        "shippingLearnerMechanicsQualification": "pending separate source-bound approval and clean-archive verification; no invocations counted in this packet",
        "noFurtherRun": True,
    }
    return result


def gae_probe(raw, gamma=GAMMA, lam=GAE_LAMBDA):
    mask = [bool(x) for x in flatten(raw, "target_role_mask")]
    actions = [int(x) for x in flatten(raw, "actions")]
    input_mask = [bool(x) for x in flatten(raw, "input_role_mask")]
    padding = [bool(x) for x in flatten(raw, "padding_mask")]
    terminated = [bool(x) for x in flatten(raw, "terminated")]
    truncated = [bool(x) for x in flatten(raw, "truncated")]
    old = [float(x) for x in flatten(raw, "old_values")]
    post = [float(x) for x in flatten(raw, "post_action_values")]
    rewards = [float(x) for x in flatten(raw, "rewards")]
    terminal_reward = [float(x) for x in flatten(raw, "terminal_reward")]
    deltas = [0.0] * len(mask)
    advantages = [0.0] * len(mask)
    next_advantage = 0.0
    for t in range(len(mask) - 1, -1, -1):
        if not mask[t]:
            next_advantage = 0.0
            continue
        bootstrap = 0.0 if terminated[t] else post[t]
        deltas[t] = rewards[t] + gamma * bootstrap - old[t]
        continues = not terminated[t] and not truncated[t]
        advantages[t] = deltas[t] + (gamma * lam * next_advantage if continues else 0.0)
        next_advantage = advantages[t]
    targets = [old[t] + advantages[t] if mask[t] else 0.0 for t in range(len(mask))]
    valid_rewards = [terminal_reward[i] for i, ok in enumerate(mask) if ok]
    return {
        "actions": actions,
        "targetRoleMask": mask,
        "inputRoleMask": input_mask,
        "paddingMask": padding,
        "terminatedMask": terminated,
        "truncatedMask": truncated,
        "terminalRewardByAction": terminal_reward,
        "terminalRewardScore": sum(valid_rewards),
        "oldCriticValues": old,
        "postActionCriticValues": post,
        "deltas": deltas,
        "advantages": advantages,
        "valueTargets": targets,
    }


def boundary_packet(evidence, recipe):
    probes = {}
    for key in ("eos", "timeout"):
        probes[key] = gae_probe(
            evidence["boundaryProbes"][key],
            recipe["training"]["gamma"],
            recipe["training"]["gaeLambda"],
        )
    return {
        "schemaVersion": 1,
        "kind": "transformer-lineage-v5-forced-boundary-probes",
        "label": "Synthetic forced contract probes with zero optimizer updates; these are not trained model quality measurements.",
        "gamma": recipe["training"]["gamma"],
        "gaeLambda": recipe["training"]["gaeLambda"],
        "units": "Rewards and critic values are synthetic reward units; advantages and value targets use the same units.",
        "probes": probes,
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


def training_rows(evidence, receipt):
    updates = evidence["primary"]["updates"]
    files = receipt["outputFileHashes"]
    rows = []
    for i, update in enumerate(updates):
        mask = update["target_role_mask"]
        term = update["terminated"]
        trunc = update["truncated"]
        padding = update["padding_mask"]
        valid_count = sum(bool(x) for row in mask for x in row)
        eos_count = sum(bool(x) for row in term for x in row)
        timeout_count = sum(bool(x) for row in trunc for x in row)
        pad_count = sum(bool(x) for row in padding for x in row)
        sample_values = [float(v) for row, mr in zip(update["sampled_kl"], mask) for v, ok in zip(row, mr) if ok]
        conditional_values = [float(v) for row, mr in zip(update["exact_restricted_conditional_kl"], mask) for v, ok in zip(row, mr) if ok]
        row = {
            "update": int(update["update"]),
            "valid_action_count": valid_count,
            "eos_count": eos_count,
            "timeout_count": timeout_count,
            "padding_count": pad_count,
            "signed_policy_loss_epoch_0_per_valid_action": float(update["policy_loss_by_epoch"][0]),
            "signed_policy_loss_epoch_1_per_valid_action": float(update["policy_loss_by_epoch"][1]),
            "clipped_critic_loss_epoch_0_per_valid_action": float(update["critic_loss_by_epoch"][0]),
            "clipped_critic_loss_epoch_1_per_valid_action": float(update["critic_loss_by_epoch"][1]),
            "mean_sampled_kl_per_valid_action": sum(sample_values) / len(sample_values),
            "mean_exact_restricted_conditional_kl_per_valid_action": sum(conditional_values) / len(conditional_values),
            "reward_fit_steps": "",
            "reward_fit_initial_pairwise_loss": "",
            "reward_fit_final_pairwise_loss": "",
            "split_state_json_sha256": "",
            "split_state_npz_sha256": "",
            "split_primary_replay_bytes_equal": "",
            "final_state_json_sha256": "",
            "final_state_npz_sha256": "",
            "final_primary_replay_bytes_equal": "",
        }
        if i == 0:
            row.update(
                reward_fit_steps=int(evidence["reward_fit"]["fit_updates"]),
                reward_fit_initial_pairwise_loss=float(evidence["reward_fit"]["initial_pairwise_loss"]),
                reward_fit_final_pairwise_loss=float(evidence["reward_fit"]["final_pairwise_loss"]),
            )
        if i == 3:
            row.update(
                split_state_json_sha256=files["primary-split-state/state.json"]["sha256"],
                split_state_npz_sha256=files["primary-split-state/state.npz"]["sha256"],
                split_primary_replay_bytes_equal="true",
            )
        if i == 7:
            row.update(
                final_state_json_sha256=files["primary-final-state/state.json"]["sha256"],
                final_state_npz_sha256=files["primary-final-state/state.npz"]["sha256"],
                final_primary_replay_bytes_equal="true",
            )
        rows.append(row)
    return rows


def trace_csv_bytes(rows):
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_svg(evaluation, trace_rows):
    dev = evaluation["splits"]["dev"]["summary"]
    held = evaluation["splits"]["heldout"]["summary"]
    delta_dev = dev["meanChildFittedRewardPerEpisode"] - dev["meanParentFittedRewardPerEpisode"]
    delta_held = held["meanChildFittedRewardPerEpisode"] - held["meanParentFittedRewardPerEpisode"]
    policies = [float(row["signed_policy_loss_epoch_1_per_valid_action"]) for row in trace_rows]
    y_min, y_max = -0.02, 0.06
    def py(value):
        return 475 - (value - y_min) / (y_max - y_min) * 270
    zero_y = py(0.0)
    points = " ".join(f"{650 + i * 38},{py(value):.2f}" for i, value in enumerate(policies))
    ticks = "".join(
        f'<line x1="{650 + i * 38}" y1="475" x2="{650 + i * 38}" y2="482" class="axis"/><text x="{650 + i * 38}" y="501" text-anchor="middle" class="small">{i}</text>'
        for i in range(len(policies))
    )
    circles = "".join(
        f'<circle cx="{650 + i * 38}" cy="{py(value):.2f}" r="6" fill="#2459a6"/><text x="{650 + i * 38}" y="{py(value) + (-12 if i % 2 == 0 else 19):.2f}" text-anchor="middle" class="small">{value:.3f}</text>'
        for i, value in enumerate(policies)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620" role="img" aria-labelledby="title desc">
<title id="title">Synthetic saved-decoder PPO results: quality gate failed</title>
<desc id="desc">Left: mean child minus matched-parent fitted reward is zero in development and heldout; heldout equality fails the strict positive improvement gate. Right: primary-lane signed clipped policy loss per valid action from the second inner optimizer epoch (epoch index 1), at primary update numbers 0 through 7. The exact restricted categorical KL is 0.000123197 per valid action at sampled child-visited contexts, below the 0.8 ceiling.</desc>
<style>text{{font-family:Arial,Helvetica,sans-serif;fill:#17212b}}.title{{font-size:25px;font-weight:700}}.panel{{font-size:20px;font-weight:700}}.label{{font-size:18px}}.small{{font-size:15px}}.muted{{fill:#51616f}}.axis{{stroke:#45525e;stroke-width:2}}.grid{{stroke:#cbd4dc;stroke-width:1}}.bad{{fill:#a12b2b}}.blue{{fill:#2459a6}}</style>
<rect width="1000" height="620" fill="#fff"/>
<text x="30" y="40" class="title">Bounded synthetic sequence-PPO run: quality gate failed</text>
<text x="30" y="72" class="label muted">One frozen synthetic reward head; 8 matched episodes per split; all reported KL is restricted to four eligible actions.</text>
<line x1="500" y1="100" x2="500" y2="500" stroke="#d6dde3"/>
<text x="45" y="120" class="panel">Mean reward improvement</text>
<text x="45" y="147" class="label muted">Reward units; 8 matched episodes per split</text>
<line x1="105" y1="225" x2="445" y2="225" class="grid"/><line x1="105" y1="325" x2="445" y2="325" class="axis"/><line x1="105" y1="425" x2="445" y2="425" class="grid"/>
<line x1="105" y1="205" x2="105" y2="445" class="axis"/><text x="94" y="231" text-anchor="end" class="small">+0.01</text><text x="94" y="331" text-anchor="end" class="small">0</text><text x="94" y="431" text-anchor="end" class="small">−0.01</text>
<circle cx="210" cy="{325 - delta_dev*10000:.2f}" r="9" fill="#2459a6"/><circle cx="350" cy="{325 - delta_held*10000:.2f}" r="9" fill="#a12b2b"/>
<text x="210" y="365" text-anchor="middle" class="small">dev Δ {delta_dev:+.3f}</text><text x="350" y="365" text-anchor="middle" class="small bad">heldout Δ {delta_held:+.3f}</text>
<text x="45" y="465" class="small bad">Floor requires heldout difference &gt; 0; observed equality fails.</text>
<text x="530" y="120" class="panel">Primary policy loss · epoch 2</text>
<text x="530" y="147" class="label muted">Per valid action; training diagnostic, not quality</text>
<line x1="610" y1="205" x2="930" y2="205" class="grid"/><line x1="610" y1="340" x2="930" y2="340" class="axis"/><line x1="610" y1="475" x2="930" y2="475" class="grid"/>
<line x1="610" y1="190" x2="610" y2="490" class="axis"/><text x="600" y="211" text-anchor="end" class="small">+0.06</text><text x="600" y="346" text-anchor="end" class="small">+0.02</text><text x="600" y="481" text-anchor="end" class="small">−0.02</text>
<polyline points="{points}" fill="none" stroke="#2459a6" stroke-width="3"/>{ticks}{circles}
<text x="770" y="535" text-anchor="middle" class="label">Primary update number</text>
<text x="30" y="555" class="label">Heldout exact restricted KL: {held['meanExactRestrictedConditionalKlPerValidActionAtChildVisitedContexts']:.9f} per valid action (ceiling {KL_CEILING:.1f}).</text>
<text x="30" y="585" class="small muted">Conditional over eligible actions at sampled child-visited contexts; not full-vocabulary or full-sequence KL.</text>
</svg>\n'''.encode("utf-8")


def inventory(approval):
    return approval["filesSha256"]


def verify_inventory(repo_root: Path, approval: dict, expected_sha: str, label: str):
    files = inventory(approval)
    if len(files) != 15:
        raise ValueError(f"{label} inventory must contain exactly 15 files")
    for relative, digest in files.items():
        path = repo_root / relative
        if not path.is_file() or sha256(path.read_bytes()) != digest:
            raise ValueError(f"{label} source inventory mismatch: {relative}")
    return files


def build(args):
    for path, expected, label in (
        (args.contract, CONTRACT_SHA256, "contract"),
        (args.raw_evidence, RAW_SHA256, "raw evidence"),
        (args.review_receipt, REVIEW_SHA256, "Sol review receipt"),
        (args.result_receipt, RESULT_RECEIPT_SHA256, "campaign result receipt"),
        (args.run_approval, RUN_APPROVAL_SHA256, "historical run approval"),
        (args.shipping_approval, SHIPPING_APPROVAL_SHA256, "shipping non-run approval"),
    ):
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f"{label} bytes do not match pinned SHA-256")
    contract = read_json(args.contract)
    evidence = read_json(args.raw_evidence)
    review = read_json(args.review_receipt)
    receipt = read_json(args.result_receipt)
    run_approval = read_json(args.run_approval)
    shipping_approval = read_json(args.shipping_approval)
    if contract.get("kind") != "sol-transformer-lineage-v5-public-teaching-contract-v3":
        raise ValueError("unexpected public teaching contract")
    if evidence.get("identity", {}).get("recipe_sha256") != RECIPE_SHA256 or evidence.get("identity", {}).get("fixture_sha256") != FIXTURE_SHA256:
        raise ValueError("raw evidence does not identify the frozen recipe and fixture")
    as_run_files = run_approval["filesSha256"]
    shipping_files = shipping_approval["filesSha256"]
    if len(as_run_files) != 15 or len(shipping_files) != 15:
        raise ValueError("source inventories must each contain exactly 15 entries")
    if as_run_files.get("projects/posttraining-audit/transformer_lineage/starter.py") != AS_RUN_STARTER_SHA256:
        raise ValueError("historical as-run starter pin differs")
    if shipping_files.get("projects/posttraining-audit/transformer_lineage/starter.py") != SHIPPING_STARTER_SHA256:
        raise ValueError("shipping starter pin differs")
    if set(as_run_files) != set(shipping_files):
        raise ValueError("as-run and shipping source file sets differ")
    if any(as_run_files[path] != shipping_files[path] for path in as_run_files if path != "projects/posttraining-audit/transformer_lineage/starter.py"):
        raise ValueError("a source outside the disclosed starter helper changed")
    as_run_root = args.as_run_source_root
    for relative, digest in as_run_files.items():
        if sha256((as_run_root / relative).read_bytes()) != digest:
            raise ValueError(f"historical as-run source snapshot mismatch: {relative}")
    verified_shipping = verify_inventory(args.repo_root, shipping_approval, SHIPPING_APPROVAL_SHA256, "shipping")
    if (
        review.get("decision", {}).get("recordedStatus") != "QUALITY_GATE_FAILED"
        or receipt.get("attempt", {}).get("result") != "QUALITY_GATE_FAILED"
        or receipt.get("quality", {}).get("passed") is not False
    ):
        raise ValueError("campaign quality status is not the accepted failed result")
    evaluation = evaluation_packet(evidence)
    boundary = boundary_packet(evidence, read_json(args.repo_root / "projects/posttraining-audit/transformer_lineage/recipe.json"))
    rows = training_rows(evidence, receipt)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    generated_names = {"evaluation.json", "boundary-probes.json", "training-trace.csv", "figure.svg", "packet-manifest.json"}
    if any((output / name).exists() for name in generated_names):
        raise ValueError("refusing to overwrite an existing numeric packet")
    write_json(output / "evaluation.json", evaluation)
    write_json(output / "boundary-probes.json", boundary)
    (output / "training-trace.csv").write_bytes(trace_csv_bytes(rows))
    (output / "figure.svg").write_bytes(render_svg(evaluation, rows))
    payloads = {}
    for name in ("evaluation.json", "training-trace.csv", "boundary-probes.json", "figure.svg", "build_numeric_packet.py", "check_numeric_packet.py"):
        path = output / name
        if not path.is_file():
            raise ValueError(f"public packet payload not present yet: {name}")
        raw = path.read_bytes()
        payloads[name] = {"bytes": len(raw), "sha256": sha256(raw)}
    changed = {
        "projects/posttraining-audit/transformer_lineage/starter.py": {
            "asRunSha256": AS_RUN_STARTER_SHA256,
            "shippingSha256": SHIPPING_STARTER_SHA256,
            "change": "One postrun source-only repair changes evaluate's matched baseline rollout critic argument from reference_params to the supplied critic_params. The 15-file as-run inventory and v5 measurements remain historical and unchanged.",
        }
    }
    manifest = {
        "schemaVersion": 1,
        "kind": "transformer-lineage-v5-numeric-teaching-packet",
        "qualityGateFailed": True,
        "qualityStatus": "QUALITY_GATE_FAILED",
        "publicTeachingContractSha256": CONTRACT_SHA256,
        "historicalAsRun": {
            "runApprovalSha256": RUN_APPROVAL_SHA256,
            "sourceReviewReceiptSha256": REVIEW_SHA256,
            "recipeSha256": RECIPE_SHA256,
            "fixtureSha256": FIXTURE_SHA256,
            "rawEvidenceSha256": RAW_SHA256,
            "campaignResultReceiptSha256": RESULT_RECEIPT_SHA256,
            "filesSha256": dict(sorted(as_run_files.items())),
            "boundFileCount": len(as_run_files),
        },
        "shippingSource": {
            "portableNonRunApprovalSha256": SHIPPING_APPROVAL_SHA256,
            "filesSha256": dict(sorted(verified_shipping.items())),
            "boundFileCount": len(verified_shipping),
            "asRunByteIdenticalFileCount": 14,
            "changedFiles": changed,
            "runAuthorized": False,
            "sourceReviewRequired": False,
            "status": "source-reviewed-nonrun",
        },
        "sourceRepairQualification": {
            "starterSha256": SHIPPING_STARTER_SHA256,
            "helperAstEqualToReferenceCount": 20,
            "learnerTodoFunctionCount": 6,
            "completedSixTodoSemantic": "passed with zero optimizer updates, cold-loaded checkpoint evaluation, and matched dev/heldout smoke",
            "untouchedStarter": "rejected at learner TODO",
            "historicalCriticHelperMutant": "rejected by fresh-process reloaded evaluation smoke",
            "candidateSha256": "02503c4b04cbf4a4bd026a2df830c7db9ad419bad89eaff1bc0328f52cc80849",
            "astProofSha256": sha256(args.ast_proof.read_bytes()),
            "semanticReceiptSha256": sha256(args.semantic_receipt.read_bytes()),
        },
        "campaignExposure": {
            "v1V5OuterUpdateExecutions": 24,
            "v1V5RewardOptimizerSteps": 240,
            "v1MatchedEvaluation": False,
            "v1ExposureDisclosure": "Eight v1 updates were executed in process, only update clock 4 was durable, and one child development rollout occurred before the baseline critic error; no matched parent comparison or completed heldout quality result.",
            "v5NewRewardHeadFits": 1,
            "v5RewardFitSteps": 120,
            "v5PrimaryUpdates": 8,
            "v5ReplayUpdates": 8,
            "v5QualityScoredLane": "primary-uninterrupted",
            "historicalPreShippingMechanicsInvocations": 5,
            "shippingLearnerMechanicsInvocations": 0,
            "shippingLearnerMechanicsStatus": "pending clean-archive qualification",
        },
        "payloads": payloads,
        "manifestSelfExcluded": True,
    }
    write_json(output / "packet-manifest.json", manifest)
    print(json.dumps({"packet": str(output), "qualityStatus": manifest["qualityStatus"], "payloads": payloads, "manifestSha256": sha256((output / "packet-manifest.json").read_bytes())}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--result-receipt", type=Path, required=True)
    parser.add_argument("--run-approval", type=Path, required=True, help="Historical v5 approval used only to verify the as-run source snapshot.")
    parser.add_argument("--shipping-approval", type=Path, required=True, help="Source-reviewed non-run record; runAuthorized must remain false.")
    parser.add_argument("--as-run-source-root", type=Path, required=True, help="Root of the preserved 15-file historical as-run source copy.")
    parser.add_argument("--repo-root", type=Path, required=True, help="Current shipping source tree root; verifies all 15 approved files.")
    parser.add_argument("--contract", type=Path, required=True, help="Private retained contract used for full regeneration; public packet verification does not need it.")
    parser.add_argument("--ast-proof", type=Path, required=True)
    parser.add_argument("--semantic-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
