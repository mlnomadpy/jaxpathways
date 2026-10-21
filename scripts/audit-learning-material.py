"""Inventory current learning material without turning execution into teaching approval."""

from collections import Counter
import hashlib
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def project_digest(directory):
    result = hashlib.sha256()
    for file in sorted(
        p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ):
        result.update(
            str(file.relative_to(ROOT)).encode() + b"\0" + file.read_bytes() + b"\0"
        )
    return result.hexdigest()


def lesson_check(content, content_hash, receipt, generated_hash):
    """An old receipt, missing notebook run or failed run cannot confer current validation."""
    if not content:
        return "No lesson body"
    if (
        not receipt
        or receipt.get("contentHash") != content_hash
        or generated_hash != content_hash
    ):
        return "CPU check pending for this source"
    if receipt.get("script") != "passed" or receipt.get("notebookCode") != "passed":
        return "CPU check failed or incomplete"
    count = receipt.get("runtime", {}).get("deviceCount", 1)
    return "Logical CPU reference checked" if count > 1 else "CPU reference checked"


def accepted_reviews(reviews, lesson_id, content_hash):
    return [
        r
        for r in reviews
        if r.get("lessonId") == lesson_id
        and r.get("contentHash") == content_hash
        and r.get("outcome") == "accepted"
        and r.get("kind") in ("expert", "novice")
        and all(r.get(k) for k in ("reviewer", "date", "changedCondition", "evidence"))
        and (ROOT / r["evidence"]).is_file()
    ]


def build():
    course = read("curriculum/course.json")
    review = read("curriculum/content-review.json")
    validations = {
        r["lessonId"]: r for r in read("curriculum/validation.json")["lessons"]
    }
    project_validations = {
        r["projectId"]: r
        for r in read("curriculum/project-validation.json")["projects"]
    }
    teaching_reviews = read("curriculum/teaching-reviews.json")["reviews"]
    phase_notes = {r["id"]: r for r in review["phases"]}
    project_notes = {r["id"]: r for r in review["projects"]}
    project_manifests = [read(path) for path in read("curriculum/projects.json")]
    assert set(phase_notes) == {
        p["id"] for p in course["phases"]
    }, "Update the phase scope review"
    assert set(project_notes) == {
        p["id"] for p in project_manifests
    }, "Update the project scope review"
    items, tasks = [], []

    def add(
        kind, id, title, url, source, level, summary, contribution, validation, **extra
    ):
        assert level in review["levels"]
        item = dict(
            kind=kind,
            id=id,
            title=title,
            url=url,
            source=source,
            level=level,
            summary=summary,
            contribution=contribution,
            validation=validation,
            review="Teaching review pending",
            **extra,
        )
        items.append(item)
        return item

    for phase in course["phases"]:
        children = []
        for lesson in phase["lessons"]:
            source = f'{lesson["path"]}/lesson.json'
            body = read(source) if (ROOT / source).exists() else {}
            content = body.get("content", {})
            content_hash = digest(content)
            generated = ROOT / f'public/api/v1/lessons/{lesson["id"]}.json'
            generated_hash = (
                json.loads(generated.read_text()).get("contentHash")
                if generated.exists()
                else None
            )
            validation = lesson_check(
                content, content_hash, validations.get(lesson["id"]), generated_hash
            )
            missing = [
                key
                for key in (
                    "problem",
                    "objectives",
                    "sections",
                    "diagnosis",
                    "references",
                    "exercise",
                    "solution",
                    "question",
                    "options",
                    "explanation",
                )
                if not content.get(key)
            ]
            if not content.get("setupBundle"):
                missing += [
                    key for key in ("experiments", "practice") if not content.get(key)
                ]
            missing_files = []
            for key, value in lesson.get("artifacts", {}).items():
                file = (
                    ROOT / value
                    if key.endswith("Source") or key == "source"
                    else ROOT / "public" / value
                )
                if not file.is_file():
                    missing_files.append(value)
            level = (
                "needs-repair"
                if not content or missing or missing_files
                else "available-draft"
            )
            if level != "needs-repair" and "pending" in validation:
                level = "in-development"
            if "failed" in validation:
                level = "needs-repair"
            current_reviews = accepted_reviews(
                teaching_reviews, lesson["id"], content_hash
            )
            expert = any(r["kind"] == "expert" for r in current_reviews)
            novice = any(r["kind"] == "novice" for r in current_reviews)
            metrics = dict(
                sections=len(content.get("sections", [])),
                experiments=len(content.get("experiments", [])),
                practice=len(content.get("practice", [])),
                references=len(content.get("references", [])),
                expertReviewed=expert,
                novicePiloted=novice,
                missing=missing,
                missingFiles=missing_files,
                choiceSpecificFeedback=(
                    len(content.get("optionFeedback", []))
                    == len(content.get("options", []))
                    and len(set(content.get("optionFeedback", [])))
                    == len(content.get("options", []))
                    and all(
                        isinstance(item, str) and item.strip()
                        for item in content.get("optionFeedback", [])
                    )
                ),
            )
            item = add(
                "lessons",
                lesson["id"],
                lesson["title"],
                f'lesson-{lesson["id"]}.html',
                source,
                level,
                lesson["objective"],
                phase_notes[phase["id"]]["contribution"],
                validation,
                phaseId=phase["id"],
                sourceHash=content_hash,
                metrics=metrics,
            )
            item["review"] = (
                "Expert review recorded" if expert else "Expert review pending"
            ) + ("; learner pilot recorded" if novice else "; learner pilot pending")
            children.append(item)
            task_rows = [
                ("exercise", "Main exercise", content.get("exercise", "")),
                ("checkpoint", "Understanding checkpoint", content.get("question", "")),
            ]
            task_rows += [
                (
                    f"practice-{i+1}",
                    p.get("title", f"Practice {i+1}"),
                    p.get("prompt", ""),
                )
                for i, p in enumerate(content.get("practice", []))
            ]
            for task_id, title, prompt in task_rows:
                tasks.append(
                    dict(
                        id=f'{lesson["id"]}:{task_id}',
                        lessonId=lesson["id"],
                        phaseId=phase["id"],
                        title=title,
                        prompt=prompt,
                        level=level,
                        source=source,
                        validation=(
                            "Authored answer and rationale; uncalibrated checkpoint"
                            if task_id == "checkpoint"
                            else validation
                            + "; reference solution, not learner grading"
                        ),
                    )
                )
        note = phase_notes[phase["id"]]
        level = (
            "needs-repair"
            if any(x["level"] == "needs-repair" for x in children)
            else (
                "in-development"
                if any(x["level"] == "in-development" for x in children)
                else note["level"]
            )
        )
        add(
            "phases",
            phase["id"],
            phase["title"],
            f'phase-guides/{phase["id"]}.html',
            "curriculum/course.json",
            level,
            note["strength"],
            note["contribution"],
            f'{sum("reference checked" in x["validation"] for x in children)}/{len(children)} current CPU references',
            metrics=dict(
                lessons=len(children),
                estimatedMinutes=sum(
                    read(f'{l["path"]}/lesson.json').get("minutes", 0)
                    for l in phase["lessons"]
                ),
            ),
        )

    for project in project_manifests:
        note = project_notes[project["id"]]
        directory = ROOT / Path(project["source"]).parent
        source_hash = project_digest(directory)
        receipt = project_validations.get(project["id"], {})
        current = (
            receipt.get("sourceHash") == source_hash
            and receipt.get("status") == "passed"
        )
        preparation = shlex.split(project.get("prepare", [{}])[0].get("command", ""))
        workspace = project["workspaceFile"]
        starter = (
            preparation[1]
            if preparation and preparation[0] == "cp"
            else (
                workspace
                if "/starter/" in workspace
                else f'projects/{project["id"]}/starter/{Path(workspace).name.removeprefix("my_")}'
            )
        )
        solution = starter.replace("/starter/", "/solution/")
        needed = [
            directory / name for name in ("README.md", "project.json", "tests/check.py")
        ]
        needed += [ROOT / starter, ROOT / solution]
        missing = [str(path.relative_to(ROOT)) for path in needed if not path.is_file()]
        project_item = add(
            "projects",
            project["id"],
            project["title"],
            f'project-guides/{project["id"]}.html',
            project["source"],
            (
                "needs-repair"
                if missing
                or (
                    receipt.get("sourceHash") == source_hash
                    and receipt.get("status") == "failed"
                )
                else (note["level"] if current else "in-development")
            ),
            note["strength"],
            note["contribution"],
            (
                "CPU reference checker passed for current sources"
                if current
                else (
                    "Current reference checker failed"
                    if receipt.get("sourceHash") == source_hash
                    and receipt.get("status") == "failed"
                    else "Reference check pending for this source"
                )
            ),
            sourceHash=source_hash,
            metrics=dict(
                stages=len(project["stages"]),
                currentReference=current,
                missingFiles=missing,
            ),
        )
        for stage in project["stages"]:
            tasks.append(
                dict(
                    id=f'{project["id"]}:stage-{stage["id"]}',
                    projectId=project["id"],
                    phaseId="",
                    title=stage["title"],
                    prompt=stage["evidence"],
                    source=project["source"],
                    level=project_item["level"],
                    validation=(
                        "Public reference stage checked; no independent learner assessment"
                        if current
                        else (
                            "Required project material missing or current checker failed"
                            if project_item["level"] == "needs-repair"
                            else "Current reference stage check pending"
                        )
                    ),
                )
            )

    for plan in read("curriculum/project-plans.json"):
        add(
            "plans",
            plan["id"],
            plan["title"],
            f'project-plans/{plan["id"]}.html',
            plan["source"],
            "planned",
            plan["summary"],
            "Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale.",
            "Specification only; no runnable capstone",
            sourceHash=hashlib.sha256((ROOT / plan["source"]).read_bytes()).hexdigest(),
        )
    for assessment in read("curriculum/assessments.json"):
        add(
            "assessments",
            assessment["id"],
            assessment["title"],
            assessment["url"],
            assessment["source"],
            "available-draft",
            assessment["description"],
            "Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning.",
            "Written tasks, public scaffolds and rubric; no calibrated grading study",
        )
    phase_index = {x["id"]: x for x in items if x["kind"] == "phases"}
    for id in course["pathwayIds"]:
        route = read(f"learning-paths/{id}.json")
        in_progress = [
            p for p in route["phaseIds"] if phase_index[p]["level"] != "available-draft"
        ]
        add(
            "pathways",
            id,
            route["title"],
            f"learning-paths/{id}.html",
            f"learning-paths/{id}.json",
            "in-development" if in_progress else "available-draft",
            route["description"],
            "Pilot this complete route, verify its prerequisite effort and review the final learner evidence. "
            + (
                "Parts still in development: " + ", ".join(in_progress) + "."
                if in_progress
                else "Teaching review is still pending."
            ),
            "Linked practice and capstone references; route comprehension unvalidated",
        )
    for role in course["roles"]:
        add(
            "careers",
            role["id"],
            role["title"],
            f'career-guides/{role["id"]}.html',
            "curriculum/course.json",
            "available-draft",
            role["description"],
            "Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role.",
            "Internal capability map; no external role or hiring validation",
        )
    for track in read("curriculum/modality-tracks.json")["tracks"]:
        add(
            "tracks",
            track["id"],
            track["title"],
            f'tracks/{track["id"]}.html',
            "curriculum/modality-tracks.json",
            "in-development",
            track["available"],
            "Connect the runnable harness to licensed natural data, meaningful task benchmarks and a separately qualified deployment target.",
            "Bounded harness practice; full modality capstones planned",
        )
    guides = [
        (
            "project-workflow",
            "Tools, folders and experiment organization",
            "content/guides/project-workflow.md",
        ),
        ("tpu-gcp", "TPU and Google Cloud workflow", "content/guides/tpu-gcp.md"),
        (
            "tpu-performance",
            "TPU precision and profiling",
            "content/guides/tpu-performance.md",
        ),
    ]
    for id, title, source in guides:
        add(
            "guides",
            id,
            title,
            id + ".html",
            source,
            "available-draft",
            "A practical guide with a downloadable practice workspace.",
            "Pilot the setup and evidence handoff with a new learner; review platform-specific and hardware claims.",
            "Workspace checks available; novice review pending",
        )
    add(
        "book",
        "reader",
        "JAX Pathways book",
        "book.html",
        "src/pages/downloads/jax-foundations.astro",
        "in-development",
        "HTML and EPUB include all authored chapters, figures, practical appendices and solutions.",
        "Review math/code on native EPUB readers, print/PDF and assistive technology; contribute chapter-level teaching reviews.",
        "Automated export checks; native reader and teaching qualification incomplete",
    )
    for item in items:
        if item["kind"] == "plans":
            item["review"] = "Implementation and review pending"
        elif item["kind"] == "careers":
            item["review"] = "Practitioner review pending"
        elif item["kind"] == "assessments":
            item["review"] = "Rubric and learner pilot review pending"
    hardware = []
    for target in read("curriculum/hardware-validation.json")["qualifications"]:
        current = (
            target.get("status") == "current"
            and bool(target.get("sources"))
            and all(
                (ROOT / source).is_file()
                and hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == h
                for source, h in target["sources"].items()
            )
        )
        hardware.append(
            dict(
                id=target["id"],
                scope=target["scope"],
                evidence=target["evidence"],
                current=current,
            )
        )
    lesson_items = [x for x in items if x["kind"] == "lessons"]
    summary = dict(
        byKind=dict(Counter(x["kind"] for x in items)),
        phaseLevels=dict(Counter(x["level"] for x in items if x["kind"] == "phases")),
        itemLevels=dict(Counter(x["level"] for x in items)),
        tasks=len(tasks),
        lessonTasks=sum("lessonId" in t for t in tasks),
        projectStages=sum("projectId" in t for t in tasks),
        currentCpuLessons=sum(
            "reference checked" in x["validation"] for x in lesson_items
        ),
        currentProjectReferences=sum(
            x.get("metrics", {}).get("currentReference", False) for x in items
        ),
        expertReviewedLessons=sum(x["metrics"]["expertReviewed"] for x in lesson_items),
        novicePilotedLessons=sum(x["metrics"]["novicePiloted"] for x in lesson_items),
        choiceSpecificFeedbackLessons=sum(
            x["metrics"]["choiceSpecificFeedback"] for x in lesson_items
        ),
        boundedHardwareQualifications=sum(x["current"] for x in hardware),
    )
    report = dict(
        schemaVersion=1,
        auditedAt=review["auditedAt"],
        baseCommit=review["baseCommit"],
        method=review["method"],
        levels=review["levels"],
        findings=review["findings"],
        summary=summary,
        hardware=hardware,
        items=items,
        tasks=tasks,
    )
    return report


def render_report(report):
    summary = report["summary"]
    md = [
        "# Learning-material audit: 21 October 2026",
        "",
        report["method"],
        "",
        f'Base revision: `{report["baseCommit"]}`. This is an internal editorial audit, not independent teaching approval.',
        "",
        "## Current assessment",
        "",
        "The book is a substantial collection of bounded, runnable practice drafts. It is not yet a fully reviewed curriculum or a set of production-model releases. The foundational mechanisms and numerical contracts are stronger than the broader model-lifecycle, target deployment and career-readiness evidence.",
        "",
        f'{summary["currentCpuLessons"]} current CPU lesson references; {summary["currentProjectReferences"]} source-matching public project receipts; {summary["boundedHardwareQualifications"]} bounded physical target qualifications. {summary["expertReviewedLessons"]} independent expert reviews and {summary["novicePilotedLessons"]} novice pilots are recorded.',
        "",
        f'The inventory covers {sum(summary["byKind"].values())} material entries, {summary["lessonTasks"]} lesson exercises/checkpoints and {summary["projectStages"]} project stages. Section counts describe coverage, not quality scores. Duration estimates have not been calibrated with learners.',
        "",
        "## Learner labels",
        "",
    ]
    for level, meaning in report["levels"].items():
        md.append(f'- **{level.replace("-", " ").title()}**: {meaning}')
    md += [
        "",
        "CPU checks, hardware scope, human review and learner completion are separate. Nothing is promoted to reviewed merely because files or a passing reference exist.",
        "",
        "## Priority contribution work",
        "",
    ]
    for finding in report["findings"]:
        md += [
            f'### {finding["priority"]}: {finding["title"]}',
            "",
            f'Status: **{finding.get("status", "open")}**.',
            "",
            finding["action"],
            "",
            "Evidence: "
            + ", ".join(f"[{p}](../{p})" for p in finding["evidence"])
            + ".",
            "",
        ]
    headings = {
        "phases": "Every course phase",
        "projects": "Every runnable project",
        "plans": "Every planned capstone",
        "assessments": "Every synthesis assessment",
        "pathways": "Every learning path",
        "careers": "Every career guide",
        "tracks": "Every modality track",
        "guides": "Every practical guide",
        "book": "Book formats",
        "lessons": "Every lesson and its practice",
    }
    for kind, heading in headings.items():
        md += [
            f"## {heading}",
            "",
            "| Material | Maturity | Current evidence and useful scope | Next contribution |",
            "| --- | --- | --- | --- |",
        ]
        for item in (x for x in report["items"] if x["kind"] == kind):

            def cell(text):
                return str(text).replace("|", "\\|").replace("\n", " ")

            md.append(
                f'| [{cell(item["id"] + ": " + item["title"])}](https://www.tahabouhsine.com/jaxpathways/{item["url"]}) | {item["level"]} | {cell(item["validation"] + ". " + item["summary"])} | {cell(item["contribution"])} |'
            )
        md.append("")
    md += [
        "## Every exercise, checkpoint and project stage",
        "",
        "These are public practice tasks, not independently administered exams. A reference run does not establish learner mastery.",
        "",
        "| Task | Material | Status | Scope |",
        "| --- | --- | --- | --- |",
    ]
    for task in report["tasks"]:
        md.append(
            f'| `{task["id"]}` | {task["title"].replace("|", "/")} | {task["level"]} | {task["validation"]} |'
        )
    md += [
        "",
        "## Source-bound hardware evidence",
        "",
        "These records qualify their declared experiments only; they do not certify an entire lesson, generation, production model or career path.",
        "",
    ]
    for target in report["hardware"]:
        md.append(
            f'- [{target["id"]}](../{target["evidence"]}): {target["scope"]} Source hashes {"match" if target["current"] else "do not match"}.'
        )
    md += [
        "",
        "## Limits and next review",
        "",
        "All canonical sources, task prompts, stage contracts, companion existence and stored source hashes were inventoried. Per-phase scope judgments are recorded in curriculum/content-review.json. Technical close reading and visual inspection are selective; no new human study, paid accelerator run or full native-reader qualification was performed. The two corrected recovery captions were followed by fresh CPU script/notebook runs and regenerated figures. Other numerical checks come from source-matching receipts and the completed 0.8.2 deployment, rather than new executions during this audit. Wrong-choice feedback, realistic datasets, target-specific outcomes and human review need further work.",
        "",
        "Do not solve these gaps by inflating word counts, randomizing quiz choices, or labelling public fixture passes as professional readiness. Start with one contribution and retain inspectable evidence.",
        "",
    ]
    return "\n".join(md)


if __name__ == "__main__":
    report = build()
    (ROOT / "docs/learning-material-audit-2026-10-21.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (ROOT / "docs/learning-material-audit-2026-10-21.md").write_text(
        render_report(report)
    )
    display = {key: value for key, value in report.items() if key != "tasks"}
    destination = ROOT / "public/api/v1/content-status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(display, indent=2, ensure_ascii=False) + "\n")
    # Browser views need labels and evidence, not the entire editorial audit or task prompts.
    labels = [
        {key: item[key] for key in ("kind", "id", "level", "validation", "review")}
        for item in report["items"]
    ]
    (destination.parent / "material-labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))
