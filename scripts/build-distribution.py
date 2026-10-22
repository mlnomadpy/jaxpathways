"""Build offline reading and portable learning bundles from canonical content."""
import hashlib
import html
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
from epub_identity import FINGERPRINT_PROPERTY, modified_at, primary_identifier, rendered_fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/downloads'
OUT.mkdir(exist_ok=True)
REPOSITORY_ONLY_TPU_FILES = {
    "resources/tpu-gcp/multihost_target.py",
    "resources/tpu-gcp/check_multihost_target.py",
}
PROJECT_EXTRA_FILES = {
    # kernel-audit's README documents this optional, separately pinned TPU environment.
    "kernel-audit": ("resources/tpu-gcp/requirements-qualification.txt",),
}
C01_SOURCE_FILES = {
    "README.md",
    "EVALUATION.md",
    "MODALITIES.md",
    "MONITORING.md",
    "RELEASE.md",
    "TRAINING_RECIPE.md",
    "baseline.py",
    "config.py",
    "evaluation.py",
    "model.py",
    "objectives.py",
    "package_runtime.py",
    "plots.py",
    "run.py",
    "trainer.py",
    "requirements-embedding-lock.txt",
    "requirements-embedding.txt",
    "data/LICENSES.md",
    "data/RECIPE_AMENDMENTS.md",
    "data/dataset.py",
    "data/frozen_recipe.json",
    "data/prepare.py",
    "data/recipe-history/c01-original-pretraining-recipe.json",
    "tools/build_evidence_index.py",
    "tools/pack_reference_evidence.py",
    "tools/render_teaching_objectives.py",
    "starter/README.md",
    "starter/model.py",
    "tests/check.py",
    "tests/resume_worker.py",
    "tests/test_plots.py",
}


C02_SOURCE_FILES = {
    'README.md',
    'config.py',
    'consumer.py',
    'data/RIGHTS.md',
    'data/foundation-reuse.json',
    'data/frozen_recipe.json',
    'data/lm_eval_tasks/LICENSE.md',
    'data/lm_eval_tasks/NOTICE.md',
    'data/lm_eval_tasks/arc_easy/arc_easy.yaml',
    'data/lm_eval_tasks/hellaswag/hellaswag.yaml',
    'data/lm_eval_tasks/hellaswag/utils.py',
    'data/prepare.py',
    'evaluate_pilot.py',
    'evaluate_pythia_baseline.py',
    'evaluate_pythia_qa.py',
    'evaluate_qa_diagnostics.py',
    'export_consumer.py',
    'fetch_pythia_baseline.py',
    'lm_eval_adapter.py',
    'model.py',
    'pythia_pin.py',
    'render_c02_figures.py',
    'requirements-c02-baseline.txt',
    'requirements-c02-figures.txt',
    'requirements-c02-lm-eval.txt',
    'requirements-c02.txt',
    'run_lm_eval.py',
    'run_pilot.py',
    'serve_local.py',
    'squad_scoring.py',
    'starter/README.md',
    'starter/model.py',
    'tests/check.py',
    'tests/replay_branch.py',
    'tests/resume_worker.py',
    'tests/test_consumer.py',
    'tests/test_data_freeze.py',
    'tests/test_evaluation.py',
    'tests/test_evidence_reconstruction.py',
    'tests/test_export_consumer.py',
    'tests/test_lm_eval_adapter.py',
    'tests/test_model_math.py',
    'tests/test_pythia_baseline.py',
    'tests/test_qa_diagnostics.py',
    'tests/test_runner_integrity.py',
    'tests/test_service_live.py',
    'tests/test_starter_contract.py',
    'tests/test_training_recovery.py',
    'tools/build_review_packet.py',
    'training.py',
}


def c01_pilot_packet(root):
    """Return the explicitly allowlisted C01 source and public evidence files."""
    project = root / "projects" / "embedding-model-300m"
    index_path = project / "reference" / "evidence-index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Required C01 public evidence index is missing: {index_path}")
    index = json.loads(index_path.read_text())
    public_files = index.get("public_files")
    source_files = index.get("source_files_sha256")
    if not isinstance(public_files, dict) or not public_files:
        raise ValueError("C01 evidence index must contain a nonempty public_files map")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("C01 evidence index must contain a nonempty source_files_sha256 map")

    def verified(relative, expected):
        candidate = (project / relative).resolve()
        if project.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError(f"C01 evidence path is missing or unsafe: {relative}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"C01 evidence hash mismatch: {relative}")
        return candidate

    files = {index_path}
    for relative, expected in source_files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("C01 source hash entries must map paths to SHA-256 strings")
        project_relative = relative.removeprefix("projects/embedding-model-300m/")
        files.add(verified(project_relative, expected))
    for relative, expected in public_files.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith("reference/")
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError(f"Unsafe C01 public evidence entry: {relative}")
        lowered = relative.lower()
        payload_names = {
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        }
        payload_suffixes = {".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".gguf", ".msgpack", ".bin"}
        if (
            Path(lowered).name in payload_names
            or Path(lowered).suffix in payload_suffixes
            or "baseline_snapshot" in Path(lowered).parts
            or ("checkpoints" in Path(lowered).parts and Path(lowered).suffix != ".json")
        ):
            raise ValueError(f"C01 source package cannot include model payload: {relative}")
        files.add(verified(relative, expected))
    for key in ("historical_receipt", "checker_amendment"):
        reference = index.get(key)
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise ValueError(f"C01 evidence index is missing {key}")
        if reference["path"] not in public_files:
            raise ValueError(f"C01 {key} must be included in public_files")
        if public_files[reference["path"]] != reference.get("sha256"):
            raise ValueError(f"C01 {key} digest does not match public_files")
    source_files_to_bundle = {project / relative for relative in C01_SOURCE_FILES}
    for path in source_files_to_bundle:
        if not path.is_file():
            raise FileNotFoundError(f"Required C01 source file is missing: {path.relative_to(project)}")
    for relative in source_files:
        project_relative = relative.removeprefix("projects/embedding-model-300m/")
        if Path(project_relative).as_posix() not in C01_SOURCE_FILES:
            raise ValueError(f"C01 source receipt names a file outside the package allowlist: {relative}")
    files.update(source_files_to_bundle)
    return project, index_path, index, files


def c02_pilot_packet(root):
    """Verify the complete C02 source set and its weights-free review evidence."""
    project = root / "projects/causal-model-300m"
    index_path = project / "reference/evidence-index.json"
    index = json.loads(index_path.read_text())
    expected_sources = {"projects/causal-model-300m/" + path for path in C02_SOURCE_FILES}
    if set(index["source_files_sha256"]) != expected_sources:
        raise ValueError("C02 source index differs from its complete package allowlist")
    if not index.get("public_files"):
        raise ValueError("C02 public evidence inventory is empty")
    files = {index_path}
    for entries, is_source in ((index["source_files_sha256"], True), (index["public_files"], False)):
        for relative, digest in entries.items():
            relative = relative.removeprefix("projects/causal-model-300m/") if is_source else relative
            path = (project / relative).resolve()
            if not path.is_relative_to(project.resolve()) or not path.is_file():
                raise ValueError(f"C02 evidence is missing or unsafe: {relative}")
            if not is_source:
                if not relative.startswith("reference/") or path.suffix not in {".md", ".json", ".jsonl", ".svg", ".png", ".log"}:
                    raise ValueError(f"C02 public evidence is outside the weights-free allowlist: {relative}")
                if path.name in {"tokenizer.json", "tokenizer_config.json", "config.json", "weights.json"}:
                    raise ValueError(f"C02 model/tokenizer payload is not public evidence: {relative}")
            if not isinstance(digest, str) or len(digest) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"C02 evidence hash mismatch: {relative}")
            files.add(path)
    return project, index_path, index, files


def c03_pilot_packet(root):
    """Verify the accepted C03 source freeze and numeric-only public packet."""
    project = root / "projects/image-model-300m"
    packet = project / "reference/evidence/v14"
    manifest_path = packet / "packet-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    recipe_path = packet / "approved-v14-recipe.json"
    approval_path = packet / "approved-v14-approval.json"
    recipe = json.loads(recipe_path.read_text())
    approval = json.loads(approval_path.read_text())
    identity = json.loads((packet / "source-and-run-identities.json").read_text())
    expected_recipe = "df0a1a138f0635a39fcaa1b03f154403ed272560df9181fb5ad37722614d3c4c"
    expected_approval = "f4fb56a70ec453ffd995c36c56c9b80cbd7cd9f739deb59f4db1fdd95c91f82b"
    expected_manifest = "3b1dcfe589fffc25a43da336beb07d82334b6f8971a73084e2b989b6e8f2a794"

    def verified(relative, expected):
        candidate = (project / relative).resolve()
        if not candidate.is_relative_to(project.resolve()) or not candidate.is_file():
            raise ValueError(f"C03 evidence path is missing or unsafe: {relative}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"C03 evidence hash mismatch: {relative}")
        return candidate

    if hashlib.sha256(recipe_path.read_bytes()).hexdigest() != expected_recipe:
        raise ValueError("C03 recipe differs from the accepted v14 bytes")
    if hashlib.sha256(approval_path.read_bytes()).hexdigest() != expected_approval:
        raise ValueError("C03 approval differs from the accepted v14 bytes")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest:
        raise ValueError("C03 packet manifest differs from the accepted v14 bytes")
    if approval.get("recipe_sha256") != expected_recipe or approval.get("source_files_sha256") != recipe.get("source_sha256"):
        raise ValueError("C03 approval does not bind the accepted recipe and source inventory")
    source_sha256 = recipe.get("source_sha256")
    if not isinstance(source_sha256, dict) or len(source_sha256) != 28:
        raise ValueError("C03 accepted recipe must enumerate exactly 28 source files")
    if set(identity.get("as_run_source_inventory", {})) != set(source_sha256):
        raise ValueError("C03 run identity source inventory differs from the accepted recipe")
    if len(identity.get("as_run_candidate_python_inventory", {})) != 11:
        raise ValueError("C03 candidate source inventory must contain exactly 11 Python files")
    if approval.get("candidate_api_source_sha256") != {
        path.split("/", 1)[1]: row["sha256"]
        for path, row in identity["as_run_candidate_python_inventory"].items()
    }:
        raise ValueError("C03 candidate source inventory differs from accepted approval")
    if not isinstance(manifest.get("files"), dict) or len(manifest["files"]) != 29:
        raise ValueError("C03 public packet manifest must enumerate exactly 29 files")

    expected_files = {manifest_path}
    for relative, digest in source_sha256.items():
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe C03 source entry: {relative}")
        expected_files.add(verified(relative, digest))
    for relative, digest in manifest["files"].items():
        if (
            not isinstance(relative, str)
            or not (relative == "TEACHING.md" or relative.startswith("reference/") or relative.startswith("tools/"))
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"Unsafe C03 public evidence entry: {relative}")
        lowered = relative.lower()
        if Path(lowered).suffix not in {".md", ".json", ".svg", ".png", ".py", ".txt"}:
            raise ValueError(f"C03 public evidence is outside the numeric packet allowlist: {relative}")
        if lowered.startswith("tools/") and Path(lowered).suffix not in {".py", ".txt"}:
            raise ValueError(f"C03 tool file is outside the checker allowlist: {relative}")
        if Path(lowered).suffix == ".png" and "figures" not in Path(lowered).parts:
            raise ValueError(f"C03 PNG is outside the checked figure directory: {relative}")
        if any(token in Path(lowered).name for token in ("weight", "checkpoint", "feature-vector", "embedding")):
            raise ValueError(f"C03 model payload is not public evidence: {relative}")
        expected_files.add(verified(relative, digest))
    actual_project_files = {
        path.resolve()
        for path in project.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual_project_files != expected_files:
        raise ValueError(
            "C03 source directory differs from the exact accepted package set: "
            f"extra={sorted(str(p.relative_to(root)) for p in actual_project_files - expected_files)}, "
            f"missing={sorted(str(p.relative_to(root)) for p in expected_files - actual_project_files)}"
        )
    return project, manifest_path, manifest, expected_files


course = json.loads((ROOT / 'public/curriculum.json').read_text())
release = json.loads((ROOT / 'curriculum/releases.json').read_text())[0]
lessons = [lesson for phase in course['phases'] for lesson in phase['lessons'] if lesson['status'] == 'authored']
phases = [phase for phase in course['phases'] if any(lesson['status'] == 'authored' for lesson in phase['lessons'])]
esc = html.escape
material_status = {(item['kind'], item['id']): item for item in json.loads((ROOT / 'public/api/v1/content-status.json').read_text())['items']}
def maturity(kind, id):
    item = material_status[(kind, id)]
    label = item['level'].replace('-', ' ').capitalize()
    return f'<p><a href="https://www.tahabouhsine.com/jaxpathways/content-status.html#{esc(kind)}-{esc(id)}">{esc(label)}</a> · {esc(item["review"])}. Applies to this material and its practice tasks.</p>'
math_markup = json.loads(subprocess.check_output(["node", "scripts/render-book-math.mjs"], cwd=ROOT, text=True))
def rich(text):
    return math_markup["prose"].get(text, esc(text))
style = 'figure{margin:1.5rem 0}figure img{width:100%;height:auto}figcaption{font-size:.85em}body{max-width:760px;margin:3rem auto;padding:0 1.5rem;font:18px/1.7 Georgia,serif;color:#172334}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.3}pre{padding:1rem;background:#eef1f5;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}article{page-break-before:always;margin-top:4rem}a{color:#2456a6}.solution{page-break-before:always}table{border-collapse:collapse}td,th{border:1px solid #bbc4d0;padding:.35rem}@media print{nav{display:none}body{margin:0;max-width:none}}'
intro = maturity('book', 'reader') + f'<h1>JAX Pathways: course reader</h1><p>Edition {esc(release["version"])} · {esc(release["date"])} · <a href="https://www.tahabouhsine.com/jaxpathways/updates.html#v{esc(release["version"])}">Edition changes</a></p><p>{len(lessons)} authored lessons from the shared course. CPU reference execution status is recorded separately in validation.json. Bounded physical TPU target receipts are described in the <a href="https://www.tahabouhsine.com/jaxpathways/tpu-performance.html">hardware evidence guide</a>; they do not qualify every lesson. Independent expert review remains outstanding.</p><p>Follow phases in order when new to JAX. Each phase guide names preparation, a useful artifact and evidence to keep. Exercises and checkpoints are practice; project completion and reviewed competency are separate. Solutions are collected in a separate appendix.</p><h2>Choose a route</h2><ul>'+''.join(f'<li><a href="https://www.tahabouhsine.com/jaxpathways/course.html?path={esc(r["id"])}">{esc(r["title"])}</a> · {esc(r["description"])}</li>' for r in course.get('pathways',[]))+'</ul><p>Career routes and milestones: <a href="https://www.tahabouhsine.com/jaxpathways/careers.html">online career guide</a>.</p>'

def prose(text):
    return ''.join(f'<p>{rich(p)}</p>' for p in text.split('\n\n'))

def code_block(text):
    return '<pre><code>'+esc(text)+'</code></pre>'

def learner_starter(contract, scope):
    """Render only an authored learner contract in the main EPUB chapter."""
    if not isinstance(contract, dict) or not all(
        isinstance(contract.get(key), list) and contract[key]
        and all(isinstance(item, str) and item.strip() for item in contract[key])
        for key in ('inputs', 'outputs', 'invariants', 'hints')
    ) or not isinstance(contract.get('code'), str) or not contract['code'].strip():
        return '<aside class="learner-starter-pending"><strong>Learner starter pending.</strong> This task has no complete, separately authored input/output contract, checks, hints and TODO scaffold yet.</aside>'
    def field(title, items):
        return '<h4>'+esc(title)+'</h4><ul>'+''.join('<li>'+rich(item)+'</li>' for item in items)+'</ul>'
    return ('<section class="learner-starter-contract"><p><strong>'+esc(scope)+' · learner-only starter</strong></p>'
            +field('Inputs', contract['inputs'])
            +field('Outputs', contract['outputs'])
            +field('Invariants to check', contract['invariants'])
            +field('Hints', contract['hints'])
            +'<h4>Your TODO scaffold</h4>'+code_block(contract['code'])+'</section>')

book_guides = json.loads(subprocess.check_output(["node", "scripts/render-book-guides.mjs"], cwd=ROOT, text=True))

solutions=[]
def chapter(lesson):
    body = f'<h1>{esc(lesson["title"])}</h1><p>{esc(lesson["objective"])}</p>'
    body += maturity('lessons', lesson['id'])
    c = lesson['content']
    if c.get('setupBundle'):
        body += '<p><a href="https://www.tahabouhsine.com/jaxpathways/'+esc(c['setupBundle']['url'])+'">'+esc(c['setupBundle']['title'])+'</a>. Extract the archive and work inside its <code>jax-start-here</code> folder. The setup instructions refer to <code>first_experiment.py</code> in that folder.</p>'
    if c.get('objectives'):
        body += '<h2>What you will be able to do</h2><ul>'+''.join('<li>'+rich(o)+'</li>' for o in c['objectives'])+'</ul>'
    if c.get('problem'): body += '<h2>The problem</h2>'+prose(c['problem'])
    body += '<h2>The idea</h2>'+prose(c['idea'])
    for section in c.get('sections',[]):
        body += '<h2>'+esc(section['title'])+'</h2>'+prose(section['body'])
        if section.get('id')=='guided-reasoning' and c.get('diagram'):
            d=c['diagram']; artifact=lesson.get('visualArtifact',{})
            body += '<h3>'+esc(d['title'])+'</h3><p><strong>Predict:</strong> '+rich(d['prediction'])+'</p>'
            if artifact.get('mechanismPng'):
                body += '<figure><img src="figures/'+esc(lesson['id'])+'-mechanism.png" alt="'+esc(d.get('alt',d['title']))+'"/><figcaption>'+esc(d['scope'])+'</figcaption></figure>'
            body += prose(d['reading'])
        if section.get('check'):
            check=section['check']
            body += '<h3>Pause and reason</h3>'+prose(check['prompt'])+'<details><summary>Compare your reasoning</summary>'+prose(check['answer'])+'</details>'
        for command in section.get('commands',[]):
            body += '<h3>'+esc(command['label'])+'</h3>'+code_block(command['code'])+'<p><strong>Expected:</strong></p>'+prose(command['expected'])
        if section.get('math'): body += math_markup['equations'][section['math']]
        if section.get('formula'): body += code_block(section['formula'])
    if c.get('visual') and lesson.get('visualArtifact'):
        visual=c['visual']; artifact=lesson['visualArtifact']
        if visual['kind']=='executed':
            scope='Plot from a recorded CPU computation · JAX '+esc(artifact['environment']['jax'])+' · '+str(artifact['environment']['deviceCount'])+' CPU device'+('s' if artifact['environment']['deviceCount'] != 1 else '')
        elif visual['kind']=='analytic': scope='Calculated scenario; no device execution was performed'
        else: scope='Conceptual workflow diagram'
        body += '<h2>'+esc(visual['title'])+'</h2><p><strong>Predict:</strong> '+rich(visual['prediction'])+'</p>'
        body += '<figure><img src="figures/'+esc(lesson['id'])+'.png" alt="'+esc(visual.get('alt',visual['title']))+'"/><figcaption>'+scope+'</figcaption></figure>'
        body += '<h3>Read the figure</h3>'+prose(visual['reading'])+'<h3>Connect it to the computation</h3>'+prose(visual['connection'])
    for step in c.get('buildSteps',[]):
        body += '<h2>'+esc(step['title'])+'</h2>'+prose(step['instruction'])+code_block(step['code'])+prose(step['explanation'])
    if c.get('buildCode'): body += '<h2>Build a numerical estimate</h2>'+code_block(c['buildCode'])+prose(c['buildOutput'])
    body += '<h2>Run the example</h2><p>The assembled script is in the separate solutions and code appendix.</p>'+prose(c['output'])
    if lesson.get('execution'):
        run=lesson['execution']
        body += '<h2>Recorded reference execution</h2><p>The saved instructor run is in the separate solutions and execution appendix.</p>'
    for experiment in c.get('experiments',[]):
        body += '<h2>'+esc(experiment['title'])+'</h2><p><strong>Predict before running:</strong> '+rich(experiment['prediction'])+'</p>'+code_block(experiment['code'])+prose(experiment['output'])+prose(experiment['explanation'])
    body += '<h2>Your exercise</h2>'+prose(c['exercise'])+learner_starter(c.get('learnerStarter'), 'Exercise')+'<p>Attempt this before consulting the separate solutions appendix.</p>'
    solutions.append(f'<section><h2>{esc(lesson["title"])}</h2><h3>Assembled example code</h3>{code_block(c["code"])}<h3>Exercise solution</h3>{code_block(c["solution"])}')
    if lesson.get('execution'):
        run=lesson['execution']
        solutions.append('<h3>Recorded reference execution</h3><p>Saved instructor CPU run · '+esc(run['executedAt'])+'</p>'+code_block(run['stdout']))
    for practice in c.get('practice',[]):
        body += '<h2>'+esc(practice['title'])+'</h2><p>'+esc(practice['difficulty'])+'</p>'+prose(practice['prompt'])+'<h3>Hint</h3>'+prose(practice['hint'])+learner_starter(practice.get('learnerStarter'), 'Practice: '+practice['title'])
        solutions.append('<h3>'+esc(practice['title'])+'</h3>'+code_block(practice['solution'])+prose(practice['explanation']))
    body += f'<h2>Checkpoint</h2><p>{rich(c["question"])}</p><ol>' + ''.join(f'<li>{rich(o)}</li>' for o in c['options']) + '</ol>'
    body += '<p>Answer and explanation are in the separate solutions appendix.</p>'
    solutions.append(f'<h3>Checkpoint</h3><p>Answer: {rich(c["options"][c["answer"]])}. {rich(c["explanation"])}</p><h4>Reason through the other choices</h4><ul>' + "".join(f'<li><strong>{rich(option)}</strong>{prose(c["optionFeedback"][index])}</li>' for index, option in enumerate(c["options"]) if index != c["answer"]) + "</ul></section>")
    if c.get('diagnosis'): body += '<h2>Diagnose the result</h2>'+prose(c['diagnosis'])
    if c.get('takeaways'): body += '<h2>Carry forward</h2><ul>'+''.join('<li>'+rich(t)+'</li>' for t in c['takeaways'])+'</ul>'
    body += '<h2>Your evidence</h2>'+prose(lesson['evidence'])
    if c.get('references'): body += '<h2>Primary references</h2><ul>'+''.join('<li><a href="'+esc(r['url'])+'">'+esc(r['title'])+'</a></li>' for r in c['references'])+'</ul>'
    return body

chapters = [chapter(l) for l in lessons]
guide_bodies=[]
for phase in phases:
    guide=phase.get('studyGuide',{})
    milestones=''.join('<li><h3>'+rich(item['title'])+'</h3><p>'+rich(item['task'])+'</p></li>' for item in guide.get('milestones',[]))
    projects=[phase.get('projectId'),*(phase.get('additionalProjectIds') or [])]
    project_links=''.join(f'<li>{esc(project)} · <a href="https://www.tahabouhsine.com/jaxpathways/project.html?id={esc(project)}">online workspace</a> · <a href="https://www.tahabouhsine.com/jaxpathways/project-guides/{esc(project)}.html">project guide</a></li>' for project in projects if project)
    stage_ids=', '.join(str(value) for value in (phase.get('projectStageIds') or []))
    stage_note=f' Start with project stage{("s" if len(phase.get("projectStageIds") or []) != 1 else "")} {esc(stage_ids)}.' if stage_ids else ''
    guide_bodies.append(maturity('phases', phase['id']) + f'<h1>Phase {esc(phase["number"])}: {esc(phase["title"])}</h1><h2>{rich(guide.get("question","Phase guide"))}</h2><p>{rich(guide.get("why",""))}</p><h2>Prepare and work through the phase</h2><ol>{milestones}</ol><h2>Readiness check</h2><p>{rich(guide.get("readiness",{}).get("prompt",""))}</p><p>Answer: {rich(guide.get("readiness",{}).get("answer",""))}</p><h2>Transfer question</h2><p>{rich(guide.get("transfer",{}).get("prompt",""))}</p><p>Approach: {rich(guide.get("transfer",{}).get("approach",""))}</p><h2>Common pitfall</h2><p><strong>{rich(guide.get("pitfall",{}).get("symptom",""))}</strong> {rich(guide.get("pitfall",{}).get("check",""))}</p><h2>Phase artifact</h2><p>{rich(guide.get("completion",""))}</p><p>{rich(guide.get("furtherWork",""))}</p><p><strong>Phase project:</strong> {esc(phase.get("project",""))}. {esc(phase.get("checkpoint",""))}{stage_note}</p><ul>{project_links}</ul>')
nav = '<nav><h2>Contents</h2><ol><li><a href="intro.xhtml">About this edition and routes</a></li>' + ''.join(f'<li><a href="phase-{i}.xhtml">Phase {esc(p["number"])}: {esc(p["title"])}</a><ol>'+''.join(f'<li><a href="chapter-{lessons.index(l)}.xhtml">{esc(l["title"])}</a></li>' for l in p['lessons'] if l['status']=='authored')+'</ol></li>' for i,p in enumerate(phases)) + '<li><a href="appendix-workflow.xhtml">Appendix: project structure and evidence</a></li><li><a href="appendix-experiments.xhtml">Appendix: reproducible experiments workshop</a></li><li><a href="solutions.xhtml">Solutions</a></li></ol></nav>'
xhtml = lambda body, title='JAX Pathways: course reader': '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" lang="en"><head><title>'+esc(title)+'</title><link rel="stylesheet" href="style.css"/></head><body>'+body+'</body></html>'
identity = primary_identifier(release, lessons)
payload = {
    'mimetype': b'application/epub+zip',
    'META-INF/container.xml': b'<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
    'OEBPS/style.css': style.encode(),
    'OEBPS/intro.xhtml': xhtml(intro).encode(),
    'OEBPS/nav.xhtml': xhtml(nav.replace('<nav>','<nav xmlns:epub="http://www.idpf.org/2007/ops" epub:type="toc">',1)).encode(),
}
for i,body in enumerate(guide_bodies): payload[f'OEBPS/phase-{i}.xhtml'] = xhtml(body, f'Phase {phases[i]["number"]}: {phases[i]["title"]}').encode()
for i,body in enumerate(chapters): payload[f'OEBPS/chapter-{i}.xhtml'] = xhtml(body, lessons[i]['title']).encode()
workflow=book_guides['workflow']+'<p><a href="https://www.tahabouhsine.com/jaxpathways/project-workflow.html">Online guide and linked resources</a></p>'
workshop=book_guides['experiments']+'<p><a href="https://www.tahabouhsine.com/jaxpathways/project-guides/reproducible-experiments.html">Online workshop and workspace</a></p>'
payload['OEBPS/appendix-workflow.xhtml'] = xhtml('<h1>Appendix: project structure and evidence</h1>'+workflow, 'Appendix: project structure and evidence').encode()
payload['OEBPS/appendix-experiments.xhtml'] = xhtml('<h1>Appendix: reproducible experiments workshop</h1>'+workshop, 'Appendix: reproducible experiments workshop').encode()
payload['OEBPS/solutions.xhtml'] = xhtml('<h1>Solutions</h1><p>Attempt each exercise and checkpoint before reading this appendix.</p>'+''.join(solutions), 'Solutions').encode()
items=''.join(f'<item id="p{i}" href="phase-{i}.xhtml" media-type="application/xhtml+xml"'+(' properties="mathml"' if '<math ' in body else '')+'/>' for i,body in enumerate(guide_bodies))
items += ''.join(f'<item id="c{i}" href="chapter-{i}.xhtml" media-type="application/xhtml+xml"'+(' properties="mathml"' if '<math ' in body else '')+'/>' for i,body in enumerate(chapters))
for name, body in [('appendix-workflow', workflow), ('appendix-experiments', workshop), ('solutions', ''.join(solutions))]:
    properties = ' properties="mathml"' if '<math ' in body else ''
    items += f'<item id="{name}" href="{name}.xhtml" media-type="application/xhtml+xml"{properties}/>'
for lesson in lessons:
    if lesson.get('visualArtifact'):
        payload['OEBPS/figures/'+lesson['id']+'.png'] = (ROOT/'public'/lesson['visualArtifact']['png']).read_bytes()
        items += '<item id="figure-'+lesson['id']+'" href="figures/'+lesson['id']+'.png" media-type="image/png"/>'
    if lesson.get('visualArtifact',{}).get('mechanismPng'):
        payload['OEBPS/figures/'+lesson['id']+'-mechanism.png'] = (ROOT/'public'/lesson['visualArtifact']['mechanismPng']).read_bytes()
        items += '<item id="mechanism-'+lesson['id']+'" href="figures/'+lesson['id']+'-mechanism.png" media-type="image/png"/>'
spine='<itemref idref="intro"/>' + ''.join(f'<itemref idref="p{i}"/>'+''.join(f'<itemref idref="c{lessons.index(l)}"/>' for l in p['lessons'] if l['status']=='authored') for i,p in enumerate(phases)) + '<itemref idref="appendix-workflow"/><itemref idref="appendix-experiments"/><itemref idref="solutions"/>'
modified = modified_at(release)
package_template = f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" unique-identifier="book-id" prefix="jp: https://www.tahabouhsine.com/jaxpathways/epub-meta#"><metadata><dc:identifier id="book-id">{esc(identity)}</dc:identifier><dc:title>JAX Pathways: course reader {esc(release["version"])} · {esc(release.get("title", ""))}</dc:title><dc:language>en</dc:language><dc:creator>Taha Bouhsine</dc:creator><dc:publisher>JAX Pathways</dc:publisher><dc:description>A free JAX and Flax book with worked math, explained figures, executable Python examples and staged projects.</dc:description><dc:source>https://www.tahabouhsine.com/jaxpathways/book.html</dc:source><dc:date>{esc(release["date"])}</dc:date><meta property="dcterms:modified">{modified}</meta><meta property="belongs-to-collection" id="collection">JAX Pathways</meta><meta refines="#collection" property="group-position">{esc(release["version"])}</meta><meta property="{FINGERPRINT_PROPERTY}">sha256:pending</meta></metadata><manifest><item id="intro" href="intro.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="style" href="style.css" media-type="text/css"/>{items}</manifest><spine>{spine}</spine></package>'
fingerprint = rendered_fingerprint(payload, package_template.encode())
package = package_template.replace('sha256:pending', 'sha256:' + fingerprint)
with ZipFile(OUT / 'jax-foundations.epub','w',compression=ZIP_DEFLATED) as book:
    book.writestr('mimetype', payload.pop('mimetype'), compress_type=ZIP_STORED)
    for name in sorted(payload):
        book.writestr(name, payload[name])
    book.writestr('OEBPS/package.opf', package)
(OUT / 'jax-foundations.build.json').write_text(json.dumps({
    'schema': 'jaxpathways-epub-build-v1',
    'releaseVersion': release['version'],
    'primaryPublicationId': identity,
    'renderedEditionFingerprint': 'sha256:' + fingerprint,
    'modifiedAt': modified,
}, indent=2, sort_keys=True) + '\n')
for manifest in json.loads((ROOT/'curriculum/projects.json').read_text()):
    project_path = ROOT/manifest
    project = json.loads(project_path.read_text())
    with ZipFile(OUT/(project['id']+'.zip'),'w',compression=ZIP_DEFLATED) as bundle:
        included=set()
        for path in sorted(project_path.parent.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive_name=path.relative_to(ROOT).as_posix()
                if archive_name not in included:
                    bundle.write(path,archive_name)
                    included.add(archive_name)
        requirements_path=Path(project['environment']['requirementsPath'])
        requirements_name=requirements_path.as_posix()
        if requirements_name not in included:
            bundle.write(ROOT/requirements_path,requirements_name)
            included.add(requirements_name)
        for relative in PROJECT_EXTRA_FILES.get(project['id'], ()):
            extra_path = ROOT / relative
            extra_name = Path(relative).as_posix()
            if extra_name not in included:
                bundle.write(extra_path, extra_name)
                included.add(extra_name)
with ZipFile(OUT/'jax-tutor-skills.zip','w',compression=ZIP_DEFLATED) as bundle:
    for path in sorted((ROOT/'skills').glob('*/SKILL.md')): bundle.write(path,path.relative_to(ROOT))
    for path in sorted((ROOT/'skills').glob('*/references/*.md')): bundle.write(path,path.relative_to(ROOT))
    bundle.write(ROOT/'skills/README.md','README.md')
print(f'Built offline HTML/EPUB ({len(lessons)} lessons), staged project, and tutor bundles.')

# C01 remains a planned full capstone. This separate source-only ZIP exposes its
# bounded CPU workflow and only the public, hash-verified reference packet.
c01_root, c01_index_path, c01_index, c01_files = c01_pilot_packet(ROOT)
c01_public_paths = {
    (c01_root / relative).resolve()
    for relative in c01_index["public_files"]
}
c01_public_paths.add(c01_index_path.resolve())
with ZipFile(OUT / "embedding-model-300m-pilot.zip", "w", compression=ZIP_DEFLATED) as bundle:
    for path in sorted(c01_files):
        bundle.write(path, path.relative_to(ROOT).as_posix())
c02_root, c02_index_path, c02_index, c02_files = c02_pilot_packet(ROOT)
with ZipFile(OUT / "causal-model-300m-pilot.zip", "w", compression=ZIP_DEFLATED) as bundle:
    for path in sorted(c02_files):
        bundle.write(path, path.relative_to(ROOT).as_posix())
c03_root, c03_manifest_path, c03_manifest, c03_files = c03_pilot_packet(ROOT)
with ZipFile(OUT / "image-model-300m-pilot.zip", "w", compression=ZIP_DEFLATED) as bundle:
    for path in sorted(c03_files):
        bundle.write(path, path.relative_to(ROOT).as_posix())

setup=next(l for l in lessons if l['id']=='welcome-01')
with ZipFile(OUT/'jax-start-here.zip','w',compression=ZIP_DEFLATED) as bundle:
    bundle.writestr('jax-start-here/first_experiment.py',setup['content']['code']+'\n')
    bundle.write(ROOT/'requirements-beginner.txt','jax-start-here/requirements-beginner.txt')
    setup_text=(ROOT/setup['artifacts']['source']).read_text().replace('../outputs/figure.svg','figure.svg').replace('../outputs/mechanism.svg','mechanism.svg')
    bundle.writestr('jax-start-here/START-HERE.md',setup_text)
    if setup.get('visualArtifact'):bundle.write(ROOT/setup['path']/'outputs/figure.svg','jax-start-here/figure.svg')
    if setup.get('visualArtifact',{}).get('mechanismImage'):bundle.write(ROOT/setup['path']/'outputs/mechanism.svg','jax-start-here/mechanism.svg')

# A portable course root for tutors: skills alone do not include their lesson sources.
workspace_files = {ROOT/'requirements-cpu.txt', ROOT/'requirements-beginner.txt',
                   ROOT/'requirements-training.txt', ROOT/'requirements-figures.txt',
                   ROOT/'requirements-vision.txt', ROOT/'requirements-conversion.txt',
                   ROOT/'requirements-research.txt', ROOT/'requirements-tracking.txt', ROOT/'scripts/course.py',
                   ROOT/'public/curriculum.json', ROOT/'public/validation.json'}
for directory in ('curriculum', 'learning-paths', 'phases', 'projects', 'skills', 'assessments', 'resources/tpu-gcp', 'content/guides'):
    workspace_files.update(path for path in (ROOT/directory).rglob('*')
                           if path.is_file() and path.suffix in ('.md', '.json', '.py', '.ipynb', '.txt', '.svg', '.png')
                           and not (directory == 'projects' and path.is_relative_to(c01_root / 'reference'))
                           and not (directory == 'projects' and path.is_relative_to(c02_root / 'reference'))
                           and not (directory == 'projects' and path.is_relative_to(c03_root))
                           and '__pycache__' not in path.parts)
# Keep the coordinated multi-host target repository-only. Its rendezvous and shared
# GCS checkpoint setup are not part of the portable workspace contract.
workspace_files = {
    path for path in workspace_files
    if path.relative_to(ROOT).as_posix() not in REPOSITORY_ONLY_TPU_FILES
}
# Registered project bundles include their executed binary artifacts and media.
for manifest in json.loads((ROOT/'curriculum/projects.json').read_text()):
    workspace_files.update(path for path in (ROOT/manifest).parent.rglob('*')
                           if path.is_file() and '__pycache__' not in path.parts
                           and not (Path(manifest).parts[1] == 'embedding-model-300m' and path.is_relative_to(c01_root / 'reference')))
workspace_files.update(c01_public_paths)
workspace_files.update(c02_files)
workspace_files.update(c03_files)
for directory in ('public/figures','public/executions','public/guides'):
    workspace_files.update(path for path in (ROOT/directory).glob('*') if path.is_file())
start_guide = '''# Start JAX Pathways with your agent

This folder contains the current course curriculum, lesson sources, projects and tutor skills.

1. Extract the archive. Open this jaxpathways folder in your coding agent.
2. Open a terminal in this folder. With Node.js/npm available, run:

```sh
npx skills add . --skill start-learning learn-jax jax-course-guide check-jax
```

Choose your agent when prompted. Installation is scoped to this project; it does not install Python.

3. In the agent chat say:

> Use start-learning to help me learn JAX from this workspace.

Codex also supports $start-learning; Claude Code supports /start-learning.
Keep this folder open so the tutor can read the course files. It starts with guided setup and
keeps your plan in LEARNING.md. Say “Use learn-jax to continue my plan” to resume.

Read skills/README.md for the full skill instructions. requirements-cpu.txt records the course's
tested Python package versions; the setup lesson explains installation. Browser progress and
the tutor's study plan do not synchronize automatically. Planned topics are not authored lessons.
This is a portable learning workspace, not a Git checkout or a website build environment.
'''
with ZipFile(OUT/'jax-course-workspace.zip','w',compression=ZIP_DEFLATED) as bundle:
    bundle.writestr('jaxpathways/START-HERE.md',start_guide)
    for path in sorted(workspace_files):
        bundle.write(path,'jaxpathways/'+str(path.relative_to(ROOT)))
print('Built course workspace with canonical lessons, projects and tutor skills.')
