"""Independent aggregation/binding audit; no candidate scoring imports."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import re
import string
from tokenizers import Tokenizer

OUT = Path(__file__).resolve().parent
DATA = Path('/private/tmp/jaxpathways-c02-datafreeze-luna-20261021')
RUN = Path('/private/tmp/jaxpathways-c02-pilot-luna-approved-mechanics-20261021')
EVAL = Path('/private/tmp/jaxpathways-c02-evaluation-20261021-final2')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = json.loads((EVAL / 'metrics.json').read_text())
gold = {r['question_id']: r for r in map(json.loads, (DATA / 'qa-final.jsonl').read_text().splitlines())}
rows = list(map(json.loads, (EVAL / 'qa-generations.jsonl').read_text().splitlines()))
tokenizer = Tokenizer.from_file(str(DATA / 'tokenizer.json'))

def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def score(prediction, answers):
    p = normalize(prediction)
    exact, f1 = [], []
    for answer in answers:
        g = normalize(answer)
        exact.append(float(p == g))
        overlap = sum((Counter(p.split()) & Counter(g.split())).values())
        f1.append(0.0 if overlap == 0 else 2 * overlap / (len(p.split()) + len(g.split())))
    return max(exact), max(f1)

assert score('', ['the']) == (1, 0)
assert score('cat cat', ['cat dog']) == (0, .5)
assert score('A cat!', ['the cat']) == (1, 1)
assert report['recipe_sha256'] == sha(DATA / 'frozen_recipe.json')
assert report['tokenizer_sha256'] == sha(DATA / 'tokenizer.json')
assert report['evaluator_source_sha256'] == sha(OUT / 'source/evaluate_pilot.py')
assert report['scorer']['reference_bytes_sha256'] == sha(OUT / 'source/squad_scoring.py')
assert len(rows) == 128 * 6 and len(gold) == 128
summary = {}
for branch, claimed in report['branches'].items():
    subset = [r for r in rows if r['branch'] == branch]
    assert len(subset) == 128
    assert {r['question_id'] for r in subset} == set(gold)
    exact, f1 = [], []
    for r in subset:
        g = gold[r['question_id']]
        expected_prompt = f"Question: {g['question']}\nContext: {g['context']}\nAnswer:"
        assert r['prompt'] == expected_prompt
        assert r['human_answers'] == g['answers']
        assert r['prompt_token_ids'] == [1, *tokenizer.encode(expected_prompt, add_special_tokens=False).ids]
        assert len(r['prompt_token_ids']) + 32 <= 512
        generated = r['generated_token_ids']
        assert 0 < len(generated) <= 32 and all(type(i) is int and 0 <= i < 512 for i in generated)
        assert 2 not in generated[:-1]
        assert tokenizer.decode(generated, skip_special_tokens=True) == r['prediction']
        assert r['finish_reason'] == ('eos' if generated[-1] == 2 else 'max_new_tokens')
        em, tf1 = score(r['prediction'], g['answers'])
        assert math.isclose(em, r['exact_match'], abs_tol=1e-15)
        assert math.isclose(tf1, r['token_f1'], abs_tol=1e-15)
        exact.append(em); f1.append(tf1)
    identity = json.loads((RUN / branch / 'identity.json').read_text())
    assert claimed['checkpoint_sha256'] == sha(RUN / branch / 'final-state.npz')
    assert claimed['checkpoint_identity_sha256'] == sha(RUN / branch / 'identity.json')
    assert claimed['candidate_source_sha256'] == identity['candidate_source_sha256']
    actual = {'questions': len(subset), 'exact_match': sum(exact)/128, 'token_f1': sum(f1)/128}
    qa = claimed['custom_squad_qa']
    assert actual['questions'] == qa['questions']
    assert math.isclose(actual['exact_match'], qa['exact_match'], abs_tol=1e-15)
    assert math.isclose(actual['token_f1'], qa['token_f1'], abs_tol=1e-15)
    per_question = {r['question_id']: r for r in qa['per_question']}
    assert len(per_question) == 128
    for r in subset:
        assert per_question[r['question_id']]['exact_match'] == r['exact_match']
        assert per_question[r['question_id']]['token_f1'] == r['token_f1']
    actual['passes_predeclared_qa_floors'] = actual['exact_match'] >= .05 and actual['token_f1'] >= .1
    summary[branch] = actual
result = {
    'status': 'pass',
    'scope': 'Independent QA score aggregation and raw record/prompt/tokenizer/answers/checkpoint binding for 768 outputs. Does not independently replay generation or reconstruct NLL logits.',
    'input_sha256': {name: sha(EVAL/name) for name in ('metrics.json','qa-generations.jsonl')},
    'checker_sha256': sha(Path(__file__)),
    'branches': summary,
}
(OUT/'qa-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
