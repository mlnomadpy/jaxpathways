"""Small CPU-only content counterexamples; no training, profiler or cloud calls.

Use the audited checkout/export, not an arbitrary dirty workspace:
    JAX_PLATFORMS=cpu python reproduce.py --source-root /path/to/audited/source
Requires NumPy and the course's reviewed JAX environment (recorded run: JAX 0.9.2).
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'verification')
    args = parser.parse_args()
    packet = Path(__file__).parent
    inventory = json.loads((packet/'audit.json').read_text())
    lessons = {l['id']: l for c in inventory['chapters'] for l in c['lessons']}
    namespaces = {}
    hashes = {}
    # Imports and function definitions only: never execute the main lesson body.
    for id in ['tpu-01', 'tpu-04', 'tpu-06']:
        path = args.source_root/lessons[id]['source']
        hashes[id] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[id] != lessons[id]['sha256']:
            raise RuntimeError(f'{id}: source does not match the audited content')
        code = json.loads(path.read_text())['content']['code']
        tree = ast.parse(code)
        module = ast.Module(body=[n for n in tree.body if isinstance(
            n, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        namespaces[id] = namespace

    first = namespaces['tpu-01']['verify_runtime_contract']()
    second = namespaces['tpu-04']['build_runtime_receipt']()
    policies = namespaces['tpu-06']['evaluate_precision_policies']()
    # Evaluate the exact step_fn expression, without invoking capture_warmed_trace.
    capture = ast.parse(json.loads((args.source_root/lessons['tpu-06']['source']).read_text())
                        ['content']['code'])
    capture = next(n for n in capture.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'capture_warmed_trace')
    assignment = next(n for n in capture.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'step_fn' for t in n.targets))
    step = eval(compile(ast.Expression(assignment.value), '<exact-step_fn>', 'eval'),
                namespaces['tpu-06'])
    x = jnp.ones((256,256), dtype=jnp.bfloat16)
    finite = []
    for _ in range(5):
        x = jax.block_until_ready(step(x))
        finite.append(bool(np.isfinite(np.asarray(x,dtype=np.float32)).all()))
    runtime = dict(source=inventory['sourceRevision'], sourceHashes=hashes,
                   method='Exact function definitions and step expression; no lesson main/training loop or profiler executed.',
                   jax=jax.__version__, observed_backend=jax.default_backend(),
                   tpu01_CPU_contract_ok=first['contract_ok'],
                   tpu04_CPU_verified=second['verified'],
                   precision_policy_rows=policies, finite_after_repeated_dot=finite)
    assert runtime['observed_backend'] == 'cpu'
    assert first['contract_ok'] and second['verified']
    assert all(r['weight_bytes'] == 16384 and r['scale_bytes'] == 256 for r in policies)
    assert finite == [True,True,True,True,False]

    def attention(q,k,v):
        scores = q@k.T/np.sqrt(q.shape[-1])
        raw = np.exp(scores-scores.max(axis=-1,keepdims=True))
        return (raw/raw.sum(axis=-1,keepdims=True))@v

    q = np.zeros((2,2))
    k = np.array([[1.,0.],[0.,1.],[1.,1.]])
    v = np.array([[2.,0.],[0.,4.],[4.,2.]])
    perm = [1,2,0]
    output = attention(q,k,v)
    altered = attention(q,k,v[perm])
    q_new = np.array([[2.,0.]])
    assert np.allclose(output,altered)
    assert not np.allclose(attention(q_new,k,v),attention(q_new,k,v[perm]))
    rewards = np.array([1.,1.])
    undiscounted = np.array([2.,1.,0.])
    discounted = np.array([1.9,1.,0.])
    td = lambda values: rewards+.9*values[1:]-values[:-1]
    np.testing.assert_allclose(td(undiscounted),[-.1,0],atol=1e-14)
    np.testing.assert_allclose(td(discounted),[0,0],atol=1e-14)
    exponents = []
    e = 0
    for _ in range(5):
        e = 8+2*e
        exponents.append(e)
    counterexamples = dict(source=inventory['sourceRevision'],
        attention=dict(zero_query_output=output.tolist(),value_only_permutation_output=altered.tolist(),
                       required_change_occurs=False,nonzero_query_changes_under_value_only_permutation=True),
        rl_discount=dict(two_step_rewards=rewards.tolist(),gamma=.9,undiscounted_V=undiscounted.tolist(),
                         TD_with_undiscounted_V=td(undiscounted).tolist(),correct_discounted_V=discounted.tolist(),
                         TD_with_correct_discounted_V=td(discounted).tolist()),
        precision_bytes=dict(shape=[256,64],reported_weight_bytes_all_rows=16384,
                             BF16_operand_weight_bytes=256*64*2,INT8_weight_bytes=256*64,
                             INT8_FP32_scale_bytes=64*4,FP32_dequantized_weight_bytes=256*64*4),
        trace_recurrence=dict(matrix_width=256,element_log2_by_completed_dot=exponents,
                              BF16_finite_exponent_limit_about=128,first_overflow_step=5))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in [('runtime-probe.json',runtime),('counterexamples.json',counterexamples)]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(backend=jax.default_backend(),jax=jax.__version__,
                         sourceChecks=3,counterexamples='passed',output=str(args.output_dir)),indent=2))


if __name__ == '__main__':
    main()
