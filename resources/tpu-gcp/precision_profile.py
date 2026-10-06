"""Measure rounding, integer arithmetic and synchronized single-device execution.

CPU results are not TPU performance predictions. All operands and quantization
are prepared before timing; this is an inner-dot experiment, not a model server.
"""
import argparse
import datetime
import platform
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('cpu', 'tpu'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--outputs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--trace', action='store_true', help='capture a short warmed profile after timing')
    args = parser.parse_args()
    if any(not 1 <= value <= 2048 for value in (args.batch, args.features, args.outputs)):
        parser.error('matrix dimensions must be between 1 and 2048')
    if args.batch * args.features * args.outputs > 16_777_216:
        parser.error('keep this teaching experiment below 16,777,216 multiply positions')
    if not 1 <= args.steps <= 200:
        parser.error('--steps must be between 1 and 200')
    return args


def quantize(array, axis):
    """Symmetric INT8, one scale per activation row or weight output column."""
    import numpy as np
    peak = np.max(np.abs(array), axis=axis, keepdims=True)
    scale = np.where(peak == 0, 1.0, peak / 127.0).astype(np.float32)
    integer = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return integer, scale


def measures(actual, reference):
    import numpy as np
    delta = actual.astype(np.float64) - reference
    norm = np.linalg.norm(reference)
    return {'max_absolute_error': float(np.max(np.abs(delta))),
            'relative_l2_error': float(np.linalg.norm(delta) / norm) if norm else None}


def experiment(args, jax, np):
    import jax.numpy as jnp
    if jax.default_backend() != args.platform:
        raise RuntimeError('requested backend was not selected')
    device = jax.local_devices()[0]
    rng = np.random.default_rng(17)
    base_x = rng.normal(size=(args.batch, args.features)).astype(np.float32)
    base_w = (rng.normal(size=(args.features, args.outputs)) / np.sqrt(args.features)).astype(np.float32)
    results = []
    trace_jobs = []
    for name in ('balanced', 'outlier'):
        host_x, host_w = base_x.copy(), base_w.copy()
        if name == 'outlier':
            host_x[:, 0] *= 30  # controlled perturbation, not a real model distribution
        reference = host_x.astype(np.float64) @ host_w.astype(np.float64)
        qx, sx = quantize(host_x, axis=1)
        qw, sw = quantize(host_w, axis=0)
        original = (jax.device_put(host_x, device), jax.device_put(host_w, device))
        operands = {
            'fp32': original,
            'bf16': tuple(value.astype(jnp.bfloat16) for value in original),
            'int8': (jax.device_put(qx, device), jax.device_put(qw, device)),
        }
        scales = (jax.device_put(sx, device), jax.device_put(sw, device))
        # This checks actual integer accumulation, independently of dequantization.
        integer_dot = jax.jit(lambda a, b: jnp.matmul(a, b, preferred_element_type=jnp.int32))
        raw = np.asarray(integer_dot(*operands['int8']).block_until_ready())
        np.testing.assert_array_equal(raw, qx.astype(np.int64) @ qw.astype(np.int64))
        for mode, values in operands.items():
            if mode == 'int8':
                fn = jax.jit(lambda a, b, s, t: jnp.matmul(a, b, preferred_element_type=jnp.int32).astype(jnp.float32) * s * t)
                call_args = (*values, *scales)
            else:
                fn = jax.jit(lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST,
                                                   preferred_element_type=jnp.float32))
                call_args = values
            for value in call_args:
                value.block_until_ready()
            executable = fn.lower(*call_args).compile()
            executable(*call_args).block_until_ready()  # compile and warm outside timer
            timings = []
            for _ in range(args.steps):
                start = time.perf_counter()
                actual = executable(*call_args).block_until_ready()
                timings.append((time.perf_counter() - start) * 1000)
            observed = np.asarray(actual)
            if not np.isfinite(observed).all():
                raise ArithmeticError('nonfinite output')
            assert observed.dtype == np.float32
            if mode != 'int8':
                rounded_reference = np.asarray(values[0], dtype=np.float64) @ np.asarray(values[1], dtype=np.float64)
                np.testing.assert_allclose(observed, rounded_reference, rtol=3e-5, atol=3e-5)
            stats = executable.memory_analysis()
            memory = None if stats is None else {key: getattr(stats, key) for key in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            identifier = name + '-' + mode
            (args.run_dir / (identifier + '.hlo.txt')).write_text(executable.as_text())
            results.append({'case': name, 'mode': mode, **measures(observed, reference),
                            'median_ms': statistics.median(timings), 'samples_ms': timings,
                            'stored_operand_bytes': sum(value.size * value.dtype.itemsize for value in call_args),
                            'output_dtype': str(observed.dtype), 'compiler_memory_estimate': memory,
                            'input_sha256': hashlib.sha256(host_x.tobytes() + host_w.tobytes()).hexdigest(),
                            'integer_accumulation_checked': mode == 'int8'})
            trace_jobs.append((identifier, executable, call_args))
    if args.trace:
        # This separate span cannot contaminate the timing samples above.
        with jax.profiler.trace(str(args.run_dir / 'trace'), create_perfetto_trace=True):
            for step, (identifier, fn, values) in enumerate(trace_jobs):
                with jax.profiler.StepTraceAnnotation('precision_case', step_num=step):
                    with jax.profiler.TraceAnnotation(identifier):
                        fn(*values).block_until_ready()
    return {'schema': 1, 'executed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python_version': platform.python_version(), 'host_system': platform.system(), 'scope': 'single-process, single-device prequantized inner dot',
            'platform': jax.default_backend(), 'device_kind': device.device_kind,
            'local_device_count': jax.local_device_count(), 'global_device_count': jax.device_count(),
            'process_count': jax.process_count(), 'jax_version': jax.__version__, 'numpy_version': np.__version__,
            'dimensions': {'batch': args.batch, 'features': args.features, 'outputs': args.outputs},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference': 'NumPy float64 dot of original float32 operands',
            'timing_scope': 'warmed synchronized call; excludes transfer, quantization and compilation',
            'outlier_rule': 'multiply input feature zero by 30 for every row', 'results': results}


def main():
    args = arguments()
    # Exclusive creation protects evidence from accidental replacement.
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / 'config.json').write_text(json.dumps({**vars(args), 'run_dir': str(args.run_dir)}, indent=2) + '\n')
    os.environ['JAX_PLATFORMS'] = args.platform
    try:
        import jax
        import numpy as np
        report = experiment(args, jax, np)
        (args.run_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('Saved', args.run_dir / 'report.json', 'on', report['platform'], report['device_kind'])
    except Exception as error:
        (args.run_dir / 'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
