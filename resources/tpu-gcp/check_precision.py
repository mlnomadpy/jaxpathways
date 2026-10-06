"""Independent CPU arithmetic and failure checks for the precision lab."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('precision_profile', ROOT / 'precision_profile.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def main():
    import numpy as np
    for axis in (0, 1):
        integer, scale = lab.quantize(np.zeros((3, 7), dtype=np.float32), axis)
        assert not integer.any() and np.all(scale == 1) and np.isfinite(scale).all()
    tiny = np.array([[0., 1., -2.]], dtype=np.float32)
    integer, scale = lab.quantize(tiny, 1)
    np.testing.assert_array_equal(integer, [[0, 64, -127]])
    np.testing.assert_allclose(scale, 2 / 127)
    assert lab.measures(np.zeros((2, 2)), np.zeros((2, 2)))['relative_l2_error'] is None
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index, shape in enumerate(((1, 7, 3), (5, 17, 9))):
            folder = root / str(index)
            command = [sys.executable, str(ROOT / 'precision_profile.py'), '--platform', 'cpu', '--run-dir', str(folder),
                       '--batch', str(shape[0]), '--features', str(shape[1]), '--outputs', str(shape[2]), '--steps', '3']
            if index == 0:
                command.append('--trace')
            result = subprocess.run(command, env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads((folder / 'report.json').read_text())
            assert report['platform'] == 'cpu' and len(report['results']) == 6
            for row in report['results']:
                assert len(row['samples_ms']) == 3 and all(value > 0 for value in row['samples_ms'])
                assert row['output_dtype'] == 'float32'
                # Independently count the supplied integer arrays and scales.
                m, k, n = shape
                expected = (m*k+k*n)*{'fp32': 4, 'bf16': 2, 'int8': 1}[row['mode']]
                if row['mode'] == 'int8':
                    expected += 4 * (m+n)
                    assert row['integer_accumulation_checked']
                assert row['stored_operand_bytes'] == expected
            if index == 0:
                paths = list((folder / 'trace').rglob('perfetto_trace.json.gz'))
                assert paths, 'actual profiler did not emit a Perfetto trace'
                trace = json.loads(gzip.decompress(paths[0].read_bytes()))
                names = {event.get('name') for event in trace['traceEvents']}
                assert all(case+'-'+mode in names for case in ('balanced', 'outlier') for mode in ('fp32', 'bf16', 'int8'))
            before = (folder / 'report.json').read_bytes()
            refused = subprocess.run(command, capture_output=True, text=True, timeout=120)
            assert refused.returncode != 0 and before == (folder / 'report.json').read_bytes()
        invalid = subprocess.run([sys.executable, str(ROOT/'precision_profile.py'), '--platform', 'cpu',
                                  '--run-dir', str(root/'invalid'), '--steps', '0'], capture_output=True, text=True)
        assert invalid.returncode != 0 and not (root/'invalid').exists()
        # Demonstrate unavailable requested backends fail with retained diagnostics.
        missing = root / 'missing-tpu'
        result = subprocess.run([sys.executable, str(ROOT/'precision_profile.py'), '--platform', 'tpu', '--run-dir', str(missing)],
                                env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'JAX_TPU_LIBRARY_PATH': str(root/'absent-libtpu.so')},
                                capture_output=True, text=True, timeout=120)
        assert result.returncode != 0 and (missing/'failure.json').exists() and not (missing/'report.json').exists()
    print('PASS: zero scales, independent integer/rounded-float checks, changed shapes, synchronized samples, real trace, overwrite and backend failure')


if __name__ == '__main__':
    main()
