"""Optional TensorFlow desktop conversion lab; not an edge-device benchmark.

Run in a separate TensorFlow environment. Writes artifacts only to --output.
The desktop tf.lite.Interpreter compatibility API may emit a migration warning;
use the target platform's current LiteRT API for device integration.
"""
import argparse
import json
import os
from pathlib import Path
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('edge-artifacts'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
w = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
b = np.array([.1, -.2], np.float32)
# A concrete function makes the input contract explicit and avoids Keras
# archive/export format assumptions across converter versions.
class Dense(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 3], tf.float32, name='features')])
    def serve(self, features):
        return tf.matmul(features, tf.constant(w)) + tf.constant(b)
model = Dense()
concrete = model.serve.get_concrete_function()
calibration = np.random.default_rng(12).uniform(-1, 1, (256, 3)).astype(np.float32)
# Explicit corners cover the domain; held-out rows below are separate.
calibration = np.vstack((calibration, [[-1, -1, -1], [1, 1, 1]])).astype(np.float32)
held_out = np.random.default_rng(31).uniform(-.8, .8, (20, 3)).astype(np.float32)
def representative_dataset():
    for row in calibration:
        yield [row[None, :]]
def convert(integer):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    if integer:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()
def invoke(interpreter, row):
    input_spec, = interpreter.get_input_details()
    output_spec, = interpreter.get_output_details()
    values = row[None, :]
    if np.issubdtype(input_spec['dtype'], np.integer):
        scale, zero = input_spec['quantization']
        assert scale > 0
        limits = np.iinfo(input_spec['dtype'])
        values = np.clip(np.rint(values / scale + zero), limits.min, limits.max).astype(input_spec['dtype'])
    interpreter.set_tensor(input_spec['index'], values)
    interpreter.invoke()
    output = interpreter.get_tensor(output_spec['index'])
    if np.issubdtype(output_spec['dtype'], np.integer):
        scale, zero = output_spec['quantization']
        assert scale > 0
        output = (output.astype(np.float32) - zero) * scale
    return output
report = {'tensorflow': tf.__version__, 'device_validated': False, 'variants': []}
for integer in (False, True):
    name = 'dense-int8' if integer else 'dense-fp32'
    artifact = convert(integer)
    (args.output / (name + '.tflite')).write_bytes(artifact)
    interpreter = tf.lite.Interpreter(model_content=artifact)
    interpreter.allocate_tensors()
    input_spec, = interpreter.get_input_details()
    output_spec, = interpreter.get_output_details()
    if integer:
        assert input_spec['dtype'] == output_spec['dtype'] == np.int8
    predicted = np.concatenate([invoke(interpreter, row) for row in held_out])
    expected = held_out @ w + b
    tolerance = .08 if integer else 2e-6  # fixed before comparison, for this fixture
    np.testing.assert_allclose(predicted, expected, atol=tolerance, rtol=0)
    report['variants'].append({'name': name, 'bytes': len(artifact),
        'input_dtype': str(input_spec['dtype']), 'output_dtype': str(output_spec['dtype']),
        'input_quantization': input_spec['quantization'],
        'output_quantization': output_spec['quantization'],
        'max_abs_error': float(np.max(np.abs(predicted-expected))), 'tolerance': tolerance})
(args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
