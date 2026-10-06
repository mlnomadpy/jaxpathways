"""Run each backend in a fresh process. No downloaded models or datasets."""
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['jax', 'tensorflow', 'torch'], required=True)
args = parser.parse_args()
os.environ['KERAS_BACKEND'] = args.backend
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import keras

x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
w = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
b = np.array([.1, -.2], np.float32)
model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(2)])
model.set_weights([w, b])
y = keras.ops.convert_to_numpy(model(x, training=False))
np.testing.assert_allclose(y[0], [3.1, -.45], atol=1e-6)
np.testing.assert_allclose(y, x @ w + b, atol=1e-6)
changed = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
np.testing.assert_allclose(keras.ops.convert_to_numpy(model(changed, training=False)), changed @ w + b, atol=2e-6)
for policy in ['mixed_float16', 'mixed_bfloat16']:
    # Layer-scoped policy avoids changing global defaults for later models.
    layer = keras.layers.Dense(2, dtype=policy)
    layer(x)
    layer.set_weights([w, b])
    result = keras.ops.convert_to_numpy(keras.ops.cast(layer(x), 'float32'))
    assert layer.variable_dtype == 'float32'
    np.testing.assert_allclose(result, y, atol=.06, rtol=.01)
    print(policy, 'variables:', layer.variable_dtype, 'compute:', layer.compute_dtype,
          'max error:', float(np.max(np.abs(result-y))))
if args.backend == 'torch':
    import torch
    native = torch.nn.Linear(3, 2)
    with torch.no_grad():
        native.weight.copy_(torch.from_numpy(w.T.copy()))
        native.bias.copy_(torch.from_numpy(b))
    np.testing.assert_allclose(native(torch.from_numpy(x)).detach().numpy(), y, atol=1e-6)
    print('Native torch Linear layout verified; torch', torch.__version__)
print('PASS Keras', keras.__version__, 'backend', keras.backend.backend())
