"""Debug unstable learning: worked experiments and reference solutions. CPU checks."""

# 1. Define the controlled objective
import jax
import jax.numpy as jnp
import numpy as np
x=jnp.array([1.,2.,3.]);y=2*x
def loss(w,features=x,targets=y):return jnp.mean((w*features-targets)**2)
def analytic_gradient(w):return (28/3)*(w-2)
np.testing.assert_allclose(loss(0.),56/3,rtol=1e-6)
np.testing.assert_allclose(jax.grad(loss)(0.),-56/3,rtol=1e-6)

# 2. Instrument the update loop
def run(rate,steps,features=x,targets=y,clip=None):
    w=jnp.array(0.);history=[]
    objective=lambda value:loss(value,features,targets)
    for _ in range(steps):
        value,grad=jax.value_and_grad(objective)(w)
        history.append([float(w),float(value),float(jnp.abs(grad))])
        assert np.all(np.isfinite(history[-1]))
        update=jnp.clip(grad,-clip,clip) if clip is not None else grad
        w=w-rate*update
    return float(w),np.asarray(history)
stable,good=run(.1,6)
unstable,bad=run(.3,6)
assert abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100

# 3. Match the recorded failure to the exact recurrence
for rate,history in [(.1,good),(.3,bad)]:
    expected_w=2-2*(1-rate*28/3)**np.arange(len(history))
    expected_loss=(14/3)*(expected_w-2)**2
    np.testing.assert_allclose(history[:,0],expected_w,rtol=2e-5,atol=1e-5)
    np.testing.assert_allclose(history[:,1],expected_loss,rtol=2e-5,atol=1e-8)
np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol=1e-5)
print('Stable final weight:',stable,'unstable pre-update [weight,loss,|gradient|]:\n',bad)

import jax
import jax.numpy as jnp
import numpy as np
x=jnp.array([1.,2.,3.]);y=2*x
def loss(w,features=x,targets=y):return jnp.mean((w*features-targets)**2)
def analytic_gradient(w):return (28/3)*(w-2)
np.testing.assert_allclose(loss(0.),56/3,rtol=1e-6)
np.testing.assert_allclose(jax.grad(loss)(0.),-56/3,rtol=1e-6)

def run(rate,steps,features=x,targets=y,clip=None):
    w=jnp.array(0.);history=[]
    objective=lambda value:loss(value,features,targets)
    for _ in range(steps):
        value,grad=jax.value_and_grad(objective)(w)
        history.append([float(w),float(value),float(jnp.abs(grad))])
        assert np.all(np.isfinite(history[-1]))
        update=jnp.clip(grad,-clip,clip) if clip is not None else grad
        w=w-rate*update
    return float(w),np.asarray(history)
stable,good=run(.1,6)
unstable,bad=run(.3,6)
assert abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100

for rate,history in [(.1,good),(.3,bad)]:
    expected_w=2-2*(1-rate*28/3)**np.arange(len(history))
    expected_loss=(14/3)*(expected_w-2)**2
    np.testing.assert_allclose(history[:,0],expected_w,rtol=2e-5,atol=1e-5)
    np.testing.assert_allclose(history[:,1],expected_loss,rtol=2e-5,atol=1e-8)
np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol=1e-5)
print('Stable final weight:',stable,'unstable pre-update [weight,loss,|gradient|]:\n',bad)

# Figure data experiment
visual_data = {'kind': 'line', 'x': list(range(len(good))), 'xlabel': 'pre-update step', 'ylabel': 'mean squared loss', 'yscale': 'log', 'series': [{'label': 'rate 0.1', 'y': good[:, 1].tolist()}, {'label': 'rate 0.3', 'y': bad[:, 1].tolist()}]}

# Experiment: Bounded clipped updates
clipped,clipped_history=run(.3,6,clip=1.)
assert np.all(np.abs(np.diff(clipped_history[:,0]))<=.300001)
assert clipped_history[-1,1]<clipped_history[0,1]
assert clipped_history[0,2]>1.

# Experiment: Feature scaling changes curvature
scaled_final,scaled=run(.1,3,x*10,y*10)
assert scaled[-1,1]>scaled[0,1]
adjusted_final,adjusted=run(.001,6,x*10,y*10)
np.testing.assert_allclose(adjusted[:,0],good[:,0],rtol=1e-5,atol=1e-5)
np.testing.assert_allclose(adjusted[:,1],good[:,1]*100,rtol=1e-4,atol=1e-8)

# Reference solution. Try the exercise before reading this.
ones=jnp.ones(3)
exact,history=run(.5,3,ones,2*ones)
divergent,divergent_history=run(1.1,6,ones,2*ones)
np.testing.assert_allclose(exact,2.,atol=1e-6)
assert divergent_history[-1,1]>divergent_history[0,1]
np.testing.assert_allclose(divergent_history[1:,1]/divergent_history[:-1,1],1.44,rtol=1e-5)

# Reference practice: Rescale the objective and compensate the update
w0=jnp.array(.25)
base_gradient=jax.grad(loss)(w0)
scaled_gradient=jax.grad(lambda w:5*loss(w))(w0)
np.testing.assert_allclose(scaled_gradient,5*base_gradient,rtol=1e-6)
reference=w0-.1*base_gradient
compensated=w0-(.1/5)*scaled_gradient
uncompensated=w0-.1*scaled_gradient
np.testing.assert_allclose(compensated,reference,rtol=1e-6)
assert not np.isclose(uncompensated,reference)
print('Reference / compensated / unchanged-rate updates:',reference,compensated,uncompensated)

# Reference practice: Diagnose a nonfinite input before an update
corrupted=x.at[1].set(jnp.nan)
bad_value,bad_grad=jax.value_and_grad(lambda w:loss(w,corrupted,y))(0.)
assert not np.isfinite(float(bad_value)) and not np.isfinite(float(bad_grad))
repaired=jnp.array([1.,2.,3.])
repaired_value,repaired_grad=jax.value_and_grad(lambda w:loss(w,repaired,y))(0.)
np.testing.assert_allclose(repaired_grad,-56/3,rtol=1e-6)
assert np.isfinite(float(repaired_value))
print("PASS: networks-05")
