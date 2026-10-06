"""Fit parameters and scale the simulation: worked experiments and reference solutions. CPU checks."""

# Define the simulator and immutable observations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

train_initials = jnp.array([0.7,1.5,2.3])
times = jnp.arange(41)*0.05
true_rate = 0.7
observations = train_initials[:,None]*jnp.exp(-true_rate*times)
def predict(rate, initials):
    return jax.vmap(lambda initial: solve(rate,initial))(initials)
def objective(log_rate):
    residual = predict(jnp.exp(log_rate),train_initials)-observations
    return jnp.mean(residual**2)

# Fit a positive rate with an explicit update
@jax.jit
def update(log_rate):
    return log_rate - 0.4*jax.grad(objective)(log_rate)
log_rate = jnp.log(1.4)
loss_history = [float(objective(log_rate))]
rate_history = [float(jnp.exp(log_rate))]
for _ in range(160):
    log_rate = update(log_rate)
    loss_history.append(float(objective(log_rate)))
    rate_history.append(float(jnp.exp(log_rate)))
fitted_rate = float(jnp.exp(log_rate))
assert abs(fitted_rate-true_rate) < 2e-7
assert loss_history[-1] < 1e-20
print("Initial/fitted rate:",rate_history[0],fitted_rate)
print("Initial/final loss:",loss_history[0],loss_history[-1])

# Validate on unseen initial conditions and a finer grid
held_initials=jnp.array([0.4,1.1,3.2])
held_truth=np.asarray(held_initials)[:,None]*np.exp(-true_rate*np.asarray(times))
held_prediction=np.asarray(predict(fitted_rate,held_initials))
held_rmse=float(np.sqrt(np.mean((held_prediction-held_truth)**2)))
assert held_rmse < 1e-7
fine=np.asarray(jax.vmap(lambda initial:solve(fitted_rate,initial,80,0.025))(held_initials))[:,::2]
fine_rmse=float(np.sqrt(np.mean((fine-held_truth)**2)))
assert fine_rmse < 1e-7
initial_only_gradient=jax.grad(lambda p:jnp.mean((predict(jnp.exp(p),train_initials)[:,0]-observations[:,0])**2))(jnp.log(1.4))
assert float(initial_only_gradient)==0.0
print("Held-out RMSE:",held_rmse,"finer-grid RMSE:",fine_rmse)
print("Initial-time-only gradient:",float(initial_only_gradient))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

train_initials = jnp.array([0.7,1.5,2.3])
times = jnp.arange(41)*0.05
true_rate = 0.7
observations = train_initials[:,None]*jnp.exp(-true_rate*times)
def predict(rate, initials):
    return jax.vmap(lambda initial: solve(rate,initial))(initials)
def objective(log_rate):
    residual = predict(jnp.exp(log_rate),train_initials)-observations
    return jnp.mean(residual**2)

@jax.jit
def update(log_rate):
    return log_rate - 0.4*jax.grad(objective)(log_rate)
log_rate = jnp.log(1.4)
loss_history = [float(objective(log_rate))]
rate_history = [float(jnp.exp(log_rate))]
for _ in range(160):
    log_rate = update(log_rate)
    loss_history.append(float(objective(log_rate)))
    rate_history.append(float(jnp.exp(log_rate)))
fitted_rate = float(jnp.exp(log_rate))
assert abs(fitted_rate-true_rate) < 2e-7
assert loss_history[-1] < 1e-20
print("Initial/fitted rate:",rate_history[0],fitted_rate)
print("Initial/final loss:",loss_history[0],loss_history[-1])

held_initials=jnp.array([0.4,1.1,3.2])
held_truth=np.asarray(held_initials)[:,None]*np.exp(-true_rate*np.asarray(times))
held_prediction=np.asarray(predict(fitted_rate,held_initials))
held_rmse=float(np.sqrt(np.mean((held_prediction-held_truth)**2)))
assert held_rmse < 1e-7
fine=np.asarray(jax.vmap(lambda initial:solve(fitted_rate,initial,80,0.025))(held_initials))[:,::2]
fine_rmse=float(np.sqrt(np.mean((fine-held_truth)**2)))
assert fine_rmse < 1e-7
initial_only_gradient=jax.grad(lambda p:jnp.mean((predict(jnp.exp(p),train_initials)[:,0]-observations[:,0])**2))(jnp.log(1.4))
assert float(initial_only_gradient)==0.0
print("Held-out RMSE:",held_rmse,"finer-grid RMSE:",fine_rmse)
print("Initial-time-only gradient:",float(initial_only_gradient))

# Figure data experiment
indices=list(range(61))
visual_data={'panels':[{'kind':'line','x':indices,'xlabel':'completed updates','ylabel':'rate (inverse seconds)','series':[{'label':'fitted rate','y':rate_history[:61]},{'label':'known synthetic rate','y':[true_rate]*61}]},{'kind':'line','x':indices,'xlabel':'completed updates','ylabel':'mean squared trajectory error','yscale':'log','series':[{'label':'training error','y':loss_history[:61]}]}]}

# Experiment: A perfect Euler fit can infer the wrong rate
h=0.5
biased_rate=(1-np.exp(-true_rate*h))/h
coarse_times=np.arange(5)*h
matched=2*(1-biased_rate*h)**np.arange(5)
np.testing.assert_allclose(matched,2*np.exp(-true_rate*coarse_times),rtol=1e-12)
assert abs(biased_rate-true_rate)>0.1
print("True rate:",true_rate,"perfect coarse Euler rate:",biased_rate)

# Experiment: Measure fixed-shape CPU batches
import time
measurements=[]
for batch_size in (1,16,256):
    starts=jnp.linspace(0.4,3.2,batch_size)
    compiled=jax.jit(lambda values:predict(fitted_rate,values))
    compiled(starts).block_until_ready()
    samples=[]
    for _ in range(7):
        beginning=time.perf_counter()
        compiled(starts).block_until_ready()
        samples.append((time.perf_counter()-beginning)*1000)
    median_ms=float(np.median(samples))
    assert median_ms>0
    measurements.append((batch_size,median_ms,median_ms/batch_size))
print("batch, median milliseconds, milliseconds per trajectory:",measurements)

# Reference solution. Try the exercise before reading this.
changed_truth=0.4
changed_observations=train_initials[:,None]*jnp.exp(-changed_truth*times)
def changed_objective(theta):
    return jnp.mean((predict(jnp.exp(theta),train_initials)-changed_observations)**2)
changed_update=jax.jit(lambda theta:theta-0.4*jax.grad(changed_objective)(theta))
changed_theta=jnp.log(0.9)
for _ in range(200):
    changed_theta=changed_update(changed_theta)
changed_rate=float(jnp.exp(changed_theta))
assert abs(changed_rate-changed_truth)<1e-6
changed_held=np.asarray(held_initials)[:,None]*np.exp(-changed_truth*np.asarray(times))
np.testing.assert_allclose(predict(changed_rate,held_initials),changed_held,rtol=2e-6,atol=1e-7)
print("Changed true/recovered rate:",changed_truth,changed_rate)

# Reference practice: Infer a different physical quantity
basis=np.exp(-0.7*np.asarray(times))
unknown_initial=2.8
measured=unknown_initial*basis
estimate=float(np.dot(basis,measured)/np.dot(basis,basis))
assert abs(estimate-unknown_initial)<1e-12
def amplitude_loss(amplitude):
    return jnp.mean((amplitude*jnp.asarray(basis)-jnp.asarray(measured))**2)
assert abs(float(jax.grad(amplitude_loss)(estimate)))<1e-12
print("Recovered initial amplitude:",estimate)

# Reference practice: Compare noisy inference with an independent search
noisy=np.asarray(observations)+0.005*np.random.default_rng(19).normal(size=observations.shape)
candidates=np.linspace(0.65,0.75,1001)
reference_predictions=np.asarray(train_initials)[None,:,None]*np.exp(-candidates[:,None,None]*np.asarray(times)[None,None,:])
reference_losses=np.mean((reference_predictions-noisy[None,:,:])**2,axis=(1,2))
best=float(candidates[np.argmin(reference_losses)])
assert abs(best-0.7)<0.01
assert float(np.min(reference_losses))>0
print("Noisy grid-search estimate:",best,"residual MSE:",float(np.min(reference_losses)))
print("PASS: science-04")
