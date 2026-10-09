# Build a Bayesian regression

Phase 11: Probabilistic modeling · about 110 minutes · CPU

## What you will be able to do

- State the prior, likelihood and observation-noise assumptions
- Derive posterior precision and check a closed-form special case
- Compute latent and observation predictive variances separately
- Demonstrate prior strength and extrapolation effects

## The problem

You have three noisy measurements and want to predict at a new input. A fitted line gives one answer, but how uncertain should that answer be? We will write a prior, combine it with a likelihood, solve the posterior exactly, and separate uncertainty about the line from noise in a new observation. This exact answer will become our reference when we build an approximate sampler.

## The idea

Bayesian regression represents uncertainty about parameters and propagates it to predictions. A latent-function prediction and a future noisy observation are different random quantities, so their intervals generally have different widths.

## Separate uncertainty about the function from observation noise

For features $x$, posterior parameter covariance $\Sigma$ gives latent predictive variance $x^T\Sigma x$. If independent observation noise has variance $\sigma^2$, a new observation has variance $x^T\Sigma x+\sigma^2$. Name each term before combining them.

Moving away from well-observed feature combinations can increase parameter-related uncertainty. It does not require the assumed observation-noise variance to change. Plot the posterior mean with separately labeled latent and observation bands to make this distinction visible.

The saved spread curves show how uncertainty varies with the input. They are conditional on the model, prior and noise assumptions. Check held-out behavior and model adequacy instead of reading any narrow posterior band as a guarantee.

### Pause and reason

Which interval is appropriate for a new noisy measurement?

<details><summary>Compare your reasoning</summary>

The posterior predictive interval that includes observation noise, under the model's assumptions. A latent-function interval omits that additional variability.

</details>

## State what is random

The features are fixed inputs. We treat the intercept and slope as uncertain, and condition on the observed targets. The prior scale $\tau=2$ expresses uncertainty about each weight before seeing these targets; it is not the sensor noise. The noise scale $\sigma=1$ describes repeated readings around the same latent line. We deliberately assume it known. Learning that scale would remove this exact two-parameter shortcut unless we used a richer conjugate model.

$$
w\sim\mathcal N(0,\tau^2 I),\qquad y\mid w,X\sim\mathcal N(Xw,\sigma^2 I)
$$

## Combine information through precision

Precision is inverse covariance: larger precision means less uncertainty. The prior contributes $I/\tau^2$; the observations contribute $X^\top X/\sigma^2$. The linear term is $X^\top y/\sigma^2$. Completing the square gives a linear system for the posterior mean. This resembles ridge regression because the posterior mode and mean coincide for this Gaussian model. That equivalence does not turn every regularized optimizer into a complete uncertainty model.

$$
\Lambda=\tau^{-2}I+\sigma^{-2}X^\top X,\quad \Sigma=\Lambda^{-1},\quad m=\Lambda^{-1}\sigma^{-2}X^\top y
$$

## Check the special case by hand

The inputs are $-1,0,1$, so their sum is zero and $X^\top X=\operatorname{diag}(3,2)$. The targets give $X^\top y=(3,4)$. Adding prior precision $1/4$ gives diagonal entries $13/4$ and $9/4$. Therefore $m=(12/13,16/9)$ and $\Sigma=\operatorname{diag}(4/13,4/9)$. These rational numbers provide an independent oracle; comparing two calls to the same posterior function would not.

## Two uncertainty bands answer two questions

At a design row $x_*$, the uncertain latent line value has variance $x_*^\top\Sigma x_*$. A new target adds independent observation variance $\sigma^2$. At input zero the latent variance is $4/13$, whereas the observation variance is $17/13$. At input two, the slope contribution increases latent variance to about $2.09$. A nominal Gaussian interval is conditional on this model and known noise scale; it is not proof that future real observations achieve nominal coverage.

$$
\mathbb E[y_*\mid D]=x_*^\top m,\qquad \operatorname{Var}(y_*\mid D)=x_*^\top\Sigma x_*+\sigma^2
$$

## Use a tiny exact model as an inference test

The next lesson compares approximate samples with this kind of known answer. This is a useful engineering habit: test an inference mechanism on a model you can solve before trusting it on one you cannot. Our design is synthetic and unusually well conditioned. Retain the prior, noise assumption and extrapolation range with your result; a narrow posterior can be confidently wrong when the likelihood is wrong.

## 1. Construct a small linear model

Create a fresh main.py. The first feature is an intercept. Targets contain three deliberately simple synthetic observations; the known noise scale is an assumption.

```python
# Step 1 — 1. Construct a small linear model: The design has three rows and two columns.
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Construct `X` via `jnp.array([[1.,-1.],[1.,0.],[1.,1.]])`
X = jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
# Construct `y` via `jnp.array([-1.,1.,3.])`
y = jnp.array([-1.,1.,3.])
# Compute `sigma, prior_scale` from `1., 2.`
sigma, prior_scale = 1., 2.
```

The design has three rows and two columns. The weight vector contains intercept then slope.

## 2. Solve the posterior system

Append the conjugate update. Use a linear solve rather than writing an explicit matrix inverse for the mean.

```python
# Step 2 — 2. Solve the posterior system: The independent fractions follow because centered inputs make the...
def posterior(X,y,sigma,prior_scale):
    # Compute `precision` from `jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2`
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2
    # Perform matrix contraction / projection to compute `mean`.
    mean = jnp.linalg.solve(precision,X.T@y/sigma**2)
    # Compute `covariance` from `jnp.linalg.solve(precision,jnp.eye(X.shape[1]))`
    covariance = jnp.linalg.solve(precision,jnp.eye(X.shape[1]))
    # Return `(mean, covariance)` to the caller.
    return mean,covariance
# Run `posterior` to compute `(mean, cov)`.
mean,cov = posterior(X,y,sigma,prior_scale)
# Assert that `jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)`.
assert jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)
# Assert that `jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)`.
assert jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)
```

The independent fractions follow because centered inputs make the precision matrix diagonal.

## 3. Predict both the latent mean and a new observation

Append the prediction calculation, then run python main.py. Retain the two variance components separately.

```python
# Step 3 — 3. Predict both the latent mean and a new observation: The variance at the extrapolation input is larger because slope...
def predictive(design,mean,cov,sigma):
    # Perform matrix / vector contraction (`@`) to compute `center`.
    center = design@mean
    # Perform matrix contraction / projection to compute `latent_variance`.
    latent_variance = jnp.einsum('ni,ij,nj->n',design,cov,design)
    # Return `(center, latent_variance, latent_variance + sigma ** 2)` to the caller.
    return center,latent_variance,latent_variance+sigma**2
# Construct `query` via `jnp.array([[1.,0.],[1.,2.]])`
query = jnp.array([[1.,0.],[1.,2.]])
# Run `predictive` to compute `(center, latent_var, observation_var)`.
center,latent_var,observation_var = predictive(query,mean,cov,sigma)
# Assert that `jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)`.
assert jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)
# Assert that `jnp.allclose(observation_var-latent_var,1.)`.
assert jnp.allclose(observation_var-latent_var,1.)
# Print the observed values to compare against the expected result.
print('posterior mean:',mean,'covariance:',cov)
# Print diagnostic summary of the computed outputs.
print('latent variance:',latent_var,'observation variance:',observation_var)
```

The variance at the extrapolation input is larger because slope uncertainty contributes more strongly there.

## Run the example

```python
# Step 1 — 1. Construct a small linear model: The design has three rows and two columns.
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Construct `X` via `jnp.array([[1.,-1.],[1.,0.],[1.,1.]])`
X = jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
# Construct `y` via `jnp.array([-1.,1.,3.])`
y = jnp.array([-1.,1.,3.])
# Compute `sigma, prior_scale` from `1., 2.`
sigma, prior_scale = 1., 2.

# Step 2 — 2. Solve the posterior system: The independent fractions follow because centered inputs make the...
def posterior(X,y,sigma,prior_scale):
    # Compute `precision` from `jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2`
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2
    # Perform matrix contraction / projection to compute `mean`.
    mean = jnp.linalg.solve(precision,X.T@y/sigma**2)
    # Compute `covariance` from `jnp.linalg.solve(precision,jnp.eye(X.shape[1]))`
    covariance = jnp.linalg.solve(precision,jnp.eye(X.shape[1]))
    # Return `(mean, covariance)` to the caller.
    return mean,covariance
# Run `posterior` to compute `(mean, cov)`.
mean,cov = posterior(X,y,sigma,prior_scale)
# Assert that `jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)`.
assert jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)
# Assert that `jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)`.
assert jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)

# Step 3 — 3. Predict both the latent mean and a new observation: The variance at the extrapolation input is larger because slope...
def predictive(design,mean,cov,sigma):
    # Perform matrix / vector contraction (`@`) to compute `center`.
    center = design@mean
    # Perform matrix contraction / projection to compute `latent_variance`.
    latent_variance = jnp.einsum('ni,ij,nj->n',design,cov,design)
    # Return `(center, latent_variance, latent_variance + sigma ** 2)` to the caller.
    return center,latent_variance,latent_variance+sigma**2
# Construct `query` via `jnp.array([[1.,0.],[1.,2.]])`
query = jnp.array([[1.,0.],[1.,2.]])
# Run `predictive` to compute `(center, latent_var, observation_var)`.
center,latent_var,observation_var = predictive(query,mean,cov,sigma)
# Assert that `jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)`.
assert jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)
# Assert that `jnp.allclose(observation_var-latent_var,1.)`.
assert jnp.allclose(observation_var-latent_var,1.)
# Print the observed values to compare against the expected result.
print('posterior mean:',mean,'covariance:',cov)
# Print diagnostic summary of the computed outputs.
print('latent variance:',latent_var,'observation variance:',observation_var)
```

Expected: The posterior mean is approximately $(0.9231,1.7778)$. Latent variances at inputs $0,2$ are approximately $0.3077,2.0855$; observation variances are each larger by $1$.

## Prediction uncertainty grows away from the observed center

**Predict:** Will an observation interval be wider or narrower than an interval for the latent line?

![Prediction uncertainty grows away from the observed center](../../phases/11-probability/02-build-a-bayesian-regression/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is the input value. The vertical axis is predictive standard deviation, in target units. The lower curve propagates posterior weight uncertainty; the upper curve also adds the assumed observation noise before taking a square root. At input zero they are about $0.555$ and $1.144$. The curves rise symmetrically toward both ends because centered training inputs produce zero intercept-slope covariance.

### Connect it to the computation

The gap is not a fixed standard deviation of one: variances add, then we take the square root. The lower curve is narrowest near the observed center and grows under extrapolation. These are exact Gaussian-model standard deviations, not measured error bars or empirical coverage on a real test set.

```python
# Compute figure data for: Prediction uncertainty grows away from the observed center
# Generate a uniform grid of points in `grid`.
grid=jnp.linspace(-3,3,61)
# Allocate initialized array `(_, lv, ov)` with the specified shape and dtype.
_,lv,ov=predictive(jnp.stack([jnp.ones_like(grid),grid],axis=1),mean,cov,sigma)
# Compute `visual_data` from `{"kind":"line","x":grid.tolist(),"xlabel":"input val...`
visual_data={"kind":"line","x":grid.tolist(),"xlabel":"input value","ylabel":"predictive standard deviation (target units)","series":[{"label":"latent line","y":jnp.sqrt(lv).tolist()},{"label":"new observation","y":jnp.sqrt(ov).tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:04:47.348581+00:00. JAX 0.9.2.

```text
posterior mean: [0.923077  1.7777778] covariance: [[0.30769232 0.        ]
 [0.         0.44444445]]
latent variance: [0.30769232 2.0854702 ] observation variance: [1.3076923 3.0854702]
posterior mean: [0.923077  1.7777778] covariance: [[0.30769232 0.        ]
 [0.         0.44444445]]
latent variance: [0.30769232 2.0854702 ] observation variance: [1.3076923 3.0854702]
[0.16       0.23529412]
PASS: probability-02

```

## Repeat independent observations

**Predict before running:** If each observed design and target is repeated once as an independent measurement under the model, does posterior uncertainty increase or decrease?

```python
# Experiment — Repeat independent observations: The calculation treats rows as independent new evidence.
repeat_mean,repeat_cov = posterior(jnp.tile(X,(2,1)),jnp.tile(y,2),sigma,prior_scale)
# Assert invariant `jnp.all(jnp.diag(repeat_cov)<jnp.diag(cov))` holds
assert jnp.all(jnp.diag(repeat_cov)<jnp.diag(cov))
# Print the observed values to compare against the expected result.
print(jnp.diag(repeat_cov))
```

**Expected:** Both posterior marginal variances decrease.

The calculation treats rows as independent new evidence. Duplicating a stored file does not create new evidence in a real dataset.

## Check the prior-only boundary

**Predict before running:** With no observations, what should the posterior return?

```python
# Experiment — Check the prior-only boundary: The posterior recovers the prior when no likelihood information...
empty_mean,empty_cov = posterior(jnp.empty((0,2)),jnp.empty((0,)),sigma,prior_scale)
# Assert that `jnp.allclose(empty_mean,0.)`.
assert jnp.allclose(empty_mean,0.)
# Assert that `jnp.allclose(empty_cov,4*jnp.eye(2))`.
assert jnp.allclose(empty_cov,4*jnp.eye(2))
```

**Expected:** The mean is zero and covariance is $4I$.

The posterior recovers the prior when no likelihood information is present.

## Make it yours

Strengthen the prior by changing its scale from two to one half. Predict the direction of change in the weight norm and check it. Explain why a tighter posterior need not mean a better model.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `posterior(...)` — Call `posterior` with your updated parameters or inputs from this lesson's workspace.
- `linalg.norm(...)` — Call `linalg.norm` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Assert invariant `jnp.linalg.norm(strong_mean)<jnp.linalg.norm(mean)` holds
2. Assert invariant `jnp.all(jnp.diag(strong_cov)<jnp.diag(cov))` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Strengthen the prior by changing its scale from two to one half.
strong_mean,strong_cov = posterior(...)  # TODO: compute strong_mean,strong_cov
# Assert invariant `jnp.linalg.norm(strong_mean)<jnp.linalg.norm(mean)` holds
assert jnp.linalg.norm(strong_mean)  # TODO: complete assertion check
# Assert invariant `jnp.all(jnp.diag(strong_cov)<jnp.diag(cov))` holds
assert jnp.all(jnp.diag(strong_cov)  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Strengthen the prior by changing its scale from two to one half.
strong_mean,strong_cov = posterior(X,y,sigma,.5)
# Assert invariant `jnp.linalg.norm(strong_mean)<jnp.linalg.norm(mean)` holds
assert jnp.linalg.norm(strong_mean)<jnp.linalg.norm(mean)
# Assert invariant `jnp.all(jnp.diag(strong_cov)<jnp.diag(cov))` holds
assert jnp.all(jnp.diag(strong_cov)<jnp.diag(cov))
```

</details>

## Move away from centered inputs

**Transfer / diagnosis**

Use design rows $(1,0),(1,1),(1,2)$. Compare the JAX result with a host NumPy linear solve and show the posterior covariance is not diagonal.

<details><summary>Hint</summary>

The intercept and slope can compensate for each other when features are not centered.

</details>

### How to write: Move away from centered inputs — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `shift_X` via `jnp.array([[1.,0.],[1.,1.],[1.,2.]])`
2. Run `posterior` to compute `(shift_m, shift_c)`.
3. Convert `host_X` to a host NumPy array for inspection or verification.
4. Construct an identity matrix `host_P`.
5. Convert `` to a host NumPy array for inspection or verification.

**Starter code scaffold (fill in the TODOs):**

```python
# Move away from centered inputs (Transfer / diagnosis): Nonzero covariance affects prediction uncertainty; keeping...
# Construct `shift_X` via `jnp.array([[1.,0.],[1.,1.],[1.,2.]])`
shift_X = jnp.array(...)  # TODO: compute shift_X
# Run `posterior` to compute `(shift_m, shift_c)`.
shift_m,shift_c = posterior(...)  # TODO: compute shift_m,shift_c
# Convert `host_X` to a host NumPy array for inspection or verification.
host_X = np.asarray(...)  # TODO: compute host_X
# Construct an identity matrix `host_P`.
host_P = np.eye(...)  # TODO: compute host_P
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(shift_m,np.linalg.solve(host_P,host_X.T@np.asarray(y)),rtol=1e-5,atol=1e-6)
# Assert invariant `shift_c[0,1]<0` holds
assert shift_c[0,1]  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Move away from centered inputs (Transfer / diagnosis): Nonzero covariance affects prediction uncertainty; keeping...
# Construct `shift_X` via `jnp.array([[1.,0.],[1.,1.],[1.,2.]])`
shift_X = jnp.array([[1.,0.],[1.,1.],[1.,2.]])
# Run `posterior` to compute `(shift_m, shift_c)`.
shift_m,shift_c = posterior(shift_X,y,sigma,prior_scale)
# Convert `host_X` to a host NumPy array for inspection or verification.
host_X=np.asarray(shift_X,dtype=np.float64)
# Construct an identity matrix `host_P`.
host_P=np.eye(2)/4+host_X.T@host_X
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(shift_m,np.linalg.solve(host_P,host_X.T@np.asarray(y)),rtol=1e-5,atol=1e-6)
# Assert invariant `shift_c[0,1]<0` holds
assert shift_c[0,1]<0
```

Nonzero covariance affects prediction uncertainty; keeping only marginal variances would lose the compensating relationship.

</details>

## Verify total predictive variance

**Transfer / diagnosis**

Draw posterior weights and independent observation noise. Compare empirical prediction variance at input zero to the analytic observation variance.

<details><summary>Hint</summary>

Use separate keys for weights and noise; sample with a Cholesky factor.

</details>

### How to write: Verify total predictive variance — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jax.random.PRNGKey(seed) & jax.random.split(key)` — Manages explicit, stateless PRNG keys—always split a key before passing subkeys into independent random draws.

**Step-by-step implementation plan:**
1. Create or split explicit PRNG key(s) (`(kw, kn)`) for reproducible randomness.
2. Sample deterministic random values into `weights` using an explicit PRNG key.
3. Sample deterministic random values into `replicated` using an explicit PRNG key.
4. Assert that `abs(float(replicated.var())-float(observation_var[0]))<.07`.

**Starter code scaffold (fill in the TODOs):**

```python
# Verify total predictive variance (Transfer / diagnosis): A simulation verifies that the independent noise term...
# Create or split explicit PRNG key(s) (`(kw, kn)`) for reproducible randomness.
kw,kn = jax.random.split(...)  # TODO: compute kw,kn
# Sample deterministic random values into `weights` using an explicit PRNG key.
weights = ...  # TODO: compute weights
# Sample deterministic random values into `replicated` using an explicit PRNG key.
replicated = ...  # TODO: compute replicated
# Assert that `abs(float(replicated.var())-float(observation_var[0]))<.07`.
assert abs(float(replicated.var())-float(observation_var[0]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Verify total predictive variance (Transfer / diagnosis): A simulation verifies that the independent noise term...
# Create or split explicit PRNG key(s) (`(kw, kn)`) for reproducible randomness.
kw,kn=jax.random.split(jax.random.key(37))
# Sample deterministic random values into `weights` using an explicit PRNG key.
weights=mean+jax.random.normal(kw,(30000,2))@jnp.linalg.cholesky(cov).T
# Sample deterministic random values into `replicated` using an explicit PRNG key.
replicated=weights[:,0]+sigma*jax.random.normal(kn,(30000,))
# Assert that `abs(float(replicated.var())-float(observation_var[0]))<.07`.
assert abs(float(replicated.var())-float(observation_var[0]))<.07
```

A simulation verifies that the independent noise term belongs in predictions for observed targets.

</details>

## Check your understanding

Which variance describes a new noisy reading at a given input?

1. Only the posterior variance of the intercept
2. The propagated weight uncertainty plus observation-noise variance
3. The training mean squared error divided by the number of weights

<details><summary>Answer and explanation</summary>

The propagated weight uncertainty plus observation-noise variance

Both the unknown line and a new independent observation error contribute. Omitting the latter gives an interval for the latent mean, not for a future reading.

</details>

## Diagnose the result

If an interval becomes implausibly narrow, print the latent and noise variance terms separately. If the slope is unexpectedly shrunk, inspect whether the prior scale was confused with prior precision. If targets have an extra column axis, reject the shape before a matrix operation silently changes the mean shape.

## Carry forward

- Posterior precision combines prior information with likelihood information.
- Latent uncertainty and future observation noise are separate quantities.
- An exact small model supplies an independent reference for approximate inference.

## Keep your evidence

Precision derivation, exact fractional reference, host solve and predictive variance plot. Keep the environment, observed outputs and your explanation of the figure.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX linear solve](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.solve.html)
- [Stan posterior predictive checks](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html)

