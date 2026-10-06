# Train a CNN on a small dataset

Phase 05: Neural network training · about 110 minutes · CPU

## What you will be able to do

- Generate images with an explicit label rule
- Derive the convolution output shape
- Pool spatial responses and retain class scores
- Read a confusion matrix rather than a success label

## The problem

Can a small model tell horizontal bars from vertical bars in noisy images? We’ll create the images locally and train a convolutional classifier on CPU. You’ll then test images from a different random seed and read a confusion matrix. The task is small enough to inspect without downloading a dataset.

## The idea

A convolution reuses one local filter at every image position. Nonlinear activation and spatial pooling turn those local responses into features for a linear classification head. This tiny bar task makes the mechanism inspectable on a CPU; success here does not establish handwritten-digit or natural-image accuracy.

## Generate images with an explicit label rule

Our images have height eight, width eight, and one channel. Class zero contains a horizontal bright bar; class one contains a vertical bright bar. The bar position varies among interior rows or columns, and independent Gaussian noise perturbs every pixel. Alternate labels so both splits are balanced. Generate training and held-out sets from different seeds; identical seeds would reproduce images and create leakage. Every observation follows a known rule, so inspect a few images and labels before training. This synthetic distribution contains no occlusion, perspective, or multiple objects and should never be described as realistic vision evaluation.

## Derive the convolution output shape

NNX Conv accepts channels-last arrays: batch, height, width, channels. A $3\times3$ filter with stride one and VALID padding slides only where the whole patch fits. An $8\times8$ image therefore yields $6\times6$ spatial outputs. Four output channels mean four independently learned filters; the kernel shape is $(3, 3, 1, 4)$, followed by four biases. Each output is a patch dot product plus a bias. Deep-learning convolution uses cross-correlation here: do not reverse the filter in the independent NumPy reference. We check one interior output explicitly before fitting, which catches axis or padding misunderstandings.

```text
(B,8,8,1) → Conv3×3/VALID → (B,6,6,4) → ReLU → spatial mean → (B,4) → Linear → (B,2)
```

## Pool spatial responses and retain class scores

ReLU replaces negative filter responses with zero. Averaging axes one and two summarizes how strongly each learned pattern appears anywhere in the remaining image. Pooling discards exact position, which suits orientation classification but would hurt a task asking for bar location. The classifier head emits two logits per image. softmax_cross_entropy_with_integer_labels consumes those scores and integer labels directly; do not apply softmax beforehand. A stable independent NumPy reference subtracts the largest score before computing log-sum-exp. The objective is averaged over observations, not over classes.

## Read a confusion matrix rather than a success label

Use rows for true classes and columns for predicted classes. The diagonal counts correct classifications; off-diagonal cells identify orientation-specific mistakes. A balanced 24-image held-out set contains $12$ examples per class, so a model always predicting horizontal has accuracy $0.5$. After training, print both held-out mean loss and accuracy, along with all four confusion counts. Check that the matrix sums to $24$ and that its trace divided by $24$ equals accuracy. This independent accounting prevents a reduction-axis error from looking like good performance. Keep the split and hyperparameters fixed; a new data distribution is a new experiment, not another chance to select the reported test result.

## 1. Generate two disjoint synthetic splits

Create main.py. Print a training image as an $8\times8$ array if you want to inspect the task before fitting.

```python
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def bars(seed,n,noise=.08):
    rng=np.random.default_rng(seed)
    images=np.zeros((n,8,8,1),np.float32)
    labels=np.arange(n,dtype=np.int32)%2
    for i,label in enumerate(labels):
        position=int(rng.integers(2,6))
        if label==0:images[i,position,:,0]=1.
        else:images[i,:,position,0]=1.
    images+=rng.normal(0,noise,images.shape).astype(np.float32)
    return jnp.array(images),jnp.array(labels)
train_x,train_y=bars(20,48)
test_x,test_y=bars(21,24)
assert train_x.shape==(48,8,8,1) and test_x.shape==(24,8,8,1)
assert not np.array_equal(np.asarray(train_x[:24]),np.asarray(test_x))
```

Two seeds produce separate examples. Both labels are balanced by construction; noise can make orientation harder as its scale grows.

## 2. Build and independently inspect the convolution

Append the small CNN and the patch dot-product check before defining an optimizer.

```python
class BarCNN(nnx.Module):
    def __init__(self):
        rngs=nnx.Rngs(0)
        self.conv=nnx.Conv(1,4,(3,3),padding='VALID',rngs=rngs)
        self.head=nnx.Linear(4,2,rngs=rngs)
    def __call__(self,x):
        features=jnp.mean(nnx.relu(self.conv(x)),axis=(1,2))
        return self.head(features)
model=BarCNN()
raw=model.conv(train_x[:1])
assert raw.shape==(1,6,6,4)
patch=np.asarray(train_x[0,:3,:3,:])
kernel=np.asarray(model.conv.kernel[...])
expected=np.sum(patch*kernel[:,:,:,0])+np.asarray(model.conv.bias[...])[0]
np.testing.assert_allclose(raw[0,0,0,0],expected,rtol=1e-5,atol=1e-6)
def loss(m,x,y):return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(x),y))
optimizer=nnx.Optimizer(model,optax.adam(.03),wrt=nnx.Param)
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(loss)(m,x,y)
    o.update(m,grads)
    return value
```

The first output at $(0,0)$ consumes exactly the top-left $3\times3$ patch. The NumPy result validates filter orientation and layout.

## 3. Fit, then measure the held-out split once

Append $80$ full-batch updates, stable NumPy loss, and confusion accounting. Every update uses only train_x/train_y.

```python
initial=float(loss(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
scores=np.asarray(model(test_x));labels=np.asarray(test_y)
predictions=scores.argmax(axis=-1)
shifted=scores-scores.max(axis=-1,keepdims=True)
reference_loss=np.mean(np.log(np.exp(shifted).sum(axis=-1))-shifted[np.arange(len(labels)),labels])
np.testing.assert_allclose(loss(model,test_x,test_y),reference_loss,rtol=1e-5,atol=1e-6)
confusion=np.zeros((2,2),dtype=int)
np.add.at(confusion,(labels,predictions),1)
accuracy=np.mean(predictions==labels)
assert confusion.sum()==24 and np.isclose(np.trace(confusion)/24,accuracy)
assert accuracy>=.9 and float(loss(model,train_x,train_y))<initial*.3
print('Held-out loss:',reference_loss,'accuracy:',accuracy,'confusion:\n',confusion)
```

The output is a measured CPU result for synthetic bars. No MNIST, TPU, or throughput claim is implied.

## Run the example

```python
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def bars(seed,n,noise=.08):
    rng=np.random.default_rng(seed)
    images=np.zeros((n,8,8,1),np.float32)
    labels=np.arange(n,dtype=np.int32)%2
    for i,label in enumerate(labels):
        position=int(rng.integers(2,6))
        if label==0:images[i,position,:,0]=1.
        else:images[i,:,position,0]=1.
    images+=rng.normal(0,noise,images.shape).astype(np.float32)
    return jnp.array(images),jnp.array(labels)
train_x,train_y=bars(20,48)
test_x,test_y=bars(21,24)
assert train_x.shape==(48,8,8,1) and test_x.shape==(24,8,8,1)
assert not np.array_equal(np.asarray(train_x[:24]),np.asarray(test_x))

class BarCNN(nnx.Module):
    def __init__(self):
        rngs=nnx.Rngs(0)
        self.conv=nnx.Conv(1,4,(3,3),padding='VALID',rngs=rngs)
        self.head=nnx.Linear(4,2,rngs=rngs)
    def __call__(self,x):
        features=jnp.mean(nnx.relu(self.conv(x)),axis=(1,2))
        return self.head(features)
model=BarCNN()
raw=model.conv(train_x[:1])
assert raw.shape==(1,6,6,4)
patch=np.asarray(train_x[0,:3,:3,:])
kernel=np.asarray(model.conv.kernel[...])
expected=np.sum(patch*kernel[:,:,:,0])+np.asarray(model.conv.bias[...])[0]
np.testing.assert_allclose(raw[0,0,0,0],expected,rtol=1e-5,atol=1e-6)
def loss(m,x,y):return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(x),y))
optimizer=nnx.Optimizer(model,optax.adam(.03),wrt=nnx.Param)
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(loss)(m,x,y)
    o.update(m,grads)
    return value

initial=float(loss(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
scores=np.asarray(model(test_x));labels=np.asarray(test_y)
predictions=scores.argmax(axis=-1)
shifted=scores-scores.max(axis=-1,keepdims=True)
reference_loss=np.mean(np.log(np.exp(shifted).sum(axis=-1))-shifted[np.arange(len(labels)),labels])
np.testing.assert_allclose(loss(model,test_x,test_y),reference_loss,rtol=1e-5,atol=1e-6)
confusion=np.zeros((2,2),dtype=int)
np.add.at(confusion,(labels,predictions),1)
accuracy=np.mean(predictions==labels)
assert confusion.sum()==24 and np.isclose(np.trace(confusion)/24,accuracy)
assert accuracy>=.9 and float(loss(model,train_x,train_y))<initial*.3
print('Held-out loss:',reference_loss,'accuracy:',accuracy,'confusion:\n',confusion)
```

Expected: Held-out accuracy is at least $0.9$ for the fixed fixture; confusion counts total $24$. The runnable script prints the actual measured loss and counts.

## See the image task and its classification errors

**Predict:** Which axis of the confusion matrix is the actual label?

![See the image task and its classification errors](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper panel is one held-out image, with pixel columns across and rows down. Its dark horizontal stripe is near row $3$; the faint variation elsewhere is added noise. This color bar measures pixel value, so the picture is an input to the classifier.

The lower panel summarizes all held-out predictions. Rows are actual classes and columns are predicted classes. The two diagonal cells contain $12$ each; both off-diagonal cells contain zero. The lower color bar measures example count, which is a different quantity from pixel intensity.

### Connect it to the computation

The upper-left confusion cell counts horizontal images classified as horizontal; the lower-right counts vertical images classified as vertical. Off-diagonal counts would identify the two kinds of confusion. Here the $24$ held-out images are all classified correctly.

Use the sample image to understand the task and the confusion matrix to understand the observed errors. The single image cannot establish accuracy, and a zero in the matrix means no observed mistakes of that kind in this set, not that such a mistake is impossible. This is a small synthetic stripe task, not evidence of general image recognition.

```python
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Held-out image 0', 'values': test_x[0, :, :, 0].tolist(), 'unit': 'pixel value'}, {'kind': 'heatmap', 'title': 'Held-out confusion', 'values': confusion.tolist(), 'rows': ['actual horizontal', 'actual vertical'], 'columns': ['pred. horizontal', 'pred. vertical'], 'unit': 'example count'}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:23:44.342965+00:00. JAX 0.9.2.

```text
Held-out loss: 0.015700625 accuracy: 1.0 confusion:
 [[12  0]
 [ 0 12]]
Held-out loss: 0.015700625 accuracy: 1.0 confusion:
 [[12  0]
 [ 0 12]]
True-class counts: [12 12] row-normalized confusion: [[1. 0.]
 [0. 1.]]
Absent class recall: undefined, not zero.
PASS: networks-04

```

## Trace a second patch and a second filter

**Predict before running:** If the spatial location changes to $(2,3)$, which input patch produces that output?

```python
patch=np.asarray(test_x[0,2:5,3:6,:])
weights=np.asarray(model.conv.kernel[...])[:,:,:,2]
expected=np.sum(patch*weights)+np.asarray(model.conv.bias[...])[2]
np.testing.assert_allclose(model.conv(test_x[:1])[0,2,3,2],expected,rtol=1e-5,atol=1e-6)
```

**Expected:** The output uses rows $2$–$4$ and columns $3$–$5$ of the input.

The same learned filter is reused at each valid location; changing the output coordinate changes the patch, not the filter.

## Predict the chance classifier

**Predict before running:** On a balanced held-out split, what is the accuracy of always predicting class zero?

```python
always_zero=np.zeros_like(labels)
baseline=np.mean(always_zero==labels)
assert baseline==.5
baseline_confusion=np.zeros((2,2),dtype=int)
np.add.at(baseline_confusion,(labels,always_zero),1)
np.testing.assert_array_equal(baseline_confusion,np.array([[12,0],[12,0]]))
```

**Expected:** Accuracy is $0.5$ with confusion $[[12,0]$,$[12,0]]$.

A majority or constant-label baseline is part of evaluation. A good-looking class-specific count alone can hide a useless predictor.

## Make it yours

Apply the trained model to one $10\times10$ image. Predict intermediate shape and parameter count before executing. Do not claim accuracy from an unlabeled input.

<details><summary>Reference solution</summary>

```python
larger=jnp.zeros((1,10,10,1));larger=larger.at[0,4,:,0].set(1.)
assert model.conv(larger).shape==(1,8,8,4)
assert model(larger).shape==(1,2)
assert sum(a.size for a in jax.tree.leaves(nnx.state(model,nnx.Param)))==50
```

</details>

## Read a confusion matrix as conditional rates

**Transfer**

Normalize each confusion-matrix row by its true-class count. Explain the diagonal and verify that weighting class recalls by class counts reproduces overall accuracy. Then remove a class in a constructed case and report its recall as undefined.

<details><summary>Hint</summary>

Keep the integer counts; a normalized row without a count hides how much evidence supports it.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
counts=confusion.sum(axis=1)
rates=np.divide(confusion,counts[:,None],out=np.full(confusion.shape,np.nan),where=counts[:,None]!=0)
np.testing.assert_allclose(rates.sum(axis=1),np.ones(2))
np.testing.assert_allclose((np.diag(rates)*counts).sum()/counts.sum(),accuracy)
missing=np.array([[3,1],[0,0]])
n=missing.sum(axis=1)
missing_rates=np.divide(missing,n[:,None],out=np.full(missing.shape,np.nan),where=n[:,None]!=0)
assert np.isnan(missing_rates[1]).all()
print('True-class counts:',counts,'row-normalized confusion:',rates)
print('Absent class recall: undefined, not zero.')
```

Each row asks where examples of one true class went. A missing class has no observed recall; assigning zero confuses missing evidence with demonstrated failure.

</details>

## Catch a channel-order mistake

**Intermediate**

Transpose NHWC images to NCHW and show why this convolution rejects them. Repair the layout and compare predictions.

<details><summary>Hint</summary>

The final axis is the input-channel dimension for NNX Conv.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
wrong=jnp.transpose(test_x[:2],(0,3,1,2))
caught=False
try:model(wrong)
except (ValueError,TypeError):caught=True
assert caught
repaired=jnp.transpose(wrong,(0,2,3,1))
np.testing.assert_allclose(model(repaired),model(test_x[:2]),rtol=1e-6)
```

The mistaken array has eight channels where the layer expects one. Explicit layout conversion is the repair; changing the declared channel count would silently change the model task.

</details>

## Check your understanding

What does spatial mean pooling discard in this classifier?

1. The exact location of local responses
2. The number of output channels
3. All orientation-sensitive features

<details><summary>Answer and explanation</summary>

The exact location of local responses

Pooling summarizes each feature channel over positions. This can support orientation classification, but it cannot preserve exact bar coordinates.

</details>

## Diagnose the result

Before tuning, inspect sample labels and input layout. Validate a convolution patch numerically, inspect prediction counts, and compare the constant baseline. If all predictions choose one class, examine class balance, score ranges, and per-class errors. If held-out samples equal training samples, repair the split before reporting metrics.

## Carry forward

- Generate images with an explicit label rule
- Derive the convolution output shape
- Pool spatial responses and retain class scores
- Read a confusion matrix rather than a success label

## Keep your evidence

Keep the 8-by-8-to-6-by-6 shape diagram, NumPy patch dot product, seeds 20/21, independent held-out cross-entropy, 24-observation confusion matrix, constant-class baseline, and layout failure repair.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Flax NNX Conv API](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html)
- [Optax classification losses](https://optax.readthedocs.io/en/latest/api/losses.html)

