"""Implement one stage at a time. Reference answers are in ../solution/model.py."""
def predict(params, x):
    raise NotImplementedError('Stage 1: implement a linear prediction')

def loss(params, x, y):
    raise NotImplementedError('Stage 1: implement scalar MSE with shape checks')

def gradient_check(params, x, y, h=1e-2):
    raise NotImplementedError('Stage 2: compare autodiff with independent differences')

def train(params, x, y, rate=0.15, steps=200):
    raise NotImplementedError('Stage 3: carry parameters through a compiled loop')

def report(params, x, y, rate=0.15, steps=200):
    raise NotImplementedError('Stage 4: report fit, held-out error, and numerical health')
