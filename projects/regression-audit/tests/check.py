"""Stage checks use held-out expectations, structural checks, and an unstable run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--implementation',choices=['starter','solution'],default='starter')
parser.add_argument('--stage',choices=['1','2','3','4','all'],default='all')
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('learner_model',ROOT/args.implementation/'model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
x=jnp.linspace(-1.,1.,21);y=2*x+1
start={'weight':jnp.array(0.),'bias':jnp.array(0.)}
try:
    if args.stage in ['1','all']:
        actual=model.predict({'weight':jnp.array(2.),'bias':jnp.array(1.)},x)
        assert actual.shape==y.shape and jnp.allclose(actual,y), 'Prediction must match the known linear function'
        assert jnp.allclose(model.loss(start,x,y),jnp.mean(y**2)), 'MSE must match an independent calculation'
        try:model.loss(start,x,y[:,None])
        except ValueError:pass
        else:raise AssertionError('Reject mismatched targets instead of silently broadcasting')
        print('PASS stage 1: prediction, objective, and shape contract')
    if args.stage in ['2','all']:
        for params in [start,{'weight':jnp.array(-1.2),'bias':jnp.array(0.3)}]:
            automatic,finite=model.gradient_check(params,x,y)
            residual=params['weight']*x+params['bias']-y
            expected={'weight':2*jnp.mean(residual*x),'bias':2*jnp.mean(residual)}
            for key in params:assert jnp.allclose(automatic[key],expected[key],atol=1e-5),f'Analytic gradient mismatch: {key}'
            for key in params:assert jnp.allclose(automatic[key],finite[key],atol=1e-3,rtol=1e-3),f'Gradient mismatch: {key}'
        print('PASS stage 2: independent gradients at two parameter settings')
    if args.stage in ['3','all']:
        fitted,history=model.train(start,x,y)
        assert history.shape==(200,), 'Record one loss per update'
        assert model.loss(fitted,x,y)<1e-8, 'Default training must fit the known function'
        assert jnp.allclose(fitted['weight'],2.,atol=1e-4) and jnp.allclose(fitted['bias'],1.,atol=1e-4)
        other,_=model.train(start,x,-1.5*x+0.25)
        assert jnp.allclose(other['weight'],-1.5,atol=1e-4) and jnp.allclose(other['bias'],0.25,atol=1e-4), 'Train must respond to the supplied targets'
        graph=jax.make_jaxpr(lambda p:model.train(p,x,y,steps=4))(start).jaxpr
        assert any(eq.primitive.name=='scan' for eq in graph.eqns), 'Express the repeated update as scan'
        print('PASS stage 3: compiled training and parameter recovery')
    if args.stage in ['4','all']:
        report=model.report(start,x,y)
        assert report['heldout_loss']<1e-8 and report['finite'], 'Verify an unseen input set and numerical health'
        replay=model.report(start,x,y)
        assert report==replay, 'Same explicit inputs should reproduce the report'
        unstable=model.report(start,x,y,rate=1.1,steps=30)
        assert unstable['training_loss']>unstable['initial_loss'], 'The report must expose unstable training'
        print('PASS stage 4: held-out checks, replay, and failure diagnosis')
        print(json.dumps(report,indent=2))
except (AssertionError,NotImplementedError) as error:
    print(f'FAIL: {error}');raise SystemExit(1)
