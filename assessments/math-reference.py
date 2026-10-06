import itertools
import numpy as np
import jax
import jax.numpy as jnp
import optax
X=np.array([[1.,0.],[1.,1.],[1.,2.]])
y=np.array([1.,2.,2.])
w=np.linalg.lstsq(X,y,rcond=None)[0]
np.testing.assert_allclose(w,[7/6,.5])
np.testing.assert_allclose(np.mean((X@w-y)**2),1/18)
H=2*X.T@X/3
np.testing.assert_allclose(np.linalg.eigvalsh(H),[.55848156,4.77485177],rtol=1e-7)
for initial in [np.zeros(2),np.array([.3,-.4])]:
 loss=lambda w:jnp.mean((jnp.asarray(X)@w-jnp.asarray(y))**2)
 np.testing.assert_allclose(jax.grad(loss)(jnp.asarray(initial)),2*X.T@(X@initial-y)/3,rtol=1e-6)
for rate in [.1,.5]:
 current=np.zeros(2);history=[]
 for i in range(20):
  history.append(np.mean((X@current-y)**2));current-=rate*2*X.T@(X@current-y)/3
 assert (history[-1]<history[0]) == (rate==.1)
ridge=np.linalg.solve(X.T@X+3*np.eye(2),X.T@y)
np.testing.assert_allclose(ridge,[22/39,21/39])
np.testing.assert_allclose((np.array([1,3])@ridge-2.5)**2,625/6084)
g=-2*y[:,None]*X
cov=np.cov(g,rowvar=False,bias=True)
np.testing.assert_allclose(cov,[[8/9,8/3],[8/3,32/3]])
for pairs,factor in [(itertools.product(range(3),repeat=2),.5),(itertools.combinations(range(3),2),.25)]:
 batches=np.array([g[list(pair)].mean(0) for pair in pairs]);np.testing.assert_allclose(batches.mean(0),g.mean(0));np.testing.assert_allclose(np.cov(batches,rowvar=False,bias=True),factor*cov)
for clipped in [False,True]:
 for rates in [[.1,.1],[.1,.01]]:
  tx=optax.chain(*([optax.clip_by_global_norm(1.)] if clipped else []),optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8))
  params=jnp.zeros(2);state=tx.init(params);m=np.zeros(2);v=np.zeros(2)
  for i,(grad,rate) in enumerate(zip([[3.,4.],[.3,.4]],rates),1):
   g=np.asarray(grad);g=g*min(1,1/np.linalg.norm(g)) if clipped else g
   m=.9*m+.1*g;v=.999*v+.001*g*g
   expected=-rate*(m/(1-.9**i))/(np.sqrt(v/(1-.999**i))+1e-8)
   update,state=tx.update(jnp.array(grad),state,params)
   np.testing.assert_allclose(-rate*np.asarray(update),expected,rtol=2e-5,atol=1e-7)
   params=params-rate*update
print('PASS: geometry, curvature, ridge, exact sampling covariance and two-step Adam references')
