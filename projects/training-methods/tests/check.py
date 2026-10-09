"""Cumulative objective checks on changed data plus actual synthetic training runs."""
import argparse,importlib.util
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser()
a.add_argument('--implementation',default='solution')
a.add_argument('--stage',default='all')
args=a.parse_args()
path=ROOT/'solution/methods.py' if args.implementation=='solution' else Path(args.implementation).resolve()
spec=importlib.util.spec_from_file_location('learner',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
last=8 if args.stage=='all' else int(args.stage)
assert 1<=last<=8
rng=np.random.default_rng(33)
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('invalid contract accepted')
def stable_logp(a):
    a=np.asarray(a,np.float64)
    z=a-a.max(-1,keepdims=True)
    return z-np.log(np.exp(z).sum(-1,keepdims=True))
for stage,name in enumerate(['mlm','mim','contrastive','sft','lora','reward','rlhf','dpo'],1):
    if stage>last:break
    if stage==1:
        for b,t,v in [(2,3,4),(1,5,2)]:
            logits=rng.normal(size=(b,t,v)).astype(np.float32)
            targets=rng.integers(v,size=(b,t))
            mask=rng.uniform(size=(b,t))>.4
            mask[0,0]=True
            expected=np.mean([-stable_logp(logits)[i,j,targets[i,j]] for i,j in zip(*np.where(mask))])
            np.testing.assert_allclose(m.masked_ce(logits,targets,mask),expected,atol=1e-6)
            corrupted=m.corrupt_tokens(jnp.asarray(targets),jnp.asarray(mask),v)
            np.testing.assert_array_equal(corrupted,np.where(mask,v,targets))
        reject(lambda:m.validate_mask(np.zeros((2,3),bool),(2,3)))
        reject(lambda:m.validate_mask(np.ones((2,3)),(2,3)))
    elif stage==2:
        images=np.arange(2*6*4*2,dtype=np.float32).reshape(2,6,4,2)
        patches=m.patchify(jnp.asarray(images),2)
        expected=np.array([[images[b,i:i+2,j:j+2,:].reshape(-1) for i in range(0,6,2) for j in range(0,4,2)] for b in range(2)])
        np.testing.assert_array_equal(patches,expected)
        hidden=np.array([[True,False,False,True,False,True],[False,True,True,False,True,False]])
        pred=np.asarray(patches)+np.arange(6)[None,:,None]/10
        oracle=np.mean([np.mean((pred[b,k]-expected[b,k])**2) for b,k in zip(*np.where(hidden))])
        np.testing.assert_allclose(m.masked_mse(pred,patches,hidden),oracle,atol=2e-6)
        reject(lambda:m.patchify(jnp.zeros((1,5,4,1))))
    elif stage==3:
        for n in [2,5]:
            left=rng.normal(size=(n,3))
            right=rng.normal(size=(n,3))
            tau=.4
            score=(left/np.linalg.norm(left,axis=1,keepdims=True))@(right/np.linalg.norm(right,axis=1,keepdims=True)).T/tau
            oracle=-.5*(np.diag(stable_logp(score)).mean()+np.diag(stable_logp(score.T)).mean())
            np.testing.assert_allclose(m.paired_contrastive(left,right,tau),oracle,atol=1e-6)
    elif stage==4:
        tokens=np.array([[0,1,2,3],[3,2,1,0]])
        mask=np.array([[False,False,True,True],[False,True,False,True]])
        logits=rng.normal(size=(2,4,4)).astype(np.float32)
        oracle=np.array([sum(stable_logp(logits)[b,t,tokens[b,t+1]] for t in range(3) if mask[b,t+1]) for b in range(2)])
        np.testing.assert_allclose(m.response_logps(logits,tokens,mask),oracle,atol=1e-6)
    elif stage==5:
        for rank in [1,2]:
            x=rng.normal(size=(3,5))
            base=rng.normal(size=(5,4))
            a=rng.normal(size=(5,rank))
            b=rng.normal(size=(rank,4))
            oracle=x@(base+3./rank*a@b)
            np.testing.assert_allclose(m.lora_forward(x,base,a,b,3.),oracle,rtol=2e-6,atol=2e-6)
    elif stage==6:
        w=rng.normal(size=3)
        chosen=rng.normal(size=(5,3))
        rejected=rng.normal(size=(5,3))
        oracle=np.logaddexp(0,-(chosen-rejected)@w).mean()
        np.testing.assert_allclose(m.preference_loss(w,chosen,rejected),oracle,atol=1e-6)
        # A coordinate finite difference catches the sign and reduction, independently.
        eps=1e-4
        direction=np.array([1.,0.,0.])
        fn=lambda z:np.logaddexp(0,-(chosen-rejected)@z).mean()
        fd=(fn(w+eps*direction)-fn(w-eps*direction))/(2*eps)
        np.testing.assert_allclose(jax.grad(m.preference_loss)(jnp.asarray(w),chosen,rejected)[0],fd,atol=1e-5)
    elif stage==7:
        theta=np.array([.3,-.2,.1])
        old=stable_logp(np.array([0.,.1,-.1]))
        ref=stable_logp(np.array([.2,0.,-.1]))
        actions=np.array([0,2,1,2])
        adv=np.array([1.,-2.,.3,-.5])
        lp=stable_logp(theta)
        ratios=np.exp(lp[actions]-old[actions])
        kl=np.sum(np.exp(lp)*(lp-ref))
        oracle=-np.minimum(ratios*adv,np.clip(ratios,.8,1.2)*adv).mean()+.3*kl
        np.testing.assert_allclose(m.ppo_loss(theta,old[actions],actions,adv,ref,.3,.2),oracle,atol=1e-6)
    else:
        policy=stable_logp(np.array([1.,-.4,.3]))
        ref=stable_logp(np.array([-.2,.2,.4]))
        chosen=np.array([0,2])
        rejected=np.array([1,1])
        beta=.7
        margin=(policy[chosen]-policy[rejected])-(ref[chosen]-ref[rejected])
        np.testing.assert_allclose(m.dpo_loss(policy,ref,chosen,rejected,beta),np.logaddexp(0,-beta*margin).mean(),atol=1e-6)
    # Training uses the supplied implementation, not an imported reference solution.
    namespace=dict(vars(m))
    exec(compile((ROOT/'labs'/f'{name}.py').read_text(),name,'exec'),namespace)
    print(f'PASS stage {stage}: {name} independent contract and actual CPU training',flush=True)
