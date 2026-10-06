# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens=jnp.array([[0,2,4],[1,3,4]],jnp.int32)
roles=jnp.array([[False,True,True]]*2)
validate_mask(roles[:,1:],tokens[:,1:].shape)
p=jnp.zeros((5,5));history=[]
def sft_objective(p):
    return -jnp.sum(response_logps(p[tokens],tokens,roles))/jnp.sum(roles[:,1:])
step=jax.jit(jax.value_and_grad(sft_objective))
for _ in range(100):
    value,g=step(p);history.append(float(value));p=p-.5*g
np.testing.assert_allclose(history[0],np.log(5),atol=1e-6)
assert history[-1]<.15
assert np.array_equal(np.argmax(np.asarray(p)[[0,1]],axis=1),[2,3])
# The final position predicts nothing and has no contribution.
base_logits=p[tokens]
changed=base_logits.at[:,-1,:].set(100.)
np.testing.assert_allclose(response_logps(changed,tokens,roles),response_logps(base_logits,tokens,roles))
print('Response-only token NLL initial/final:',history[0],history[-1])
