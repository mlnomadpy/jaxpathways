tokens = jnp.array([[0,0,0,0],[1,1,1,1],[2,2,2,2]],jnp.int32)
selected = jnp.array([[False,True,False,False]]*3)
corrupted = corrupt_tokens(tokens,selected,3)
key = jax.random.key(7)
p = {'embedding':jax.random.normal(key,(4,6))*.2,'head':jnp.zeros((6,3))}
loss = lambda p: masked_ce(mlm_logits(p,corrupted),tokens,selected)
step = jax.jit(jax.value_and_grad(loss)); history=[]
for _ in range(100):
    value,g = step(p); history.append(float(value)); p=jax.tree.map(lambda a,b:a-.4*b,p,g)
assert history[-1] < history[0]*.15
np.testing.assert_allclose(history[0],np.log(3),atol=1e-6)
# Changing the clean target after constructing corrupted input cannot change the forward pass.
changed_targets=tokens.at[:,1].set((tokens[:,1]+1)%3)
assert not np.isclose(float(masked_ce(mlm_logits(p,corrupted),changed_targets,selected)),history[-1])
held_mask=jnp.array([[False,False,True,False]]*3)
held_loss=float(masked_ce(mlm_logits(p,corrupt_tokens(tokens,held_mask,3)),tokens,held_mask))
assert held_loss < .2
print('MLM initial/final/changed-mask:',history[0],history[-1],held_loss)
