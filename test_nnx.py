import jax
from flax import nnx
class MyModule(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jax.numpy.zeros(()))
m = MyModule()
graphdef, state = nnx.split(m)
print(type(graphdef), type(state))
m2 = nnx.merge(graphdef, state)
# Modify m2
m2.w.value = jax.numpy.ones(())
print(m.w.value, m2.w.value)
