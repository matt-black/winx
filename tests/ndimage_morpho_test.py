import jax
import jax.numpy as jnp

from winx.ndimage import binary_dilation, binary_erosion

jax.config.update("jax_platform_name", "cpu")


_b = jnp.ones((5, 5))
_b = _b.at[2, 2].set(0)
_b = _b.astype(bool)


def test_binary_dilation():
    d = binary_dilation(_b, None, border_value=True)
    assert jnp.all(d)


def test_binary_erosion():
    e = binary_erosion(_b, None, border_value=False)
    assert jnp.all(jnp.logical_not(e))
