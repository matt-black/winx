"""testing suite for the ndimage `generic_filter` function

tests basic functionality/correctness of filtering coordinates, etc.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import Array

from winx.filter import filter_window

jax.config.update("jax_platform_name", "cpu")


def identity(x: Array) -> float:
    r, c = x.shape[0] // 2, x.shape[1] // 2
    return x[r, c]


def window_identity(x: Array) -> Array:
    return x


def test_padding_2d_valid_odd():
    x = jnp.zeros((5, 5))
    y = filter_window(x, identity, 3, None, "valid", "constant", 1)
    n_row, n_col = y.shape
    assert n_row == 3 and n_col == 3


def test_padding_2d_same_odd():
    x = jnp.zeros((5, 5))
    y = filter_window(x, identity, 3, None, "same", "constant", 1)
    n_row, n_col = y.shape
    assert n_row == 5 and n_col == 5


def test_padding_2d_same_even():
    x = jnp.zeros((5, 5))
    y = filter_window(x, identity, 2, None, "same", "constant", 1)
    nrow, ncol = y.shape
    assert nrow == 5 and ncol == 5


def test_footprint():
    x = jr.normal(jr.key(10), (5, 5))
    fp = jnp.zeros((3, 3)).at[1, 1].set(1).astype(jnp.bool)
    y = filter_window(x, window_identity, 3, fp, "valid", "constant", 0)
    z = filter_window(x, identity, 3, None, "valid", "constant", 0)
    assert jnp.allclose(y, z).item()


def test_vmap_map_match():
    x = jr.normal(jr.key(10), (5, 5))
    fun = Partial(
        filter_window, x, identity, 3, None, "valid", mode="constant", cval=0
    )
    y = fun(batch_size=None)
    z = fun(batch_size=2)
    assert jnp.allclose(y, z).item()
