"""testing suite for the ndimage api"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy
from jaxtyping import Array
from scipy import ndimage as ndi
from scipy.datasets import ascent

from kilroy.ndimage import generic_filter, median_filter

jax.config.update("jax_platform_name", "cpu")


def identity(x: Array) -> float:
    r, c = x.shape[0] // 2, x.shape[1] // 2
    return x[r, c]


def window_identity(x: Array) -> Array:
    return x


def test_padding_2d_valid():
    x = jnp.zeros((5, 5))
    y = generic_filter(x, identity, 3, None, "valid", mode="constant", cval=1)
    n_row, n_col = y.shape
    assert n_row == 3 and n_col == 3


def test_padding_2d_same():
    x = jnp.zeros((5, 5))
    y = generic_filter(x, identity, 3, None, "same", mode="constant", cval=1)
    n_row, n_col = y.shape
    assert n_row == 5 and n_col == 5


def test_footprint():
    x = jr.normal(jr.key(10), (5, 5))
    fp = jnp.zeros((3, 3)).at[1, 1].set(1).astype(jnp.bool)
    y = generic_filter(x, window_identity, 3, fp, "valid", "constant", 0)
    z = generic_filter(x, identity, 3, None, "valid", "constant", 0)
    assert jnp.allclose(y, z)


def test_median_same():
    a = ascent().astype(numpy.float32)
    sc = ndi.median_filter(a, 3, None, None, "constant", cval=0)
    my = median_filter(jnp.asarray(a), 3, None, "same", "constant", cval=0)
    my = numpy.asarray(my)
    assert numpy.allclose(sc, my)
