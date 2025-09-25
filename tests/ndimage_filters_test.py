"""testing suite for the ndimage api

tests that our implementations of the filter return the same results as those given by `scipy.ndimage`
"""

import jax
import jax.numpy as jnp
import numpy
from jaxtyping import Array
from scipy import ndimage as ndi
from scipy.datasets import ascent

from winx.ndimage import (
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter,
)

jax.config.update("jax_platform_name", "cpu")


_a = ascent().astype(numpy.float32)


def identity(x: Array) -> Array:
    r, c = x.shape[0] // 2, x.shape[1] // 2
    return x[r, c]


def window_identity(x: Array) -> Array:
    return x


def test_median_same():
    sc = ndi.median_filter(_a, 3, None, None, "constant", 0)
    my = median_filter(jnp.asarray(_a), 3, None, "constant", 0)
    my = numpy.asarray(my)
    assert numpy.allclose(sc, my)


def test_median_same_pad_not_constant():
    sc = ndi.median_filter(_a, 3, None, None, mode="reflect")
    my = median_filter(jnp.asarray(_a), 3, None, "symmetric")
    my = numpy.asarray(my)
    assert numpy.allclose(sc, my)


def test_maximum_same():
    sc = ndi.maximum_filter(_a, 3, None, None, "constant", 0)
    my = maximum_filter(jnp.asarray(_a), 3, None, "constant", 0)
    my = numpy.asarray(my)
    assert numpy.allclose(sc, my)


def test_minimum_same():
    sc = ndi.minimum_filter(_a, 3, None, None, "constant", 0)
    my = minimum_filter(jnp.asarray(_a), 3, None, "constant", 0)
    my = numpy.asarray(my)
    assert numpy.allclose(sc, my)


# def test_percentile_same():
#     sc = ndi.percentile_filter(_a, 60, 5, None, None, 'constant', 0)
#     my = percentile_filter(_a, 60, 5, None, 'same', 'constant', 0)
#     my = numpy.asarray(my)
#     assert numpy.allclose(sc, my)


# def test_rank_same():
#     sc = ndi.rank_filter(_a, 2, 5, None, None, 'constant', 0)
#     my = rank_filter(_a, 2, 5, None, 'same', 'constant', 0)
#     my = numpy.asarray(my)
#     assert numpy.allclose(sc, my)


def test_uniform_same():
    sc = ndi.uniform_filter(_a, 3, None, "constant", 0)
    my = uniform_filter(jnp.asarray(_a), 3, None, "constant", 0)
    my = numpy.asarray(my)
    assert numpy.allclose(sc, my)
