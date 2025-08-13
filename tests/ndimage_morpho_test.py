import jax
import jax.numpy as jnp

from winx.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_hit_or_miss,
    grey_dilation,
    grey_erosion,
)

jax.config.update("jax_platform_name", "cpu")


_b = jnp.ones((5, 5))
_b = _b.at[2, 2].set(0)
_b = _b.astype(bool)


def test_binary_dilation():
    d = binary_dilation(_b, 3, None, border_value=True)
    assert jnp.all(d)


def test_binary_erosion():
    e = binary_erosion(_b, 3, None, border_value=False)
    assert jnp.all(jnp.logical_not(e))


def test_grey_dilation():
    x = jnp.array([[5, 2, 3], [7, 1, 4], [0, 8, 5]])
    res = jnp.array([[7, 7, 4], [8, 8, 8], [8, 8, 8]])
    y = grey_dilation(x, 3, None, 1, 0, 0, None)
    assert jnp.array_equal(y, res)


def test_grey_erosion():
    x = jnp.array([[5, 2, 3], [7, 1, 4], [0, 8, 5]])
    res = jnp.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
    y = grey_erosion(x, 3, None, 1, 10, 0, None)
    assert jnp.array_equal(y, res)


def test_binary_hit_or_miss_origin0():
    a = jnp.zeros((7, 7), dtype=int)
    a = a.at[1, 1].set(1)
    a = a.at[2:4, 2:4].set(1)
    a = a.at[4:6, 4:6].set(1)
    a = a.astype(jnp.bool)
    s1 = jnp.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]]).astype(jnp.bool)
    y = binary_hit_or_miss(a, s1, None, 0, 0, None)
    res = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(jnp.bool)
    assert jnp.array_equal(y, res)
