import jax
import numpy
from jaxtyping import Array
from scipy.datasets import ascent

from kilroy.ndimage import generic_filter

jax.config.update("jax_platform_name", "cpu")


_a = ascent().astype(numpy.float32)
_b = _a[None, ...]
_c = _a[None, None, ...]
_d = _a[None, :, :, None]
_e = _a[:, None, :, None]


def identity(x: Array) -> float:
    r, c = x.shape[0] // 2, x.shape[1] // 2
    return x[r, c]


def test_axes_same_shape0():
    y = generic_filter(_b, identity, 3, None, "constant", 0, 0, axes=(1, 2))
    assert len(y.shape) == 3 and all(
        [s0 == s1 for s0, s1 in zip(_b.shape, y.shape)]
    )


def test_axes_same_shape1():
    z = generic_filter(_c, identity, 3, None, "constant", 0, 0, axes=(2, 3))
    assert len(z.shape) == 4 and all(
        [s0 == s1 for s0, s1 in zip(_c.shape, z.shape)]
    )


def test_axes_same_shape2():
    w = generic_filter(_d, identity, 3, None, "constant", 0, 0, axes=(1, 2))
    assert len(w.shape) == 4 and all(
        [s0 == s1] for s0, s1 in zip(_d.shape, w.shape)
    )


def test_axes_right_value():
    v = generic_filter(_d, identity, 3, None, "constant", 0, 0, axes=(1, 2))[
        0, :, :, 0
    ]
    assert numpy.all(v == _a)


def test_axes_mixedin_shape_and_value():
    u = generic_filter(_e, identity, 3, None, "constant", 0, 0, axes=(0, 2))
    u2 = u[:, 0, :, 0]
    assert (
        numpy.all(u2 == _a)
        and len(u.shape) == 4
        and all([s0 == s1 for s0, s1 in zip(_e.shape, u.shape)])
    )
