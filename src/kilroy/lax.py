"""A lax-like interface that can be used to apply filters similar to how convolutions are applied in ``jax.lax``.

Instead of ``conv`` and ``conv_general_dilated``, ``filt`` and ``filt_general_dilated`` are provided.
"""

from collections.abc import Callable
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

from jaxtyping import Array

__all__ = ["filt", "filt_general_dilated"]


def filt(
    x: Array,
    fun: Callable[[Array], Union[Array, Number]],
    window_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[Tuple[int, int]],
) -> Array:
    return filt_general_dilated(
        x, fun, window_shape, window_strides, padding, None, None, 1, 1
    )


def filt_general_dilated(
    x: Array,
    fun: Callable[[Array], Union[Array, Number]],
    window_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[Tuple[int, int]],
    lhs_dilation: Optional[Sequence[int]] = None,
    rhs_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
) -> Array:
    raise NotImplementedError("todo")
