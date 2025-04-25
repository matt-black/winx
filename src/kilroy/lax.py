"""A lax-like interface that can be used to apply filters similar to how convolutions are applied in ``jax.lax``.

Instead of ``conv`` and ``conv_general_dilated``, ``filt`` and ``filt_general_dilated`` are provided.
"""

from collections.abc import Callable
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import jax
from jax.tree_util import Partial
from jaxtyping import Array

from .filter import filter_window

__all__ = ["filt", "filt_general_dilated"]


def filt(
    x: Array,
    fun: Callable[[Array], Union[Array, Number]],
    window_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[Tuple[int, int]],
) -> Array:
    """Convenience wrapper around `filt_general_dilated`.

    Args:
        x (Array): A rank *n+2* dimensional input array.
        fun (Callable[[Array], Union[Array, Number]]): Filtering function to be applied to each window.
        window_shape (Sequence[int]): Shape of the filtering window.
        window_strides (Sequence[int]): Inter-window strides.
        padding (str | Sequence[Tuple[int, int]]): Either the string 'same' or 'valid' or a sequence of tuples specifying padding applied to the left and right side of each dimension. Zero padding is used.

    Returns:
        Array: Filtered array.
    """
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
    """General n-dimensional filtering operation, with optional dilation.

    Args:
        x (Array): Rank *n+2* dimensional input array.
        fun (Callable[[Array], Union[Array, Number]]): Filtering function to be applied to each window.
        window_shape (Sequence[int]): Shape of the window.
        window_strides (Sequence[int]): Inter-window strides.
        padding (str | Sequence[Tuple[int, int]]): Either the string 'same' or 'valid' or a sequence of tuples specifying padding applied to the left and right side of each dimension. Zero padding is used.
        lhs_dilation (Optional[Sequence[int]], optional): The dilation factor to apply in each spatial dimension of the input. If None, no dilation is done. Defaults to None.
        rhs_dilation (Optional[Sequence[int]], optional): The dilation factor to apply in each spatial dimension to the window (pre-filtering). If None, no dilation is done. Defaults to None.
        feature_group_count (int, optional): TODO. Defaults to 1.
        batch_group_count (int, optional): TODO. Defaults to 1.

    Raises:
        ValueError: if input array, `x` is not at least 3-dimensional.
        NotImplementedError: if `feature_group_count` or `batch_group_count` > 1.

    Returns:
        Array: Filtered array.

    Notes:
        Right hand side dilation (`rhs_dilation`) is done, under the hood, by constructing a footprint similar to the mechanism used in `ndimage.generic_filter`. Thus, the filtering function will receive a 1D (sorted) array of values, not an ND array.
    """
    if len(x.shape) < 3:
        raise ValueError("invalid number of dimensions for input array.")

    n_batch = x.shape[0]
    n_batch_grps = n_batch // batch_group_count
    n_feat = x.shape[1]
    n_feat_grps = n_feat // feature_group_count
    new_window_shape = tuple([n_batch_grps, n_feat_grps] + list(window_shape))
    fun = Partial(
        filter_window,
        fun=fun,
        size=new_window_shape,
        footprint=None,
        padding=padding,
        mode="constant",
        cval=0,
        origin=0,
        axes=None,
        batch_size=None,
        window_strides=window_strides,
        base_dilation=lhs_dilation,
        window_dilation=rhs_dilation,
    )
    if feature_group_count == 1 and batch_group_count == 1:
        return jax.vmap(jax.vmap(fun, 0, 0), 1, 1)(x)
    else:
        raise NotImplementedError("to finish")
