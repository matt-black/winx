"""A ``scipy.ndimage``-like interface."""

from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array

from ..filter import filter_window


def generic_filter(
    x: Array,
    fun: Callable[[Array], Union[Array, Number]],
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Calculate a multidimensional filter using the given function.

    At each element of the input array, a window is generated, centered around that element, and the function is evaluated on that window.

    Args:
        x (Array): The input array.
        fun (Callable[[Array],Union[Array,Number]]): Function to apply at each element.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Raises:
        ValueError: If ``size`` is specified as a sequence and the number of elements is not the same as the number of dimensions.

    Returns:
        Array: The filtered array.

    Notes:
        When using the `footprint` argument, the values passed to the filtering function will be a 1d array sorted in descending order.
    """
    stride = [
        1,
    ] * len(x.shape)
    if axes is None:
        return filter_window(
            x,
            fun,
            size,
            footprint,
            "same",
            mode,
            cval,
            origin,
            stride,
            None,
            None,
            None,
        )
    else:
        all_ax = list(range(len(x.shape)))
        map_ax = [a for a in all_ax if a not in axes]
        vmap_ax, rest_vmap_ax = map_ax[0], map_ax[1:]
        if len(rest_vmap_ax) == 0:
            pfun = Partial(
                filter_window,
                fun=fun,
                size=size,
                footprint=footprint,
                padding="same",
                mode=mode,
                cval=cval,
                origin=origin,
                window_strides=stride,
                base_dilation=None,
                window_dilation=None,
                batch_size=None,
            )
            return jax.vmap(pfun, vmap_ax, vmap_ax)(x)
        else:
            new_ax = [(a if a < vmap_ax else a - 1) for a in axes]
            print([vmap_ax, new_ax])
            pfun = Partial(
                generic_filter,
                fun=fun,
                size=size,
                footprint=footprint,
                mode=mode,
                cval=cval,
                origin=origin,
                axes=new_ax,
            )
            return jax.vmap(pfun, vmap_ax, vmap_ax)(x)


def maximum_filter(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional maximum filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Maximum-filtered array.
    """
    return generic_filter(x, jnp.max, size, footprint, mode, cval, origin, axes)


def median_filter(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional median filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Median-filtered array.
    """
    return generic_filter(
        x,
        jnp.median,
        size,
        footprint,
        mode,
        cval,
        origin,
        axes,
    )


def minimum_filter(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional minimum filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Minimum-filtered array.
    """
    return generic_filter(x, jnp.min, size, footprint, mode, cval, origin, axes)


def percentile_filter(
    x: Array,
    percentile: Union[Array, Number],
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional percentile filter.

    Args:
        x (Array): The input array.
        percentile (Union[Array, Number]): The percentile of the window to return at each position. Should contain integer or floating point values between 0 and 100.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Percentile-filtered array.
    """
    return generic_filter(
        x,
        Partial(jnp.percentile, q=percentile, method="nearest"),
        size,
        footprint,
        mode,
        cval,
        origin,
        axes,
    )


@partial(jax.jit, static_argnums=(1,))
def _rank(rank: int, x: Array) -> Number:
    return jnp.sort(x)[rank]


def rank_filter(
    x: Array,
    rank: int,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional rank filter.

    Rank filtering takes a window of values, flattens the values, sorts them, and then selects the `rank`th element in the sorted array.

    Args:
        x (Array): The input array.
        rank (int): The rank parameter. May be negative, which will select the `-nth` element in the intermediate sorted array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Rank-filtered array.
    """
    return generic_filter(
        x,
        Partial(_rank, rank),
        size,
        footprint,
        mode,
        cval,
        origin,
        axes,
    )


def uniform_filter(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional uniform (AKA "mean") filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Uniform-filtered array.
    """
    return generic_filter(
        x,
        jnp.mean,
        size,
        footprint,
        mode,
        cval,
        origin,
        axes,
    )


def variance(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Array:
    """Compute a multidimensional variance filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.

    Returns:
        Array: Variance-filtered array.
    """
    return generic_filter(x, jnp.var, size, footprint, mode, cval, origin, axes)
