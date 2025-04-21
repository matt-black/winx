"""A ``scipy.ndimage``-like interface."""

from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Int

from ._util import make_extract_footprint_fun_text, make_extract_fun_text

__all__ = [
    "generic_filter",
    "median_filter",
    "percentile_filter",
    "rank_filter",
    "uniform_filter",
]


def generic_filter(
    x: Array,
    fun: Callable[[Array], Union[Array, Number]],
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    padding: str = "same",
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
    batch_size: Optional[int] = None,
    window_strides: Optional[Sequence[int]] = None,
) -> Array:
    """generic_filter Calculate a multidimensional filter using the given function.

    At each element of the input array, a window is generated, centered around that element, and the function is evaluated on that window.

    Args:
        x (Array): The input array.
        fun (Callable[[Array],Union[Array,Number]]): Function to apply at each element.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        padding (str, optional): Either the strings 'same' or 'valid'. 'same' adds padding to produce the same output size as the input. 'valid' does no padding, so only elements where the window fits completely in the image will be evaluated. Defaults to 'same'.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.
        batch_size (int, optional): Integer specifying the size of the batch for each step to execute in parallel. If specified, `jax.lax.map` will be used. If `None`, `jax.vmap` will be used. Defaults to `None`.

    Raises:
        ValueError: If ``size`` is specified as a sequence and the number of elements is not the same as the number of dimensions.
        ValueError: If any of the dimensions in ``size`` are even.
        NotImplementedError: If ``axes`` is not ``None``.

    Returns:
        Array: The filtered array.

    Notes:
        When using the `footprint` argument, the values passed to the filtering function will be a 1d array sorted in descending order.
    """
    if axes is not None:
        raise NotImplementedError("TODO")
    if footprint is not None:
        size = footprint.shape

    n_dim = len(x.shape)
    # make sure size is specified for all dimensions
    if isinstance(size, int):
        size = [
            size,
        ] * n_dim
    else:
        if len(size) != n_dim:
            raise ValueError(
                f"you specified {len(size)} dims for size but the input array is {n_dim}-dimensional"
            )
    for ws in size:
        if ws % 2 == 0:
            raise ValueError(
                f"window must be odd in all dimensions, shape input was {size}"
            )
    if window_strides is None:
        window_strides = [
            1,
        ] * n_dim
    # make the window extraction function
    window_radii = [ws // 2 for ws in size]
    if footprint is None:
        fun_txt = make_extract_fun_text(n_dim)
    else:
        fun_txt = make_extract_footprint_fun_text(n_dim)
    lcls = locals()
    exec(fun_txt, globals(), lcls)
    extract_window = lcls["extract_window"]
    # pad the input array according to the padding argument
    if padding == "same":
        padding = tuple((r, r) for r in window_radii)
        x = jnp.pad(x, padding, mode=mode, constant_values=cval)
    else:  # padding is assumed 'valid'
        pass
    out_shape = [s - 2 * r for s, r in zip(x.shape, window_radii)]
    # generate an array of coordinates that will be shaped like [prod(x.shape), n_dim]
    # this function works by `vmap`'ing over these coords, extracting a window around
    # the coordinate, and applying the function to that window.
    coords = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(r, s - r, strd)
                for s, r, strd in zip(x.shape, window_radii, window_strides)
            ],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, n_dim)
    # fill in static arguments for the `extract_window` function
    # these are the array to be filtered and the window radii
    # we're left with a function that takes in a coordinate and returns a slice from the array
    if footprint is None:
        get_window: Callable[[Int[Array, " {n_dim}"]], Array] = Partial(
            extract_window, x, *window_radii
        )
    else:
        n_fp_out = jnp.sum(footprint).item()
        get_window: Callable[[Int[Array, " {n_dim}"]], Array] = Partial(
            extract_window, x, footprint, n_fp_out, *window_radii
        )

    # compose the input function with this `get_window` function so that we can
    # vmap over coordinates instead of windows, themselves
    def compose_fun(coord: Int[Array, " {n_dim}"]) -> Union[Array, Number]:
        return fun(get_window(coord))

    if batch_size is None:
        return jax.vmap(compose_fun, 0, 0)(coords).reshape(out_shape)
    else:
        return jax.lax.map(compose_fun, coords, batch_size=batch_size).reshape(
            out_shape
        )


def median_filter(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    padding: str = "same",
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
    batch_size: Optional[int] = None,
) -> Array:
    """median_filter Compute a multidimensional median filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        padding (str, optional): Either the strings 'same' or 'valid'. 'same' adds padding to produce the same output size as the input. 'valid' does no padding, so only elements where the window fits completely in the image will be evaluated. Defaults to 'same'.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.
        batch_size (int, optional): Integer specifying the size of the batch for each step to execute in parallel. If specified, `jax.lax.map` will be used. If `None`, `jax.vmap` will be used. Defaults to `None`.

    Returns:
        Array: Median-filtered array.
    """
    return generic_filter(
        x,
        jnp.median,
        size,
        footprint,
        padding,
        mode,
        cval,
        origin,
        axes,
        batch_size,
    )


def percentile_filter(
    x: Array,
    percentile: Union[Array, Number],
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    padding: str = "same",
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
    batch_size: Optional[int] = None,
) -> Array:
    """percentile_filter Compute a multidimensional percentile filter.

    Args:
        x (Array): The input array.
        percentile (Union[Array, Number]): The percentile of the window to return at each position. Should contain integer or floating point values between 0 and 100.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        padding (str, optional): Either the strings 'same' or 'valid'. 'same' adds padding to produce the same output size as the input. 'valid' does no padding, so only elements where the window fits completely in the image will be evaluated. Defaults to 'same'.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.
        batch_size (int, optional): Integer specifying the size of the batch for each step to execute in parallel. If specified, `jax.lax.map` will be used. If `None`, `jax.vmap` will be used. Defaults to `None`.

    Returns:
        Array: Percentile-filtered array.
    """
    return generic_filter(
        x,
        Partial(jnp.percentile, q=percentile, method="nearest"),
        size,
        footprint,
        padding,
        mode,
        cval,
        origin,
        axes,
        batch_size,
    )


@partial(jax.jit, static_argnums=(1,))
def _rank(rank: int, x: Array) -> Number:
    return jnp.sort(x)[rank]


def rank_filter(
    x: Array,
    rank: int,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    padding: str = "same",
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
    batch_size: Optional[int] = None,
) -> Array:
    """rank_filter Compute a multidimensional rank filter.

    Rank filtering takes a window of values, flattens the values, sorts them, and then selects the `rank`th element in the sorted array.

    Args:
        x (Array): The input array.
        rank (int): The rank parameter. May be negative, which will select the `-nth` element in the intermediate sorted array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        padding (str, optional): Either the strings 'same' or 'valid'. 'same' adds padding to produce the same output size as the input. 'valid' does no padding, so only elements where the window fits completely in the image will be evaluated. Defaults to 'same'.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.
        batch_size (int, optional): Integer specifying the size of the batch for each step to execute in parallel. If specified, `jax.lax.map` will be used. If `None`, `jax.vmap` will be used. Defaults to `None`.

    Returns:
        Array: Rank-filtered array.
    """
    return generic_filter(
        x,
        Partial(_rank, rank),
        size,
        footprint,
        padding,
        mode,
        cval,
        origin,
        axes,
        batch_size,
    )


def uniform_filter(
    x: Array,
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    padding: str = "same",
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
    batch_size: Optional[int] = None,
) -> Array:
    """uniform_filter Compute a multidimensional uniform (AKA "mean") filter.

    Args:
        x (Array): The input array.
        size (int | Sequence[int]): Shape that is taken from the input array at every element position to define the input to the filter function. If an integer, a square (cubic, etc.) window with that size in each dimension will be used.
        footprint (Array, optional): A boolean array that delineates a window as well as which of the elements in that window gets passed to the filter function. If this is used, the values selected by the footprint are passed to ``fun`` as a 1-dimensional array. When ``footprint`` is given, ``size`` is ignored.
        padding (str, optional): Either the strings 'same' or 'valid'. 'same' adds padding to produce the same output size as the input. 'valid' does no padding, so only elements where the window fits completely in the image will be evaluated. Defaults to 'same'.
        mode (str, optional): Determines how the input array will be padded beyond its boundaries. Defaults to 'constant'. For valid values, see ``jax.numpy.pad``.
        cval (Number, optional): Value to fill past edges of input if ``mode`` is 'constant'. Defaults to 0.
        origin (int | Sequence[int], optional): Controls the placement of the filter on the input's elements. A value of 0 (the default) centers the filter over the pixel. Positive values shift the filter to the left. Negative values shift the filter to the right.
        axes (int | Sequence[int], optional): Axes of input array to filter along. If ``None``, the input is filtered along all axes.
        batch_size (int, optional): Integer specifying the size of the batch for each step to execute in parallel. If specified, `jax.lax.map` will be used. If `None`, `jax.vmap` will be used. Defaults to `None`.

    Returns:
        Array: Uniform-filtered array.
    """
    return generic_filter(
        x,
        jnp.mean,
        size,
        footprint,
        padding,
        mode,
        cval,
        origin,
        axes,
        batch_size,
    )
