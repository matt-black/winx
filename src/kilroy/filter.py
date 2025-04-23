"""Main functionality of the package, implementing a general mechanism for window-based filtering.

Implements a single function, `filter_window` that does window-based filtering. `ndimage.generic_filter` is a convenience wrapper for this function. `lax.filt_general_dilated` is also a convenience wrapper around this, except it handles the `feature_group_count` and `batch_group_count` and expects inputs to have 2 leading dimensions corresponding to batches and features.
"""

from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Int, Num

__all__ = ["filter_window"]


def filter_window(
    x: Array,
    fun: Callable[[Array], Union[Array, Number]],
    size: int | Sequence[int],
    footprint: Optional[Array] = None,
    padding: str = "same",
    mode: str = "constant",
    cval: Number = 0,
    origin: int | Sequence[int] = 0,
    axes: Optional[int | Sequence[int]] = None,
    window_strides: Optional[Sequence[int]] = None,
    lhs_dilation: Optional[Sequence[int]] = None,
    rhs_dilation: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
) -> Array:
    """filter Calculate a multidimensional filter using the given function.

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
        window_strides (Sequence[int], optional): Defaults to `None`.
        lhs_dilation (Sequence[int], optional): Defaults to `None`.
        rhs_dilation (Sequence[int], optional): Defaults to `None`.
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
    n_dim = len(x.shape)
    if rhs_dilation is not None:
        # this is like dilating a convolution kernel
        # here, create a square footprint (or take the footprint) and interleave
        # zeros of the appropriate width to do a quasi-dilation
        if footprint is None:
            footprint = jnp.ones(
                [
                    size,
                ]
                * n_dim
            )
        for ax, rhs_dil_size in enumerate(rhs_dilation):
            sze = rhs_dil_size * (footprint.shape[ax] - 1) + footprint.shape[ax]
            footprint = jnp.apply_along_axis(
                Partial(dilate_vector, dilation=rhs_dil_size, size=sze),
                ax,
                footprint,
            )
    if footprint is not None:
        size = footprint.shape

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
    lcls = locals()
    exec(make_extract_fun_text(n_dim), globals(), lcls)
    extract_window = lcls["extract_window"]
    # pad the input array according to the padding argument
    if padding == "same":
        padding = tuple((r, r) for r in window_radii)
        x = jnp.pad(x, padding, mode=mode, constant_values=cval)
    else:  # padding is assumed 'valid'
        pass
    out_shape = [
        (s - 2 * r) // strd
        for s, r, strd in zip(x.shape, window_radii, window_strides)
    ]
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
    ext_window: Callable[[Int[Array, " {n_dim}"]], Array] = Partial(
        extract_window, x, *window_radii
    )
    if footprint is None:
        if lhs_dilation is not None:
            szes = [
                2 * wr + 1 + lhd * (2 * wr)
                for wr, lhd in zip(window_radii, lhs_dilation)
            ]

            def get_window(coord: Int[Array, " {n_dim}"]) -> Array:
                return dilate_array(
                    ext_window(coord),
                    lhs_dilation,
                    szes,
                    list(range(len(szes))),
                )

        else:
            get_window = ext_window
    else:
        if lhs_dilation is not None:
            if rhs_dilation is not None:
                raise ValueError(
                    "can't use left- and right-hand dilation simultaneously"
                )
            else:
                raise ValueError("lhs dilation and footprint are incompatible")
        n_fp_out = jnp.sum(footprint).item()

        def get_window(coord: Int[Array, " {n_dim}"]) -> Array:
            return apply_footprint(ext_window(coord), footprint, n_fp_out)

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


def make_extract_fun_text(n_dim: int) -> str:
    """_make_extract_fun_text generate a string that defines a function called ``extract_window`` that can extract fixed size, N-dimensional windows from an input array.

    Args:
        n_dim (int): number of dimensions in the array

    Returns:
        str: text defining the function (can be ``exec'd``)
    """
    static_argnums = tuple(list(range(1, n_dim + 1)))
    var_names = [f"rad{d:d}" for d in range(n_dim)]
    var_name_txt = ",".join(var_names)
    out_shape_txt = "(" + ",".join(map(lambda r: f"{r:s}*2+1", var_names)) + ")"
    return f"@partial(jax.jit, static_argnums={static_argnums})\ndef extract_window(arr,{var_name_txt},coord):\n    out_shape = {out_shape_txt}\n    top_left = coord - jnp.array([{var_name_txt}])\n    return jax.lax.dynamic_slice(arr, top_left, out_shape)"


def make_extract_footprint_fun_text(n_dim: int) -> str:
    """_make_extract_fun_text generate a string that defines a function called ``extract_window`` that can extract fixed size, N-dimensional windows from an input array.

    Args:
        n_dim (int): number of dimensions in the array
        footprint (Array): masking array

    Returns:
        str: text defining the function (can be ``exec'd``)
    """
    static_argnums = tuple(list(range(2, n_dim + 3)))
    var_names = [f"rad{d:d}" for d in range(n_dim)]
    var_name_txt = ",".join(var_names)
    out_shape_txt = "(" + ",".join(map(lambda r: f"{r:s}*2+1", var_names)) + ")"
    return f"@partial(jax.jit, static_argnums={static_argnums})\ndef extract_window(arr,fp,size,{var_name_txt},coord):\n    out_shape = {out_shape_txt}\n    top_left = coord - jnp.array([{var_name_txt}])\n    win = jax.lax.dynamic_slice(arr, top_left, out_shape)\n    win_f = jnp.where(fp, win, -jnp.inf)\n    return jnp.sort(win_f.flatten(), descending=True)[:size]"


@partial(jax.jit, static_argnums=(2,))
def apply_footprint(arr: Array, footprint: Bool[Array, "..."], size: int):
    win_f = jnp.where(footprint, arr, -jnp.inf)
    return jnp.sort(win_f.flatten(), descending=True)[:size]


@partial(jax.jit, static_argnums=(1, 2))
def dilate_vector(vec: Num[Array, " n"], dilation: int, size: int):
    z = jnp.zeros((vec.shape[0], dilation))

    def concat(a, b):
        return jnp.concatenate([a, b])

    return jax.vmap(concat, 0, 0)(vec[:, None], z).flatten()[:size]


@partial(jax.jit, static_argnums=(1, 2, 3))
def dilate_array(
    arr: Array,
    dilation: Sequence[int],
    szes: Sequence[int],
    axes: Sequence[int],
):
    for ax, dil, sze in zip(axes, dilation, szes):
        arr = jnp.apply_along_axis(
            Partial(dilate_vector, dilation=dil, size=sze), ax, arr
        )
    return arr
