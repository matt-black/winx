"""morphological filters"""

from functools import partial
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from .._types import Numeric
from ._ndimage import generic_filter

__all__ = ["binary_dilation", "binary_erosion"]


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _erode_or_dilate(
    x: Bool[Array, "..."],
    structure: Optional[Array],
    op: str = "erode",
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    window_op = jnp.all if op == "erode" else jnp.any
    if iterations == 1:
        return generic_filter(
            x, window_op, 3, structure, "constant", border_value, origin, axes
        )
    else:

        def scan_fun(y: Bool[Array, "..."], _) -> Bool[Array, "..."]:
            return generic_filter(
                y,
                window_op,
                3,
                structure,
                "constant",
                border_value,
                origin,
                axes,
            )

        val, _ = jax.lax.scan(scan_fun, x, None, iterations)  # type: ignore
        return val


def binary_erosion(
    x: Bool[Array, "..."],
    structure: Optional[Array],
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Multidimensional binary erosion with the given structuring element.

    Args:
        x (Bool[Array, &quot;...&quot;]): Input array.
        structure (Optional[Array]): Structuring element. If `None`, a 3x3 array of 1's is used.
        iterations (int, optional): Numeric of times to repeat the erosion. Defaults to 1.
        border_value (bool, optional): Value at the border when padding. Defaults to False.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]: Erosion of the input.
    """
    return _erode_or_dilate(
        x, structure, "erode", iterations, border_value, origin, axes
    )


def binary_dilation(
    x: Bool[Array, "..."],
    structure: Optional[Array],
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Multidimensional binary dilation with the given structuring element.

    Args:
        x (Bool[Array, &quot;...&quot;]): Input array.
        structure (Optional[Array]): Structuring element. If `None`, a 3x3 array of 1's is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (bool, optional): Value at the border when padding. Defaults to False.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]: Dilation of the input.
    """
    return _erode_or_dilate(
        x, structure, "dilate", iterations, border_value, origin, axes
    )


def binary_closing(
    x: Bool[Array, "..."],
    structure: Optional[Array],
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    closed = binary_erosion(
        binary_dilation(x, structure, 1, border_value, origin, axes),
        structure,
        1,
        border_value,
        origin,
        axes,
    )
    if iterations == 1:
        return closed
    else:
        return binary_closing(
            closed, structure, iterations - 1, border_value, origin, axes
        )


def binary_opening(
    x: Bool[Array, "..."],
    structure: Optional[Array],
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    opened = binary_dilation(
        binary_erosion(x, structure, 1, border_value, origin, axes),
        structure,
        1,
        border_value,
        origin,
        axes,
    )
    if iterations <= 1:
        return opened
    else:
        return binary_opening(
            opened, structure, iterations - 1, border_value, origin, axes
        )
