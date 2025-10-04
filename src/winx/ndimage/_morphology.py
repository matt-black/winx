"""morphological filters"""

from functools import partial
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Num

from .._types import Numeric
from ._ndimage import generic_filter

__all__ = [
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
    "binary_closing",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "grey_closing",
    "morphological_gradient",
]


@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6))
def _erode_or_dilate_binary(
    x: Bool[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    op: str = "erode",
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    window_op = jnp.all if op == "erode" else jnp.any
    if iterations == 1:
        return generic_filter(
            x,
            window_op,
            size,
            structure,
            "constant",
            border_value,
            origin,
            axes,
        )
    else:

        def scan_fun(
            y: Bool[Array, "..."], _
        ) -> tuple[Bool[Array, "..."], None]:
            return (
                generic_filter(
                    y,
                    window_op,
                    size,
                    structure,
                    "constant",
                    border_value,
                    origin,
                    axes,
                ),
                None,
            )

        val, _ = jax.lax.scan(scan_fun, x, None, iterations)
        return val


def binary_erosion(
    x: Bool[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Compute the multidimensional binary erosion with the given structuring element.

    Args:
        x (Bool[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the erosion. Defaults to 1.
        border_value (bool, optional): Value at the border when padding. Defaults to False.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]: Erosion of the input.
    """
    return _erode_or_dilate_binary(
        x, size, structure, "erode", iterations, border_value, origin, axes
    )


def binary_dilation(
    x: Bool[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Compute the binary dilation of the input array with the given structuring element.

    Args:
        x (Bool[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (bool, optional): Value at the border when padding. Defaults to False.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]: Dilation of the input.
    """
    return _erode_or_dilate_binary(
        x, size, structure, "dilate", iterations, border_value, origin, axes
    )


def binary_closing(
    x: Bool[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Compute the binary closing of the input array.

    Morphological closing is the erosion of the dilation of the input.

    Args:
        x (Bool[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (bool, optional): Value at the border when padding. Defaults to False.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]: morphological closing of input
    """

    def scan_fun(
        carry: Bool[Array, "..."], _
    ) -> tuple[Bool[Array, "..."], None]:
        return (
            binary_erosion(
                binary_dilation(
                    carry, size, structure, 1, border_value, origin, axes
                ),
                size,
                structure,
                1,
                border_value,
                origin,
                axes,
            ),
            None,
        )

    closed, _ = jax.lax.scan(scan_fun, x, None, length=iterations)
    return closed


def binary_opening(
    x: Bool[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: bool = False,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Compute the binary opening of the input array.

    Morphological opening is the dilation of the erosion of the input.

    Args:
        x (Bool[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (bool, optional): Value at the border when padding. Defaults to False.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]: morphological closing of input
    """

    def scan_fun(
        carry: Bool[Array, "..."], _
    ) -> tuple[Bool[Array, "..."], None]:
        return (
            binary_dilation(
                binary_erosion(
                    carry, size, structure, 1, border_value, origin, axes
                ),
                size,
                structure,
                1,
                border_value,
                origin,
                axes,
            ),
            None,
        )

    opened, _ = jax.lax.scan(scan_fun, x, None, length=iterations)
    return opened


def binary_hit_or_miss(
    x: Bool[Array, "..."],
    structure1: Bool[Array, "..."],
    structure2: Optional[Bool[Array, "..."]] = None,
    origin1: Union[int, Sequence[int]] = 0,
    origin2: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Bool[Array, "..."]:
    """Compute the binary hit or miss transform for the input using the two structures.

    Args:
        x (Bool[Array, "..."]): Input array.
        structure1 (Bool[Array, "..."]): pattern in the input array to look for (matches foreground).
        structure2 (Optional[Bool[Array, "..."]]): Structure that must not match any of the foreground pixels.
        origin1 (Union[int, Sequence[int]]): Placement of first structuring element. Defaults to 0.
        origin2 (Union[int, Sequence[int]]): Placement of second structuring element. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Bool[Array]
    """
    if structure2 is None:
        structure2 = jnp.logical_not(structure1)
    match_struct1 = Partial(jnp.array_equal, structure1)
    match_struct2 = Partial(jnp.array_equal, jnp.logical_not(structure2))
    return jnp.logical_and(
        generic_filter(
            x, match_struct1, structure1.shape, None, origin=origin1, axes=axes
        ),
        generic_filter(
            x, match_struct2, structure2.shape, None, origin=origin2, axes=axes
        ),
    )


@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6))
def _erode_or_dilate_greyscale(
    x: Num[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    op: str = "erode",
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Num[Array, "..."]:
    window_op = jnp.min if op == "erode" else jnp.max
    if iterations == 1:
        return generic_filter(
            x,
            window_op,
            size,
            structure,
            "constant",
            border_value,
            origin,
            axes,
        )
    else:

        def scan_fun(y: Num[Array, "..."], _) -> tuple[Num[Array, "..."], None]:
            return (
                generic_filter(
                    y,
                    window_op,
                    size,
                    structure,
                    "constant",
                    border_value,
                    origin,
                    axes,
                ),
                None,
            )

        val, _ = jax.lax.scan(scan_fun, x, None, iterations)
        return val


def grey_erosion(
    x: Num[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Num[Array, "..."]:
    """Multidimensional greyscale erosion with the given structuring element.

    Args:
        x (Num[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the erosion. Defaults to 1.
        border_value (Numeric, optional): Value at the border when padding. Defaults to 0.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Num[Array]: Erosion of the input.
    """
    return _erode_or_dilate_greyscale(
        x, size, structure, "erode", iterations, border_value, origin, axes
    )


def grey_dilation(
    x: Num[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Num[Array, "..."]:
    """Multidimensional greyscale dilation with the given structuring element.

    Args:
        x (Num[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (Numeric, optional): Value at the border when padding. Defaults to 0.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Num[Array]: Dilation of the input.
    """
    return _erode_or_dilate_greyscale(
        x, size, structure, "dilate", iterations, border_value, origin, axes
    )


def grey_closing(
    x: Num[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Num[Array, "..."]:
    """Multidimensional greyscale closing with the given structuring element.

    Args:
        x (Num[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (Numeric, optional): Value at the border when padding. Defaults to 0.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Num[Array]: Greyscale closing of the input.
    """

    def scan_fun(carry: Num[Array, "..."], _) -> tuple[Num[Array, "..."], None]:
        return (
            grey_erosion(
                grey_dilation(
                    carry, size, structure, 1, border_value, origin, axes
                ),
                size,
                structure,
                1,
                border_value,
                origin,
                axes,
            ),
            None,
        )

    closed, _ = jax.lax.scan(scan_fun, x, None, length=iterations)
    return closed


def grey_opening(
    x: Num[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    iterations: int = 1,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Num[Array, "..."]:
    """Multidimensional greyscale opening with the given structuring element.

    Args:
        x (Num[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        iterations (int, optional): Numeric of times to repeat the dilation. Defaults to 1.
        border_value (Numeric, optional): Value at the border when padding. Defaults to 0.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Num[Array]: Greyscale opening of the input.
    """

    def scan_fun(carry: Num[Array, "..."], _) -> tuple[Num[Array, "..."], None]:
        return (
            grey_dilation(
                grey_erosion(
                    carry, size, structure, 1, border_value, origin, axes
                ),
                size,
                structure,
                1,
                border_value,
                origin,
                axes,
            ),
            None,
        )

    opened, _ = jax.lax.scan(scan_fun, x, None, length=iterations)
    if iterations <= 1:
        return opened
    else:
        return grey_opening(
            opened, size, structure, iterations - 1, border_value, origin, axes
        )


def morphological_gradient(
    x: Num[Array, "..."],
    size: int | Sequence[int],
    structure: Array | None = None,
    border_value: Numeric = 0,
    origin: Union[int, Sequence[int]] = 0,
    axes: Optional[int | Sequence[int]] = None,
) -> Num[Array, "..."]:
    """Compute the multidimensional morphological gradient.

    Args:
        x (Num[Array, "..."]): Input array.
        size (int | Sequence[int]): Shape of the array of ones that will be used as a structuring element.
        structure (Optional[Array]): Structuring element. If `None`, the argument from `size` is used.
        border_value (Numeric, optional): Value at the border when padding. Defaults to 0.
        origin (Union[int, Sequence[int]], optional): Placement of the filter. Defaults to 0.
        axes (Optional[int  |  Sequence[int]], optional): The axes over which to apply the filter. Defaults to None.

    Returns:
        Num[Array]: difference between greyscale dilation and erosion.
    """
    return jnp.subtract(
        grey_dilation(x, size, structure, 1, border_value, origin, axes),
        grey_erosion(x, size, structure, 1, border_value, origin, axes),
    )
