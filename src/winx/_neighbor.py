"""Internal utilities for looking up neighbors"""

import jax.numpy as jnp
from jaxtyping import Array, Int


def neighbors(dim: Int, conn: Int, *args: Int) -> Int[Array, "n {dim}"]:
    """Get `conn`-connected neighbors of the element being queried.

    Args:
        dim (Int): dimension of input array to query
        conn (Int): connectedness

    Returns:
        Int[Array, "n {dim}"]: array of coordinates of neighbors, where `n` is the number of neighbors.
    """
    if dim == 1:
        return coords_1d_conn(*args)
    elif dim == 2:
        if conn == 1:
            return coords_2d_4conn(*args)
        elif conn == 2:
            return coords_2d_8conn(*args)
        else:
            raise ValueError("invalid connectivity, `conn` must be in {1,2}")
    elif dim == 3:
        if conn == 1:
            return coords_3d_6conn(*args)
        elif conn == 2:
            return coords_3d_26conn(*args)
        else:
            raise ValueError("invalid connectivity, `conn` must be in {1,2}")
    elif dim == 4:
        if conn == 1:
            return coords_4d_8conn(*args)
        elif conn == 2:
            return coords_4d_80conn(*args)
        else:
            raise ValueError("invalid connectivity, `conn` must be in {1,2}")
    else:
        raise ValueError(
            "neighbor lookup only implemented for <=4 dimensional arrays"
        )


def coords_1d_conn(x: Int) -> Int[Array, "2 1"]:
    """Get coordinates of all grid neighbors in 1D for the input grid coordinates.

    Volume being queried is assumed to be of shape (x,)

    Args:
        x (int): coordinate

    Returns:
        Int[Array, "2 1"]
    """
    return jnp.array([[x - 1], [x + 1]])


def coords_2d_4conn(y: Int, x: Int) -> Int[Array, "4 2"]:
    """Get coordinates of 1-connected grid neighbors in 2D for the input grid coordinates.

    Array being queried is assumed to be of shape (y, x)

    Args:
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "4 2"]
    """
    return jnp.array([[y, x - 1], [y, x + 1], [y - 1, x], [y + 1, x]])


def coords_2d_8conn(y: Int, x: Int) -> Int[Array, "8 2"]:
    """Get coordinates of all grid neighbors in 2D for the input grid coordinates.

    Volume being queried is assumed to be of shape (y, x)

    Args:
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "8 2"]
    """
    return jnp.array(
        [
            [y - 1, x - 1],
            [y - 1, x],
            [y - 1, x + 1],
            [y, x - 1],
            [y, x + 1],
            [y + 1, x - 1],
            [y + 1, x],
            [y + 1, x + 1],
        ]
    )


def coords_3d_6conn(z: Int, y: Int, x: Int) -> Int[Array, "6 3"]:
    """Get coordinates of 1-connected grid neighbors in 3D for the input grid coordinates.

    Array being queried is assumed to be of shape (z, y, x)

    Args:
        z (int): z coordinate
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "6 3"]
    """
    return jnp.array(
        [
            [z + 1, y, x],
            [z - 1, y, x],
            [z, y - 1, x],
            [z, y + 1, x],
            [z, y, x - 1],
            [z, y, x + 1],
        ]
    )


def coords_3d_26conn(z: Int, y: Int, x: Int) -> Int[Array, "26 3"]:
    """Get coordinates of all grid neighbors in 3D for the input grid coordinates.

    Volume being queried is assumed to be of shape (z, y, x)

    Args:
        z (int): z coordinate
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "26 3"]
    """
    return jnp.array(
        [
            [z - 1, y - 1, x - 1],
            [z - 1, y - 1, x],
            [z - 1, y - 1, x + 1],
            [z - 1, y, x - 1],
            [z - 1, y, x],
            [z - 1, y, x + 1],
            [z - 1, y + 1, x - 1],
            [z - 1, y + 1, x],
            [z - 1, y + 1, x + 1],
            [z, y - 1, x - 1],
            [z, y - 1, x],
            [z, y - 1, x + 1],
            [z, y, x - 1],
            [z, y, x + 1],
            [z, y + 1, x - 1],
            [z, y + 1, x],
            [z, y + 1, x + 1],
            [z + 1, y - 1, x - 1],
            [z + 1, y - 1, x],
            [z + 1, y - 1, x + 1],
            [z + 1, y, x - 1],
            [z + 1, y, x],
            [z + 1, y, x + 1],
            [z + 1, y + 1, x - 1],
            [z + 1, y + 1, x],
            [z + 1, y + 1, x + 1],
        ]
    )


def coords_4d_8conn(s: Int, z: Int, y: Int, x: Int) -> Int[Array, "8 4"]:
    """Get coordinates of 1-connected grid neighbors in 4D for the input grid coordinates.

    Array being queried is assumed to be of shape (s, z, y, x)

    Args:
        s (int): s coordinate
        z (int): z coordinate
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "8 4"]
    """
    return jnp.array(
        [
            [s - 1, z, y, x],
            [s + 1, z, y, x],
            [s, z + 1, y, x],
            [s, z - 1, y, x],
            [s, z, y, x - 1],
            [s, z, y, x + 1],
            [s, z, y + 1, x],
            [s, z, y - 1, x],
        ]
    )


def coords_4d_80conn(s: Int, z: Int, y: Int, x: Int) -> Int[Array, "80 4"]:
    """Get coordinates of all grid neighbors in 4D for the input grid coordinates.

    Array being queried is assumed to be of shape (s, z, y, x)

    Args:
        s (int): s coordinate
        z (int): z coordinate
        y (int): y coordinate
        x (int): x coordinate

    Returns:
        Int[Array, "80 4"]
    """
    return jnp.array(
        [
            [s - 1, z - 1, y - 1, x - 1],
            [s - 1, z - 1, y - 1, x],
            [s - 1, z - 1, y - 1, x + 1],
            [s - 1, z - 1, y, x - 1],
            [s - 1, z - 1, y, x],
            [s - 1, z - 1, y, x + 1],
            [s - 1, z - 1, y + 1, x - 1],
            [s - 1, z - 1, y + 1, x],
            [s - 1, z - 1, y + 1, x + 1],
            [s - 1, z, y - 1, x - 1],
            [s - 1, z, y - 1, x],
            [s - 1, z, y - 1, x + 1],
            [s - 1, z, y, x - 1],
            [s - 1, z, y, x],
            [s - 1, z, y, x + 1],
            [s - 1, z, y + 1, x - 1],
            [s - 1, z, y + 1, x],
            [s - 1, z, y + 1, x + 1],
            [s - 1, z + 1, y - 1, x - 1],
            [s - 1, z + 1, y - 1, x],
            [s - 1, z + 1, y - 1, x + 1],
            [s - 1, z + 1, y, x - 1],
            [s - 1, z + 1, y, x],
            [s - 1, z + 1, y, x + 1],
            [s - 1, z + 1, y + 1, x - 1],
            [s - 1, z + 1, y + 1, x],
            [s - 1, z + 1, y + 1, x + 1],
            [s, z - 1, y - 1, x - 1],
            [s, z - 1, y - 1, x],
            [s, z - 1, y - 1, x + 1],
            [s, z - 1, y, x - 1],
            [s, z - 1, y, x],
            [s, z - 1, y, x + 1],
            [s, z - 1, y + 1, x - 1],
            [s, z - 1, y + 1, x],
            [s, z - 1, y + 1, x + 1],
            [s, z, y - 1, x - 1],
            [s, z, y - 1, x],
            [s, z, y - 1, x + 1],
            [s, z, y, x - 1],
            [s, z, y, x + 1],
            [s, z, y + 1, x - 1],
            [s, z, y + 1, x],
            [s, z, y + 1, x + 1],
            [s, z + 1, y - 1, x - 1],
            [s, z + 1, y - 1, x],
            [s, z + 1, y - 1, x + 1],
            [s, z + 1, y, x - 1],
            [s, z + 1, y, x],
            [s, z + 1, y, x + 1],
            [s, z + 1, y + 1, x - 1],
            [s, z + 1, y + 1, x],
            [s, z + 1, y + 1, x + 1],
            [s + 1, z - 1, y - 1, x - 1],
            [s + 1, z - 1, y - 1, x],
            [s + 1, z - 1, y - 1, x + 1],
            [s + 1, z - 1, y, x - 1],
            [s + 1, z - 1, y, x],
            [s + 1, z - 1, y, x + 1],
            [s + 1, z - 1, y + 1, x - 1],
            [s + 1, z - 1, y + 1, x],
            [s + 1, z - 1, y + 1, x + 1],
            [s + 1, z, y - 1, x - 1],
            [s + 1, z, y - 1, x],
            [s + 1, z, y - 1, x + 1],
            [s + 1, z, y, x - 1],
            [s + 1, z, y, x],
            [s + 1, z, y, x + 1],
            [s + 1, z, y + 1, x - 1],
            [s + 1, z, y + 1, x],
            [s + 1, z, y + 1, x + 1],
            [s + 1, z + 1, y - 1, x - 1],
            [s + 1, z + 1, y - 1, x],
            [s + 1, z + 1, y - 1, x + 1],
            [s + 1, z + 1, y, x - 1],
            [s + 1, z + 1, y, x],
            [s + 1, z + 1, y, x + 1],
            [s + 1, z + 1, y + 1, x - 1],
            [s + 1, z + 1, y + 1, x],
            [s + 1, z + 1, y + 1, x + 1],
        ]
    )
