from numbers import Number
from typing import List, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.lax.lax import PrecisionLike
from jaxtyping import Array, Float, Num


def gaussian_kernel_1d(
    sigma: float, order: int, radius: int
) -> Float[Array, " 2 * {radius} + 1"]:
    """1-D gaussian kernel.

    Args:
        sigma (float): standard deviation
        order (int): 0 order is a Gaussian, order>0 corresp. to derivatives
        radius (int): radius of the kernel

    Raises:
        ValueError: if order < 0

    Returns:
        Float[Array]
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    sigma2 = jnp.square(sigma)
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        exponent_range = jnp.arange(order + 1)
        q = jnp.zeros(order + 1).at[0].set(1)
        diag_exp = jnp.diag(exponent_range[1:], 1)  # diag_exp @ q(x) = q'(x)
        diag_p = jnp.diag(
            jnp.ones(order) / -sigma2, -1
        )  # diag_p @ q(x) = q(x) * p'(x)
        qmat_deriv = diag_exp + diag_p
        for _ in range(order):
            q = qmat_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter1d(
    x: Num[Array, " a"],
    sigma: float,
    axis: int = -1,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: int = 0,
    precision: PrecisionLike | None = None,
) -> Num[Array, " a"]:
    """Compute a 1-D gaussian filter.

    Args:
        x (Array): input array
        sigma (float): standard deviation of Gaussian
        axis (int): axis of `input` along which to calculate, optional. Defaults to -1.
        order (int): order of 0 is Gaussian, higher orders are derivatives, optional. Defaults to 0.
        truncate (float): truncate filter at this many std. dev's, optional. Defaults to 4.
        mode (str): how input array is extended beyond boundaries, optional. Defaults to 'constant'.
        cval (float): value to use for `mode='constant'`, optional. Defaults to 0.
        radius (int): radius of the filter. Overrides `truncate` if >0. Defaults to 0.
        precision (PrecisionLike): precision to use for calculation, optional. Defaults to None.

    Raises:
        ValueError: if radius < 0

    Returns:
        Array
    """
    if axis < 0:
        axis = x.ndim + axis
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sigma + 0.5)
    if radius > 0.0:
        lw = radius
    if lw < 0:
        raise ValueError(f"Radius must be a nonnegative integer. Got {lw}.")
    weights = gaussian_kernel_1d(sigma, order, lw)[::-1]
    # pad the input array appropriately
    pad_width = [((lw, lw) if a == axis else (0, 0)) for a in range(x.ndim)]
    if mode == "constant":
        x_pad = jnp.pad(x, pad_width, mode=mode, constant_values=cval)
    else:
        x_pad = jnp.pad(x, pad_width, mode=mode)
    # Be careful that modes in signal.convolve refer to the
    # 'same' 'full' 'valid' modes, while in gaussian_filter1d refers to the
    # way the padding is done 'constant' 'reflect' etc.
    return jnp.apply_along_axis(
        jsp.signal.convolve,
        axis,
        x_pad,
        weights,
        mode="valid",
        method="fft",
        precision=precision,
    )


def gaussian_filter(
    x: Array,
    sigma: float | List[float] | Num[Array, " b"],
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: int = 0,
    axes: int | Tuple[int, ...] | None = None,
    precision: PrecisionLike | None = None,
) -> Array:
    """Compute a multi-dimensional Gaussian filter.

    Args:
        x (Num[Array]): Input array
        sigma (float): Standard deviation of Gaussian
        order (int): Order of 0 is Gaussian, higher orders are derivatives, optional. Defaults to 0.
        mode (str): How input array is extended beyond boundaries, optional. Defaults to 'constant'.
        cval (float): Value to use for `mode='constant'`, optional. Defaults to 0.
        truncate (float): Truncate filter at this many std. dev's, optional. Defaults to 4.
        radius (int): Radius of the filter. Overrides `truncate`, if provided.
        axes (int | Tuple[int,...] | None): Axis of `input` along which to calculate, optional. Defaults to -1.
        precision (PrecisionLike | None): Precision to use for calculation, optional. Defaults to None.

    Returns:
        Array
    """
    if axes is None:
        axes = list(range(x.ndim))
    if isinstance(sigma, Number):
        sigma = [
            sigma,
        ] * len(axes)

    for ax, _sigma in zip(axes, sigma):
        x = gaussian_filter1d(
            x,
            _sigma,
            axis=ax,
            order=order,
            truncate=truncate,
            radius=radius,
            mode=mode,
            precision=precision,
            cval=cval,
        )
    return x
