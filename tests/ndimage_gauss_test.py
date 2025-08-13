import jax
import jax.numpy as jnp
import numpy
import scipy.ndimage as ndi
from scipy.datasets import ascent

from winx.ndimage import gaussian_filter, gaussian_filter1d

jax.config.update("jax_platform_name", "cpu")


_a = ascent().astype(numpy.float32)


def test_gauss1d_same():
    # padding mode constant
    b = ndi.gaussian_filter1d(_a, 2, mode="constant", cval=0)
    c = gaussian_filter1d(jnp.asarray(_a), 2, mode="constant", cval=0)
    assert numpy.allclose(b, c)


def test_gauss_same():
    b = ndi.gaussian_filter(_a, 2, mode="constant", cval=0)
    c = gaussian_filter(jnp.asarray(_a), 2, mode="constant", cval=0)
    assert numpy.allclose(b, c)
