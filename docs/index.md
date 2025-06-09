# winx

N-dimensional window filtering in JAX.

## Introduction

Application of array filters that compute over the entire window (_e.g._ windowed median-filtering an image) cannot be done in `jax`. The closest mechanism is `jax.lax.reduce_window` but this requires the function to be a binary operation that compares elements within the window. `winx` provides this missing functionality, allowing users to construct arbitrary, window-based filters to an n-dimensional array.

We provide an API that matches that of `scipy.ndimage`, notably a JAX-transformable implementation of `generic_filter`.

We also provide an API similar to that of `jax.lax.conv_general_dilated` and `jax.lax.conv`, allowing for a similar approach to nonlinear filtering as used in (linear) convolution.

## Examples

```
import winx.ndimage as wndi
```

## Installation
