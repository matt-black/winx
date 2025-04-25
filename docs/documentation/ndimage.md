# `ndimage`

A `scipy.ndimage`-like API where the functions are JAX-transformable.

## Filters

::: kilroy.ndimage.generic_filter
handler: python
options:
show_source: false
show_root_heading: true

::: kilroy.ndimage.median_filter
handler: python
options:
show_source: false
show_root_heading: true

::: kilroy.ndimage.uniform_filter
handler: python
options:
show_source: false
show_root_heading: true

## Morphology

Important differences with the `scipy.ndimage` API:

- `border_value` arguments do not set the value at the border in the output array, like in `scipy`, but set the value when the input array is padded so that the output shape matches the input shape.
- `mask` is not included as an input argument. To use a mask, while maintaining JAX-transformability, one can use `jnp.where` in combination with the `binary_` operations.

::: kilroy.ndimage.binary_dilation
handler: python
options:
show_source: false
show_root_heading: true

::: kilroy.ndimage.binary_erosion
handler: python
options:
show_source: false
show_root_heading: true
