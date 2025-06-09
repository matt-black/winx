# winx

N-dimensional windowed filtering in JAX.

## Introduction

Application of array filters that compute over the entire window (_e.g._ 3x3 median-filtering an image) cannot be done in `jax`. The closest mechanism is `jax.lax.reduce_window` but this requires the function to be a binary operation that compares elements within the window. `winx` provides this missing functionality, allowing users to construct and apply arbitrary, window-based filters to an n-dimensional array.

## Documentation

Coming soon...

## Dependencies

- [JAX](https://github.com/jax-ml/jax)
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
