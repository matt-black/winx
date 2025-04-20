"""Private utility functions."""


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
