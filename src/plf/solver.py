import jax.numpy as jnp
from jaxtyping import Array, Float


def solve_weighted_linear_regression(
    x: Float[Array, "n"],
    y: Float[Array, "n"],
    weights: Float[Array, "n"]
) -> tuple[float, float]:
    """
    Solves for the slope and intercept using weighted least squares.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        weights: 1D array of weights for each data point.

    Returns:
        A tuple containing the calculated (slope, intercept).
    """
    x_arr, y_arr, w_arr = jnp.asarray(x), jnp.asarray(y), jnp.asarray(weights)

    sqrt_w = jnp.sqrt(w_arr)
    A = jnp.vstack([x_arr, jnp.ones_like(x_arr)]).T

    A_w = sqrt_w[:, None] * A
    y_w = sqrt_w * y_arr

    params, _, _, _ = jnp.linalg.lstsq(A_w, y_w, rcond=None)
    slope, intercept = params

    return slope, intercept

