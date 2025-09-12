import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Tuple
from functools import partial



@partial(eqx.filter_jit, static_argnames=["n_segments", "smoothing_window"])

def init_curvature(x_data: Float[Array, "n"], y_data: Float[Array, "n"], n_segments: Int[Array, ""],
                       smoothing_window: Int[Array, ""] = jnp.array(10), min_separation_ratio: Float[Array, ""] = jnp.array(0.05)) -> Tuple[Float[Array, "m"], Float[Array, "m"]]:
    """
    Initializes breakpoints in regions of high curvature with minimum separation.
    This version is fully compatible with JAX's JIT compiler.

    Args:
        x_data: The x-coordinates of the data, assumed to be sorted.
        y_data: The y-coordinates of the data.
        n_segments: The total number of linear segments.
        smoothing_window: The size of the moving average window.
        min_separation_ratio: The minimum distance between breakpoints as a fraction
                              of the total data length.

    Returns:
        A tuple containing fixed-size JAX arrays for breakpoint x and y positions.
    """
    num_breakpoints = n_segments - 1

    kernel = jnp.ones(smoothing_window) / smoothing_window
    y_smoothed = jnp.convolve(y_data, kernel, mode='same')

    first_derivative = jnp.gradient(y_smoothed)
    second_derivative = jnp.gradient(first_derivative)
    curvature = jnp.abs(second_derivative)

    pad_size = smoothing_window
    curvature = curvature.at[:pad_size].set(0).at[-pad_size:].set(0)

    
    min_separation_indices = (x_data.shape[0] * min_separation_ratio)
    min_separation_indices = jnp.maximum(1, jnp.floor(min_separation_indices))

    
    initial_indices = jnp.zeros(num_breakpoints, dtype=jnp.int32)
    initial_state = (curvature, initial_indices)

    def body_fun(i, state):
        current_curvature, indices_so_far = state
        max_idx = jnp.argmax(current_curvature)
        updated_indices = indices_so_far.at[i].set(max_idx)

        
        start = jnp.maximum(0, max_idx - min_separation_indices)
        end = jnp.minimum(len(current_curvature), max_idx + min_separation_indices + 1)

        mask_range = jnp.arange(len(current_curvature))
        mask = (mask_range >= start) & (mask_range < end)

        new_curvature = jnp.where(mask, 0.0, current_curvature)
        return new_curvature, updated_indices

    
    _, breakpoint_indices = jax.lax.fori_loop(0, num_breakpoints, body_fun, initial_state)

    breakpoint_indices = jnp.sort(breakpoint_indices)

    init_breakpoints_x = x_data[breakpoint_indices]
    init_breakpoints_y = y_data[breakpoint_indices]

    return init_breakpoints_x, init_breakpoints_y







if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    
    x = jnp.linspace(0, 10, 200)
    y = jnp.piecewise(x,
                      [x < 2, (x >= 2) & (x < 5), x >= 5],
                      [lambda x: 2 * x,
                       lambda x: 4 - (x - 2) * 3,
                       lambda x: -5 + (x - 5) * 0.5])
    y+= 0.3 * jax.random.normal(key, shape=x.shape)
    

    N_SEGMENTS = 3


    bx_c, by_c = init_curvature(x, y, N_SEGMENTS, smoothing_window=10)
    print("\n--- Curvature ---")
    print("x:", bx_c)
    print("y:", by_c)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data', alpha=0.5)
    
    plt.scatter(bx_c, by_c, color='green', label='Curvature Init', zorder=5)
    plt.legend()
    plt.title('Breakpoint Initialization Methods')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('breakpoint_initialization.png')

