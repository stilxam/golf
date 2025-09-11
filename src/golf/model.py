import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Optional


class PiecewiseModel(eqx.Module):
    """
    An Equinox module representing a continuous piecewise linear function (a spline).

    This model learns the optimal (x, y) coordinates of the "corners"
    of the piecewise function to best fit the data.
    """
    internal_breakpoints_x: Float[Array, " n_internal_breakpoints"]

    breakpoints_y: Float[Array, " n_breakpoints"]

    x_range: Float[Array, " 2"] = eqx.field(static=True)

    def __init__(
            self,
            n_segments: int,
            x_range: Tuple[float, float] | Float[Array, " 2"],
            init_random: bool = False,
            *,
            key: PRNGKeyArray,
            init_breakpoints_x: Optional[Float[Array, " n_internal_breakpoints"]] = None,
            init_breakpoints_y: Optional[Float[Array, " n_breakpoints"]] = None,
    ):
        """
        Initializes the model.

        Args:
            n_segments: The number of linear segments.
            x_range: A tuple or array (min, max) defining the fixed endpoints of the function.
            init_random: Whether to initialize breakpoints randomly or evenly spaced.
            key: A JAX random key for initializing the heights.
            init_breakpoints_x: Optional array to initialize internal breakpoints.
            init_breakpoints_y: Optional array to initialize breakpoints y values.
        """

        keys = jax.random.split(key, 2)
        self.x_range = jnp.asarray(x_range)
        if init_breakpoints_x is not None:
            self.internal_breakpoints_x = jnp.array(init_breakpoints_x)
        elif init_random:
            self.internal_breakpoints_x = jax.random.uniform(
                keys[0], (n_segments - 1,), minval=self.x_range[0], maxval=self.x_range[1]
            )
        else:
            self.internal_breakpoints_x = jnp.linspace(self.x_range[0], self.x_range[1], n_segments + 1)[1:-1]

        if init_breakpoints_y is not None:
            self.breakpoints_y = jnp.array(init_breakpoints_y)
        else:
            self.breakpoints_y = jax.random.normal(keys[1], (n_segments + 1,))

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """
        Differentiable prediction for a single scalar input x.
        This will be vmapped in the loss function for batch processing.
        """
        sorted_internal_x = jnp.sort(self.internal_breakpoints_x)
        full_x = jnp.concatenate([
            jnp.array([self.x_range[0]]),
            sorted_internal_x,
            jnp.array([self.x_range[1]])
        ])

        # jnp.interp is the core of our differentiable model.
        return jnp.interp(x, full_x, self.breakpoints_y)

