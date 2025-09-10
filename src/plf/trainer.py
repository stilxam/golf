import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from .model import PiecewiseModel


@eqx.filter_jit
def _compute_loss(
        model: PiecewiseModel, x_data: Float[Array, "num_points"], y_data: Float[Array, "num_points"]
) -> Float[Array, ""]:
    """Computes the mean squared error. The output is a scalar array."""
    y_pred = jax.vmap(model)(x_data)
    return jnp.mean(jnp.square(y_data - y_pred))


@eqx.filter_jit
def _make_step(
        model: PiecewiseModel,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        x_data: Float[Array, "num_points"],
        y_data: Float[Array, "num_points"]
):
    """Performs a single gradient update step."""
    loss, grads = eqx.filter_value_and_grad(_compute_loss)(model, x_data, y_data)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def fit(
        model: PiecewiseModel,
        x_data: Float[Array, "num_points"],
        y_data: Float[Array, "num_points"],
        n_iterations: int = 300,
        learning_rate: float = 0.01,
        patience: int = 10,
        verbose: bool = True
) -> PiecewiseModel:
    """Trains the PiecewiseModel to fit the provided data."""
    x_arr = jnp.asarray(x_data)
    y_arr = jnp.asarray(y_data)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    if verbose:
        print("--- Starting Training ---")

    # initialize early-stopping variables
    best_loss = jnp.inf
    best_model = model
    wait = 0

    for i in range(n_iterations):
        model, opt_state, loss = _make_step(model, optimizer, opt_state, x_arr, y_arr)
        if verbose and (i == 0 or (i + 1) % max(1, (n_iterations // 10)) == 0):
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = loss
            print(f"Iteration {i + 1:5d}: Loss = {loss_val:.6f}")

        if loss < best_loss:
            best_loss = loss
            best_model = model
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose:
                print(f"Early stopping at iteration {i + 1}")
            break

    if verbose:
        print("--- Finished Training ---")

    return best_model

