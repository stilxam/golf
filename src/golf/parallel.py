from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .model import PiecewiseModel
from .trainer import _compute_loss


def fit_parallel(
        models: List[PiecewiseModel],
        data_pairs: List[tuple[jax.Array, jax.Array]],
        n_iterations: int = 300,
        learning_rate: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
) -> List[PiecewiseModel]:
    """
    Trains multiple PiecewiseModel instances in parallel on multiple data pairs on a single device (e.g., GPU).

    This function leverages `jax.vmap` to vectorize the training process, allowing
    for efficient parallel updates to all models in a single execution pass.

    Args:
        models: A list of `PiecewiseModel` instances to be trained.
        data_pairs: A list of (x_data, y_data) tuples for training.
        n_iterations: The maximum number of training iterations.
        learning_rate: The learning rate for the Adam optimizer.
        patience: The number of iterations to wait for improvement before early stopping.
        verbose: Whether to print training progress.

    Returns:
        A list of trained `PiecewiseModel` instances.
    """
    if len(models) != len(data_pairs):
        raise ValueError("The number of models must match the number of data pairs.")

    x_data, y_data = zip(*data_pairs)
    x_arr = jnp.stack([jnp.asarray(x) for x in x_data])
    y_arr = jnp.stack([jnp.asarray(y) for y in y_data])


    # Stack the list of models into a single, batched PyTree model.
    batched_model = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *models)

    optimizer = optax.adam(learning_rate)
    batched_opt_state = optimizer.init(eqx.filter(batched_model, eqx.is_array))

    # Vmap the gradient calculation function to operate over the batch of models.
    vmapped_grad_fn = jax.vmap(
        eqx.filter_value_and_grad(_compute_loss),
        in_axes=(0, 0, 0)  # (model, x_data, y_data)
    )

    # Define the parallel step function.
    @eqx.filter_jit
    def parallel_step(model, opt_state, x, y):
        losses, grads = vmapped_grad_fn(model, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, losses

    if verbose:
        print("--- Starting Parallel Training ---")

    best_avg_loss = jnp.inf
    best_batched_model = batched_model
    wait = 0

    for i in range(n_iterations):
        batched_model, batched_opt_state, losses = parallel_step(
            batched_model, batched_opt_state, x_arr, y_arr
        )
        avg_loss = jnp.mean(losses)

        if verbose and (i == 0 or (i + 1) % max(1, (n_iterations // 10)) == 0):
            print(f"Iteration {i + 1:5d}: Average Loss = {f'{avg_loss:.6f}'}")

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_batched_model = batched_model
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"Early stopping at iteration {i + 1}")
            break

    if verbose:
        print("--- Finished Parallel Training ---")

    # Unstack the batched model back into a list of individual models.
    num_models = len(models)
    trained_models = [
        jax.tree_util.tree_map(lambda leaf: leaf[i], best_batched_model)
        for i in range(num_models)
    ]

    return trained_models
