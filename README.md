# GOLF — Gradient Optimized piecewise Linear Fitting

plf is a small Python package built with JAX and Equinox for fitting continuous piecewise-linear functions (splines) to 1D data. It provides a differentiable model representing a continuous piecewise-linear function, a trainer that optimizes the model parameters with Optax, and a small solver for weighted linear regression.

## Key features

- Differentiable `PiecewiseModel` (continuous piecewise-linear spline) implemented as an Equinox module.
- Simple training loop with early stopping using Optax (Adam).
- Utility solver for weighted linear regression.

## Installation

The package depends on JAX, Equinox, Optax, and jaxtyping. Install with pip (pick the correct JAX wheel for your platform/GPU):

```bash
pip install jax jaxlib equinox optax jaxtyping
```

## Usage (quickstart)

```python
from golf import PiecewiseModel, fit
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(420)
x_data = jnp.linspace(0, 10, 100)
y_data = jnp.where(x_data < 3, 1.0, jnp.where(x_data < 6, 3.0, -1.0))
y_data += jax.random.normal(key, (100,)) * 0.2


initial_model = PiecewiseModel(n_segments=5, x_range=(0, 10), init_random=False , key=key)


trained_model = fit(
    initial_model,
    x_data,
    y_data,
    n_iterations=2000,
    learning_rate=0.01,
    patience=200,
)

y_pred_final = jax.vmap(trained_model)(x_data)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Original Data', alpha=0.6, s=20, zorder=1)
plt.plot(x_data, y_pred_final, color='red', label='Fitted Piecewise Model', zorder=2)
plt.legend()
```
![img.png](img.png)



## API reference (brief)

- `plf.model.PiecewiseModel`
  - A differentiable Equinox module representing a continuous piecewise-linear function.
  - Constructor: `PiecewiseModel(n_segments: int, x_range: tuple[float, float], *, key, init_random=False, init_breakpoints_x=None, init_breakpoints_y=None)`
  - Callable: `model(x)` returns interpolated y for scalar or vmapped over arrays via `jax.vmap`.

- `plf.trainer.fit`
  - Trains a `PiecewiseModel` on (x, y).
  - Signature: `fit(model, x_data, y_data, n_iterations=300, learning_rate=0.01, patience=10, verbose=True) -> trained_model`
  - Uses Optax Adam optimizer and early stopping based on validation of loss during training.

- `plf.solver.solve_weighted_linear_regression`
  - Solves for slope and intercept using weighted least squares.
  - Signature: `solve_weighted_linear_regression(x, y, weights) -> (slope, intercept)`

## Development

- The project is lightweight and intended as a starting point for experiments with differentiable splines in JAX.
- Tests and additional examples can be added in the repository root.

