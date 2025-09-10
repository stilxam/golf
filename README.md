# plf — Piecewise Linear Fitting

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
import jax
import jax.numpy as jnp
from plf.model import PiecewiseModel
from plf.trainer import fit

# Create synthetic data
key = jax.random.PRNGKey(0)
x = jnp.linspace(0.0, 1.0, 200)
y = jnp.sin(2 * jnp.pi * x) + 0.1 * jax.random.normal(key, x.shape)

# Initialize a piecewise model with 4 segments on [0, 1]
model = PiecewiseModel(n_segments=4, x_range=(0.0, 1.0), key=key)

# Fit the model to data
trained = fit(model, x, y, n_iterations=500, learning_rate=1e-2)

# Predict with the trained model (vectorized)
y_pred = jax.vmap(trained)(x)
```



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

