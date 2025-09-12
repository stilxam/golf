from src.golf import PiecewiseModel, fit, init_curvature
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(420)
x_data = jnp.linspace(0, 10, 100)
y_data = jnp.where(x_data < 3, 1.0, jnp.where(x_data < 6, 3.0, -1.0))
y_data += jax.random.normal(key, (100,)) * 0.2

n_segments = 5
init_bx, init_by = init_curvature(x_data, y_data, n_segments)

initial_model = PiecewiseModel(
    n_segments=n_segments,
    x_range=(0, 10),
    init_breakpoints_x=init_bx,
    init_breakpoints_y=init_by,
    key=key
)

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
plt.scatter(init_bx, init_by[1:-1], color='green', label='Initial Breakpoints', zorder=5)
plt.legend()
plt.savefig("smart_init.png")
plt.show()
