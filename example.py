from src.golf import PiecewiseModel, fit_parallel, init_curvature
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create a key for reproducibility
key = jax.random.PRNGKey(420)

# --- Dataset 1 ---
x_data1 = jnp.linspace(0, 10, 100)
y_data1 = jnp.sin(x_data1) + jax.random.normal(key, (100,)) * 0.2

# --- Dataset 2 ---
x_data2 = jnp.linspace(-5, 5, 100)
y_data2 = jnp.cos(x_data2) + jax.random.normal(key, (100,)) * 0.1

# --- Model Initialization ---
n_segments = 5

# Model 1
init_bx1, init_by1 = init_curvature(x_data1, y_data1, n_segments)
initial_model1 = PiecewiseModel(
    n_segments=n_segments,
    x_range=(0, 10),
    init_breakpoints_x=init_bx1,
    init_breakpoints_y=init_by1,
    key=key
)

# Model 2
init_bx2, init_by2 = init_curvature(x_data2, y_data2, n_segments)
initial_model2 = PiecewiseModel(
    n_segments=n_segments,
    x_range=(-5, 5),
    init_breakpoints_x=init_bx2,
    init_breakpoints_y=init_by2,
    key=key
)

# --- Parallel Training ---
models = [initial_model1, initial_model2]

data_pairs = [(x_data1, y_data1), (x_data2, y_data2)]

trained_models = fit_parallel(
    models,
    data_pairs,
    n_iterations=2000,
    learning_rate=0.01,
    patience=200,
)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Plot for Model 1
y_pred1 = jax.vmap(trained_models[0])(x_data1)
axes[0].scatter(x_data1, y_data1, label='Original Data 1', alpha=0.6, s=20)
axes[0].plot(x_data1, y_pred1, color='red', label='Fitted Model 1')
axes[0].set_title('Dataset 1')
axes[0].legend()

# Plot for Model 2
y_pred2 = jax.vmap(trained_models[1])(x_data2)
axes[1].scatter(x_data2, y_data2, label='Original Data 2', alpha=0.6, s=20)
axes[1].plot(x_data2, y_pred2, color='blue', label='Fitted Model 2')
axes[1].set_title('Dataset 2')
axes[1].legend()

plt.savefig("fit_parallel_example.png")
plt.show()