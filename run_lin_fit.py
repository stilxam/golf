import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from plf import PiecewiseModel, fit

def main():
    """
    An example script to demonstrate the usage of the piecewise fitter.
    """
    # 1. Create synthetic non-linear data
    key = jax.random.PRNGKey(420)
    x_data = jnp.linspace(0, 10, 100)
    # x_data = jax.random.uniform(key, (100,), minval=0.0, maxval=10.0)
    # y_data = jnp.sin(x_data) + x_data * 0.2 + jax.random.normal(key, (100,)) * 0.2
    # synthetic data with steps and noise
    y_data = jnp.where(x_data < 3, 1.0, jnp.where(x_data < 6, 3.0, 2.0))
    y_data += jax.random.normal(key, (100,)) * 0.2
    # Shuffle the data to avoid any ordering effects



    # 2. Instantiate the initial model
    initial_model = PiecewiseModel(n_segments=4, x_range=(0.0, 10.0), init_random=False , key=key)

    print("Initial internal breakpoints (x-coordinates):", initial_model.internal_breakpoints_x)
    print("Initial breakpoint heights (y-coordinates):", initial_model.breakpoints_y)

    # 3. Train the model using the fit function
    trained_model = fit(
        initial_model,
        x_data,
        y_data,
        n_iterations=2000,
        learning_rate=0.01,
        patience=200,
    )
    print("\nOptimized internal breakpoints (x-coordinates):", trained_model.internal_breakpoints_x)
    print("Optimized breakpoint heights (y-coordinates):", trained_model.breakpoints_y)

    # 4. Visualize the results
    # Generate predictions from the trained model across the data range
    y_pred_final = jax.vmap(trained_model)(x_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Original Data', alpha=0.6, s=20, zorder=1)
    plt.plot(x_data, y_pred_final, color='red', label='Fitted Piecewise Model', zorder=2)

    # Plot the optimized breakpoints as vertical lines
    # Reconstruct the full set of breakpoints (including endpoints)
    sorted_internal_x = jnp.sort(trained_model.internal_breakpoints_x)
    all_breakpoints_x = jnp.concatenate([
        jnp.array([trained_model.x_range[0]]),
        sorted_internal_x,
        jnp.array([trained_model.x_range[1]])
    ])

    for bp in all_breakpoints_x[1:-1]:  # Skip the fixed endpoints
        plt.axvline(x=bp, color='gray', linestyle='--', linewidth=1, zorder=0)

    plt.title('Differentiable Piecewise Fitter')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()