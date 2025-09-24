# EXPERIMENTAL:DOESNT WORK

import os
import sys
sys.path.append('/home/maxwell/golf/experiment')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
from beartype import beartype as typechecked

from model import PiecewiseModel
from initialization import init_curvature
from variable_segments_trainer import fit

# Import the Gray-Scott dataset generator
from diff_data import dataset_generator, SimParams


def get_true_function(x: Float[Array, ""]) -> Float[Array, ""]:
    """A simple piecewise function with 3 segments."""
    return jnp.where(x < 3, x, jnp.where(x < 7, 2 * x + 7, -x + 18))


def get_gray_scott_data(
    key: PRNGKeyArray,
    grid_size: int = 256,
    T: float = 2000.0
) -> tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """
    Generates Gray-Scott reaction-diffusion data using the dataset from diff_data.py.
    Returns the final spatial profile of species V as training data.
    """
    params = SimParams(grid_size=grid_size, T=T)

    # Generate the full spatiotemporal dataset
    history_array = dataset_generator(key, params, num_saves=50)

    # Extract spatial coordinates and final V concentration profile
    x_data = jnp.linspace(0, params.L, params.grid_size)
    y_data = history_array[-1, 1, :]  # Final profile of species V

    return x_data, y_data


def generate_data(
    key: PRNGKeyArray, n_points: int = 100, noise_std: float = 0.5
) -> tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """Generates noisy data from the true piecewise function."""
    x_data = jnp.linspace(0, 10, n_points)
    y_true = jax.vmap(get_true_function)(x_data)
    noise = jax.random.normal(key, (n_points,)) * noise_std
    y_data = y_true + noise
    return x_data, y_data


def count_effective_segments(model: PiecewiseModel, tol: float = 1e-2) -> int:
    """Counts the number of segments with significantly different slopes."""
    sorted_internal_x = jnp.sort(model.internal_breakpoints_x)
    full_x = jnp.concatenate([
        jnp.array([model.x_range[0]]),
        sorted_internal_x,
        jnp.array([model.x_range[1]])
    ])
    slopes = jnp.diff(model.breakpoints_y) / jnp.diff(full_x)
    slope_changes = jnp.diff(slopes)
    return jnp.sum(jnp.abs(slope_changes) > tol) + 1


def merge_insignificant_breakpoints(
    model: PiecewiseModel, tol: float = 1e-2
) -> PiecewiseModel:
    """Merges insignificant breakpoints from a fitted model."""
    sorted_internal_x = jnp.sort(model.internal_breakpoints_x)
    full_x = jnp.concatenate([
        jnp.array([model.x_range[0]]),
        sorted_internal_x,
        jnp.array([model.x_range[1]])
    ])
    slopes = jnp.diff(model.breakpoints_y) / jnp.diff(full_x)
    slope_changes = jnp.diff(slopes)

    significant_mask = jnp.abs(slope_changes) > tol

    new_internal_breakpoints_x = sorted_internal_x[significant_mask]

    # Keep the y-breakpoints corresponding to significant slope changes
    # The first y-breakpoint is always kept.
    # For internal breakpoints, we keep the one *after* a significant slope change.
    # The slope_changes are aligned with the internal breakpoints.
    # A change at index i corresponds to internal breakpoint i.
    # The y-breakpoints are at indices 1 to n-1 for internal breakpoints.
    y_mask = jnp.concatenate([jnp.array([True]), significant_mask, jnp.array([True])])
    new_breakpoints_y = model.breakpoints_y[y_mask]

    new_n_segments = new_internal_breakpoints_x.shape[0] + 1

    # Generate a new key for the merged model
    key = jax.random.PRNGKey(42)

    return PiecewiseModel(
        n_segments=new_n_segments,
        x_range=model.x_range,
        key=key,
        init_breakpoints_x=new_internal_breakpoints_x,
        init_breakpoints_y=new_breakpoints_y,
    )


def main():
    """Main function to run the example with Gray-Scott dataset."""
    key = jax.random.PRNGKey(0)

    # Generate Gray-Scott reaction-diffusion data
    print("Generating Gray-Scott reaction-diffusion data...")
    x_data, y_data = get_gray_scott_data(key, grid_size=256, T=2000.0)

    # Update x_range to match the Gray-Scott domain
    x_range = (float(x_data.min()), float(x_data.max()))
    print(f"Data range: x ∈ [{x_range[0]:.1f}, {x_range[1]:.1f}], y ∈ [{y_data.min():.4f}, {y_data.max():.4f}]")

    n_initial_segments = 50  # Start with more segments for complex RD patterns

    # Initialize model with curvature-based breakpoints
    init_bx, init_by = init_curvature(x_data, y_data, n_initial_segments)

    model = PiecewiseModel(
        n_segments=n_initial_segments,
        x_range=x_range,
        key=key,
        init_breakpoints_x=init_bx,
        init_breakpoints_y=init_by
    )

    print(f"Starting with {n_initial_segments} segments.")

    # Fit the model with lighter L1 regularization to preserve complexity
    fitted_model = fit(
        model,
        x_data,
        y_data,
        n_iterations=3000,
        learning_rate=0.0005,  # Slower learning for better convergence
        l1_lambda=0.000,  # Much lighter regularization to preserve RD structure
        verbose=True
    )

    # Merge insignificant breakpoints with tighter tolerance
    merged_model = merge_insignificant_breakpoints(fitted_model, tol=5e-4)

    # Count the remaining segments
    n_final_segments = count_effective_segments(fitted_model)
    print(f"Finished with {n_final_segments} effective segments.")
    print(f"Model after merging has {merged_model.n_segments} segments.")

    # Plot the results
    plt.figure(figsize=(14, 10))

    # Subsample data for clearer visualization
    n_plot_points = min(800, len(x_data))  # Show more data points
    indices = jnp.linspace(0, len(x_data)-1, n_plot_points, dtype=int)
    x_plot_data = x_data[indices]
    y_plot_data = y_data[indices]

    plt.scatter(x_plot_data, y_plot_data, label="Gray-Scott Data", alpha=0.7, s=12, c='blue')

    # Generate smooth prediction curves
    x_smooth = jnp.linspace(x_range[0], x_range[1], 500)
    y_fit_smooth = jax.vmap(fitted_model)(x_smooth)
    y_merged_smooth = jax.vmap(merged_model)(x_smooth)

    plt.plot(x_smooth, y_fit_smooth, "r-", label=f"Fitted Model ({fitted_model.n_segments} segments)", linewidth=2.5)
    plt.plot(x_smooth, y_merged_smooth, "g--", lw=3, label=f"Merged Model ({merged_model.n_segments} segments)")

    # Mark breakpoints
    sorted_bp_x = jnp.sort(fitted_model.internal_breakpoints_x)
    bp_y = jax.vmap(fitted_model)(sorted_bp_x)
    plt.scatter(sorted_bp_x, bp_y, color='red', s=60, zorder=5, marker='o',
               edgecolor='darkred', linewidth=1, label='Breakpoints')

    plt.title("Variable Segments Fit on Gray-Scott Reaction-Diffusion Data", fontsize=16)
    plt.xlabel("Spatial Coordinate (x)", fontsize=14)
    plt.ylabel("Concentration of Species V", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("variable_segments_gray_scott.png", dpi=200, bbox_inches='tight')
    plt.show()
    print("Plot saved to variable_segments_gray_scott.png")

    # Print some statistics about the fit
    mse_fitted = jnp.mean((jax.vmap(fitted_model)(x_data) - y_data)**2)
    mse_merged = jnp.mean((jax.vmap(merged_model)(x_data) - y_data)**2)
    print(f"\nFit Quality:")
    print(f"Fitted model MSE: {mse_fitted:.6f}")
    print(f"Merged model MSE: {mse_merged:.6f}")

    # Show segment complexity reduction
    complexity_reduction = (n_initial_segments - merged_model.n_segments) / n_initial_segments * 100
    print(f"Complexity reduction: {complexity_reduction:.1f}% ({n_initial_segments} → {merged_model.n_segments} segments)")


if __name__ == "__main__":
    main()
