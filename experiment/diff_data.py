import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, jaxtyped, PRNGKeyArray
from tqdm.auto import trange

from beartype import beartype as typechecked

GridSize = int
NumSpecies = 2
Grid = Float[Array, "grid_size"]
State = Float[Array, "num_species grid_size"]



class SimParams(eqx.Module):
    
    Du: float
    Dv: float
    feed_rate: float
    kill_rate: float

    L: float
    grid_size: GridSize = eqx.static_field()

    T: float
    dt: float

    dx: float

    def __init__(self, L=100.0, grid_size=512, T=8000.0, safety_factor=0.9):
        self.Du = 0.08
        self.Dv = 0.04
        self.feed_rate = 0.022
        self.kill_rate = 0.051
        self.L = L
        self.grid_size = grid_size
        self.T = T
        self.dx = L / grid_size
        D_max = max(self.Du, self.Dv)
        self.dt = safety_factor * (self.dx ** 2) / (2 * D_max)


@jaxtyped(typechecker=typechecked)
@eqx.filter_jit
def laplacian_neumann(y: Grid, dx: float) -> Grid:
    y_padded = jnp.pad(y, 1, mode='edge')
    return (y_padded[:-2] + y_padded[2:] - 2 * y) / (dx ** 2)


@jaxtyped(typechecker=typechecked)
@eqx.filter_jit
def gray_scott_reaction(y: State, params: SimParams) -> State:
    u, v = y[0], y[1]
    reaction_term = u * v ** 2
    du_dt = -reaction_term + params.feed_rate * (1 - u)
    dv_dt = reaction_term - (params.feed_rate + params.kill_rate) * v
    return jnp.stack([du_dt, dv_dt])


@jaxtyped(typechecker=typechecked)
@eqx.filter_jit
def dydt(y: State, params: SimParams) -> State:
    D = jnp.array([params.Du, params.Dv]).reshape(2, 1)
    diffusion = D * jax.vmap(laplacian_neumann, in_axes=(0, None))(y, params.dx)
    reaction = gray_scott_reaction(y, params)
    return diffusion + reaction


@jaxtyped(typechecker=typechecked)
@eqx.filter_jit
def rk4_step(y: State, params: SimParams) -> State:
    dt = params.dt
    k1 = dt * dydt(y, params)
    k2 = dt * dydt(y + 0.5 * k1, params)
    k3 = dt * dydt(y + 0.5 * k2, params)
    k4 = dt * dydt(y + k3, params)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def dataset_generator(key: PRNGKeyArray, params: SimParams = SimParams(grid_size=512, T=4000.0), pert_width_frac: float = 0.1) -> Float[Array, "num_saves num_species grid_size"]:
    """
    Generate a Gray-Scott dataset using the given random key and simulation parameters.
    Returns the history array of shape (num_saves, num_species, grid_size).
    """
    x_coords: Grid = jnp.linspace(0, params.L, params.grid_size)
    u_initial: Grid = jnp.ones(params.grid_size)
    v_initial: Grid = jnp.zeros(params.grid_size)

    pert_width = int(params.grid_size * pert_width_frac)
    mid = params.grid_size // 2
    u_initial = u_initial.at[mid - pert_width: mid + pert_width].set(0.5)
    v_initial = v_initial.at[mid - pert_width: mid + pert_width].set(0.25)
    y_initial: State = jnp.stack([u_initial, v_initial])

    n_steps = int(params.T / params.dt)
    save_every = n_steps // 100
    history = []
    y_current = y_initial

    for step in range(n_steps):
        y_current = rk4_step(y_current, params)
        if step % save_every == 0:
            history.append(y_current)

    history_array: Float[Array, "num_saves num_species grid_size"] = jnp.stack(history)
    return history_array

if __name__ == "__main__":
    params = SimParams(grid_size=512, T=4000.0)
    n_steps = int(params.T / params.dt)
    save_every = n_steps // 100

    print(f"Grid size: {params.grid_size}, Spatial step (dx): {params.dx:.4f}")
    print(f"Stable time step (dt): {params.dt:.4f}, Total steps: {n_steps}")

    key = jax.random.PRNGKey(0)
    history_array = dataset_generator(key, params)

    x_coords: Grid = jnp.linspace(0, params.L, params.grid_size)
    final_profile_v: Grid = history_array[-1, 1, :]

    print("\n--- Data Ready for GOLF ---")
    print(f"x_data shape: {x_coords.shape}")
    print(f"y_data shape: {final_profile_v.shape}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, final_profile_v, label=f'Concentration of V at t={params.T}')
    plt.title("1D Snapshot from JAX-Native RD Solver", fontsize=16)
    plt.xlabel("Spatial Coordinate (x)", fontsize=12)
    plt.ylabel("Concentration", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(history_array[:, 1, :], aspect='auto', origin='lower',
               extent=[0, params.L, 0, params.T], cmap='viridis')
    plt.title("Space-Time Kymograph of V", fontsize=16)
    plt.xlabel("Spatial Coordinate (x)", fontsize=12)
    plt.ylabel("Time (t)", fontsize=12)
    plt.colorbar(label="Concentration of V")
    plt.show()