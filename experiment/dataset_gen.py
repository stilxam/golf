import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
from typing import Dict, Tuple, Union


@eqx.filter_jit
def generate_single_stress_strain_curve(key, n_points=300) -> Dict[str, Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]]:
    """
    Generates a single, realistic, and noisy stress-strain curve.
    Uses the Hollomon equation for more realistic plastic deformation.
    """
    key_params, key_noise = jax.random.split(key)
    p_keys = jax.random.split(key_params, 6)

    # --- 1. Generate Varied Material Properties ---
    # Young's Modulus (E) in GPa
    youngs_modulus_gpa = jax.random.uniform(p_keys[0], minval=180.0, maxval=220.0)
    
    # Yield Strength in MPa
    yield_strength_mpa = jax.random.uniform(p_keys[1], minval=350.0, maxval=550.0)
    
    # Strain at the yield point
    yield_strain = yield_strength_mpa / (youngs_modulus_gpa * 1000)

    strain_hardening_n = jax.random.uniform(p_keys[2], minval=0.1, maxval=0.25)

    strain_uts_factor = jax.random.uniform(p_keys[3], minval=20.0, maxval=50.0)
    strain_uts = yield_strain * strain_uts_factor
    
    # Fracture happens after UTS
    strain_frac = strain_uts * 1.15
    
    # Noise level in MPa
    noise_level = jax.random.uniform(p_keys[4], minval=8.0, maxval=20.0)

    # --- 2. Define the Ground-Truth Function using the Hollomon Model ---
    strain = jnp.linspace(0, strain_frac, n_points)

    # Region 1: Elastic (linear)
    stress_elastic = (youngs_modulus_gpa * 1000) * strain

    # Region 2: Plastic Strain Hardening (Hollomon Equation: σ = K * ε^n)
    # First, find K such that the curve is continuous at the yield point
    K = yield_strength_mpa / (yield_strain**strain_hardening_n)
    stress_plastic = K * (strain**strain_hardening_n)
    
    # Calculate the Ultimate Tensile Strength (UTS) from the Hollomon curve
    uts_mpa = K * (strain_uts**strain_hardening_n)

    # Region 3: Necking/Failure (downward slope after UTS)
    stress_necking = uts_mpa - (uts_mpa * 0.3) * ((strain - strain_uts) / (strain_frac - strain_uts))

    # Combine the pieces
    stress_true = jnp.where(
        strain < yield_strain,
        stress_elastic,
        jnp.where(strain < strain_uts, stress_plastic, stress_necking)
    )

    # --- 3. Add Noise ---
    noise = jax.random.normal(key_noise, shape=(n_points,)) * noise_level
    stress_noisy = stress_true + noise

    # --- 4. Ground Truth Points ---
    # UTS point
    true_uts_point = (strain_uts, uts_mpa)
    # Fracture point (stress at strain_frac)
    stress_frac = uts_mpa - (uts_mpa * 0.3) * ((strain_frac - strain_uts) / (strain_frac - strain_uts))
    true_fracture_point = (strain_frac, stress_frac)

    return {
        "strain": strain,
        "stress_noisy": stress_noisy,
        "true_yield_point": (yield_strain, yield_strength_mpa),
        "true_uts_point": true_uts_point,
        "true_fracture_point": true_fracture_point,
    }

def make_dataset(n_curves, key)-> Dict[str, Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]]:
    keys = jax.random.split(key, n_curves)
    return jax.vmap(generate_single_stress_strain_curve)(keys)



if __name__ == "__main__":
    key = jax.random.PRNGKey(120)
    N_CURVES = 10

    dataset = make_dataset(N_CURVES, key)


    print(f"Dataset generated with {N_CURVES} curves.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    all_true_yield_strains, all_true_yield_stresses = dataset['true_yield_point']
    all_true_uts_strains, all_true_uts_stresses = dataset['true_uts_point']
    all_true_frac_strains, all_true_frac_stresses = dataset['true_fracture_point']

    for i in range(N_CURVES):
        strain = dataset['strain'][i]
        stress = dataset['stress_noisy'][i]
        true_yield_strain = all_true_yield_strains[i]
        true_yield_stress = all_true_yield_stresses[i]
        true_uts_strain = all_true_uts_strains[i]
        true_uts_stress = all_true_uts_stresses[i]
        true_frac_strain = all_true_frac_strains[i]
        true_frac_stress = all_true_frac_stresses[i]

        line, = ax.plot(strain, stress, alpha=0.7, label=f'Curve {i+1}')

        ax.plot(
            true_yield_strain,
            true_yield_stress,
            marker='*',
            markersize=12,
            color=line.get_color(),
            markeredgecolor='black',
            label=f'Yield Point {i+1}' if i == 0 else None
        )
        ax.plot(
            true_uts_strain,
            true_uts_stress,
            marker='o',
            markersize=10,
            color=line.get_color(),
            markeredgecolor='black',
            label=f'UTS Point {i+1}' if i == 0 else None
        )
        ax.plot(
            true_frac_strain,
            true_frac_stress,
            marker='X',
            markersize=10,
            color=line.get_color(),
            markeredgecolor='black',
            label=f'Fracture Point {i+1}' if i == 0 else None
        )

    ax.set_title(f'Generated Dataset: {N_CURVES} Stress-Strain Curves', fontsize=16)
    ax.set_xlabel('Strain (dimensionless)', fontsize=14)
    ax.set_ylabel('Stress (MPa)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.show()
