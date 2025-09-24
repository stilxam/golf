
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PyTree
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


@jax.jit
def solve_and_get_rss(B: Float[Array, "n_samples n_features"], y: Float[Array, "n_samples"]) -> Float:
    """Solves the least squares problem and returns the Residual Sum of Squares."""
    coeffs, residuals, _, _ = jnp.linalg.lstsq(B, y, rcond=None)
    
    return jnp.sum(residuals)


class HingeBasis(eqx.Module):
    """
    Represents a single hinge basis function: max(0, sign * (x_i - knot)).
    """
    feature_idx: int = eqx.field(static=True)
    knot: float
    sign: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, "n_samples n_features"]) -> Float[Array, "n_samples"]:
        """Evaluates the basis function on input data x."""
        term = x[:, self.feature_idx] - self.knot
        return jnp.maximum(0, self.sign * term)
    
    def __repr__(self):
        s = '+' if self.sign > 0 else '-'
        return f"max(0, {s}(x_{self.feature_idx} - {self.knot:.2f}))"



class MARSModel(eqx.Module):
    """
    The complete MARS model, composed of an intercept and a set of basis functions
    combined by a linear layer.
    """
    basis_functions: list[HingeBasis]
    linear: eqx.nn.Linear

    def __init__(self, n_basis_fns: int):
        
        
        key = jax.random.PRNGKey(0) 
        self.basis_functions = []
        self.linear = eqx.nn.Linear(n_basis_fns, 1, use_bias=True, key=key)

        
    def __call__(self, x: Float[Array, "n_samples n_features"]) -> Float[Array, "n_samples"]:
        """Predicts the output for input data x."""
        if not self.basis_functions:
            
            return jnp.full((x.shape[0], 1), self.linear.bias)

        
        
        B = jnp.stack([bf(x) for bf in self.basis_functions], axis=1)
    
        
        
        
        return self.linear(B.T).T
    def __repr__(self):
        if not self.basis_functions:
            return f"MARSModel(intercept={self.linear.bias.item():.3f})"
        
        
        w = self.linear.weight.flatten()
        b = self.linear.bias.item()
        
        terms = [f"{b:.3f}"]
        terms.extend([f"{w[i]:.3f} * {bf}" for i, bf in enumerate(self.basis_functions)])
        return "MARSModel(\n  " + " +\n  ".join(terms) + "\n)"



class MARSFitter:
    """
    Drives the two-stage process of building and pruning a MARS model.
    """
    def __init__(self, max_terms: int = 10, max_interaction_degree: int = 1):
        if max_interaction_degree > 1:
            raise NotImplementedError("Interaction terms are not implemented in this example.")
        self.max_terms = max_terms
    
    def _get_design_matrix(self, model: MARSModel, x: Array) -> Array:
        """Constructs the design matrix B from the model's basis functions."""
        B = jnp.ones((x.shape[0], 1)) 
        if model.basis_functions:
            basis_evals = jnp.stack([bf(x) for bf in model.basis_functions], axis=1)
            B = jnp.concatenate([B, basis_evals], axis=1)
        return B

    def _refit_model(self, model: MARSModel, x: Array, y: Array) -> tuple[MARSModel, Float]:
        """Fits the linear coefficients of a model with a fixed basis."""
        B = self._get_design_matrix(model, x)
        coeffs, residuals, _, _ = jnp.linalg.lstsq(B, y, rcond=None)
        
        
        new_bias = coeffs[0]
        new_weights = coeffs[1:].reshape(1, -1) if len(coeffs) > 1 else jnp.empty((1,0))
        
        
        n_basis = len(model.basis_functions)
        key = jax.random.PRNGKey(0) 
        new_linear = eqx.nn.Linear(
            in_features=n_basis, out_features=1, use_bias=True, key=key
        )
        new_linear = eqx.tree_at(lambda l: l.weight, new_linear, new_weights)
        new_linear = eqx.tree_at(lambda l: l.bias, new_linear, new_bias)
        
        updated_model = eqx.tree_at(lambda m: m.linear, model, new_linear)
        
        rss = jnp.sum(residuals)
        return updated_model, rss

    def _calculate_gcv(self, rss: Float, num_samples: int, num_params: int) -> Float:
        """Calculates the Generalized Cross-Validation score."""
        denominator = (1.0 - num_params / num_samples)**2
        return rss / (num_samples * denominator)

    def fit(self, x: Float[Array, "n_samples n_features"], y: Float[Array, "n_samples"]) -> MARSModel:
        """
        Fits the MARS model using the forward and backward passes.
        """
        num_samples, num_features = x.shape
        
        
        
        initial_model = MARSModel(n_basis_fns=0)
        model, current_rss = self._refit_model(initial_model, x, y)
        
        forward_pass_models = [model]
        
        print("--- Starting Forward Pass ---")
        pbar = tqdm(total=self.max_terms)
        while len(forward_pass_models[-1].basis_functions) < self.max_terms:
            last_model = forward_pass_models[-1]
            best_rss_gain = float('inf')
            best_new_basis = None
            
            
            
            
            
            
            parent_bases = [None] 
            
            for parent_basis in parent_bases: 
                for feature_idx in range(num_features):
                    
                    knot_candidates = jnp.unique(x[:, feature_idx])
                    
                    for knot in knot_candidates:
                        for sign in [-1, 1]:
                            candidate_basis = HingeBasis(feature_idx=feature_idx, knot=knot, sign=sign)
                            
                            
                            temp_basis_list = last_model.basis_functions + [candidate_basis]
                            temp_model = MARSModel(n_basis_fns=len(temp_basis_list))
                            temp_model = eqx.tree_at(lambda m: m.basis_functions, temp_model, temp_basis_list)
                            
                            _, rss = self._refit_model(temp_model, x, y)
                            
                            if rss < best_rss_gain:
                                best_rss_gain = rss
                                best_new_basis = candidate_basis

            if best_new_basis and best_rss_gain < current_rss:
                new_basis_list = last_model.basis_functions + [best_new_basis]
                new_model = MARSModel(n_basis_fns=len(new_basis_list))
                new_model = eqx.tree_at(lambda m: m.basis_functions, new_model, new_basis_list)
                
                fitted_model, rss = self._refit_model(new_model, x, y)
                forward_pass_models.append(fitted_model)
                current_rss = rss
                pbar.update(1)
                pbar.set_description(f"Terms: {len(fitted_model.basis_functions)}, RSS: {rss:.4f}")
            else:
                
                break
        pbar.close()

        
        print("\n--- Starting Backward Pass ---")
        best_gcv = float('inf')
        final_model = None
        
        
        for model_to_prune in tqdm(reversed(forward_pass_models)):
            best_pruned_gcv_for_this_size = float('inf')
            best_pruned_model_for_this_size = None

            if not model_to_prune.basis_functions:
                _, rss = self._refit_model(model_to_prune, x, y)
                num_params = 1 
                gcv = self._calculate_gcv(rss, num_samples, num_params)
                if gcv < best_gcv:
                    best_gcv = gcv
                    final_model = model_to_prune
                continue

            
            for i in range(len(model_to_prune.basis_functions)):
                
                pruned_basis_list = [bf for j, bf in enumerate(model_to_prune.basis_functions) if i != j]
                pruned_model = MARSModel(n_basis_fns=len(pruned_basis_list))
                pruned_model = eqx.tree_at(lambda m: m.basis_functions, pruned_model, pruned_basis_list)
                
                fitted_pruned_model, rss = self._refit_model(pruned_model, x, y)
                
                num_params = len(fitted_pruned_model.basis_functions) + 1 
                gcv = self._calculate_gcv(rss, num_samples, num_params)
                
                if gcv < best_pruned_gcv_for_this_size:
                    best_pruned_gcv_for_this_size = gcv
                    best_pruned_model_for_this_size = fitted_pruned_model

            
            if best_pruned_gcv_for_this_size < best_gcv:
                best_gcv = best_pruned_gcv_for_this_size
                final_model = best_pruned_model_for_this_size

        print(f"\nBest model found with {len(final_model.basis_functions)} terms and GCV: {best_gcv:.4f}")
        return final_model



if __name__ == '__main__':
    
    key = jax.random.PRNGKey(42)
    x_key, noise_key = jax.random.split(key)
    
    n_samples = 100
    n_features = 1 
    
    X = jax.random.uniform(x_key, shape=(n_samples, n_features), minval=-5, maxval=5)
    
    y_true = jnp.sin(X[:, 0]) + 0.2 * X[:, 0]
    
    y = y_true + jax.random.normal(noise_key, shape=(n_samples,)) * 0.2

    
    fitter = MARSFitter(max_terms=20)
    best_model = fitter.fit(X, y)
    
    print("\n--- Final Model ---")
    print(best_model)
    
    x_plot = jnp.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    
    @eqx.filter_jit
    def predict(model, x_in):
        
        return model(x_in).flatten()

    y_pred = predict(best_model, x_plot)

    sort_indices = jnp.argsort(X[:, 0])
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Noisy Data')
    plt.plot(x_plot, y_pred, color='red', linewidth=3, label='Fitted MARS Model')
    plt.title('JAX/Equinox MARS Implementation')
    plt.xlabel('Feature X')
    plt.ylabel('Target y')
    plt.legend()
    plt.grid(True)
    plt.show()
