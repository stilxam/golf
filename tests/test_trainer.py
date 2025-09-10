import unittest
import jax
import jax.numpy as jnp
from plf.model import PiecewiseModel
from plf.trainer import fit

class TestTrainer(unittest.TestCase):
    def test_fit_runs(self):
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(0.0, 1.0, 100)
        y = jnp.sin(x * jnp.pi) + 0.1 * jax.random.normal(key, x.shape)

        model = PiecewiseModel(n_segments=3, x_range=(0.0, 1.0), key=key)
        
        # Run fit for a few iterations
        trained_model = fit(model, x, y, n_iterations=10, verbose=False)
        
        self.assertIsInstance(trained_model, PiecewiseModel)
        
        # Check that predictions can be made
        y_pred = jax.vmap(trained_model)(x)
        self.assertEqual(y_pred.shape, x.shape)

if __name__ == "__main__":
    unittest.main()
