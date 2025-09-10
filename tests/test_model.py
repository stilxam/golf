import unittest
import jax
import jax.numpy as jnp
from plf.model import PiecewiseModel


class TestPiecewiseModel(unittest.TestCase):
    def test_runs_and_shapes(self):
        key = jax.random.PRNGKey(0)
        x = jnp.linspace(0.0, 1.0, 20)
        model = PiecewiseModel(n_segments=4, x_range=(0.0, 1.0), key=key)
        y_pred = jax.vmap(model)(x)
        self.assertEqual(y_pred.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
