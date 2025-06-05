import unittest
from algorithms.genesis_algorithms.dummy_algorithm import DummyAlgorithm

class TestDummyAlgorithm(unittest.TestCase):

    def test_predict_returns_state(self):
        """Test that the predict method returns the input state."""
        algo = DummyAlgorithm()
        test_state = {"data": [1, 2, 3], "info": "test_state"}
        self.assertEqual(algo.predict(test_state), test_state)

    def test_initialization_with_config(self):
        """Test algorithm initialization with a configuration."""
        config = {"param_a": 10, "param_b": "active"}
        algo = DummyAlgorithm(config=config)
        self.assertEqual(algo.get_config(), config)

    def test_initialization_without_config(self):
        """Test algorithm initialization without a configuration."""
        algo = DummyAlgorithm()
        self.assertEqual(algo.get_config(), {})

if __name__ == '__main__':
    unittest.main()
