class DummyAlgorithm:
    def __init__(self, config=None):
        """
        A simple dummy algorithm for testing purposes.
        It can optionally take a configuration dictionary.
        """
        self.config = config if config is not None else {}
        print("DummyAlgorithm initialized.")

    def predict(self, state):
        """
        Given a state, returns the state itself.
        This is a placeholder for actual algorithm logic.
        """
        print(f"DummyAlgorithm predict called with state: {state}")
        return state

    def get_config(self):
        """
        Returns the configuration of the algorithm.
        """
        return self.config

if __name__ == '__main__':
    # Example usage
    algo = DummyAlgorithm(config={"param1": "value1", "mode": "test"})
    test_state = {"sensor_data": [1, 2, 3], "status": "active"}
    prediction = algo.predict(test_state)
    print(f"Prediction: {prediction}")
    print(f"Config: {algo.get_config()}")
