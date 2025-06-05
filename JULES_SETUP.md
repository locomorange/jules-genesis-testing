# Jules Development Environment Setup and Guidelines

This document provides instructions for setting up the development environment for this repository and guidelines for how Jules (the AI assistant) should interact with it, including validation and testing procedures.

## Initial Setup

To ensure a consistent development environment, follow these steps:

1.  **Clone the repository (if not already done):**
    ```bash
    # git clone <repository_url>
    # cd <repository_name>
    ```

2.  **Create and activate a Python virtual environment:**
    It is strongly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # venv\Scripts\activate    # For Windows
    ```

3.  **Install dependencies:**
    This project uses `requirements.txt` to list its dependencies. Install them using `uv` (if available) or `pip`.
    ```bash
    # If you have uv installed:
    uv pip install .
    # Alternatively, using pip:
    # pip install .
    ```
    This command should be considered the primary "do setup" step for Jules.

## Run to Validate

To validate that the core functionality of the project is working correctly after setup, you can run the main example script.

1.  **Execute the Policy Iteration example for FrozenLake-v1:**
    ```bash
    python algorithms/gymnasium_examples/model_based/policy_iteration_frozenlake.py
    ```
    **Expected Output:**
    Upon successful execution, you should see output in the console detailing:
    *   The iterative process of policy evaluation and improvement.
    *   The calculated optimal value function.
    *   The derived optimal policy.
    *   Results from test episodes run with the optimal policy (e.g., average reward, goal attainment).
    Seeing this output without errors indicates a successful basic setup and that the primary algorithm is functioning.

2.  **Test Script (Future Use):**
    A script named `run_test.sh` is intended for running automated tests.
    ```bash
    sh run_test.sh
    ```
    Currently, this repository does not have a dedicated test suite. See the "Recommendations" section for guidance on adding tests.

## Recommendations for Testing

To improve code quality and ensure robustness, adding automated tests is highly recommended.

### Recommended Test Frameworks

*   **`unittest`**: Python's built-in test framework.
*   **`pytest`**: A popular third-party framework known for its simplicity and powerful features.

### Test Design Strategies for `policy_iteration_frozenlake.py`

Consider the following types of tests for the existing `policy_iteration_frozenlake.py` script:

1.  **Unit Tests (Core Algorithm Logic):**
    *   Focus on testing individual functions or components of the policy iteration algorithm if they are modularized (e.g., separate functions for policy evaluation and policy improvement).
    *   **Example Test Cases:**
        *   **Policy Evaluation:** Given a small, predefined Markov Decision Process (MDP) (i.e., states, actions, transition probabilities, rewards) and a fixed policy, does the policy evaluation step correctly calculate the value function? You can manually calculate the expected values for a tiny MDP to verify.
        *   **Policy Improvement:** Given a value function for a small MDP, does the policy improvement step correctly identify the optimal action for each state?
        *   Test edge cases, such as an MDP where all rewards are zero, or a deterministic environment.

2.  **Integration Tests (Algorithm with `FrozenLake-v1` Environment):**
    *   Test the `policy_iteration_frozenlake.py` script as a whole, interacting with the `FrozenLake-v1` environment.
    *   **Example Test Cases:**
        *   **Deterministic FrozenLake:** If `FrozenLake-v1` is configured to be non-slippery (`is_slippery=False`), the environment becomes deterministic. In this case, the optimal policy is well-defined and known. Test if the algorithm finds this specific optimal policy.
        *   **Performance on Stochastic FrozenLake:** For the standard stochastic environment, while the exact optimal policy might vary slightly due to randomness in tie-breaking, the algorithm should consistently achieve a certain level of performance (e.g., average reward over many episodes, or a high probability of reaching the goal).
        *   Check if the algorithm converges within a reasonable number of iterations for the given environment.

### Configuring `run_test.sh`

Once tests are implemented (e.g., in a `tests/` directory):

*   If using `unittest`, `run_test.sh` could contain:
    ```bash
    python -m unittest discover -s tests
    ```
*   If using `pytest`, `run_test.sh` could simply be:
    ```bash
    pytest
    ```

This structured approach to testing will significantly benefit the maintainability and reliability of the codebase.
