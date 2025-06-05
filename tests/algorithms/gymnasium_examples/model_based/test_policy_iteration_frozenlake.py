import unittest
from unittest.mock import patch
import numpy as np
import io # For capturing print output
import gymnasium # Required for a more complete mock environment

# Assuming policy_iteration_frozenlake.py is in the parent directory and can be imported.
# Adjust the import path as necessary based on your project structure.
from algorithms.gymnasium_examples.model_based.policy_iteration_frozenlake import print_value_function, print_policy, policy_evaluation, policy_iteration, test_policy

class MockEnv(gymnasium.Env):
    def __init__(self, n_states, n_actions, policy_to_test, expected_actions=None):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(n_actions)
        self.observation_space = gymnasium.spaces.Discrete(n_states)
        self.n_states = n_states
        self.current_state = 0
        self.episodes_run = 0
        self.steps_taken_in_episode = 0
        self.max_steps_per_episode = 5 # For testing truncation
        self.policy_to_test = policy_to_test # The policy being tested by the main function
        self.expected_actions = expected_actions if expected_actions is not None else {} # state: expected_action
        self.rewards_for_actions = {} # (state, action): reward
        self.next_states_for_actions = {} # (state, action): next_state
        self.terminated_for_actions = {} # (state, action): terminated
        self.total_reward_for_ep = 0
        self.actions_taken_in_ep = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = 0 # Always start at state 0 for simplicity
        self.episodes_run += 1
        self.steps_taken_in_episode = 0
        self.total_reward_for_ep = 0
        self.actions_taken_in_ep = []
        return self.current_state, {"info": "mock_reset"}

    def step(self, action):
        # Check if the action taken is consistent with the policy being tested
        # This is a bit meta, as test_policy calls env.step(policy[state])
        # So, action here *is* policy[self.current_state]
        if self.current_state in self.expected_actions:
            assert action == self.expected_actions[self.current_state], \
                f"Action {action} taken in state {self.current_state} does not match expected action {self.expected_actions[self.current_state]}"

        self.actions_taken_in_ep.append(action)

        reward = self.rewards_for_actions.get((self.current_state, action), 0)
        next_s = self.next_states_for_actions.get((self.current_state, action), self.current_state)
        terminated = self.terminated_for_actions.get((self.current_state, action), False)

        self.total_reward_for_ep += reward
        self.current_state = next_s
        self.steps_taken_in_episode += 1

        truncated = self.steps_taken_in_episode >= self.max_steps_per_episode
        if terminated or truncated:
            # Store episode reward for verification if needed, though test_policy calculates its own avg
            pass

        return self.current_state, reward, terminated, truncated, {"info": "mock_step"}

    def render(self):
        # For 'ansi' mode, test_policy calls env.render(). We can make it return something.
        return f"Mock render: State {self.current_state}, Steps {self.steps_taken_in_episode}"

    def close(self):
        pass

    # --- Helper methods to configure the mock environment for specific tests ---
    def set_rewards_for_actions(self, rewards_map): # e.g., {(state, action): reward}
        self.rewards_for_actions = rewards_map

    def set_next_states_for_actions(self, next_states_map): # e.g., {(state, action): next_state}
        self.next_states_for_actions = next_states_map

    def set_terminated_for_actions(self, terminated_map): # e.g., {(state, action): terminated}
        self.terminated_for_actions = terminated_map

    def set_expected_actions(self, expected_actions_map): # e.g., {state: action}
        self.expected_actions = expected_actions_map

class TestPrintFunctions(unittest.TestCase):
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_value_function(self, mock_stdout):
        V = np.array([0.1, 0.2, 0.3, 0.4])
        rows, cols = 2, 2
        message = "Test Value Function:"
        print_value_function(V, rows, cols, message)
        output = mock_stdout.getvalue()
        self.assertIn(message, output)
        self.assertIn("0.1000", output) # Check if values are present
        self.assertIn("0.4000", output)
        # Check if it has roughly the correct number of lines (message + rows + separator)
        self.assertEqual(len(output.strip().split('\n')), rows + 2)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_policy(self, mock_stdout):
        policy = np.array([0, 1, 2, 3]) # Actions: Left, Down, Right, Up
        rows, cols = 2, 2
        message = "Test Policy:"
        action_symbols = {0: '<', 1: 'v', 2: '>', 3: '^'}
        print_policy(policy, rows, cols, message)
        output = mock_stdout.getvalue()
        self.assertIn(message, output)
        self.assertIn(action_symbols[0], output)
        self.assertIn(action_symbols[3], output)
        # Check if it has roughly the correct number of lines (message + rows + separator)
        self.assertEqual(len(output.strip().split('\n')), rows + 2)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_value_function_empty(self, mock_stdout):
        V = np.array([])
        rows, cols = 0, 0
        message = "Empty Value Function:"
        print_value_function(V, rows, cols, message)
        output = mock_stdout.getvalue()
        self.assertIn(message, output)
        # Should just print the message (separator becomes an empty line, stripped)
        self.assertEqual(len(output.strip().split('\n')), 1)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_policy_empty(self, mock_stdout):
        policy = np.array([])
        rows, cols = 0, 0
        message = "Empty Policy:"
        print_policy(policy, rows, cols, message)
        output = mock_stdout.getvalue()
        self.assertIn(message, output)
        # Should just print the message (separator becomes an empty line, stripped)
        self.assertEqual(len(output.strip().split('\n')), 1)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_value_function_1d(self, mock_stdout):
        V = np.array([0.1, 0.2, 0.3, 0.4])
        rows, cols = 1, 4
        message = "1D Value Function:"
        print_value_function(V, rows, cols, message)
        output = mock_stdout.getvalue()
        self.assertIn(message, output)
        self.assertIn("0.1000", output)
        self.assertIn("0.4000", output)
        self.assertEqual(len(output.strip().split('\n')), rows + 2)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_policy_1d(self, mock_stdout):
        policy = np.array([0, 1, 2, 3])
        rows, cols = 1, 4
        message = "1D Policy:"
        action_symbols = {0: '<', 1: 'v', 2: '>', 3: '^'}
        print_policy(policy, rows, cols, message)
        output = mock_stdout.getvalue()
        self.assertIn(message, output)
        self.assertIn(action_symbols[0], output)
        self.assertIn(action_symbols[3], output)
        self.assertEqual(len(output.strip().split('\n')), rows + 2)

class TestPolicyEvaluation(unittest.TestCase):
    def setUp(self):
        # Simple 2-state MDP (deterministic)
        # State 0: action 0 -> state 0 (reward 0), action 1 -> state 1 (reward 1)
        # State 1: terminal state (self-loop with reward 0 for all actions)
        self.n_states = 2
        self.n_actions = 2
        self.P = {
            0: { # State 0
                0: [(1.0, 0, 0.0, False)], # (prob, next_state, reward, terminated)
                1: [(1.0, 1, 1.0, False)]  # In a true terminal state, next_state might not matter or terminated=True
            },
            1: { # State 1 (acting as a terminal state for reward purposes, or just a sink)
                0: [(1.0, 1, 0.0, True)], # Stays in state 1, reward 0, terminated
                1: [(1.0, 1, 0.0, True)]
            }
        }
        self.gamma = 0.9
        self.theta = 1e-9

    def test_policy_evaluation_go_to_terminal(self):
        # Policy: from state 0, take action 1 (to state 1)
        # from state 1, action does not matter as it's terminal for this policy's purpose
        policy = np.array([1, 0]) # Action 1 in state 0, Action 0 in state 1 (irrelevant)

        # Expected V:
        # V(S1) = 0 (terminal)
        # V(S0) = R(S0, a1) + gamma * V(S1) = 1.0 + self.gamma * 0 = 1.0
        expected_V = np.array([1.0, 0.0])

        V = policy_evaluation(policy, self.P, self.n_states, self.n_actions, self.gamma, self.theta)
        np.testing.assert_array_almost_equal(V, expected_V, decimal=5)

    def test_policy_evaluation_stay_in_state0(self):
        # Policy: from state 0, take action 0 (to state 0)
        policy = np.array([0, 0]) # Action 0 in state 0

        # Expected V:
        # V(S1) = 0 (terminal, though not reached by this policy from S0)
        # V(S0) = R(S0, a0) + gamma * V(S0) = 0 + self.gamma * V(S0)
        # V(S0) * (1 - gamma) = 0  => V(S0) = 0
        expected_V = np.array([0.0, 0.0])

        V = policy_evaluation(policy, self.P, self.n_states, self.n_actions, self.gamma, self.theta)
        np.testing.assert_array_almost_equal(V, expected_V, decimal=5)

    def test_policy_evaluation_convergence_check(self):
        # Test that it converges even with a more complex (but still small) setup if needed
        # For this simple case, convergence is implicitly tested by the correctness of V.
        # Here, we can test with a slightly different gamma.
        policy = np.array([1, 0]) # Go to terminal
        gamma_conv_test = 0.5
        # V(S1) = 0
        # V(S0) = 1.0 + 0.5 * 0 = 1.0
        expected_V_conv = np.array([1.0, 0.0])
        V = policy_evaluation(policy, self.P, self.n_states, self.n_actions, gamma_conv_test, self.theta)
        np.testing.assert_array_almost_equal(V, expected_V_conv, decimal=5)

    def test_policy_evaluation_stochastic_rewards_and_transitions(self):
        # More complex P to ensure multiple transitions are handled
        P_complex = {
            0: { # State 0
                0: [(0.5, 0, 0.0, False), (0.5, 1, 10.0, False)], # Action 0: 50% stay, 50% go to S1 (reward 10)
                1: [(1.0, 1, 1.0, False)] # Action 1: Go to S1 (reward 1)
            },
            1: { # State 1 (Terminal state)
                0: [(1.0, 1, 0.0, True)],
                1: [(1.0, 1, 0.0, True)]
            }
        }
        # Policy: In S0, take action 0. In S1, take action 0 (irrelevant as S1 is terminal).
        policy = np.array([0, 0])
        gamma_c = 0.9

        # V(S1) = 0 (terminal)
        # V(S0) = Sum over s', r [ p(s',r | s=0, a=0) * (r + gamma_c * V(s')) ]
        # V(S0) = 0.5 * (0.0 + gamma_c * V(S0)) + 0.5 * (10.0 + gamma_c * V(S1))
        # V(S0) = 0.5 * gamma_c * V(S0) + 0.5 * (10.0 + gamma_c * 0)
        # V(S0) = 0.5 * gamma_c * V(S0) + 5.0
        # V(S0) * (1 - 0.5 * gamma_c) = 5.0
        # V(S0) = 5.0 / (1 - 0.5 * gamma_c)
        # V(S0) = 5.0 / (1 - 0.5 * 0.9) = 5.0 / (1 - 0.45) = 5.0 / 0.55
        expected_V_s0 = 5.0 / (1.0 - 0.5 * gamma_c)
        expected_V_complex = np.array([expected_V_s0, 0.0])

        V = policy_evaluation(policy, P_complex, self.n_states, self.n_actions, gamma_c, self.theta)
        np.testing.assert_array_almost_equal(V, expected_V_complex, decimal=5)

class TestPolicyIteration(unittest.TestCase):
    def setUp(self):
        # Simple 2-state MDP (deterministic)
        # State 0: Goal state (e.g., like a simplified FrozenLake goal)
        # State 1: Start state
        # Actions: 0 (e.g., UP), 1 (e.g., LEFT)
        # From S1:
        #   Action 0 (UP) -> S0 (reward 1, terminal)
        #   Action 1 (LEFT) -> S1 (reward 0, non-terminal)
        # From S0: (Terminal state, actions don't matter)
        #   Action 0 -> S0 (reward 0, terminal)
        #   Action 1 -> S0 (reward 0, terminal)

        self.n_states = 2
        self.n_actions = 2
        self.rows = 1 # For print functions, not critical for logic
        self.cols = 2 # For print functions
        self.gamma = 0.9
        self.theta = 1e-9

        self.P_test = {
            0: { # State 0 (Goal) - Terminal
                0: [(1.0, 0, 0.0, True)],
                1: [(1.0, 0, 0.0, True)]
            },
            1: { # State 1 (Start)
                0: [(1.0, 0, 1.0, True)],  # Action 0 (UP) leads to S0 (Goal) with reward 1
                1: [(1.0, 1, 0.0, False)]  # Action 1 (LEFT) stays in S1 with reward 0
            }
        }
        # Mock environment object with necessary attributes if policy_iteration uses them directly
        # However, policy_iteration primarily uses P, n_states, n_actions.
        # The 'env' argument in policy_iteration is mostly for context or if it were to call env.reset() etc.
        # which it doesn't seem to do internally based on the provided code.
        # For this test, we can pass a simple mock or None if only P, n_states, n_actions are truly used.
        # Let's create a minimal mock env for completeness.
        self.mock_env = unittest.mock.Mock()
        # policy_iteration doesn't seem to use env.observation_space.n or env.action_space.n directly,
        # as n_states and n_actions are passed separately.
        # It also doesn't use env.unwrapped.P as P is passed directly.

    def test_policy_iteration_simple_convergence(self):
        # Expected optimal policy:
        # S0: Any action (e.g., 0) as it's terminal.
        # S1: Action 0 (UP) to reach the goal S0.
        expected_optimal_policy = np.array([0, 0]) # Action 0 from S0 (irrelevant), Action 0 from S1

        # Expected optimal value function V*:
        # V*(S0) = 0 (terminal state)
        # V*(S1) = max_a Q(S1, a)
        # Q(S1, action 0) = R(S1,a0->S0) + gamma * V*(S0) = 1.0 + self.gamma * 0 = 1.0
        # Q(S1, action 1) = R(S1,a1->S1) + gamma * V*(S1)
        # If policy is to take action 0 from S1:
        # V(S1) = 1.0
        # If policy is to take action 1 from S1:
        # V(S1) = 0 + gamma * V(S1) => V(S1)*(1-gamma) = 0 => V(S1) = 0.
        # So, optimal V*(S1) = 1.0
        expected_optimal_V = np.array([0.0, 1.0]) # V for S0, V for S1

        # The `policy_iteration` function uses `print_policy` and `print_value_function`
        # which print to stdout. We should patch stdout to avoid clutter during tests.
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            optimal_policy, optimal_V = policy_iteration(
                self.mock_env, self.P_test, self.n_states, self.n_actions,
                self.rows, self.cols, self.gamma, self.theta
            )

        # Note: The order of states in my P_test and expected arrays is S0 (goal), S1 (start).
        # If the algorithm or indexing assumes state 0 is the typical start and higher states are further,
        # this might need adjustment. Let's assume state indices are just labels.
        # The policy iteration should find: policy[1] = 0 (from state 1, take action 0)
        # policy[0] can be anything as state 0 is terminal.

        # The algorithm initializes policy to np.zeros(), so policy[0] will be 0.
        np.testing.assert_array_equal(optimal_policy, expected_optimal_policy)
        np.testing.assert_array_almost_equal(optimal_V, expected_optimal_V, decimal=5)

    def test_policy_iteration_different_initial_policy(self):
        # What if the initial policy (hardcoded as np.zeros in policy_iteration)
        # was different? The algorithm should still converge to the same optimal.
        # This is harder to test without modifying the source or more complex mocking.
        # However, we can test a scenario where the initial all-zero policy is already optimal
        # or one step away.

        # Let's use a slightly different P where action 1 from S1 is initially better if V(S0) was negative (it won't be)
        # Or, more simply, confirm the result from the previous test is robust.
        # The current policy_iteration function hardcodes initial policy to all zeros.
        # We can test if it converges from that specific starting point.

        # Re-run with a different gamma to see if it affects the optimal policy outcome (it shouldn't for this MDP)
        # but values will change.
        gamma_alt = 0.5
        # V*(S0) = 0
        # V*(S1) = 1.0 + gamma_alt * 0 = 1.0
        expected_optimal_V_alt = np.array([0.0, 1.0])
        # Policy should remain the same.
        expected_optimal_policy_alt = np.array([0, 0])

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            optimal_policy_alt, optimal_V_alt = policy_iteration(
                self.mock_env, self.P_test, self.n_states, self.n_actions,
                self.rows, self.cols, gamma_alt, self.theta
            )

        np.testing.assert_array_equal(optimal_policy_alt, expected_optimal_policy_alt)
        np.testing.assert_array_almost_equal(optimal_V_alt, expected_optimal_V_alt, decimal=5)

class TestTestPolicy(unittest.TestCase):
    def setUp(self):
        self.n_states = 2
        self.n_actions = 2
        # Policy: S0 -> A0, S1 -> A1
        self.test_policy_array = np.array([0, 1])
        self.mock_env_instance = MockEnv(self.n_states, self.n_actions, self.test_policy_array)

    def test_runs_n_episodes(self):
        n_episodes = 3
        # Configure simple transitions: S0,A0 -> S0 (term); S1,A1 -> S0 (term)
        self.mock_env_instance.set_next_states_for_actions({(0,0):0, (1,1):0})
        self.mock_env_instance.set_terminated_for_actions({(0,0):True, (1,1):True})

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            test_policy(self.mock_env_instance, self.test_policy_array, n_episodes=n_episodes, render_mode=None)

        self.assertEqual(self.mock_env_instance.episodes_run, n_episodes)
        output = mock_stdout.getvalue()
        self.assertIn(f"--- {n_episodes}エピソードでの方策テスト ---", output) # Message from test_policy
        self.assertIn(f"{n_episodes}エピソードの平均報酬:", output) # Message from test_policy

    def test_reward_accumulation_and_averaging(self):
        n_episodes = 2
        # S0,A0 -> S0 (R=1, term)
        # S1,A1 -> S0 (R=2, term)
        # Policy is [0,1]. Env starts in S0.
        # Ep1: Start S0, action policy[0]=0. Reward 1. Total 1.
        # Ep2: Start S0, action policy[0]=0. Reward 1. Total 1.
        # Avg reward should be 1.0
        self.mock_env_instance.set_rewards_for_actions({(0,0):1, (1,1):2}) # S0,A0 gets R1; S1,A1 gets R2
        self.mock_env_instance.set_next_states_for_actions({(0,0):0, (1,1):0})
        self.mock_env_instance.set_terminated_for_actions({(0,0):True, (1,1):True})

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            test_policy(self.mock_env_instance, self.test_policy_array, n_episodes=n_episodes, render_mode=None)

        output = mock_stdout.getvalue()
        self.assertIn(f"エピソード 1 終了。総報酬: 1", output) # test_policy prints int for whole number reward
        self.assertIn(f"エピソード 2 終了。総報酬: 1", output)
        self.assertIn(f"{n_episodes}エピソードの平均報酬: 1.00", output)
        # Success rate check (reward > 0)
        self.assertIn(f"成功率: 1.00", output)


    def test_uses_policy_correctly(self):
        n_episodes = 1
        # Policy: S0 -> A0, S1 -> A1
        # Expected actions for mock_env to verify: S0 takes A0.
        self.mock_env_instance.set_expected_actions({0: 0}) # Expect action 0 if in state 0
        self.mock_env_instance.set_rewards_for_actions({(0,0):1})
        self.mock_env_instance.set_terminated_for_actions({(0,0):True}) # Terminate on first step

        with patch('sys.stdout', new_callable=io.StringIO):
            test_policy(self.mock_env_instance, self.test_policy_array, n_episodes=n_episodes)

        # Assertion is inside MockEnv.step(). If it fails, an AssertionError will be raised.
        # Check that at least one action was taken in the episode
        self.assertTrue(len(self.mock_env_instance.actions_taken_in_ep) > 0)
        self.assertEqual(self.mock_env_instance.actions_taken_in_ep[0], self.test_policy_array[0])


    def test_truncation(self):
        n_episodes = 1
        # Make MockEnv run for long enough for test_policy's own truncation to hit
        self.mock_env_instance.max_steps_per_episode = 105 # Must be > test_policy's limit of 100

        # Policy S0->A0. Env starts S0. Action A0.
        # S0,A0 -> S0 (R=0, term=False) - to allow multiple steps without natural termination
        self.mock_env_instance.set_next_states_for_actions({(0,0):0})
        self.mock_env_instance.set_terminated_for_actions({(0,0):False}) # Never terminate from env

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            test_policy(self.mock_env_instance, self.test_policy_array, n_episodes=n_episodes)

        output = mock_stdout.getvalue()
        # test_policy has its own step_count > 100 truncation
        self.assertIn("エピソードが最大ステップ数に達したため打ち切り。", output)
        # Verify that many steps were actually taken by the environment
        self.assertGreater(self.mock_env_instance.steps_taken_in_episode, 100)


    @patch('sys.stdout', new_callable=io.StringIO)
    def test_render_mode_ansi(self, mock_stdout):
        n_episodes = 1
        self.mock_env_instance.set_terminated_for_actions({(0,0):True}) # Quick termination

        test_policy(self.mock_env_instance, self.test_policy_array, n_episodes=n_episodes, render_mode='ansi')
        output = mock_stdout.getvalue()
        self.assertIn("Mock render:", output) # Check if our mock_env.render() was called
        self.assertIn(f"ステップ 0: 状態: 0, 行動: {self.test_policy_array[0]}", output) # Check for step details print

if __name__ == '__main__':
    unittest.main()
