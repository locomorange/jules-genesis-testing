# Policy Iteration Algorithm Explained

## 1. Introduction

Policy Iteration is a classic algorithm in Reinforcement Learning used to find an **optimal policy (π\*)** for a given Markov Decision Process (MDP). An optimal policy is one that maximizes the expected cumulative discounted reward from any starting state.

Policy Iteration is considered a **model-based algorithm**. This means it requires full knowledge of the MDP's dynamics, specifically the transition probabilities `p(s',r|s,a)` (the probability of transitioning to state `s'` and receiving reward `r`, given current state `s` and action `a`) and the discount factor `γ`.

## 2. Overall Process

The Policy Iteration algorithm works by iteratively performing two main steps until the policy no longer improves, indicating that an optimal policy has been found:

1.  **Policy Evaluation:** Given the current policy `π`, calculate the state-value function `V<sup>π</sup>(s)` for all states `s`. This function represents the expected return starting from state `s` and following policy `π`.
2.  **Policy Improvement:** Using the state-value function `V<sup>π</sup>(s)` calculated in the evaluation step, improve the current policy `π` by making it greedy with respect to `V<sup>π</sup>(s)`. This results in a new, potentially better, policy `π'`.

This cycle of evaluation and improvement is repeated:
`π<sub>0</sub> → [Evaluate] → V<sup>π<sub>0</sub></sup> → [Improve] → π<sub>1</sub> → [Evaluate] → V<sup>π<sub>1</sub></sup> → [Improve] → π<sub>2</sub> → ... → π* → V*`

The process continues until the policy `π` converges, meaning it no longer changes between iterations. At this point, `π` is guaranteed to be an optimal policy `π*`.

## 3. Policy Evaluation Step

*   **Objective:** For a given policy `π`, determine the value `V<sup>π</sup>(s)` for every state `s ∈ S`. This value is the expected discounted sum of future rewards when starting in state `s` and following policy `π` thereafter.

*   **Method:** The state-value function `V<sup>π</sup>(s)` is defined by the Bellman expectation equation for policy `π`:
    `V<sup>π</sup>(s) = Σ<sub>a</sub> π(a|s) Σ<sub>s',r</sub> p(s',r|s,a) [r + γV<sup>π</sup>(s')]`

    *   `π(a|s)`: The probability of taking action `a` in state `s` under policy `π`.
    *   `p(s',r|s,a)`: The transition probability of ending up in state `s'` with reward `r` from state `s` and action `a`.
    *   `γ`: The discount factor.

    If the policy `π` is deterministic, meaning `π(s)` directly specifies an action `a` for each state `s`, then `π(a|s)` is 1 for `a = π(s)` and 0 otherwise. The equation simplifies to:
    `V<sup>π</sup>(s) = Σ<sub>s',r</sub> p(s',r|s,π(s)) [r + γV<sup>π</sup>(s')]` (where `π(s)` is the action taken by the policy in state `s`)

*   **Iteration (Iterative Policy Evaluation):**
    To find `V<sup>π</sup>(s)`, Policy Evaluation typically involves an iterative process. Starting with arbitrary initial values `V<sub>0</sub><sup>π</sup>(s)` (often all zeros), the Bellman expectation equation is used as an update rule:
    `V<sub>k+1</sub><sup>π</sup>(s) ← Σ<sub>a</sub> π(a|s) Σ<sub>s',r</sub> p(s',r|s,a) [r + γV<sub>k</sub><sup>π</sup>(s')]`

    This update is repeatedly applied for all states `s` (a "sweep" through the state space) in each iteration `k`. The sequence `V<sub>0</sub>, V<sub>1</sub>, V<sub>2</sub>, ...` is guaranteed to converge to the true `V<sup>π</sup>` as `k → ∞`, provided `γ < 1` or the MDP eventually reaches a terminal state. In practice, the iteration stops when the maximum change in values between iterations (`max<sub>s</sub> |V<sub>k+1</sub><sup>π</sup>(s) - V<sub>k</sub><sup>π</sup>(s)|`) falls below a small threshold (theta, θ).

## 4. Policy Improvement Step

*   **Objective:** Given the state-value function `V<sup>π</sup>` for the current policy `π` (obtained from the Policy Evaluation step), generate a new policy `π'` that is better than or equal to `π`.

*   **Method:** The policy is improved by acting greedily with respect to the current value function `V<sup>π</sup>`. For each state `s`, the new policy `π'` selects the action `a` that maximizes the expected reward if we were to take action `a` and then follow the existing policy `π` for subsequent states (though in practice, we use `V<sup>π</sup>` which already accounts for future rewards under `π`).

    More precisely, for each state `s`, `π'` is updated to choose the action `a` that maximizes the **action-value function Q<sup>π</sup>(s,a)**:
    `Q<sup>π</sup>(s,a) = Σ<sub>s',r</sub> p(s',r|s,a) [r + γV<sup>π</sup>(s')]`

    The new policy `π'` is then defined as:
    `π'(s) = argmax<sub>a</sub> Q<sup>π</sup>(s,a) = argmax<sub>a</sub> Σ<sub>s',r</sub> p(s',r|s,a) [r + γV<sup>π</sup>(s')]`

    This means that for each state `s`, the new policy `π'(s)` will select the action `a` that leads to the highest expected one-step lookahead value, considering the immediate reward and the discounted value of the subsequent state (as estimated by `V<sup>π</sup>`).

*   **Policy Stability:**
    After computing the new policy `π'`, we compare it to the old policy `π`.
    *   If `π'(s) = π(s)` for all states `s`, then the policy has stabilized. This means that acting greedily with respect to `V<sup>π</sup>` does not produce a different policy. At this point, `π` is an optimal policy `π*`, and `V<sup>π</sup>` is the optimal value function `V*`. The algorithm has converged.
    *   If `π'(s) ≠ π(s)` for at least one state `s`, then the new policy `π'` is strictly better than or equal to `π`. We set `π ← π'` and repeat the process, starting with another round of Policy Evaluation for this new policy `π'`.

    This is based on the **Policy Improvement Theorem**, which states that if `π'` is the greedy policy with respect to `V<sup>π</sup>`, then `V<sup>π'</sup>(s) ≥ V<sup>π</sup>(s)` for all `s ∈ S`. If there is strict inequality for any state, then the new policy is strictly better.

## 5. Convergence

Policy Iteration is guaranteed to converge to an optimal policy `π*` and its corresponding optimal state-value function `V*` in a finite number of iterations, provided the MDP is finite (i.e., has a finite number of states and actions).

The reasoning for this convergence is as follows:
1.  **Policy Improvement:** Each policy improvement step, unless the policy is already optimal, generates a strictly better policy (i.e., `V<sup>π'</sup>(s) ≥ V<sup>π</sup>(s)` for all states, with strict inequality for at least one state if `π` is not optimal).
2.  **Finite Policies:** For a finite MDP, there is a finite number of possible deterministic policies (specifically, `|A|<sup>|S|</sup>`, where `|A|` is the number of actions and `|S|` is the number of states).
3.  **Guaranteed Convergence:** Since each iteration (that changes the policy) yields a strictly better policy, and there are only a finite number of distinct policies, Policy Iteration cannot cycle indefinitely through non-optimal policies. It must eventually reach a policy that cannot be further improved, which is, by definition, an optimal policy.

Once the policy improvement step fails to change the policy, the Bellman optimality equation holds for `V<sup>π</sup>`, and thus `π` is optimal.
`V*(s) = max<sub>a</sub> Σ<sub>s',r</sub> p(s',r|s,a) [r + γV*(s')]`
