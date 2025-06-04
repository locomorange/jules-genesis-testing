# Markov Decision Processes (MDPs)

## 1. What is an MDP and its role in Reinforcement Learning?

A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. It's a fundamental concept in Reinforcement Learning (RL).

In RL, an **agent** learns to make a sequence of decisions by interacting with an **environment**. The goal is to maximize some notion of cumulative reward. MDPs provide the formal language to describe this interaction. The environment is typically modeled as an MDP. The agent observes the state of the environment and chooses an action. The environment then transitions to a new state and provides a reward to the agent. The agent uses this feedback (new state and reward) to learn which actions are good in which states.

## 2. The Components of an MDP

An MDP is formally defined by the following components:

*   **States (S):** A set of all possible situations or configurations the agent can be in. For example, in a game of chess, a state would be the specific arrangement of all pieces on the board. In a robot navigation task, a state could be the robot's current location and orientation.

*   **Actions (A):** A set of all possible moves the agent can take in a given state. For example, in chess, actions are the legal moves for each piece. For the robot, actions could be "move forward," "turn left," or "turn right." The set of available actions might depend on the current state, denoted as A(s).

*   **Transition Probability Function (P):** This function, often denoted as `P(s' | s, a)`, defines the dynamics of the environment. It specifies the probability of transitioning from a state `s` to a new state `s'` after taking action `a`.
    *   `s`: Current state
    *   `a`: Action taken by the agent
    *   `s'`: Next state
    *   So, `P(s' | s, a)` is the probability that taking action `a` in state `s` will lead to state `s'`.
    *   The sum of probabilities for all possible next states `s'` given a state `s` and action `a` must be 1:  Σ<sub>s'∈S</sub> P(s' | s, a) = 1.

*   **Reward Function (R):** This function, often denoted as `R(s, a, s')` or sometimes `R(s)` or `R(s,a)`, defines the immediate reward received by the agent.
    *   `R(s, a, s')`: The reward received after transitioning from state `s` to state `s'` as a result of action `a`.
    *   Sometimes, the reward is simplified to depend only on the state `s` (`R(s)`) or the state-action pair (`R(s,a)`).
    *   Rewards can be positive (for desirable outcomes) or negative (for undesirable outcomes, i.e., penalties). The agent's goal is to maximize the total reward it accumulates over time.

*   **Discount Factor (γ - Gamma):** This is a value between 0 and 1 (inclusive, 0 ≤ γ ≤ 1). It determines the importance of future rewards compared to immediate rewards.
    *   If γ = 0, the agent is "myopic" and only cares about the immediate reward.
    *   If γ is close to 1, the agent is "farsighted" and takes future rewards strongly into account.
    *   The discount factor ensures that the sum of rewards over an infinite horizon (in some problems) remains finite. It also reflects the uncertainty about the future; a reward in the distant future might be less certain than an immediate one.

## 3. The Markov Property

A key assumption in MDPs is the **Markov Property**. This property states that the future is independent of the past, given the present.

More formally, the probability of transitioning to state `s'` and receiving reward `r` depends *only* on the current state `s` and the action `a` taken. It does not depend on the sequence of states and actions that led to the current state `s`.

`P(s<sub>t+1</sub> = s', R<sub>t+1</sub> = r | s<sub>t</sub>, a<sub>t</sub>, s<sub>t-1</sub>, a<sub>t-1</sub>, ..., s<sub>0</sub>, a<sub>0</sub>) = P(s<sub>t+1</sub> = s', R<sub>t+1</sub> = r | s<sub>t</sub>, a<sub>t</sub>)`

In simpler terms, the current state `s<sub>t</sub>` encapsulates all the necessary information from the history to make an optimal decision. You don't need to know how you got to state `s`; all that matters is that you are in state `s`. This property greatly simplifies the modeling and solution of RL problems.

## 4. The Goal in an MDP

The primary goal of an agent in an MDP is to find a **policy** (π) that maximizes the **expected cumulative discounted reward**.

*   **Policy (π):** A policy is a mapping from states to actions. It dictates what action the agent should take in each state.
    *   Deterministic policy: `π(s) = a` (in state `s`, always take action `a`).
    *   Stochastic policy: `π(a | s) = P(A<sub>t</sub> = a | S<sub>t</sub> = s)` (in state `s`, the probability of taking action `a`).

*   **Expected Cumulative Discounted Reward:** The agent aims to maximize the sum of discounted rewards it expects to receive over time. This is often called the **return** or **value**. For a sequence of rewards `R<sub>t+1</sub>, R<sub>t+2</sub>, R<sub>t+3</sub>, ...` starting from time `t`, the discounted return is:
    `G<sub>t</sub> = R<sub>t+1</sub> + γR<sub>t+2</sub> + γ<sup>2</sup>R<sub>t+3</sub> + ... = Σ<sub>k=0</sub><sup>∞</sup> γ<sup>k</sup>R<sub>t+k+1</sub>`

The agent learns the optimal policy (π*) by interacting with the MDP. This optimal policy π* is the one that achieves the highest possible expected return from all states. Various algorithms in RL, like Value Iteration, Policy Iteration, Q-learning, and SARSA, are designed to find or approximate this optimal policy.
