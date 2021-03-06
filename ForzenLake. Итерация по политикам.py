import gym
import numpy as np
ENV_NAME = "FrozenLake-v0"
env = gym.make(ENV_NAME)
print(type(env.observation_space.n))


def compute_value_function(policy, gamma=1.0):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-10
    while True:
        update_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = policy[state]
            value_table[state] = sum([trans_prob*(reward_prob + gamma * update_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in env.P[state][action]])
        if (np.sum(np.fabs(update_value_table - value_table))) <= threshold:
            break
    return value_table


def extract_policy(value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, done = next_sr
                Q[action] += trans_prob*(reward_prob+gamma*value_table[next_state])
        policy[state] = np.argmax(Q)
    return policy


def policy_iteration(env, gamma=1.0):
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 1000
    for i in range(no_of_iterations):
        new_value_func = compute_value_function(random_policy)
        new_policy = extract_policy(new_value_func)
        if np.all(random_policy == new_policy):
            print("Policy iteration solved at step %d." % (i+1))
            return new_policy
        random_policy = new_policy


print(policy_iteration(env))