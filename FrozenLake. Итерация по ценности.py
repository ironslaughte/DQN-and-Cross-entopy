import gym
import numpy as np
import atari_py

env = gym.make('FrozenLake-v0')
env.render()

def value_iterations(env, gamma=1.0):
    value_table = np.zeros(env.observation_space.n)
    no_of_iter = 100000
    threshold = 1e-20
    for i in range(no_of_iter):
        update_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q = []
            for action in range(env.action_space.n):
                next_state_rewards = []
                for next_sr in env.P[state][action]:  # env.P[][] возвращает вероятность перехода в
                    # следующее состояние, награду и информацию о завершении эпизода
                    # Q(state,action) = P*(R + gamma*V[next_state])
                    trans_prob, next_state, reward_prob, done = next_sr
                    next_state_rewards.append((trans_prob*(reward_prob+gamma*update_value_table[next_state])))
                Q.append(np.sum(next_state_rewards))
            value_table[state] = max(Q)
        if np.sum(np.fabs(update_value_table - value_table)) <= threshold:
            print("Value-iteration converged at iter %d." % (i+1))
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

optimal_value_func = value_iterations(env)
optimal_policy = extract_policy(optimal_value_func)

print(optimal_policy)

