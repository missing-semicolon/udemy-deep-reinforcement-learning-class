import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

import q_learning
from q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg


class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)


# replace SKLearn Regressor
q_learning.SGDRegressor = SGDRegressor

# calculate everything up to max[Q(s,a)]
# Ex.
# R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]
# def calculate_return_before_prediction(rewards, gamma):
#     ret = 0
#     for r in reversed(rewards[1:]):
#         ret += r + gamma*ret
#     ret += rewards[0]
#     return ret

# returns a list of states_and_rewards, and the total reward


def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0
    # array of [gamma^0, gamma^1, ..., gamma^(n-1)]
    multiplier = np.array([gamma] * n)**np.arange(n)

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)

        states.append(observation)
        actions.append(action)

        prev_observation = observation
        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        # update the model
        if len(rewards) >= n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + (gamma**n) * np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)

        totalreward += reward
        iters += 1

    # empty the cache
    # After running the loop above, we have no gone the full number of iters or
    # the step action returned done=True.

    # For the last n observations of the loop, we do not have a full number n of
    # iterations ot calcualte the return G

    # Let's look only at the last few rewards, states, and actions...
    rewards = rewards[-n + 1:]
    states = states[-n + 1:]
    actions = actions[-n + 1:]
    # unfortunately, new versions of gym cut off at 200 steps even if not hitting goal
    # need to check if we're really done, in which case all later rewards are 0
    if observation[0] >= 0.5:  # If the task has already been achieved by the nth from the last iteration...
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)  # G now is the discounted rewards remaining without a gamma*max_a[Q(s',a')]
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        # didn't make the goal
        while len(rewards) > 0:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return totalreward


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot optimal sate-value function
    plot_cost_to_go(env, model)
