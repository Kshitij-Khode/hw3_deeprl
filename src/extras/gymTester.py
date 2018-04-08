import gym, time, numpy as np


def main():
    # Make the environment
    env = gym.make('LunarLander-v2')

    # Record the environment
    # env = gym.wrappers.Monitor(env, '.', force=True)

    for episode in range(100):
        done = False
        obs = env.reset()

        while not done: # Start with while True
            env.render()

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('obs: %s' % obs)
            print('action: %s' % action)
            print('reward: %s' % reward)
            print('done: %s' % done)
            print('info: %s' % info)
            if obs[-1] or obs[-2]: time.sleep(1)
            if reward > 0: time.sleep(5)

if __name__ == '__main__':
    main()