import sys, argparse, keras, gym, matplotlib, time, itertools

matplotlib.use('Agg')

import numpy      as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import optimizers

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        keras.backend.tensorflow_backend.set_session(self.sess)

        self.model = model

        with tf.variable_scope('policyEst'):
            self.state  = tf.placeholder(dtype=tf.float32, name='state')
            self.action = tf.placeholder(dtype=tf.int32, name='action')
            self.target = tf.placeholder(dtype=tf.float32, name='target')

        self.actProb     = tf.get_default_graph().get_tensor_by_name('%s:0' % [n.name for n in tf.get_default_graph().as_graph_def().node][-1])
        self.pickActProb = tf.gather(self.actProb, self.action)

        self.loss = -tf.log(self.pickActProb) * self.target

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.trainOp   = self.optimizer.minimize(self.loss)

        self.model.summary()

    def train(self, env):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        gamma   = 1.0
        max_eps = 3000

        for ep in range(max_eps):
            episode = zip(self.generate_episode(env))

        # trew = [r/1e-2 for r in rewards]

        # for t, trans in enumerate(episode):
        #     Gt = sum(gamma**s * obs[2] for s, obs in enumerate(episode[t:]))
        #     self.update(trans[0], trans[1], trans[2])

    def generate_episode(self, env):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step

        states  = []
        actions = []
        rewards = []
        state   = env.reset()

        for t in itertools.count():
            actionProbs = self.predict(state)
            action      = np.random.choice(np.arange(len(actionProbs)), p=actionProbs)
            nstate, rew, term, _ = env.step(action)

            if term: break

            states.append(state)
            actions.append(action)
            rewards.append(rew)

            state = nstate

        return states, actions, rewards

    def predict(self, state):
        return self.sess.run(self.actProb, {self.state: state})

    def update(self, state, action, target):
        pass


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args              = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes      = args.num_episodes
    lr                = args.lr

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f: model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    reInfModel = Reinforce(model, lr)
    reInfModel.train(env)


if __name__ == '__main__':
    main(sys.argv)