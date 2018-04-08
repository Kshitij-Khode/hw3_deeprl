import sys, argparse, keras, gym, matplotlib, time, itertools

import numpy      as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import optimizers

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.meanRews = []
        self.stdRews  = []
        self.epsX     = []

        print('Reinforce __init__: lr:%s' % lr)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        keras.backend.tensorflow_backend.set_session(self.sess)

        self.model = model

        # with tf.variable_scope('policyEst'):
        #     self.state  = tf.placeholder(dtype=tf.float32, name='state')
        #     self.target = tf.placeholder(dtype=tf.float32, name='target')

        self.inpState    = tf.get_default_graph().get_tensor_by_name('dense_1_input:0')
        self.outActProb  = tf.get_default_graph().get_tensor_by_name('dense_4/Softmax:0')

        self.execAct     = tf.placeholder(dtype=tf.int32, name='execAct')
        self.pickActProb = tf.gather(self.outActProb, self.execAct)

        self.trgRew =  tf.placeholder(dtype=tf.float32, name='trgRew')
        self.loss   = -tf.log(self.pickActProb) * self.trgRew

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.trainOp   = self.optimizer.minimize(self.loss)

        self.model.summary()
        tf.summary.FileWriter('./store/', self.sess.graph)

    def train(self, env, numEps):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        plt.ion()
        plt.figure()

        gamma   = 1.0
        saveInt = 5000
        testInt = 1000

        for ep in xrange(numEps):

            if ep % saveInt == 0: self.save_model_weights(ep)
            if ep % testInt == 0: self.test(env, 100, ep)

            states, actions, rewards = self.generate_episode(env)
            trew                     = [r*1e-2 for r in rewards]

            Gt = [np.sum([pow(gamma,k-t)*trew[k] for k in xrange(t,len(trew))])
                                                 for t in xrange(len(trew))]
            Gt = np.tile(np.matrix(Gt), (4,1)).transpose()

            print('ep:%s, len:%s, Gt[0]:%s, cRew:%s' % (ep, len(trew), Gt[0,0], np.sum(trew)))

            _, loss = self.sess.run([self.trainOp, self.loss], feed_dict={
                self.inpState: states, self.execAct: actions, self.trgRew: Gt
            })

        self.save_model_weights(numEps)
        plt.show()

    def test(self, env, numEps, trainEps):
        epRews = []
        for ep in xrange(numEps):
            _, _, rewards = self.generate_episode(env)
            epRews.append(np.sum(rewards))

        self.meanRews.append(np.mean(epRews))
        self.stdRews.append(np.std(epRews))
        self.epsX.append(trainEps)

        print('test-%s: meanRews:%s, stdRews:%s' % (self.epsX[-1],
                                                    self.meanRews[-1],
                                                    self.stdRews[-1]))

        plt.errorbar(self.epsX, self.meanRews, yerr=self.stdRews)
        plt.pause(0.001)

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

        for _ in itertools.count():
            # env.render()

            actionProbs = self.model.predict(np.expand_dims(state,0))[0]

            action = np.random.choice(np.arange(len(actionProbs)), p=actionProbs)
            nstate, rew, term, _ = env.step(action)

            # print('actionInt:%s, actionProbs:%s, rew:%s' % (np.arange(len(actionProbs)),
            #     actionProbs,
            #     rew))

            states.append(state)
            actions.append(action)
            rewards.append(rew)

            if term: break

            state = nstate

        return states, actions, rewards

    def save_model_weights(self, prefix):
        self.model.save_weights('./store/%s_weights.ckpt' % prefix, overwrite=True)
        print('saved ./model/%s_weights.ckpt' % prefix)

    def load_model_weights(self, weight_file):
        self.model.load_weights(weight_file)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--weight-path', dest='weight_path', type=str)
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args              = parse_arguments()
    model_config_path = args.model_config_path
    # num_episodes      = args.num_episodes
    num_episodes      = 55000
    lr                = args.lr
    weight_path       = args.weight_path

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f: model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    reInfModel = Reinforce(model, lr)
    # reInfModel.load_model_weights(weight_path)
    reInfModel.train(env, num_episodes)

    # reInfModel.load_model_weights(weight_path)
    # reInfModel.test(env, num_episodes)


if __name__ == '__main__':
    main(sys.argv)
