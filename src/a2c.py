#!/usr/bin/env python

import sys, argparse, keras, gym, matplotlib, time, itertools, os

import numpy      as np
import tensorflow as tf
# import matplotlib.pyplot as plt

from keras.models       import Sequential
from keras              import optimizers
from keras.layers.core  import Dense
from keras.layers.advanced_activations import LeakyReLU

class A2C(object):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, modelPath, lr, criticLr, lModWPath, lCritWPath, sWPath, n):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.meanRews = []
        self.stdRews  = []
        self.epsX     = []

        self.tCumRews = []
        self.tEpsX    = []

        self.n        = n
        with open(modelPath, 'r') as f: self.model = keras.models.model_from_json(f.read())

        print('A2C __init__: lr:%s, criticLr:%s, n:%s' % (lr, criticLr, n))

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        keras.backend.tensorflow_backend.set_session(self.sess)

        # Load the actor model from file.
        with open(modelPath, 'r') as f: self.model  = keras.models.model_from_json(f.read())
        with open(modelPath, 'r') as f: self.critic = keras.models.model_from_json(f.read())

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr=lr))
        self.model.summary()

        self.critic = Sequential()
        self.critic.add(Dense(32, input_shape=(8,)))
        self.critic.add(LeakyReLU(alpha=0.01))
        self.critic.add(Dense(32))
        self.critic.add(LeakyReLU(alpha=0.01))
        self.critic.add(Dense(1))
        self.critic.compile(loss='mean_squared_error',
                            optimizer=keras.optimizers.Adam(lr=criticLr))
        self.critic.summary()

        if lModWPath:  self.loadWeights(self.model, lModWPath)
        if lCritWPath: self.loadWeights(self.critic, lCritWPath)

        if sWPath:
            self.storePath = sWPath
            if not os.path.exists(self.storePath): os.makedirs(self.storePath)

        tf.summary.FileWriter(self.storePath, self.sess.graph)


    def train(self, env, nTrainEps):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        # plt.ion()
        # plt.figure()

        gamma    = 1.0
        saveInt  = 1000
        testInt  = 1000
        tPlotInt = 1000

        for ep in xrange(nTrainEps):

            if ep % saveInt == 0:
                self.saveWeights(self.model, '%s%s' % (ep,'model'))
                self.saveWeights(self.critic, '%s%s' % (ep,'critic'))
            if ep % testInt == 0: self.test(env, 100, ep)

            states, actions, rewards = self.generate_episode(env)

            labelsP = []
            labelsV = []
            for t in xrange(len(states)):
                Vt = 0 if t+self.n >= len(states) \
                       else self.critic.predict(np.expand_dims(states[t+self.n],0))[0]
                Rt = (pow(gamma,self.n)*Vt) + (np.sum([pow(gamma,k)*rewards[t+k] \
                                                     if t+k < len(states)        \
                                                     else 0                      \
                                                     for k in xrange(self.n)]))
                labelsP.append([Rt-Vt if i == actions[t] else 0 for i in xrange(4)])
                labelsV.append(Rt-Vt)

            labelsP = np.matrix(labelsP)
            labelsV = np.matrix(labelsV).transpose()
            states = np.matrix(states)

            self.tEpsX.append(ep)
            self.tCumRews.append(np.sum(rewards))

            # if ep % tPlotInt == 0:
            #     plt.plot(self.tEpsX, self.tCumRews, color='Red')
            #     plt.pause(0.001)

            print('ep:%s, len:%s, cRew:%s' % (ep, len(rewards), np.sum(rewards)))

            self.model.train_on_batch(states, labelsP)
            self.critic.train_on_batch(states, labelsV)

        self.saveWeights(self.model, '%s%s' % (nTrainEps,'model'))
        self.saveWeights(self.critic, '%s%s' % (nTrainEps,'critic'))

        # plt.show(block=True)


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

        # plt.errorbar(self.epsX, self.meanRews, yerr=self.stdRews, color='Yellow')
        # plt.pause(0.001)

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
            env.render()

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

    def saveWeights(self, model, prefix):
        model.save_weights('%s%s_weights.ckpt' % (self.storePath, prefix), overwrite=True)
        print('saved %s%s_weights.ckpt' % (self.storePath, prefix))

    def loadWeights(self, model, weightFile):
        model.load_weights(weightFile)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', dest='modelPath', type=str, default='')
    parser.add_argument('--nTrainEps', dest='nTrainEps', type=int, default=55000)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    parser.add_argument('--criticLr', dest='criticLr', type=float, default=1e-4)
    parser.add_argument('--lModWPath', dest='lModWPath', type=str, default='')
    parser.add_argument('--lCritWPath', dest='lCritWPath', type=str, default='')
    parser.add_argument('--sWPath', dest='sWPath', type=str, default='./store/')
    parser.add_argument('--n', dest='n', type=int, default=20)

    return parser.parse_args()

def main(args):
    # Parse command-line arguments.
    args       = parse_arguments()
    modelPath  = args.modelPath
    nTrainEps  = args.nTrainEps
    lr         = args.lr
    criticLr   = args.criticLr
    lModWPath  = args.lModWPath
    lCritWPath = args.lCritWPath
    sWPath     = args.sWPath
    n          = args.n

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # TODO: Train the model using A2C and plot the learning curves.
    reInfModel = A2C(modelPath, lr, criticLr, lModWPath, lCritWPath, sWPath, n)

    # reInfModel.train(env, nTrainEps)

    reInfModel.test(env, nTrainEps, 0)

if __name__ == '__main__':
    main(sys.argv)
