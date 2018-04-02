import sys
import argparse
import numpy as np
import keras
import random
import gym

D1 = []
D10 = []
D50 = []
D100 = []

class Imitation():
    def __init__(self, model_config_path, expert_weights_path):
        # Load the expert model.
        with open(model_config_path, 'r') as f:
            self.expert = keras.models.model_from_json(f.read())
        self.expert.load_weights(expert_weights_path)
        
        # Initialize the cloned model (to be trained).
        with open(model_config_path, 'r') as f:
            self.model = keras.models.model_from_json(f.read())


        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternatively compile your model here.
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=keras.optimizers.Adam(lr=0.001),
                            metrics=['accuracy']
                           )

    def run_expert(self, env, render=False):
        # Generates an episode by running the expert policy on the given env.
        return Imitation.generate_episode(self.expert, env, render)

    def run_model(self, env, render=False):
        # Generates an episode by running the cloned policy on the given env.
        return Imitation.generate_episode(self.model, env, render)

    @staticmethod
    def generate_episode(model, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        envir = env.env

        nstate = envir.reset()
        max_epi_len = 1000
        for _ in xrange(max_epi_len):
            if render: envir.render()
            states += [nstate]
            n = nstate
            np_array = np.array([[n[i] for i in xrange(len(n))]])
            qvals = model.predict(np_array)[0]
            action = np.argmax(qvals)
            onehot = np.zeros(env.action_space.n)
            onehot[action] = 1.0
            actions += [onehot]
            nstate, rew, _, _ = envir.step(action)
            rewards += [rew]

        return states, actions, rewards
    
    def train(self, env, num_episodes=100, num_epochs=50, render=False):
        # Trains the model on training data generated by the expert policy.
        # Args:
        # - env: The environment to run the expert policy on. 
        # - num_episodes: # episodes to be generated by the expert.
        # - num_epochs: # epochs to train on the data generated by the expert.
        # - render: Whether to render the environment.
        # Returns the final loss and accuracy.
        # TODO: Implement this method. It may be helpful to call the class
        #       method run_expert() to generate training data.
        loss = 0
        acc = 0

        States, Actions = [], []
        for i in xrange(num_episodes):
            states, actions, _ = self.run_expert(env, render)
            States += states
            Actions += actions
            self.model.fit(x=np.array(states), y=np.array(actions), epochs=num_epochs)

        [loss, acc] = self.model.evaluate(np.array(States), np.array(Actions))


        return loss, acc


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--expert-weights-path', dest='expert_weights_path',
                        type=str, default='LunarLander-v2-weights.h5',
                        help="Path to the expert weights file.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    global D1, D10, D50, D100
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    expert_weights_path = args.expert_weights_path
    render = args.render
    
    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # TODO: Train cloned models using imitation learning, and record their
    #       performance.
    imitation1 = Imitation(model_config_path, expert_weights_path)
    (loss1, acc1) = imitation1.train(env, num_episodes=1, render = False)

    imitation10 = Imitation(model_config_path, expert_weights_path)
    (loss10, acc10) = imitation10.train(env, num_episodes=10, render = False)

    imitation50 = Imitation(model_config_path, expert_weights_path)
    (loss50, acc50) = imitation50.train(env, num_episodes=50, render = False)

    imitation100 = Imitation(model_config_path, expert_weights_path)
    (loss100, acc100) = imitation100.train(env, num_episodes=100, render = False)



    def calculate_mean_std(imit, rend, num_episodes=50):
        rewards = []
        for _ in xrange(num_episodes):
            _, _, rew = imit.run_model(env, rend)
            rewards += rew

        Sum = sum(rewards)
        avg = Sum/len(rewards)
        std = (sum([(r-avg)**2 for r in rewards])/len(rewards))**0.5

        return avg, std

    (mean1, std1) = calculate_mean_std(imitation1, True)
    (mean10, std10) = calculate_mean_std(imitation10, True)
    (mean50, std50) = calculate_mean_std(imitation50, False)
    (mean100, std100) = calculate_mean_std(imitation100, False)


    def calculate_mean_std_exp(imit, rend, num_episodes=50):
        rewards = []
        for _ in xrange(num_episodes):
            _, _, rew = imit.run_expert(env, rend)
            rewards += rew

        Sum = sum(rewards)
        avg = Sum/len(rewards)
        std = (sum([(r-avg)**2 for r in rewards])/len(rewards))**0.5

        return avg, std

    (meanE1, stdE1) = calculate_mean_std_exp(imitation1, True)
    (meanE10, stdE10) = calculate_mean_std_exp(imitation10, True)
    (meanE50, stdE50) = calculate_mean_std_exp(imitation50, False)
    (meanE100, stdE100) = calculate_mean_std_exp(imitation100, False)

    print "(loss, acc) = ", (loss1, acc1), (loss10, acc10), (loss50, acc50), (loss100, acc100)
    
    print "mean, std: (clonded)"
    print mean1, std1
    print mean10, std10
    print mean50, std50
    print mean100, std100

    print "mean, std: (expert)"
    print meanE1, stdE1
    print meanE10, stdE10
    print meanE50, stdE50
    print meanE100, stdE100


if __name__ == '__main__':
  main(sys.argv)