import collections
import gym
import numpy as np
import statistics
import wandb
import tensorflow as tf
import tqdm
from absl import flags
import sys
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple
# from threading import Thread, Lock
import threading
from collections import namedtuple
from A3C_utils import *
from A3C_models_keras import *
import json
from dmlab_maze.dm_env.A3CLabEnv import RandomMaze
import logging
from logging.handlers import RotatingFileHandler
import time
# from env.A3CLabEnv import RandomMaze
from multiprocessing import cpu_count

# Task config
flags.DEFINE_string("task_root",
                    "/home/learning/Documents/kejia/grid-cells",
                    # None,
                    "Dataset path.")
flags.DEFINE_string("data_root",
                    "/home/learning/Documents/kejia/grid-cells/dm_lab_data/",
                    "path of the dataset folder to store data")
flags.DEFINE_string("map_name",
                    "map_10_0.txt",
                    "name of the txt map")
flags.DEFINE_string("saver_results_directory",
                    "/home/learning/Documents/kejia/grid-cells/",
                    # None,
                    "Path to directory for saving results.")

# Training config
flags.DEFINE_string("training_optimizer_options",
                    "{'learning_rate': 1e-6, 'momentum': 0.99}",  # lr [1e-6, 2e-4]
                    "Defines a dict with opts passed to the optimizer.")
flags.DEFINE_float("alpha",
                   0.50,  # [0.48, 0.52]
                   "baseline cost")
flags.DEFINE_float("beta",
                   8e-5,  # [6e-5, 1e-4]
                   "entropy regularization")
flags.DEFINE_float("gamma",
                   0.99,
                   "discount factor in the value function")
flags.DEFINE_integer("backprop_len",
                     100,  # 100
                     "backpropagation steps in actor-critic learner")
flags.DEFINE_integer("save_interval",
                     50,
                     "backpropagation steps in actor-critic learner")
flags.DEFINE_integer("action_repeat",
                     4,
                     "repeat each action selected by the actor")
flags.DEFINE_integer("num_worker",
                     32,
                     "number of workers each running on one thread")
flags.DEFINE_integer("max_episode_length",
                     5400,  # 5400
                     "Number of maximum training steps in one episode.")
flags.DEFINE_integer("episode_num",
                     1000,
                     "Number of episodes.")
flags.DEFINE_string("training_optimizer_class",
                    "tf.keras.optimizers.RMSprop",
                    "The optimizer used for training.")

# Model config
flags.DEFINE_integer("model_nh_lstm",
                     256,
                     "Number of hidden units in LSTM.")

# Environment config
flags.DEFINE_float("coord_range",
                    2.5,
                    "coordinate range of the dmlab room")
flags.DEFINE_integer("dataset_size",
                     100,
                     "number of files in the dataset")
flags.DEFINE_integer("file_length",
                     100,
                     "number of trajectories in each file")
flags.DEFINE_integer("eps_length",
                     100,
                     "number of steps in each trajectory")


FLAGS = flags.FLAGS
FLAGS(sys.argv)

tf.keras.backend.set_floatx('float64')  # 64
# wandb.init(name='A3C', project="deep-rl-tf2")

# parser = argparse.ArgumentParser()
# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--update_interval', type=int, default=5)
# parser.add_argument('--actor_lr', type=float, default=0.0005)
# parser.add_argument('--critic_lr', type=float, default=0.001)
#
# args = parser.parse_args()

COORD = tf.train.Coordinator()
CUR_EPISODE = 0
# Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'pos', 'rots', 'trans_vel', 'ang_vel', 'done'))
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))


# actions in Deepmind
def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left, /degree
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
        _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
]

def make_configs():
    # mapper: sample a new maze
    random_maze = sample_maze(name=FLAGS.map_name, start_range=6)

    # initialize the maze environment
    maze_configs = set_config_level(random_maze)
    return maze_configs

def make_environment(level):
    # SET THE ENVIRONMENT
    # level name
    # level = "nav_random_maze"

    # desired observations
    observation_list = ['RGB_INTERLEAVED',
                        # 'RGB.LOOK_PANORAMA_VIEW',
                        # 'RGB.LOOK_TOP_DOWN_VIEW',
                        'DEBUG.CAMERA.TOP_DOWN',
                        'DEBUG.POS.TRANS',
                        'DEBUG.POS.ROT',
                        'VEL.TRANS',
                        'VEL.ROT',
                        ]

    # configurations
    configurations = {
        'width': str(64),
        'height': str(64),
        "fps": str(60)
    }

    # maze theme and posters
    theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN", "INVISIBLE_WALLS"]
    decal_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # decoration on the wall?

    # # mapper
    # random_maze = sample_maze(name=map_name, start_range=6)
    #
    # # initialize the maze environment
    # maze_configs = set_config_level(random_maze)

    new_maze_configs = make_configs()

    # create the map environment
    myEnv = RandomMaze(level,
                       observation_list,
                       configurations,
                       FLAGS.coord_range)

    # # initialize the maze environment
    # maze_configs = set_config_level(random_maze)

    # set the maze
    myEnv.reset(new_maze_configs)

    return myEnv


class ACModel(tf.keras.Model):
    """Network Structure"""
    def __init__(self, num_actions, num_hidden_units):
        super(ACModel, self).__init__()
        self.fc1 = layers.Dense(128)
        self.lrelu1 = layers.LeakyReLU(0.1)
        self.fc2 = layers.Dense(256)
        self.lrelu2 = layers.LeakyReLU(0.1)
        self.fc3 = layers.Dense(256)
        self.lrelu3 = layers.LeakyReLU(0.1)

        self.lstm = layers.LSTMCell(num_hidden_units)  # activation="relu")
        self.actor = layers.Dense(num_actions, activation="softmax")
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x, (ht, ct) = inputs

        x = self.fc1(x)
        x = self.lrelu1(x)
        x = self.fc2(x)
        x = self.lrelu2(x)
        x = self.fc3(x)
        x = self.lrelu3(x)

        x, (ht, ct) = self.lstm(x, states=[ht, ct])
        return self.actor(x), self.critic(x), (ht, ct)


class ActorCritic:
    def __init__(
            self,
            num_actions,
            num_hidden_units,
            optimizer,
            weights_path=None,
            pretrained=False):
        """Initialize."""
        super(ActorCritic, self).__init__()
        self.n_acts = num_actions
        self.n_units = num_hidden_units
        self.model = self.build_model(pretrained, weights_path)
        self.opt = optimizer

    def build_model(self, pretrained, weights_path):
        model = ACModel(self.n_acts, self.n_units)
        if pretrained:
            model.load_weights(weights_path)
            print("Model load weights successfully")

        # model initialization
        (ht, ct) = (tf.zeros((1, self.n_units)), tf.zeros((1, self.n_units)))
        # input to NN: [input_squence, batch_size, input_size]
        # ground truth input_size = pos + rots + action + reward = 5
        _, _, (_, _) = model((tf.random.normal([64, 5]), (ht, ct)))
        return model

    # def call(self, inputs):
    #     return self.model(inputs)

    # def compute_loss(self, actions, logits, advantages, v_pred, td_targets):
    #     # mean = tf.keras.metrics.Mean()
    #     # mean.update_state(td_targets, sample_weight=logits)
    #     # loss_policy = mean
    #     sparsece_loss = tf.keras.losses.SparseCategoricalCrossentropy(
    #         from_logits=True)
    #     ce_loss = tf.keras.losses.CategoricalCrossentropy(
    #         from_logits=True)
    #     mse = tf.keras.losses.MeanSquaredError()
    #     actions = tf.cast(actions, tf.int32)  # shpae[5,1], 0 or 1
    #     policy_loss = sparsece_loss(
    #         actions, logits, sample_weight=tf.stop_gradient(advantages))  # why?
    #     entropy_loss = ce_loss(logits, logits)  # to be maximized
    #     critic_loss = mse(td_targets, v_pred)  # to be minimized
    #
    #     loss = policy_loss + FLAGS.alpha*critic_loss - FLAGS.beta*entropy_loss
    #     return loss
    #
    # def train(self, states, actions, advantages, td_targets):
    #     with tf.GradientTape() as tape:
    #         logits, v_pred = self.model(states)
    #         assert v_pred.shape == td_targets.shape
    #         loss = self.compute_loss(actions, logits, advantages, v_pred, tf.stop_gradient(td_targets))
    #     grads = tape.gradient(loss, self.model.trainable_variables)
    #     self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    #     return loss


class LearnerAgent:
    def __init__(self, env_name, optimizer, num_worker, memory_size=10000):
        # env = gym.make(env_name)
        # self.map_name = map_name
        self.env_name = env_name
        env = make_environment(self.env_name)
        self.replay_buffer = ReplayMemory(memory_size)
        # self.state_dim = env.observation_space.shape[0]
        self.action_dim = len(ACTION_LIST)
        self.opt = optimizer

        self.global_actor_critic = ActorCritic(num_actions=self.action_dim,
                                               num_hidden_units=FLAGS.model_nh_lstm,
                                               optimizer=self.opt)
        self.num_workers = num_worker
        self.logger = self.setup_logger()

        stats_file = FLAGS.saver_results_directory + "log/drl_log/stats" + \
                     time.strftime("%m-%d_%H:%M", time.localtime()) + '.json'
        self.stats_dict = StatsDict(['episode_reward'], save_file=stats_file)

    def setup_logger(self):
        # logging configs
        log_name = 'A3C_ground_truth' + time.strftime("%m-%d_%H:%M", time.localtime())
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            # filename=FLAGS.task_root + '/log/drl_log/' + log_name + '.log',
                            # filename='/log/' + log_name + '.log',
                            filemode='w')

        # formatter
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

        # handler
        fh = logging.FileHandler(FLAGS.task_root + '/log/drl_log/' + log_name + '.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # set logger
        log = logging.getLogger(log_name)
        log.setLevel(logging.DEBUG)
        log.addHandler(fh)

        return log

    def train(self, max_episodes=1000):
        workers = []

        # # log the training
        # log_name = 'A3C_ground_truth' + time.strftime("%m-%d_%H:%M", time.localtime())
        # logging.basicConfig(level=logging.INFO,
        #                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        #                     datefmt='%a, %d %b %Y %H:%M:%S',
        #                     filename=FLAGS.task_root + '/log/drl_log/' + log_name + '.log',
        #                     # filename='/log/' + log_name + '.log',
        #                     filemode='w')
        # # logging.info('please log something')
        # log = logging.getLogger('tensorflow')
        # log.setLevel(logging.DEBUG)
        # # formatter
        # formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        # # handler
        # fh = logging.FileHandler('tensorflow.log')
        # fh.setLevel(logging.INFO)
        # fh.setFormatter(formatter)
        # log.addHandler(fh)


        # save the model
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, net=self.global_actor_critic.model)
        check_dir = FLAGS.saver_results_directory + "/result/model/ckpt_dmlab_A3C" + time.strftime("%m-%d_%H:%M",
                                                                                               time.localtime())
        manager = tf.train.CheckpointManager(checkpoint, directory=check_dir, max_to_keep=20,
                                             checkpoint_name='model_A3C.ckpt')

        for i in range(self.num_workers):
            env = make_environment(self.env_name)
            workers.append(WorkerAgent(
                env, self.global_actor_critic, max_episodes, self.opt, manager, i, self.logger, self.stats_dict))

        for worker in workers:
            worker.start()

        # for worker in workers:
        #     worker.join()

        COORD.join(workers)


class WorkerAgent(threading.Thread):
    def __init__(self, env, global_actor_critic, max_episodes, optimizer, ckpt_manager, index, logger, save_dict,
                 memory_size=10000):
        threading.Thread.__init__(self)
        self.replay_buffer = ReplayMemory(memory_size)
        self.lock = threading.Lock()
        self.env = env
        # self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = len(ACTION_LIST)

        self.max_episodes = max_episodes
        self.global_actor_critic = global_actor_critic
        self.actor_critic = ActorCritic(num_actions=self.action_dim,
                                        num_hidden_units=256,
                                        optimizer=optimizer)
        self.model = self.actor_critic.model
        # self.logger = logger
        self.manager = ckpt_manager
        # self.thread_name = threading.currentThread().getName()
        self.thread_name = "worker_" + str(index)
        self.log_name = self.thread_name + "_logger"

        self.logger = logger
        # self.logger = self.setup_logger()
        self.stats_dict = save_dict

        # initialization of weights to be the same as global network
        self.pull_param()
        # self.actor_critic.model.set_weights(self.global_actor_critic.model.get_weights())

        print("initialize on thread", self.thread_name)

    def setup_logger(self):
        # logging configs
        log_name = 'A3C_ground_truth' + time.strftime("%m-%d_%H:%M", time.localtime()) + str(self.thread_name)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            # filename=FLAGS.task_root + '/log/drl_log/' + log_name + '.log',
                            # filename='/log/' + log_name + '.log',
                            filemode='w')

        # formatter
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

        # handler
        fh = logging.FileHandler(FLAGS.task_root + '/log/drl_log/' + log_name + '.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # set logger
        log = logging.getLogger(self.log_name)
        log.setLevel(logging.DEBUG)
        log.addHandler(fh)

        return log

    def n_step_td_target(self, rewards, next_v_value, done):
        """Discounted return R for n steps"""
        if not done:
            last_value = next_v_value
        else:
            last_value = tf.zeros((1, 1), dtype="float64")
        td_targets = [last_value]

        for i in reversed(range(len(rewards))):
            td_targets.insert(0, FLAGS.gamma * td_targets[0] + rewards[i])

        # for k in reversed(range(0, len(rewards))):
        #     cumulative = FLAGS.gamma * cumulative + rewards[k]
        #     td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def pull_param(self):
        """
        pull parameters from global network
        Reference:
        https://github.com/iverxin/rl_impl
        """
        for worker_para, global_para in zip(self.model.trainable_variables,
                                            self.global_actor_critic.model.trainable_variables):
            worker_para.assign(global_para)

    def concatenate(self, state_list):
        s = []
        for x in state_list:
            if isinstance(x, list):
                s = s + x
            elif isinstance(x, int) or isinstance(x, float):
                s.append(x)
        return np.expand_dims(np.array(s), axis=0)

    def train(self):
        global CUR_EPISODE  # current episode
        # logger = self.setup_logger()

        self.logger.info("initialize on thread %s", self.thread_name)

        while not COORD.should_stop():
            if CUR_EPISODE > FLAGS.episode_num:
                done = 1
                print("Training ends at maximum number of episodes")
                break
            if (CUR_EPISODE % FLAGS.save_interval == 0) and CUR_EPISODE > 0:
                with self.lock:
                    # self.model.save_weights(save_path)
                    self.manager.save()
                    self.stats_dict.save()
                    self.logger.info("Saving model")
                    print('Saving model weights at episode {}'.format(CUR_EPISODE))

            episode_reward, done = 0, False
            statistic_dict = {'eps_reward': []}
            episode_length = 0
            new_start = True
            action_index = 6  # init action: move forward

            # last_obs, _, pos, rots, ego_vel = self.env.reset(make_configs())
            # resize_obs = -1 + (last_obs - 1) / 127
            # reward = 0
            # # last_obs, reward, if_terminal, last_dist, pos, rots, ego_vel, none_dict = self.env.observe()
            # # state = [goal_code, current_grid_code, vision/ground_truth, last_action, reward]
            #
            # state = self.concatenate((pos, rots, action_index, reward))

            # self.env.show_front_view()

            # while not done:
            while episode_length < FLAGS.max_episode_length:
                if new_start: # reach the goal or a new episode
                    # reset the env
                    last_obs, _, pos, rots, ego_vel = self.env.reset(make_configs())
                    # resize_obs = -1 + (last_obs - 1) / 127
                    reward = 0
                    state = self.concatenate((pos, rots, action_index, reward))
                    # initialize lstm
                    ht = tf.zeros((1, FLAGS.model_nh_lstm))
                    ct = tf.zeros((1, FLAGS.model_nh_lstm))

                    new_start = False

                rewards = []
                values = []
                log_probs = []
                entropies = []

                with tf.GradientTape() as tape:
                    for update_step in range(FLAGS.backprop_len):
                    # if CUR_EPISODE % FLAGS.backprop_len == 0:
                        probs, value, (ht, ct) = self.actor_critic.model((state, (ht, ct)))
                        # action_index = np.random.choice(self.action_dim, p=probs[0])
                        dist = tfp.distributions.Categorical(logits=probs)
                        action_index = ((dist.sample()).numpy()).tolist()
                        if action_index[0] == 6:
                            print("state:", state)
                            print("ht, ct:", ht, ct)
                            print("action index:", action_index)
                            print(probs)
                        # action = tf.gather(ACTION_LIST, action_index)
                        action = ACTION_LIST[action_index[0]]

                        log_prob = dist.log_prob(action_index)
                        entropy = dist.entropy()

                        last_obs, reward, done, last_dist, pos, rots, ego_vel, none_dict = self.env.step(action, FLAGS.action_repeat)
                        if done:
                            print("Reach the goal!")
                            new_start = True
                        episode_length += 1
                        resize_obs = -1 + (last_obs - 1) / 127
                        # print("pos:", pos)
                        if episode_length >= FLAGS.max_episode_length:
                            new_start = True

                        next_state = self.concatenate((pos, rots, action_index, reward))

                        # self.env.show_front_view(CUR_EPISODE)

                        episode_reward += reward
                        rewards.append(reward)
                        values.append(value)
                        log_probs.append(log_prob)
                        entropies.append(entropy)

                        state = next_state

                        if new_start:
                            break

                    # output = tf.scan(step, tf.range(FLAGS.backprop_length), first_values)

                    _, next_v_value, (_, _) = self.actor_critic.model((next_state, (ht, ct)))
                    td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    # _, baselines = self.actor_critic.model.predict(states)

                    def compute_loss(q_values, baseline):
                        policy_loss = 0.0
                        critic_loss = 0.0
                        entropy_loss = 0.0

                        for i in reversed(range(len(rewards))):
                            adv = tf.stop_gradient(q_values[i]) - baseline[i]

                            critic_loss += tf.reduce_mean(0.5 * tf.square(adv))
                            policy_loss += -tf.reduce_mean(tf.stop_gradient(adv) * log_probs[i])
                            entropy_loss += tf.reduce_mean(entropies[i])

                        total_loss = policy_loss + FLAGS.alpha * critic_loss - FLAGS.beta * entropy_loss
                        return total_loss

                    loss = compute_loss(q_values=td_targets, baseline=values)

                with self.lock:
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    # grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
                    self.global_actor_critic.opt.apply_gradients(zip(grads,
                                                                     self.global_actor_critic.model.trainable_variables))
                    self.pull_param()
                    # logger.info('Pull parameters')

                # print('Process {}/X: {}/Episode: {}/EP_Reward: {:.2f}/Loss: {:.3f}'.format(self.seed, info['x'],
                #                                                                            CUR_EPISODE,
                #                                                                            episode_reward,
                #                                                                            loss))

                # state = next_state

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            # wandb.log({'Reward': episode_reward})
            self.stats_dict.update("episode_reward", episode_reward)
            self.logger.info('Episode %i, reward %.5f, length %i, on %s', CUR_EPISODE, episode_reward,
                             episode_length, self.thread_name)
            ep_reward = 0
            CUR_EPISODE += 1

    def run(self):
        self.train()


def main():
    optimizer_class = eval(FLAGS.training_optimizer_class)
    # TODO: rmsprop with shared statistics
    optimizer = optimizer_class(**eval(FLAGS.training_optimizer_options))
    env_name = "nav_random_maze"
    # env_name = 'contributed/dmlab30/rooms_watermaze'

    agent = LearnerAgent(env_name, optimizer, num_worker=3, memory_size=10000)
    agent.train()


if __name__ == "__main__":
    main()
