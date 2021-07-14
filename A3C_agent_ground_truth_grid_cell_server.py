import collections
import gym
import numpy as np
import statistics
import wandb
import tensorflow as tf
import tqdm
import sonnet as snt
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple
# from threading import Thread, Lock
import threading
from collections import namedtuple
from A3C_utils import *
from A3C_models_snt import *
from Flags import *
import json
from dmlab_maze.dm_env.A3CLabEnv import RandomMaze
import logging
import time
from gridcell_agent import *
from multiprocessing import cpu_count

FILE_PATH = os.path.realpath(__file__)
FILE_DIR, _ = os.path.split(FILE_PATH)

# wandb.init(name='A3C', project="deep-rl-tf2")

COORD = tf.train.Coordinator()
COND = threading.Condition()
CUR_EPISODE = 0
GRID_TRAINING = 0
# Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'pos', 'rots', 'trans_vel', 'ang_vel', 'done'))
Transition = namedtuple('Transition', ('pos', 'rots', 'vel'))


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


class ActorCritic:
    def __init__(
            self,
            rnn_cell,
            num_hidden_units,
            optimizer,
            weights_path=None,
            pretrained=False):
        """Initialize."""
        super(ActorCritic, self).__init__()
        self.core = rnn_cell
        self.n_units = num_hidden_units
        self.model = self.build_model(pretrained, weights_path)
        self.opt = optimizer

    def build_model(self, pretrained, weights_path):
        model = ACModel(self.core, self.n_units)
        if pretrained:
            # TODO: change to sonnet style
            model.load_weights(weights_path)
            print("Model load weights successfully")

        # model initialization
        # TODO: 1 can be replaced by batch_size
        initial_lstmstate = snt.LSTMState(tf.zeros((1, self.n_units)), tf.zeros((1, self.n_units)))
        # input to NN: [input_squence, batch_size, input_size]
        # ground truth input_size = pos + rots + action + reward =

        # passing an example to get trainable variables

        _, _, _ = model((tf.random.normal([64, 517]), initial_lstmstate))
        return model


class LearnerAgent:
    def __init__(self, env_name, a3c_optimizer, grid_optimizer, num_worker, memory_size=1e6):
        # env = gym.make(env_name)
        # self.map_name = map_name
        self.env_name = env_name
        env = make_environment(self.env_name)
        self.replay_buffer = ReplayMemory(memory_size)
        # self.state_dim = env.observation_space.shape[0]
        self.action_dim = len(ACTION_LIST)
        self.a3c_opt = a3c_optimizer
        self.grid_opt = grid_optimizer

        self.ac_cell = ACCell(num_actions=self.action_dim,
                              num_hidden_units=FLAGS.policy_nh_lstm)
        # self.global_actor_critic = ActorCritic(num_actions=self.action_dim,
        #                                        num_hidden_units=FLAGS.model_nh_lstm)
        #                                        # optimizer=self.opt)
        self.global_actor_critic = ActorCritic(rnn_cell=self.ac_cell,
                                               num_hidden_units=FLAGS.policy_nh_lstm,
                                               optimizer=self.a3c_opt)
        # With pretrained model
        pretrained_ckpt_path = '/home/learning/Documents/kejia/grid-cells/result/model/ckpt_dmlab04-07_21:22/model_dmlab.ckpt-20'
        self.grid_cell_model = GridModel(optimizer=self.grid_opt,
                                         ckpt_path=pretrained_ckpt_path,
                                         load_model=FLAGS.load_grid_cell)

        self.num_workers = num_worker
        self.a3c_logger = self.setup_logger(file_name='A3C_ground_truth_grid', file_path='log/drl_log/')
        self.grid_logger = self.setup_logger(file_name='tensorflow_py3.7_dmlab_A3C', file_path='log/')

        stats_file = "log/drl_log/stats" + \
                     time.strftime("%m-%d_%H:%M", time.localtime()) + '.json'
        self.stats_dict = StatsDict(['episode_reward'], save_file=stats_file)

    def setup_logger(self, file_name, file_path):
        # logging configs
        log_name = file_name + time.strftime("%m-%d_%H:%M", time.localtime())
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            # filename=FLAGS.task_root + '/log/drl_log/' + log_name + '.log',
                            # filename='/log/' + log_name + '.log',
                            filemode='w')

        # formatter
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

        # handler
        fh = logging.FileHandler(file_path + log_name + '.log')
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
        # A3C
        a3c_checkpoint = tf.train.Checkpoint(optimizer=self.a3c_opt, net=self.global_actor_critic.model)
        a3c_check_dir = FLAGS.saver_results_directory + "/result/model/ckpt_dmlab_A3C_grid" + time.strftime("%m-%d_%H:%M",
                                                                                               time.localtime())
        a3c_manager = tf.train.CheckpointManager(a3c_checkpoint, directory=a3c_check_dir, max_to_keep=20,
                                                 checkpoint_name='model_A3C_grid.ckpt')
        # Grid
        grid_checkpoint = tf.train.Checkpoint(optimizer=self.grid_opt, net=self.grid_cell_model.rnn)
        grid_check_dir = FLAGS.saver_results_directory + "/result/model/ckpt_dmlab" + time.strftime("%m-%d_%H:%M",
                                                                                               time.localtime())

        grid_manager = tf.train.CheckpointManager(grid_checkpoint, directory=grid_check_dir, max_to_keep=20,
                                             checkpoint_name='model_dmlab_A3C.ckpt')

        for i in range(self.num_workers):
            env = make_environment(self.env_name)
            workers.append(WorkerAgent(env, self.global_actor_critic, max_episodes, self.a3c_opt, a3c_manager, i,
                                       self.a3c_logger, self.stats_dict, self.replay_buffer, self.grid_cell_model))

        # workers.append(GridAgent(self.replay_buffer, self.grid_cell_model))
        grid_agent = GridAgent(self.replay_buffer, self.grid_cell_model, self.grid_logger, grid_manager, COND)
        workers.append(grid_agent)

        for worker in workers:
            worker.start()

        COORD.join(workers)


class WorkerAgent(threading.Thread):
    def __init__(self, env, global_actor_critic, max_episodes, optimizer, ckpt_manager, index, logger, save_dict,
                 replay_buffer, grid_model):
        threading.Thread.__init__(self)
        self.replay_buffer = replay_buffer
        self.lock = threading.Lock()
        self.env = env
        # self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = len(ACTION_LIST)

        self.max_episodes = max_episodes
        self.global_actor_critic = global_actor_critic
        self.grid_cell = grid_model
        self.goal_grid_code = None
        # self.actor_critic = ActorCritic(num_actions=self.action_dim,
        #                                 num_hidden_units=256)
        #                                 # optimizer=optimizer)
        self.ac_cell = ACCell(num_actions=self.action_dim,
                              num_hidden_units=FLAGS.policy_nh_lstm)
        self.actor_critic = ActorCritic(rnn_cell=self.ac_cell,
                                        num_hidden_units=FLAGS.policy_nh_lstm,
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

    # def setup_logger(self):
    #     # logging configs
    #     log_name = 'A3C_ground_truth_grid' + time.strftime("%m-%d_%H:%M", time.localtime()) + str(self.thread_name)
    #     logging.basicConfig(level=logging.INFO,
    #                         format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #                         datefmt='%a, %d %b %Y %H:%M:%S',
    #                         # filename=FLAGS.task_root + '/log/drl_log/' + log_name + '.log',
    #                         # filename='/log/' + log_name + '.log',
    #                         filemode='w')
    #
    #     # formatter
    #     formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    #
    #     # handler
    #     fh = logging.FileHandler(FLAGS.task_root + '/log/drl_log/' + log_name + '.log')
    #     fh.setLevel(logging.INFO)
    #     fh.setFormatter(formatter)
    #
    #     # set logger
    #     log = logging.getLogger(self.log_name)
    #     log.setLevel(logging.DEBUG)
    #     log.addHandler(fh)
    #
    #     return log

    def n_step_td_target(self, rewards, next_v_value, done):
        """Discounted return R for n steps"""
        if not done:
            last_value = next_v_value
        else:
            last_value = tf.zeros((1, 1), dtype="float32")
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
        for worker_para, global_para in zip(self.actor_critic.model.trainable_variables,
                                            self.global_actor_critic.model.trainable_variables):
            worker_para.assign(global_para)

    def concatenate(self, state_list):
        s = []
        for x in state_list:
            if isinstance(x, list):
                s = s + x
            elif isinstance(x, int) or isinstance(x, float):
                s.append(x)
        return  tf.constant([s], dtype=tf.float32)
        # return np.expand_dims(np.array(s).astype("float32"), axis=0)

    def concat_grid(self, pos, rots, ego_vel, action_index, reward, done):
        env_state = self.concatenate((pos, rots, action_index, reward))

        # grid cell model initializaiton
        with self.lock:
            ensemble_pose = utils.encode_initial_conditions(tf.constant([pos]), tf.constant([[rots]]),
                                                            self.grid_cell.place_cell_ensembles,
                                                            self.grid_cell.head_direction_ensembles)
            # gird code, output shape [1, 1, 256]
            current_grid_code = self.grid_cell.forward_grid_code(ensemble_pose, tf.constant([[ego_vel]]))
        current_grid_code = tf.squeeze(current_grid_code, axis=0)

        if self.goal_grid_code is None:
            self.goal_gird_code = tf.zeros_like(current_grid_code, dtype=tf.float32)
        else:
            if done:
                self.goal_grid_code = current_grid_code

        state = tf.concat((self.goal_gird_code, current_grid_code, env_state),axis=-1)
        return state

    def train(self):
        global CUR_EPISODE  # current episode
        global GRID_TRAINING
        # logger = self.setup_logger()

        self.logger.info("initialize on thread %s", self.thread_name)

        while not COORD.should_stop():
            # traj_ego_vel = []
            # traj_target_pos = []
            # traj_target_hd = []
            # traj_obs_img = []

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

            # reset the env
            last_obs, _, pos, rots, ego_vel = self.env.reset(make_configs())

            # last_obs, _, pos, rots, ego_vel = self.env.reset(make_configs())
            # resize_obs = -1 + (last_obs - 1) / 127
            # reward = 0
            # # last_obs, reward, if_terminal, last_dist, pos, rots, ego_vel, none_dict = self.env.observe()
            # # state = [goal_code, current_grid_code, vision/ground_truth, last_action, reward]
            #
            # state = self.concatenate((pos, rots, action_index, reward))

            # self.env.show_front_view()

            # vel_eps_trajectory = []
            # pos_trajectory = []
            # hd_trajectory = []
            # obs_trajectory = []

            # while not done:
            while episode_length < FLAGS.max_episode_length:
                if new_start:
                    # # reset the env
                    # last_obs, _, pos, rots, ego_vel = self.env.reset(make_configs())
                    # restart from random position
                    start_pos = sample_maze(name=FLAGS.map_name, start_range=6, only_new_start=True)
                    last_obs, _, pos, rots, ego_vel = self.env.restart(start_pos)
                    # resize_obs = -1 + (last_obs - 1) / 127
                    reward = 0
                    done = 0

                    state = self.concat_grid(pos, rots, ego_vel, action_index, reward, done)
                    # state = self.concatenate((pos, rots, action_index, reward))

                    # policy lstm initial state
                    ht = tf.zeros((1, FLAGS.policy_nh_lstm))
                    ct = tf.zeros((1, FLAGS.policy_nh_lstm))
                    lstmstate = snt.LSTMState(ht, ct)

                    ego_vel_list = []
                    target_pos_list = []
                    target_hd_list = []
                    # obs_img_list = []

                    new_start = False

                rewards = []
                values = []
                log_probs = []
                entropies = []

                with tf.GradientTape() as tape:
                    for update_step in range(FLAGS.backprop_len):
                    # if CUR_EPISODE % FLAGS.backprop_len == 0:
                    #     probs, value, (ht, ct) = self.actor_critic.model((state, (ht, ct)))
                        probs, value, lstmstate = self.actor_critic.model((state, lstmstate))
                        # action_index = np.random.choice(self.action_dim, p=probs[0])
                        dist = tfp.distributions.Categorical(logits=probs)
                        action_index = ((dist.sample()).numpy()).tolist()
                        if action_index[0] == 6:
                            print("state:", state)
                            # print("ht, ct:", ht, ct)
                            print("action index:", action_index)
                            print(probs)
                        # action = tf.gather(ACTION_LIST, action_index)
                        action = ACTION_LIST[action_index[0]]

                        log_prob = dist.log_prob(action_index)
                        entropy = dist.entropy()

                        last_obs, reward, done, last_dist, pos, rots, ego_vel, none_dict = self.env.step(action, FLAGS.action_repeat)
                        resize_obs = -1 + (last_obs - 1) / 127

                        ego_vel_list.append(np.array(ego_vel, dtype="float32"))
                        target_pos_list.append(np.array(pos, dtype="float32"))
                        target_hd_list.append(rots)
                        # obs_img_list.append(resize_obs)

                        if done:
                            print("Reach the goal!")
                            new_start = True
                        episode_length += 1

                        # print("pos:", pos)
                        if episode_length >= FLAGS.max_episode_length:
                            new_start = True

                        next_state = self.concat_grid(pos, rots, ego_vel, action_index, reward, done)

                        # self.env.show_front_view(CUR_EPISODE)

                        episode_reward += reward
                        rewards.append(reward)
                        values.append(value)
                        log_probs.append(log_prob)
                        entropies.append(entropy)

                        state = next_state

                        if new_start:
                            if len(target_hd_list) > (FLAGS.sequence_length+1):
                                with self.lock:
                                    vel_eps_traj = tf.stack(ego_vel_list, axis=0)
                                    pos_eps_traj = tf.stack(target_pos_list, axis=0)
                                    hd_eps_traj = tf.stack(target_hd_list, axis=0)
                                    # obs_eps_traj = tf.stack(obs_img_list, axis=0)
                                    self.replay_buffer.push(pos_eps_traj, hd_eps_traj, vel_eps_traj)
                                    print("push trajectory to replay buffer")
                            else:
                                print("dropped short trajectory")
                            break

                    # output = tf.scan(step, tf.range(FLAGS.backprop_length), first_values)

                    _, next_v_value, lstmstate = self.actor_critic.model((next_state, lstmstate))
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
                    grads = tape.gradient(loss, self.actor_critic.model.trainable_variables)
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

            # vel_eps_trajectory = tf.stack(traj_ego_vel, axis=0)
            # pos_trajectory = tf.stack(traj_target_pos, axis=0)
            # hd_trajectory = tf.stack(traj_target_hd, axis=0)
            # obs_trajectory = tf.stack(traj_obs_img, axis=0)
            # self.replay_buffer.push(vel_trajectory, pos_trajectory, hd_trajectory, obs_trajectory)

            self.stats_dict.update("episode_reward", episode_reward)
            self.logger.info('Episode %i, reward %.5f, length %i, on %s', CUR_EPISODE, episode_reward,
                             episode_length, self.thread_name)
            self.logger.info("memory_length %s", self.replay_buffer.get_memory_length())

            if GRID_TRAINING == 0:
                if self.replay_buffer.get_memory_length() > FLAGS.training_minibatch_size:
                    COND.acquire()
                    print("notify the grid cell")
                    COND.notify()
                    GRID_TRAINING = 1
                    COND.release()
            ep_reward = 0
            CUR_EPISODE += 1

    def run(self):
        self.train()



def main():
    a3c_optimizer_class = eval(FLAGS.A3C_training_optimizer_class)
    # TODO: rmsprop with shared statistics
    a3c_optimizer = a3c_optimizer_class(**eval(FLAGS.A3C_training_optimizer_options))

    grid_optimizer_class = eval(FLAGS.training_optimizer_class)
    grid_optimizer = grid_optimizer_class(**eval(FLAGS.training_optimizer_options))

    env_name = "nav_random_maze"
    # env_name = 'contributed/dmlab30/rooms_watermaze'

    agent = LearnerAgent(env_name, a3c_optimizer, grid_optimizer, num_worker=3, memory_size=10000)
    agent.train()




if __name__ == "__main__":
    main()
