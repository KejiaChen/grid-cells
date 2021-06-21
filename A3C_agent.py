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
from threading import Thread, Lock
from collections import namedtuple
from multiprocessing import cpu_count

tf.keras.backend.set_floatx('float64')
wandb.init(name='A3C', project="deep-rl-tf2")

# Training config
flags.DEFINE_string("training_optimizer_options",
                    "{'learning_rate': 1e-5, 'momentum': 0.99}",  # lr [1e-6, 2e-4]
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
                     100,
                     "backpropagation steps in actor-critic learner")
flags.DEFINE_integer("action_repeat",
                     4,
                     "repeat each action selected by the actor")
flags.DEFINE_integer("num_worker",
                     32,
                     "number of workers each running on one thread")
flags.DEFINE_integer("training_episodes",
                     1000,
                     "Number of maximum training episodes.")
flags.DEFINE_string("training_optimizer_class",
                    "tf.keras.optimizers.RMSprop",
                    "The optimizer used for training.")
flags.DEFINE_string("data_root",
                    "/home/learning/Documents/kejia/grid-cells/dm_lab_data/",
                    "path of the dataset folder to store data")
flags.DEFINE_string("map_name",
                    "map_10_0.txt",
                    "name of the txt map")

# Model config
flags.DEFINE_integer("model_nh_lstm",
                     256,
                     "Number of hidden units in LSTM.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

tf.keras.backend.set_floatx('float64')
wandb.init(name='A3C', project="deep-rl-tf2")

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


class ReplayMemory(object):
    '''
  Replay buffer to store the experience temporarily.
  '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # self.memory[self.position] = transition_dict
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=100):
        if self.position == 0:
            print('error: empty memory when sampling')
            return []
        if self.position <= batch_size:
            return self.memory
        else:
            return self.memory[-batch_size:]

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class ACModel(tf.keras.Model):
    """Network Structure"""

    def __init__(self, num_actions, num_hidden_units):
        super(ACModel, self).__init__()
        self.common = layers.LSTM(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions, activation="softmax")
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x, (ht, ct) = inputs
        x, (ht, ct) = self.common(x, states=[ht, ct])
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
        _, _, (_, _) = model((tf.random.normal([1, 84, 84, 4]), (ht, ct)))
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
        env = gym.make(env_name)
        self.replay_buffer = ReplayMemory(memory_size)
        self.env_name = env_name
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.opt = optimizer

        self.global_actor_critic = ActorCritic(self.state_dim, self.action_dim, self.opt)
        self.num_workers = num_worker

    def train(self, max_episodes=1000):
        workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name)  # TODO: change env to dmlab
            workers.append(WorkerAgent(
                env, self.global_actor_critic, max_episodes, self.opt))

        for worker in workers:
            worker.start()

        # for worker in workers:
        #     worker.join()

        COORD.join(workers)


class WorkerAgent(Thread):
    def __init__(self, env, global_actor_critic, max_episodes, optimizer, memory_size=10000):
        Thread.__init__(self)
        self.replay_buffer = ReplayMemory(memory_size)
        self.lock = Lock()
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = len(ACTION_LIST)

        self.max_episodes = max_episodes
        self.global_actor_critic = global_actor_critic
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, optimizer)
        self.model = self.actor_critic.model

        # initialization of weights to be the same as global network
        self.pull_param()
        # self.actor_critic.model.set_weights(self.global_actor_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        """Discounted return R for n steps"""
        if not done:
            last_value = next_v_value
        else:
            last_value = tf.zeros((1, 1))
        td_targets = [last_value]

        for i in reversed(range(len(rewards))):
            td_targets.insert(0, FLAGS.gamma.gamma * td_targets[0] + rewards[i])

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
                                            self.global_worker.model.trainable_variables):
            worker_para.assign(global_para)

    def train(self):
        global CUR_EPISODE  # current episode

        while not COORD.should_stop():
            if CUR_EPISODE > self.training_episodes:
                print("process done")
                break
            # if self.save and (CUR_EPISODE % args.save_interval == 0) and CUR_EPISODE > 0:
            #     with self.lock:
            #         self.model.save_weights(save_path)
            #         print('Saving model weights at episode {}'.format(CUR_EPISODE))

            episode_reward, done = 0, False
            new_eps = True

            state = self.env.reset()

            while not done:
                if new_eps:
                    ht = tf.zeros((1, self.n_units))
                    ct = tf.zeros((1, self.n_units))
                    new_ep = False

                rewards = []
                values = []
                log_probs = []
                entropies = []

                with tf.GradientTape() as tape:
                    for step in range(FLAGS.action_repeat):
                        probs, value, (ht, ct) = self.actor_critic.model.predict((state, (ht, ct)))
                        # action_index = np.random.choice(self.action_dim, p=probs[0])
                        dist = tfp.distributions.Categorical(logits=probs)
                        action_index = dist.sample()
                        action = tf.gather(ACTION_LIST, action_index)

                        log_prob = dist.log_prob(action)
                        entropy = dist.entropy()

                        next_state, reward, done, _ = self.env.step(action)  # TODO
                        next_state = tf.cast(np.expand_dims(state, axis=0), dtype=tf.float32)

                        episode_reward += reward
                        rewards.append(reward)
                        values.append(value)
                        log_probs.append(log_prob)
                        entropies.append(entropy)

                        if done:
                            break

                    # output = tf.scan(step, tf.range(FLAGS.backprop_length), first_values)

                    _, next_v_value, (_, _) = self.actor_critic.model.predict(next_state, (ht, ct))
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
                    self.global_actor_critic.opt.apply_gradients(
                        zip(grads, self.global_actor_critic.model.trainable_variables))
                    self.pull_param()

                # print('Process {}/X: {}/Episode: {}/EP_Reward: {:.2f}/Loss: {:.3f}'.format(self.seed, info['x'],
                #                                                                            CUR_EPISODE,
                #                                                                            episode_reward,
                #                                                                            loss))

                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            wandb.log({'Reward': episode_reward})
            ep_reward = 0
            CUR_EPISODE += 1

    def run(self):
        self.train()


def main():
    optimizer_class = eval(FLAGS.training_optimizer_class)
    optimizer = optimizer_class(**eval(FLAGS.training_optimizer_options))
    env_name = 'CartPole-v1'
    agent = LearnerAgent(env_name, optimizer, FLAGS.num_worker, memory_size=10000)
    agent.train()


if __name__ == "__main__":
    main()
