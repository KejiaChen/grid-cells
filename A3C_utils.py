import random
import os
import json
from collections import namedtuple
# from env.A3CLabEnv_dmlab import RandomMaze
from collections import defaultdict
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras.optimizers as optim
from collections import defaultdict
import sys

import tempfile
import IPython.terminal.debugger as Debug
import IPython.display as display
import PIL.Image as Image

Transition = namedtuple('Transition', ('pos', 'rots', 'vel'))


def _get_dataset_files(dateset_info, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath)
    num_files = dateset_info.size
    # use_num_files = 64
    template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
    return [
            os.path.join(base, template.format(i, num_files - 1))
            for i in range(num_files)
            # for i in range(use_num_files)
    ]


# actions in Deepmind
def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left, /degree
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        # _action(0, 0, 0, -1, 0, 0, 0),  # backward
        # _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
        # _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
]


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def random_pos(map, tag, range_len=6):
    # range = 6 means agent starts from [4,5,6,7,8,9] in the map.txt
    # search a position which is still empty
    # to avoid P and G assigning to the same position
    while True:
        start_index = int((len(map)-1 - range_len)/2)  # 2
        print(tag +" range l:[%d %d]" % (2+start_index, len(map)-start_index))
        # start_l = int(random.uniform(2+start_index, len(map_txt)-start_index))
        # map_list = list(range(1, len(map_txt)+1))
        start_l_list = list(range(2+start_index, len(map)-start_index))
        start_l = random.choice(start_l_list)
        # start_l = 1
        print(tag + " range s:[%d %d]" % (2+start_index, len(map[0])-1-start_index))
        start_s_list = list(range(2+start_index, len(map[0])-1-start_index))
        start_s = random.choice(start_s_list)
        if map[start_l][start_s] == ' ':
            map[start_l] = map[start_l][:start_s] + tag + map[start_l][start_s + 1:]
            break
    # start_s = 1
    # start_s = int(random.uniform(2+start_index, len(map_txt[0])-1-start_index))
    # map[start_l] = map[start_l][:start_s] + tag + map[start_l][start_s+1:]
    return map


def sample_maze(name, start_range=6, only_new_start=False):
    """
    map_name: name of the txt map file
    start_range
    """
    # load all example maze names
    dir_path = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(dir_path, 'dmlab_maze/example/maps/')
    maze_names = os.listdir(map_path)
    # sample a maze
    # name = random.sample(maze_names, 1)[0]
    # name = 'map_10_0.txt'
    # name = FLAGS.map_name
    with open(map_path + name, 'r') as f:
        lines = f.readlines()
        map_txt = [l for l in lines]
    # obtain the maze size
    size = int(name.split('_')[1]) + 2
    # obtain the valid positions
    valid_pos = []
    # choos the range for start
    # start_range = 6
    # print("loaded map:", map_txt)

    start_map_txt = random_pos(map_txt, tag='P', range_len=start_range)
    if not only_new_start:
        goal_map_txt = random_pos(map_txt, tag='G', range_len=len(map_txt)-2)
    # for i, l in enumerate(map_txt):  # line
    #     print("start map:", l)

    for i, l in enumerate(map_txt):  # line:11
        # print("loaded map:", l)
        for j, s in enumerate(l):   # state:12
            if s == 'P':
                start_pos = [i, j]
                print("start position:[%d %d]" % (i, j))
            if s == 'G':
                goal_pos = [i, j]
                print("goal position:[%d %d]" % (i, j))
            if s == ' ' or s == 'P' or s == 'G':
                valid_pos.append([i, j])

    if only_new_start:
        print("random start")
        return start_pos + [random.uniform(-180, 180)]

    print("map loaded")
    return name, size, map_txt, valid_pos, start_range, start_pos, goal_pos


# def position_scale(map_pos):
#     # x_scale = 2.5/(883-116)
#     x_scale = FLAGS.coord_range/(1083 - 115)
#     # y_scale = 2.5 / (1083 - 116)
#     real_pos = -0.5*FLAGS.coord_range + (map_pos-116)*x_scale  # rescale position to (-1.25, 1.25)
#     return real_pos


def set_config_level(maze):
    maze_name, maze_size, maze_map_txt, maze_valid_pos, maze_start_range, maze_start_pos, maze_goal_pos = maze
    theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN"]
    decal_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    maze_configs = defaultdict(lambda: None)
    maze_configs["maze_name"] = f"maze_{maze_size}x{maze_size}"  # string type name
    maze_configs["maze_size"] = [maze_size, maze_size]  # [int, int] list
    maze_configs["maze_seed"] = '1234'  # string type number
    # maze_configs["maze_texture"] = "INVISIBLE_WALLS"
    maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]  # string type name in theme_list
    maze_configs["maze_decal_freq"] = 0
    # maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]  # float number in decal_list
    maze_configs["maze_map_txt"] = "".join(maze_map_txt)  # string type map
    # initialize the maze start and goal positions
    maze_configs["start_range"] = [maze_start_range, maze_start_range]
    maze_configs["start_pos"] = maze_start_pos + [random.uniform(-180, 180)]  # start position on the txt map [rows, cols, orientation]
    # maze_configs["start_pos"] = maze_start_pos + [0]
    maze_configs["goal_pos"] = maze_goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag

    # print_maze_info(maze_configs)
    return maze_configs


def print_maze_info(configs):
    print('----------------------------')
    print("Maze info: ")
    print("Maze name = ", configs["maze_name"])
    print("Maze size = ", configs["maze_size"])
    print("Maze seed = ", configs["maze_seed"])
    print("Maze texture = ", configs["maze_texture"])
    print("Maze decal freq = ", configs["maze_decal_freq"])
    print("Maze map txt = ")
    print(configs["maze_map_txt"])
    # set the maze start and goal positions
    print("Start range = ", configs["start_range"])
    print("Start pos = ", configs["start_pos"])
    print("Goal pos = ", configs["goal_pos"])
    print("Update flag = ", configs["update"])
    print('----------------------------')


class SharedRMSprop(optim.Optimizer):
    """Implements RMSprop algorithm with shared states.
    TODO: change to tensorflow
    """

    def __init__(self,
                 params,
                 lr=7e-4,
                 alpha=0.99,
                 eps=0.1,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        defaults = defaultdict(lr=lr, alpha=alpha, eps=eps,
                               weight_decay=weight_decay, momentum=momentum, centered=centered)
        super(SharedRMSprop, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = tf.zeros(1)
                state['grad_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['momentum_buffer'] = p.data.new(
                ).resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['square_avg'].share_memory_()
                state['step'].share_memory_()
                state['grad_avg'].share_memory_()
                state['momentum_buffer'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(
                        -1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss


class ReplayMemory(object):
    """
    Replay buffer to store the experience temporarily.
    To be stored: obs(or pos and rots), ego_vels
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0  # pointer to current position

    def push(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # self.memory[self.position] = transition_dict
        # when capacitry is reached, FIFO
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=10, sequence_length=100):
        if self.position == 0:
            print('error: empty memory when sampling')
            return []
        if self.position <= batch_size:
            batch = self.memory
            return self.memory
        else:
            sample_batch = random.choices(self.memory, k=batch_size)
        batch = Transition(*zip(*sample_batch))

        in_pos_list = []
        in_hd_list = []
        target_pos_list = []
        target_hd_list = []
        ego_vel_list = []
        for j in range(batch_size):
            length = tf.shape(batch[0][j])
            length = length.numpy()[0]
            start_index = random.randint(1, length - sequence_length)
            end_index = start_index + sequence_length

            target_pos, in_pos = self.sample_slice(batch[0][j], start_index, end_index)
            target_hd, in_hd = self.sample_slice(tf.expand_dims(batch[1][j], axis=1), start_index, end_index)
            ego_vel, _ = self.sample_slice(batch[2][j], start_index, end_index)

            in_pos_list.append(in_pos)
            in_hd_list.append(in_hd)
            target_pos_list.append(target_pos)
            target_hd_list.append(target_hd)
            ego_vel_list.append(ego_vel)

        return tf.stack(in_pos_list), tf.stack(in_hd_list), tf.stack(ego_vel_list),\
               tf.stack(target_pos_list), tf.stack(target_hd_list)

    def sample_slice(self, traj, start, end):
        slice_traj = traj[start:end, :]
        initial = traj[start-1, :]
        return slice_traj, initial


    # def pop(self, length):

    def clear(self):
        self.memory = []
        self.position = 0

    def get_coord_range(self, coord):
        coord_range = ((-coord / 2, coord / 2), (-coord / 2, coord / 2))  # coordinate range for x and y
        return coord_range

    def get_memory_length(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)


class StatsDict(object):
    def __init__(self, key_list, save_file):
        self.stats_dict = dict.fromkeys(key_list)
        for key in key_list:
            self.stats_dict[key] = []
        self.file = save_file

    def update(self, key, value):
        self.stats_dict[key].append(value)

    def save(self):
        with open(self.file, 'w') as fp:
            json.dump(self.stats_dict, fp)



