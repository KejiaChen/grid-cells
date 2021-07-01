import random
import os
import collections
from env.LabEnv import RandomMaze
from collections import defaultdict
from random_agent import SpringAgent
from absl import flags
import numpy as np
import math
import tensorflow as tf
import sys
import tempfile
import IPython.terminal.debugger as Debug
import IPython.display as display
import PIL.Image as Image

# Task config
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
flags.DEFINE_string("data_root",
                    "/home/learning/Documents/kejia/grid-cells/dm_lab_data/",
                    "path of the dataset folder to store data")
flags.DEFINE_string("map_name",
                    "map_10_0.txt",
                    "name of the txt map")

# Require flags from keyboard input
flags.mark_flag_as_required("data_root")
flags.mark_flag_as_required("map_name")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# dataset information
DatasetInfo = collections.namedtuple(
            'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
        square_room=DatasetInfo(
            # basepath='square_room_100steps_2.5m_novision_100',
            basepath='square_room_' + str(FLAGS.eps_length) + 'steps_2.5m_novision_' + str(FLAGS.file_length) + '_test',
            size=FLAGS.dataset_size,  # 100 files
            sequence_length=FLAGS.eps_length,  # 100 steps
            coord_range=((-0.5*FLAGS.coord_range, 0.5*FLAGS.coord_range),
                         (-0.5*FLAGS.coord_range, 0.5*FLAGS.coord_range))),)  # coordinate range for x and y


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


def sample_maze():
    # load all example maze names
    maze_names = os.listdir('./example/maps')
    # sample a maze
    # name = random.sample(maze_names, 1)[0]
    # name = 'map_10_0.txt'
    name = FLAGS.map_name
    with open('./example/maps/' + name, 'r') as f:
        lines = f.readlines()
        map_txt = [l for l in lines]
    # obtain the maze size
    size = int(name.split('_')[1]) + 2
    # obtain the valid positions
    valid_pos = []
    # choos the range for start
    start_range = 6
    # print("loaded map:", map_txt)

    def random_start(map, range=6):
        # range = 6 means agent starts from [4,5,6,7,8,9] in the map.txt
        # map
        start_index = int((len(map_txt)-1 - range)/2)  # 2
        print("start range l:[%d %d)" % (2+start_index, len(map_txt)-start_index))
        # start_l = int(random.uniform(2+start_index, len(map_txt)-start_index))
        start_l = random.choice([3, 4, 5, 6, 7, 8])
        # start_l = 1
        print("start range s:[%d %d)" % (2+start_index, len(map_txt[0])-1-start_index))
        start_s = random.choice([3, 4, 5, 6, 7, 8])
        # start_s = 1
        # start_s = int(random.uniform(2+start_index, len(map_txt[0])-1-start_index))
        map[start_l] = map[start_l][:start_s] + 'P' + map[start_l][start_s+1:]
        return map

    map_txt = random_start(map_txt, range=start_range)
    # for i, l in enumerate(map_txt):  # line
    #     print("start map:", l)

    for i, l in enumerate(map_txt):  # line:11
        # print("loaded map:", l)
        for j, s in enumerate(l):   # state:12
            if s == 'P':
                start_pos = [i, j]
                print("start position:[%d %d]" % (i, j))
            if s == ' ' or s == 'P' or s == 'G':
                valid_pos.append([i, j])
    print("map loaded")
    return name, size, map_txt, valid_pos, start_range, start_pos


def position_scale(map_pos):
    # x_scale = 2.5/(883-116)
    x_scale = FLAGS.coord_range/(1083 - 115)
    # y_scale = 2.5 / (1083 - 116)
    real_pos = -0.5*FLAGS.coord_range + (map_pos-116)*x_scale  # rescale position to (-1.25, 1.25)
    return real_pos


def set_config_level(maze):
    maze_name, maze_size, maze_map_txt, maze_valid_pos, maze_start_range, maze_start_pos = maze
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
    maze_configs["goal_pos"] = maze_valid_pos[-1] + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag

    print_maze_info(maze_configs)
    return maze_configs


def print_maze_info(configs):
    print("Maze info: ")
    print("Maze name = ", configs["maze_name"])
    print("Maze size = ", configs["maze_size"])
    print("Maze seed = ", configs["maze_seed"])
    print("Maze texture = ", configs["maze_texture"])
    print("Maze decal freq = ", configs["maze_decal_freq"])
    print("Maze map txt = ", configs["maze_map_txt"])
    # set the maze start and goal positions
    print("Start range = ", configs["start_range"])
    print("Start pos = ", configs["start_pos"])
    print("Goal pos = ", configs["goal_pos"])
    print("Update flag = ", configs["update"])
    print('----------------------------')


def run_demo():
    # SET THE ENVIRONMENT
    # level name
    level = "nav_random_maze"
    # level = 'contributed/dmlab30/rooms_watermaze'

    # desired observations
    observation_list = ['RGB_INTERLEAVED',
                        # 'RGB.LOOK_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW',
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

    # mapper
    random_maze = sample_maze()

    # create the map environment
    myEnv = RandomMaze(level,
                       observation_list,
                       configurations)

    # initialize the maze environment
    maze_configs = set_config_level(random_maze)

    # set the maze
    myEnv.reset(maze_configs)

    # save inital position and direction (np.array)
    last_obs, reward, if_terminal, last_dist, pos, rots, trans_vel, ang_vel, none_dict = myEnv.observe()
    # init_pose = maze_configs["start_pos"]
    # init_pos = np.array([init_pose[0], init_pose[1]])
    # init_hd = math.radians(init_pose[2])
    init_pos = position_scale(np.array([pos[0], pos[1]]))
    init_hd = math.radians(rots[1])

    # # create observation windows
    # myEnv._last_observation = myEnv.get_random_observations(myEnv.position_map2maze([1, 3, 0], myEnv.maze_size))
    # myEnv.show_panorama_view()
    myEnv.show_front_view()

    # agent = SpringAgent(myEnv.get_action_specification())\
    agent = SpringAgent(ACTION_LIST)

    # start test
    episode_length = FLAGS.eps_length  # every trajectory has 100 steps
    file_length = FLAGS.file_length  # every file contains 10000 trajectories
    time_episodes_num = FLAGS.dataset_size * episode_length * file_length * 2
    random.seed(maze_configs["maze_seed"])
    ep = 0
    file = 0
    # pos_len = 1
    reward = 0
    num_steps = 2
    # Initialize trajectory list
    ego_vel = []
    target_pos = []
    target_hd = []
    obs_img = []

    # Initialize list for each file, which includes 10000 curves
    traj_init_pos = []
    traj_init_hd = []
    traj_ego_vel = []
    traj_target_pos = []
    traj_target_hd = []
    traj_obs_img = []

    # The first random step
    # last_obs = myEnv.get_front_view()
    for t in range(time_episodes_num):
        # sample an action of translation velocity and angular velocity
        # ang_vel = random.choice([-20, -10, 0, 10, 20])
        # ang_act = _action(ang_vel, 0, 0, 0, 0, 0, 0)
        # print("angular vel:", ang_vel)
        # # in radians
        # ang_vel_rad = math.radians(ang_vel)
        #
        # trans_act = _action(0, 0, 0, 1, 0, 0, 0)  # move forward

        # ang_act = random.choice([-20, 0, 20])
        ang_act = random.uniform(-150, 150)
        trans_act = random.choice([0, 1])
        # act = _action(ang_act, 0, 0, trans_act, 0, 0, 0)
        act = _action(ang_act, 0, 0, 1, 0, 0, 0)
        # print("action", act)

        for i in range(num_steps):
            last_obs, reward, if_terminal, last_dist, pos, rots, trans_vel, ang_vel, none_dict = myEnv.step(act)

        # print("translational velocity", trans_vel)
        # print("angluar velocity", ang_vel)
        # print("hd:", rots)
        print("pos:", pos)

        # act = agent.random_step()

        # save trajectory in list
        ang_vel_rad = math.radians(ang_vel[1])  # in radians
        trans_vel_value = np.sqrt(np.square(position_scale(trans_vel[0])) + np.square(position_scale(trans_vel[1])))
        # print("vt", trans_vel_value)
        ego_vel.append(np.array([trans_vel_value, math.sin(ang_vel_rad), math.cos(ang_vel_rad)]))  # trans_vel + sine and cosine of angular velocity
        target_pos.append(position_scale(np.array([pos[0], pos[1]])))
        print("saved position:", position_scale(np.array([pos[0], pos[1]])))
        # print("current direction:", rots)
        target_hd.append(math.radians(rots[1]))
        resize_obs = -1+(last_obs-1)/127
        obs_img.append(resize_obs)  # front view, shape[64, 64, 3]

        # for a random maze view
        # myEnv._last_observation = myEnv.get_random_observations(myEnv.position_map2maze([1, 3, 0], myEnv.maze_size))
        # for the panorama view
        # myEnv.show_panorama_view(t)
        # for the front view
        myEnv.show_front_view(t)

        if t % episode_length == 0:
            # trajectory tuple
            # print("trajectory", target_pos)
            # episode_traj = (init_pos, init_hd, ego_vel, target_pos, target_hd, obs_img)
            print("init position:", init_pos)
            traj_init_pos.append(init_pos)
            traj_init_hd.append(init_hd)
            traj_ego_vel.append(ego_vel)
            traj_target_pos.append(target_pos)
            traj_target_hd.append(target_hd)
            traj_obs_img.append(obs_img)
            # array_image = np.array(obs_img)
            # print("shape:", array_image.shape) # shape(100, 64, 64, 3)

            # process trajectory into tf.Example
            def make_example(i_pos, i_hd, vel, t_pos, t_hd, img):

                feature_map = {
                    'init_pos': _float_feature(i_pos.flatten()),  # shape=(?, 2), ?=minibatch size
                    'init_hd': _float_feature([i_hd]),
                    'ego_vel': _float_feature(np.array(vel).flatten()),
                    'target_pos': _float_feature(np.array(t_pos).flatten()),
                    'target_hd': _float_feature(np.array(t_hd).flatten()),
                    'image': _bytes_feature(bytes(np.array(img))),
                }

                return tf.train.Example(features=tf.train.Features(feature=feature_map))

            # write into tfrecords
            if ep % file_length == 0:
                record_path = FLAGS.data_root
                record_file = _get_dataset_files(_DATASETS["square_room"], record_path)
                # record_file = "test00" + str(ep) + "-of-0099.tfrecord"

                with tf.io.TFRecordWriter(record_file[file]) as writer:
                    print("save traj in:" , record_file[file])
                    for i in range(len(traj_target_pos)):
                        # tf_example = make_example(init_pos, init_hd, ego_vel, target_pos, target_hd, obs_img)
                        tf_example = make_example(traj_init_pos.pop(0), traj_init_hd.pop(0), traj_ego_vel.pop(0),
                                                  traj_target_pos.pop(0), traj_target_hd.pop(0), traj_obs_img.pop(0))
                        writer.write(tf_example.SerializeToString())

                file += 1

            # a new episode
            ep += 1

            # randomly sample a new maze after each episode
            random_new_maze = sample_maze()
            # set the new maze params
            new_maze_configs = set_config_level(random_new_maze)

            # set the maze
            print("Time = {}, Ep = {}, Start = {}, Goal = {}".format(t, ep, new_maze_configs['start_pos'], new_maze_configs['goal_pos']))
            print("Setting new random maze...")
            myEnv.reset(new_maze_configs)

            # save inital position and direction (np.array)
            last_obs, reward, if_terminal, last_dist, pos, rots, trans_vel, ang_vel, none_dict = myEnv.observe()
            init_pos = position_scale(np.array([pos[0], pos[1]]))
            init_hd = math.radians(rots[1])

            # Initialize trajectory list
            ego_vel = []
            target_pos = []
            target_hd = []
            obs_img = []


if __name__ == '__main__':
    # sample_maze()
    # image = "/home/learning/Documents/kejia/grid-cells/graph/download.jpeg"
    # image_string = open(image, 'rb').read()
    # raw_image_dataset = tf.data.TFRecordDataset("/home/learning/Documents/kejia/grid-cells/dm_lab_data/square_room_100steps_2.5m_1000000/0000-of-0099.tfrecord")
    # sequence_length = 100
    #
    # feature_map = {
    #     'init_pos': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),  # shape=(?, 2), ?=minibatch size
    #     'init_hd': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    #     'ego_vel': tf.io.FixedLenFeature(shape=[sequence_length, 3], dtype=tf.float32),
    #     'target_pos': tf.io.FixedLenFeature(shape=[sequence_length, 2], dtype=tf.float32),
    #     'target_hd': tf.io.FixedLenFeature(shape=[sequence_length, 1], dtype=tf.float32),
    #     'image': tf.io.FixedLenFeature([], tf.string),
    # }
    #
    # def _parse_image_function(example_proto):
    #     # Parse the input tf.Example proto using the dictionary above.
    #     return tf.io.parse_single_example(example_proto, feature_map)
    #
    # parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    # reader_batch = parsed_image_dataset.batch(batch_size=1)
    #
    # for data in reader_batch.take(1):  # reader_batch.take(1) has only one element
    #     # print(type(batch))
    #     in_pos = data['init_pos']
    #     in_hd = data['init_hd']
    #     ego_vel = data['ego_vel']
    #     target_pos = data['target_pos']
    #     target_hd = data['target_hd']
    #     image = data['image']
    #     # decode_image = Image.frombytes("RGBA", (160, 160), (image.numpy())[0])
    #     # decode_image.save("test_image.png")
    #
    # print("read dataset")

    run_demo()
