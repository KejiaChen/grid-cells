# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Minimal queue based TFRecord reader for the Grid Cell paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import tree as nest
from absl import app
from absl import flags
import PIL.Image as Image

import sys
import utils_new as utils
# comment these lines when run train.py
# Task config
# flags.DEFINE_string("task_dataset_info", "square_room",
#                     "Name of the room in which the experiment is performed.")
# flags.DEFINE_string("task_root",
#                     "/home/learning/Documents/kejia/grid-cells/dm_lab_data",
#                     "Dataset path.")
# flags.DEFINE_integer("use_data_files", 100,
#                      "Number of files to read")
# flags.DEFINE_integer("training_minibatch_size", 10,
#                      "Size of the training minibatch.")
# flags.DEFINE_float("coord_range",
#                     2.5,
#                     "coordinate range of the dmlab room")
# flags.DEFINE_string("saver_results_directory",
#                     "/home/learning/Documents/kejia/grid-cells/result",
#                     # None,
#                     "Path to directory for saving results.")
# FLAGS = flags.FLAGS
# FLAGS(sys.argv)
# comment these lines when run train.py

# DatasetInfo = collections.namedtuple(
#             'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])
#
# _DATASETS = dict(
#         square_room=DatasetInfo(
#             basepath='square_room_100steps_2.5m_novision_100',
#             size=100,  # 100 files
#             sequence_length=100,  # 100 steps
#             coord_range=((-1.25, 1.25), (-1.25, 1.25))),)  # coordinate range for x and y


def _get_dataset_files(dateset_info, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath)
    num_files = dateset_info.size
    # num_files = 100
    use_num_files = 1
    template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
    return [
            os.path.join(base, template.format(i, num_files - 1))
            # for i in range(num_files)
            for i in range(use_num_files)
    ]


class DataReader(object):
    """Minimal queue based TFRecord reader.

    You can use this reader to load the datasets used to train the grid cell
    network in the 'Vector-based Navigation using Grid-like Representations
    in Artificial Agents' paper.
    See README.md for a description of the datasets and an example of how to use
    the reader.
    """

    def __init__(
            self,
            dataset,
            # use_size,
            root,
            dataset_size=100,
            file_length=100,
            eps_length=100,
            coord=2.5,
            # Queue params
            num_threads=4,
            capacity=256,
            min_after_dequeue=128,
            seed=None,
            vision=False):
        """Instantiates a DataReader object and sets up queues for data reading.

        Args:
            dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
                'rooms_free_camera_no_object_rotations',
                'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
                'shepard_metzler_7_parts']. type of env
            root: string, path to the root folder of the data.
            num_threads: (optional) integer, number of threads used to feed the reader
                queues, defaults to 4.
            capacity: (optional) integer, capacity of the underlying
                RandomShuffleQueue, defaults to 256.
            min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
                RandomShuffleQueue, defaults to 128.
            seed: (optional) integer, seed for the random number generators used in
                the reader.

        Raises:
            ValueError: if the required version does not exist;
        """

        self.DatasetInfo = collections.namedtuple(
            'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])
        if vision:
            self.data_folder = 'square_room_100steps_2.5m_' + 'vision_' + str(file_length)
        else:
            self.data_folder = 'square_room_100steps_2.5m_' + 'novision_' + str(file_length)
        self._DATASETS = dict(
                    square_room=self.DatasetInfo(
                        basepath=self.data_folder,
                        size=dataset_size,  # 100 files
                        sequence_length=eps_length,  # 100 steps
                        coord_range=((-coord/2, coord/2), (-coord/2, coord/2))),)  # coordinate range for x and y

        if dataset not in self._DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                             'are {}'.format(dataset, self._DATASETS.keys()))
        self._dataset_info = self._DATASETS[dataset]
        self._steps = self._DATASETS[dataset].sequence_length
        self.vision = vision

        with tf.device('/cpu'):
            file_names = _get_dataset_files(self._dataset_info, root)
            # filename_queue = tf.data.Dataset.from_tensor_slices(file_names)  # create filename queue
            self._reader = tf.data.TFRecordDataset(file_names)
            # self._reader = self._reader.repeat(num_threads)

            self._reader = self._make_read_op(self._reader, capacity, seed)

    def read_batch(self, batch_size):
        reader_batch = self._reader.batch(batch_size=batch_size)
        return reader_batch

    def read(self, batch_size):
        """Reads batch_size. read batch from dict """

        reader_batch = self._reader.batch(batch_size=batch_size)

        def preprocess_image(image):
            decode_image = Image.frombytes("RGBA", (160, 160), image.numpy())
            rgb_image = decode_image.convert('RGB')
            # resize_image = rgb_image.resize((64, 64))
            rgb_image.save("test_image_rgb.png")
            return rgb_image

        for data in reader_batch.take(1):  # reader_batch.take(1) has only one element
            # print(type(batch))
            in_pos = data['init_pos']
            in_hd = data['init_hd']
            ego_vel = data['ego_vel']
            target_pos = data['target_pos']
            target_hd = data['target_hd']
            if self.vision:
                image = data["image"]  # [64, 64, 3], RGB
                return in_pos, in_hd, ego_vel, target_pos, target_hd, image
            # train_image = preprocess_image(raw_image)
        return in_pos, in_hd, ego_vel, target_pos, target_hd

    def get_coord_range(self):
        return self._dataset_info.coord_range

    def _make_read_op(self, reader, capacity, seed):
        """Instantiates the ops used to read and parse the data into tensors."""
        # _, raw_data = reader.read_up_to(filename_queue, num_records=64)
        feature_map = {
            'init_pos':
                tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),  # shape=(?, 2), ?=minibatch size
            'init_hd':
                tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),  # shape=(?, 1)
            'ego_vel':
                tf.io.FixedLenFeature(
                    shape=[self._dataset_info.sequence_length, 3],  # shape=(?, 100, 3) for 100 steps
                    dtype=tf.float32),
            'target_pos':
                tf.io.FixedLenFeature(
                    shape=[self._dataset_info.sequence_length, 2],  # shape=(?, 100, 2) for 100 steps
                    dtype=tf.float32),
            'target_hd':
                tf.io.FixedLenFeature(
                    shape=[self._dataset_info.sequence_length, 1],  # shape=(?, 100, 1) for 100 steps
                    dtype=tf.float32),
            # 'image':
            #     tf.io.FixedLenFeature(
            #         shape=[],  # image
            #         dtype=tf.string),
        }
        if self.vision:
            feature_map = {
                'init_pos':
                    tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),  # shape=(?, 2), ?=minibatch size
                'init_hd':
                    tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),  # shape=(?, 1)
                'ego_vel':
                    tf.io.FixedLenFeature(
                        shape=[self._dataset_info.sequence_length, 3],  # shape=(?, 100, 3) for 100 steps
                        dtype=tf.float32),
                'target_pos':
                    tf.io.FixedLenFeature(
                        shape=[self._dataset_info.sequence_length, 2],  # shape=(?, 100, 2) for 100 steps
                        dtype=tf.float32),
                'target_hd':
                    tf.io.FixedLenFeature(
                        shape=[self._dataset_info.sequence_length, 1],  # shape=(?, 100, 1) for 100 steps
                        dtype=tf.float32),
                'image':
                    tf.io.FixedLenFeature(
                        shape=[],  # image
                        dtype=tf.string),
            }

        def read_and_decode(example_string):
            example = tf.io.parse_example(serialized=example_string, features=feature_map)
            return example

        # reader = reader.repeat(4)
        reader = reader.shuffle(buffer_size=capacity, seed=seed)
        reader = reader.map(read_and_decode)
        # reader = reader.batch(batch_size=10)

        return reader

    # def position_scale(map_pos):
    #     # x_scale = 2.5/(883-116)
    #     x_scale = FLAGS.coord_range / (1083 - 116)
    #     # y_scale = 2.5 / (1083 - 116)
    #     real_pos = -0.5 * FLAGS.coord_range + map_pos * x_scale  # rescale position to (-1.25, 1.25)
    #     return real_pos

    # def fix_temp_traj(traj):
    #     init_pos, init_hd, ego_vel, target_pos, target_hd = traj
    #     target_pos - 0.300

# # comment these lines when run train.py
# if __name__ == '__main__':
    # # dataset = tf.data.Dataset.range(10)
    # # for i in range(4):
    # #     dataset = dataset.shuffle(buffer_size=10)
    # #     dataset = dataset.take(3)
    # #     print(list(dataset.as_numpy_iterator()))
    #
    # dataset_info = _DATASETS[FLAGS.task_dataset_info]
    # # file_names = _get_dataset_files(dataset_info, FLAGS.task_root)
    # # file_names = ['/home/learning/Documents/kejia/grid-cells/dm_lab_data/square_room_100steps_2.5m_1000000/0001-of-0099.tfrecord']
    # #               # '/home/learning/Documents/kejia/grid-cells/dm_lab_data/square_room_100steps_2.5m_1000000/0002-of-0099.tfrecord']
    # # raw_image_dataset = tf.data.TFRecordDataset(file_names)
    # # sequence_length = 100
    #
    # # feature_map = {
    # #     'init_pos': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),  # shape=(?, 2), ?=minibatch size
    # #     'init_hd': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    # #     'ego_vel': tf.io.FixedLenFeature(shape=[sequence_length, 3], dtype=tf.float32),
    # #     'target_pos': tf.io.FixedLenFeature(shape=[sequence_length, 2], dtype=tf.float32),
    # #     'target_hd': tf.io.FixedLenFeature(shape=[sequence_length, 1], dtype=tf.float32),
    # #     'image': tf.io.FixedLenFeature([], tf.string),
    # # }
    # #
    # # def _parse_image_function(example_proto):
    # #     # Parse the input tf.Example proto using the dictionary above.
    # #     return tf.io.parse_single_example(example_proto, feature_map)
    # #
    # # parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    # # reader_batch = parsed_image_dataset.batch(batch_size=10)
    # #
    # # i = 0
    # # for data in reader_batch.take(1):  # reader_batch.take(1) has only one element
    # # # for data in parsed_image_dataset:
    # #     # print(type(batch))
    # #     in_pos = data['init_pos']
    # #     in_hd = data['init_hd']
    # #     ego_vel = data['ego_vel']
    # #     target_pos = data['target_pos']
    # #     target_hd = data['target_hd']
    # #     image = data['image']
    # #     i += 1
    # #
    # # print("end iteration")
    # # print("read dataset")
    #
    # # file_names = _get_dataset_files(dataset_info, FLAGS.task_root)
    # # print(file_names)
    # plotname = 'trajectory_dmlab_test.pdf'
    # data_reader = DataReader(FLAGS.task_dataset_info, root=FLAGS.task_root, num_threads=4)
    # for i in range(100):
    #     train_traj1 = data_reader.read(batch_size=FLAGS.training_minibatch_size)  # tuple of data
    #     init_pos1, init_hd1, ego_vel1, target_pos1, target_hd1 = train_traj1
    #     utils.plot_trajectories(target_pos1, target_pos1, 10, FLAGS.saver_results_directory, plotname)
    # #     train_traj2 = data_reader.read(batch_size=FLAGS.training_minibatch_size)  # tuple of data
    #     print(i)
    # # init_pos2, init_hd2, ego_vel2, target_pos2, target_hd2 = train_traj2
    # # print(type(init_pos))
    # # print(init_pos)
    # print('range', data_reader.get_coord_range())
# comment these lines when run train.py
