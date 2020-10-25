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

import sys
# comment these lines when run train.py
# Task config
# flags.DEFINE_string("task_dataset_info", "square_room",
#                     "Name of the room in which the experiment is performed.")
# flags.DEFINE_string("task_root",
#                     "/home/kejia/grid-cells/data",
#                     "Dataset path.")
# # flags.DEFINE_integer("use_data_files", 10,
# #                      "Number of files to read")
# flags.DEFINE_integer("training_minibatch_size", 10,
#                      "Size of the training minibatch.")
# FLAGS = flags.FLAGS
# FLAGS(sys.argv)
# comment these lines when run train.py

DatasetInfo = collections.namedtuple(
            'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
        square_room=DatasetInfo(
            basepath='square_room_100steps_2.2m_1000000',
            size=100,  # 100 files
            sequence_length=100,  # 100 steps
            coord_range=((-1.1, 1.1), (-1.1, 1.1))),)  # coordinate range for x and y


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
            # Queue params
            num_threads=4,
            capacity=256,
            min_after_dequeue=128,
            seed=None):
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

        if dataset not in _DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                             'are {}'.format(dataset, _DATASETS.keys()))

        self._dataset_info = _DATASETS[dataset]
        self._steps = _DATASETS[dataset].sequence_length

        with tf.device('/cpu'):
            file_names = _get_dataset_files(self._dataset_info, root)
            file_read_up_to = file_names[0:63]
            # filename_queue = tf.data.Dataset.from_tensor_slices(file_names)  # create filename queue
            self._reader = tf.data.TFRecordDataset(file_read_up_to)
            self._reader = self._reader.repeat(num_threads)

            self._reader = self._make_read_op(self._reader, capacity, seed)

            # read_ops = [
            #         self._make_read_op(reader, capacity, seed) for _ in range(num_threads)
            # ]
            # print('read_ops', read_ops)

            # iteration
            # i = 0
            # # reader_batch = []
            # for batch in self._reader.take(1):  # 64
            #     i = i+1
            #     # reader_batch.append(batch)
            #     print(repr(reader_batch))
            #     print("iteration", i)

            # dtypes = nest.map_structure(lambda x: x.dtype, read_ops[0])
            # shapes = nest.map_structure(lambda x: x.shape[1:], read_ops[0])

    # def read_batch(self, batch_size):
    #     reader_batch = self._reader.batch(batch_size=batch_size)
    #     return reader_batch

    def read(self, batch_size):
        """Reads batch_size. read batch from dict """

        reader_batch = self._reader.batch(batch_size=batch_size)
        for data in reader_batch.take(1):  # 64
            # print(type(batch))
            in_pos = data['init_pos']
            in_hd = data['init_hd']
            ego_vel = data['ego_vel']
            target_pos = data['target_pos']
            target_hd = data['target_hd']
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
        }

        def read_and_decode(example_string):
            # feature_dict = tf.io.parse_single_example(serialized=example_string, features=feature_map)
            #
            example = tf.io.parse_example(serialized=example_string, features=feature_map)
            # print('ego_vel', example['ego_vel'])
            batch = [
                example['init_pos'], example['init_hd'],
                example['ego_vel'][:self._steps, :],  # every 100 steps as a batch
                example['target_pos'][:self._steps, :],
                example['target_hd'][:self._steps, :]
            ]
            # print(batch)
            return example

        # reader = reader.repeat(4)
        reader = reader.shuffle(buffer_size=capacity, seed=seed)
        reader = reader.map(read_and_decode)
        # reader = reader.batch(batch_size=10)

        return reader


# # comment these lines when run train.py
# if __name__ == '__main__':
#     dataset = tf.data.Dataset.range(10)
#     for i in range(4):
#         dataset = dataset.shuffle(buffer_size=10)
#         dataset = dataset.take(3)
#         print(list(dataset.as_numpy_iterator()))
#
#     dataset_info = _DATASETS[FLAGS.task_dataset_info]
#     with tf.device('/cpu'):
#         file_names = _get_dataset_files(dataset_info, FLAGS.task_root)
#     print(file_names)
#     data_reader = DataReader(
#         FLAGS.task_dataset_info, root=FLAGS.task_root, num_threads=4)
#     train_traj1 = data_reader.read(batch_size=FLAGS.training_minibatch_size)  # tuple of data
#     # init_pos1, init_hd1, ego_vel1, target_pos1, target_hd1 = train_traj1
#     train_traj2 = data_reader.read(batch_size=FLAGS.training_minibatch_size)  # tuple of data
#     # init_pos2, init_hd2, ego_vel2, target_pos2, target_hd2 = train_traj2
#     # print(type(init_pos))
#     # print(init_pos)
#     print('range', data_reader.get_coord_range())
# comment these lines when run train.py
