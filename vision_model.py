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

"""Model for grid cells supervised training.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import sonnet as snt
import tensorflow as tf


def displaced_linear_initializer(input_size, displace, dtype=tf.float32):
    stddev = 1. / numpy.sqrt(input_size)
    return tf.compat.v1.truncated_normal_initializer(
            mean=displace*stddev, stddev=stddev, dtype=dtype)


class ConvNet(snt.Module):
    def __init__(self,
                 nh_conv_output,
                 name="convnet"):
        super(ConvNet, self).__init__(name=name)
        self.nh_output = nh_conv_output
        self.convnet = []
        self.resnet = {}
        self.structure = [(16, 2), (32, 2), (32, 2)]
        # with tf.compat.v1.variable_scope('convnet'):
        for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
            # Downscale.
            self.convnet.append(snt.Conv2D(num_ch, 3, stride=1, padding='SAME'))

            # Residual block(s).
            for j in range(num_blocks):
                with tf.compat.v1.variable_scope('residual_%d_%d' % (i, j)):
                    resblock = []
                    resblock.append(snt.Conv2D(num_ch, 3, stride=1, padding='SAME'))
                    resblock.append(snt.Conv2D(num_ch, 3, stride=1, padding='SAME'))
                    self.resnet['residual_%d_%d' % (i, j)] = resblock

        self.output_linear = snt.Linear(self.nh_output, name="conv_output_linear")

    def __call__(self, frame, training=False):
        conv_out = frame
        for i, (num_ch, num_blocks) in enumerate(self.structure):
            conv_out = self.convnet[i](conv_out)
            conv_out = tf.nn.pool(
                conv_out,
                window_shape=[3, 3],
                pooling_type='MAX',
                padding='SAME',
                strides=[2, 2])

            for j in range(num_blocks):
                residual_block = self.resnet['residual_%d_%d' % (i, j)]
                block_input = conv_out
                conv_out = tf.nn.relu(conv_out)
                conv_out = residual_block[0](conv_out)
                conv_out = tf.nn.relu(conv_out)
                conv_out = residual_block[1](conv_out)
                conv_out += block_input

            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.flatten(conv_out)

            conv_out = self.output_linear(conv_out)
            conv_out = tf.nn.relu(conv_out)
            return conv_out


class VisionModule(snt.Module):
    """Vision module to produce place cell and head direction cell activity patterns"""

    def __init__(self,
                 conv,
                 target_ensembles,
                 nh_conv,
                 nh_bottleneck,
                 init_weight_disp=0.0,
                 name="vision_module"):
        """Constructor of the Vision Network.

                Args:
                    conv: Convolutional network
                    target_ensembles: Targets, place cells and head direction cells.
                    nh_conv: Size of the convnet output.
                    nh_bottleneck: Size of the linear layer between convnet output and output.
                    dropoutrates_bottleneck: Iterable of keep rates (0,1]. The linear layer is
                        partitioned into as many groups as the len of this parameter.
                    bottleneck_weight_decay: Weight decay used in the bottleneck layer.
                    bottleneck_has_bias: If the bottleneck has a bias.
                    init_weight_disp: Displacement in the weights initialisation.
                    name: the name of the module.
                """
        super(VisionModule, self).__init__(name=name)
        self.conv_net = conv
        # self._target_ensembles = target_ensembles
        self._nh_bottleneck = nh_bottleneck
        self._nh_conv = nh_conv
        self._init_weight_disp = init_weight_disp
        # self.initial_pc = snt.Linear(self._nh_lstm, name="vision_state_init")
        # self.initial_hd = snt.Linear(self._nh_lstm, name="vision_cell_init")
        # self.bottleneck_layer = snt.Linear(self._nh_bottleneck,
        #                                    with_bias=self._bottleneck_has_bias,
        #                                    # new version of sonnet has no inner regularizers factor
        #                                    # L2 regularization is added to total_loss in train.py
        #                                    # regularizers={
        #                                    #     "w": tf.keras.regularizers.l2(
        #                                    #         0.5 * (self._bottleneck_weight_decay))},
        #                                    name="bottleneck")
        self.ens_pc, self.ens_hd = target_ensembles

        self.output_pc_layer = snt.Linear(
            self.ens_pc.n_cells,
            # new version of sonnet has no inner regularizers factor
            # L2 regularization is added to total_loss in train.py
            # regularizers={
            #         "w": tf.keras.regularizers.l2(
            #                 0.5 * (self._bottleneck_weight_decay))},
            w_init=displaced_linear_initializer(self._nh_conv,  # self._nh_bottleneck,
                                                self._init_weight_disp,
                                                dtype=tf.float32),
            name="pc_logits")
        self.output_hd_layer = snt.Linear(
            self.ens_hd.n_cells,
            # new version of sonnet has no inner regularizers factor
            # L2 regularization is added to total_loss in train.py
            # regularizers={
            #         "w": tf.keras.regularizers.l2(
            #                 0.5 * (self._bottleneck_weight_decay))},
            w_init=displaced_linear_initializer(self._nh_conv,  # self._nh_bottleneck,
                                                self._init_weight_disp,
                                                dtype=tf.float32),
            name="hd_logits")

    def __call__(self, frame, training=False):
        """
        Args:
            frames: visual inputs list (63, 64,3), with values in [-1,1]
            training: whether to update network; activates and deactivates dropout

        Returns:

        """
        # Convert to floats.
        frame = tf.cast(frame, tf.float32)
        shape = frame._shape_as_list()
        frame = tf.reshape(frame, (shape[0]*shape[1], shape[2], shape[3], shape[4]))

        # frame /= 255  # [-1,1]
        # conc_inputs = tf.concat(inputs, axis=1, name="conc_inputs")  # shape ( ,BN)

        # convnet from IMPALA
        conv_out = self.conv_net(frame)
        # with tf.variable_scope('convnet'):
        #     conv_out = frame
        #     for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
        #         # Downscale.
        #         conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
        #         conv_out = tf.nn.pool(
        #             conv_out,
        #             window_shape=[3, 3],
        #             pooling_type='MAX',
        #             padding='SAME',
        #             strides=[2, 2])
        #
        #         # Residual block(s).
        #         for j in range(num_blocks):
        #             with tf.variable_scope('residual_%d_%d' % (i, j)):
        #                 block_input = conv_out
        #                 conv_out = tf.nn.relu(conv_out)
        #                 conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
        #                 conv_out = tf.nn.relu(conv_out)
        #                 conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
        #                 conv_out += block_input
        #
        # conv_out = tf.nn.relu(conv_out)
        # conv_out = snt.BatchFlatten()(conv_out)
        #
        # conv_out = snt.Linear(256)(conv_out)
        # conv_out = tf.nn.relu(conv_out)

        # Bottleneck
        # bottleneck = self.bottleneck_layer(conv_out)

        # if not training and self._dropoutrates_bottleneck is not None:
        #     # tf.compat.v1.logging.info("Adding dropout layers")
        #     print("Adding dropout layers in Network")
        #     n_scales = len(self._dropoutrates_bottleneck)  # number of partition
        #     scale_pops = tf.split(bottleneck, n_scales, axis=1)  # partitioned bottleneck
        #     dropped_pops = [tf.nn.dropout(pop, 1 - (rate), name="dropout")
        #                     for rate, pop in zip(self._dropoutrates_bottleneck,
        #                                          scale_pops)]  # each partition with respective dropout rate
        #     bottleneck = tf.concat(dropped_pops, axis=1)

        # # Outputs place and HD ensembles with bottleneck
        # ens_pos_outputs = self.output_pc_layer(bottleneck)
        # ens_pos_outputs = tf.transpose(ens_pos_outputs, perm=[1, 0, 2])
        # ens_hd_outputs = self.output_hd_layer(bottleneck)
        # ens_hd_outputs = tf.transpose(ens_hd_outputs, perm=[1, 0, 2])
        #
        # ens_outputs = tf.tuple([ens_pos_outputs, ens_hd_outputs])
        # bottleneck = tf.transpose(bottleneck, perm=[1, 0, 2])

        # Outputs place and HD ensembles
        ens_pos_outputs = self.output_pc_layer(conv_out)
        ens_pos_outputs = tf.reshape(ens_pos_outputs, (shape[0], shape[1], self.ens_pc.n_cells))
        ens_hd_outputs = self.output_hd_layer(conv_out)
        ens_hd_outputs = tf.reshape(ens_hd_outputs, (shape[0], shape[1], self.ens_hd.n_cells))

        ens_outputs = tf.tuple([ens_pos_outputs, ens_hd_outputs])
        # bottleneck = tf.transpose(conv_out, perm=[1, 0, 2])
        conv_out = tf.reshape(conv_out, (shape[0], shape[1], self._nh_conv))

        return ens_outputs, conv_out
