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

"""Ensembles of place and head direction cells.

These classes provide the targets for the training of grid-cell networks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot_max(x, axis=-1):
    """Compute one-hot vectors setting to one the index with the maximum value."""
    return tf.one_hot(tf.argmax(x, axis=axis),
                      depth=x.get_shape()[-1],
                      dtype=x.dtype)


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    return tf.nn.softmax(x, dim=axis)


def softmax_sample(x):
    """Sample the categorical distribution from logits and sample it."""
    dist = tf.contrib.distributions.OneHotCategorical(logits=x, dtype=tf.float32)
    return dist.sample()


class CellEnsemble(object):
    """Abstract parent class for place and head direction cell ensembles."""

    def __init__(self, n_cells, soft_targets, soft_init):
        self.n_cells = n_cells
        if soft_targets not in ["softmax", "voronoi", "sample", "normalized"]:
            raise ValueError
        else:
            self.soft_targets = soft_targets
        # Provide initialization of LSTM in the same way as targets if not specified
        # i.e one-hot if targets are Voronoi
        if soft_init is None:
            self.soft_init = soft_targets
        else:
            if soft_init not in [
                    "softmax", "voronoi", "sample", "normalized", "zeros"
            ]:
                raise ValueError
            else:
                self.soft_init = soft_init

    def get_targets(self, x):
        """Type of target."""

        if self.soft_targets == "normalized":
            targets = tf.exp(self.unnor_logpdf(x))
        elif self.soft_targets == "softmax":
            lp = self.log_posterior(x)
            targets = softmax(lp)
        elif self.soft_targets == "sample":
            lp = self.log_posterior(x)
            targets = softmax_sample(lp)
        elif self.soft_targets == "voronoi":
            lp = self.log_posterior(x)
            targets = one_hot_max(lp)
        return targets

    def get_ref_points(self, s):
        """get three reference points with lagrest values in ensembles,
        i.e. smallest distances to the encoded point."""
        temp = s
        grid = tf.stack(tf.meshgrid(tf.range(10), tf.range(100), indexing='ij'), axis=2)
        grid = tf.dtypes.cast(grid, tf.int64)

        pos_list = []
        encoding_list = []
        test_list = []
        for i in range(3):
            index = tf.math.argmax(temp, axis=2)  # (10, 100)
            arg_grid = tf.concat([grid, tf.expand_dims(index, axis=-1)], axis=2)
            max_s = tf.gather_nd(temp, arg_grid)  # (10, 100)
            pos_list.append(index)
            encoding_list.append(max_s)

            # set the current max to 0, in order to get the second maximum
            flat_max_s = tf.reshape(max_s, [1000])
            flat_arg_grid = tf.reshape(arg_grid, [1000, 3])
            # test_list.append(flat_arg_grid)
            var = tf.sparse.SparseTensor(indices=flat_arg_grid, values=flat_max_s, dense_shape=[10, 100, 256])
            var = tf.sparse_tensor_to_dense(var)  # tensor with maximum values
            temp = temp - var

        # with tf.train.SingularMonitoredSession() as sess:
        #     test = sess.run({'pos1': pos_list[0],
        #                      'pos2': pos_list[1],
        #                      'pos3': pos_list[2],
        #                      'var1': encoding_list[0],
        #                      'var2': encoding_list[1],
        #                      'var3': encoding_list[2],
        #                      'flat1': test_list[0],
        #                      'flat2': test_list[1],
        #                      'flat3': test_list[2],
        #                      })

        print("select three points")

        return temp, pos_list, encoding_list

    # def decode_target(self, s):
    #     """inverse function of get_target(). s is the output, which is a result of softmax."""
    #     if self.soft_targets == "normalized":
    #         lp = tf.math.log(s)
    #     elif self.soft_targets == "softmax":
    #         lp = tf.math.log(s)
    #     # elif self.soft_targets == "sample":
    #     #     lp = self.log_posterior(x)
    #     #     targets = softmax_sample(lp)
    #     # elif self.soft_targets == "voronoi":
    #     #     lp = self.log_posterior(x)
    #     #     targets = one_hot_max(lp)
    #     return lp

    def get_init(self, x):
        """Type of initialisation."""

        if self.soft_init == "normalized":
            init = tf.exp(self.unnor_logpdf(x))
        elif self.soft_init == "softmax":  # choose softmax in train.py, how about others?
            lp = self.log_posterior(x)
            init = softmax(lp)
        elif self.soft_init == "sample":
            lp = self.log_posterior(x)
            init = softmax_sample(lp)
        elif self.soft_init == "voronoi":
            lp = self.log_posterior(x)
            init = one_hot_max(lp)
        elif self.soft_init == "zeros":
            init = tf.zeros_like(self.unnor_logpdf(x))
        return init

    def loss(self, predictions, targets):
        """Loss."""

        if self.soft_targets == "normalized":
            smoothing = 1e-2
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=(1. - smoothing) * targets + smoothing * 0.5,
                    logits=predictions,
                    name="ensemble_loss")
            loss = tf.reduce_mean(loss, axis=-1)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=targets,
                    logits=predictions,
                    name="ensemble_loss")
        return loss

    def log_posterior(self, x):
        logp = self.unnor_logpdf(x)
        log_posteriors = logp - tf.reduce_logsumexp(logp, axis=2, keep_dims=True)  # log(sum(exp())), reduce 2nd axis to length 1
        # with tf.train.SingularMonitoredSession() as sess:
        #     ob = sess.run(log_posteriors)
        #     print('watch')
        return log_posteriors

    # def inverse_log_posterior(self, s):
    #     """first_target: shape(B,1,2)"""
    #     log_posteriors = self.decode_target(s)
    #     # first_logp = log_posteriors(first_target)
    #     # first_log_posterior = log_posteriors[:, 0, :]
    #     # log_sum_exp = first_log_posterior - first_logp
    #     exp_logp_by_sum = tf.math.exp(log_posteriors)
    #     return exp_logp_by_sum


class PlaceCellEnsemble(CellEnsemble):
    """Calculates the dist over place cells given an absolute position."""

    def __init__(self, n_cells, stdev=0.35, pos_min=-5, pos_max=5, seed=None,
                 soft_targets=None, soft_init=None):
        super(PlaceCellEnsemble, self).__init__(n_cells, soft_targets, soft_init)
        # Create a random MoG with fixed cov over the position (Nx2)
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(pos_min, pos_max, size=(self.n_cells, 2))  # shape(256,2)
        self.stdev = stdev
        self.variances = np.ones_like(self.means) * stdev**2

    def unnor_logpdf(self, trajs):
        # Output the probability of each component at each point (BxTxN)
        meanx = self.means[:, 0]

        diff = trajs[:, :, tf.newaxis, :] - self.means[np.newaxis, np.newaxis, ...]  # shape (10, 100, 256, 2)
        unnor_logp = -0.5 * tf.reduce_sum((diff**2) / self.variances, axis=-1)  # shape (10, 100, 256)
        # note that matrix_A/matrix_B is done elementwise
        # with tf.train.SingularMonitoredSession() as sess:
        #     ob = sess.run(unnor_logp)
        #     print('watch')
        return unnor_logp

    def decode_position(self, s):
        temp, pos_list, encoding_list = self.get_ref_points(s)

        tensor_means = tf.convert_to_tensor(self.means, dtype=tf.float32)
        threshold = tf.constant(0.9999)
        max_encoding = encoding_list[0]

        # preparation for decoding on three points
        # exp_logp_by_sum = []
        # for i in range(3):
        #     # exp_logp_by_sum.append(self.inverse_log_posterior(encoding_list[i]))  # (10,100)*3
        #     exp_logp_by_sum.append(encoding_list[i]) # (10,100)*3

        position = tf.constant([0, 0])  # initialization
        for i in range(10):
            for j in range(100):
                temp_position_0 = tf.expand_dims(tensor_means[tf.gather_nd(pos_list[0], [i, j])], axis=0)  # (1,2)
                temp_position = []
                temp_exp_logp = []
                for k in range(3):
                    temp_position.append(tf.expand_dims(tensor_means[tf.gather_nd(pos_list[k], [i, j])], axis=0))
                    temp_exp_logp.append(tf.expand_dims(tf.gather_nd(encoding_list[k], [i, j]), axis=0))

                if i == 0 and j == 0:
                    position = temp_position_0  # assume the first point == 1
                    valid_position = temp_position_0
                else:

                    def condition(max_value, threshold):
                        return tf.math.less(threshold, max_value)

                    def f1(temp):
                        return temp

                    # def f2(valid):
                    #     return valid

                    def decode_on_three_points(pos, value):

                        point1 = [pos[0], value[0]]  # [(1,2), (1)]
                        point2 = [pos[1], value[1]]
                        point3 = [pos[2], value[2]]

                        # normal line of point1 and point2
                        slope_n1, alpha = line_equ(point1, point2)
                        # slope_n1 = tf.expand_dims(slope_n1, axis=2)

                        # normal line of point1 and point2
                        slope_n2, beta = line_equ(point1, point3)
                        # slope_n2 = tf.expand_dims(slope_n2, axis=2)

                        # get the intersection of point alpha and beta
                        # x = (beta[:, :, 1] - alpha[:, :, 1] + slope_n1 * alpha[:, :, 0] - slope_n2 * beta[:, :, 0]) / (
                        #             slope_n1 - slope_n2)
                        # y = slope_n1 * (x - alpha[:, :, 0]) + alpha[:, :, 1]  # line equation
                        x = (beta[0, 1] - alpha[0, 1] + slope_n1 * alpha[0, 0] - slope_n2 * beta[0, 0]) / (
                                    slope_n1 - slope_n2)
                        y = slope_n1 * (x - alpha[0, 0]) + alpha[0, 1]  # line equation
                        return tf.expand_dims(tf.stack([x, y]), axis=0)

                    def line_equ(p1, p2):
                        v = p1[0] - p2[0]
                        # distance between reference points
                        dist = tf.norm(v, ord='euclidean')  # alpha1+alpha2

                        # find the intersection of the line and its normal line
                        h = p1[1] / p2[1]  # propotion from decoding
                        square_dist_diff = -2 * (self.stdev ** 2) * tf.math.log(h)  # dist_1^2 - dist_2^2
                        diff_dist = square_dist_diff / dist  # dist_1-dist_2
                        dist_1 = (diff_dist + dist) / 2
                        dist_2 = dist - dist_1
                        # dist_1 = tf.concat([tf.expand_dims(dist_1, axis=2), tf.expand_dims(dist_1, axis=2)], axis=2)
                        position = p1[0] - dist*v  # (10,100,2)

                        # normal line equation
                        slope1 = ((p2[0])[0, 1] - p1[0][0, 1]) / (
                                    (p2[0])[0, 0] - p1[0][0, 0])  # (y2-y1)/(x2-x1)
                        # slope1 = ((p2[0])[:, :, 1] - p1[0][:, :, 1]) / (
                        #             (p2[0])[:, :, 0] - p1[0][:, :, 0])  # (y2-y1)/(x2-x1)
                        slope_n1 = -1 / slope1
                        # y = slope_n1*(x-alpha[0]) + alpha[1] # line function of normal vector for v1
                        return slope_n1, position

                    result = tf.cond(condition(max_encoding[i, j], threshold),
                                     lambda: f1(temp_position_0),
                                     lambda: decode_on_three_points(temp_position, temp_exp_logp))
                    valid_position = result

                    position = tf.concat([position, valid_position], axis=0)  # (10*100,2)
        return position

    def decode_on_three_points(self, pos, value):
        points = []
        for i in range(3):
            exp_logp_by_sum = self.inverse_log_posterior(value[i])  # (10,100,256)
            contaned = [pos[i], exp_logp_by_sum]
            points.append(contaned)

        # # find the 3 nearst position to calcaulte the euclidean position
        # points = []
        # temp = exp_logp_by_sum
        # for i in range(3):
        #     index = tf.math.argmax(temp, axis=2)  # (10, 100)
        #     shape = tf.shape(index)
        #     # position = tf.stack([tf.zeros(shape=shape), tf.zeros(shape=shape)], axis=-1)
        #     # position = tf.map_fn(lambda x: self.means[x], index)
        #
        #     # get max value from argmax:
        #     # B = tf.constant([[[0, 10, 30, 6],
        #     #                   ...[0, 21, 14, 15],
        #     #                   ...[0, 12, 7, 5]],
        #     #                  ...[[5, 8, 9, 15],
        #     #                      ...[11, 3, 27, 9],
        #     #                      ...[13, 42, 3, 2]]])
        #     # indexB = tf.argmax(B, axis=2)
        #     # grid = tf.stack(tf.meshgrid(tf.range(2), tf.range(3), indexing='ij'), axis=2)
        #     # grid = tf.dtypes.cast(grid, tf.int64)
        #     # position = tf.concat([grid, tf.expand_dims(indexB, axis=-1)], axis=2)
        #     # Bmax = tf.gather_nd(B, position)
        #
        #     grid = tf.stack(tf.meshgrid(tf.range(10), tf.range(100), indexing='ij'), axis=2)
        #     grid = tf.dtypes.cast(grid, tf.int64)
        #     grid_value = tf.concat([grid, tf.expand_dims(index, axis=-1)], axis=2)
        #     exp_logp = tf.gather_nd(exp_logp_by_sum, grid_value)
        #
        #     tensor_means = tf.convert_to_tensor(self.means, dtype=tf.float32)
        #
        #     for j in range(10):
        #         for k in range(100):
        #             temp_position = tf.expand_dims(tensor_means[tf.gather_nd(index, [j, k])], axis=0)  # (1,2)
        #             if j == 0 and k == 0:
        #                 position = temp_position
        #             else:
        #                 position = tf.concat([position, temp_position], axis=0)  # (10*100,2)
        #     # temp[index] = 0
        #     position = tf.reshape(position, [10, 100, 2])
        #     contaned = [position, exp_logp]
        #     points.append(contaned)
        # print("get 3 nearst points")  # every three points determine a position

        point1 = points[0]  # [(10,100,2), (10, 100)]
        point2 = points[1]
        point3 = points[2]

        # v1 = point1[0] - point2[0]  # vector between points
        # v2 = point1[0] - point3[0]

        # # distance between reference points
        # dist1 = np.linalg.norm(v1, axis=2)  # alpha1+alpha2
        # dist2 = np.linalg.norm(v2, axis=2)  # beta1+beta2
        # angle = np.arccos(np.dot(v1, v2)/(dist1*dist2))

        # normal line of point1 and point2
        slope_n1, alpha = self.line_equ(point1, point2)
        # slope_n1 = tf.expand_dims(slope_n1, axis=2)

        # normal line of point1 and point2
        slope_n2, beta = self.line_equ(point1, point3)
        # slope_n2 = tf.expand_dims(slope_n2, axis=2)

        # get the intersection of point alpha and beta
        x = (beta[:, :, 1] - alpha[:, :, 1] + slope_n1 * alpha[:, :, 0] - slope_n2 * beta[:, :, 0]) / (
                    slope_n1 - slope_n2)
        y = slope_n1 * (x - alpha[:, :, 0]) + alpha[:, :, 1]  # line equation
        return tf.stack([x, y], axis=2)

    def line_equ(self, p1, p2):
        v = p1[0] - p2[0]
        # distance between reference points
        dist = tf.norm(v, ord='euclidean', axis=2)  # alpha1+alpha2

        # find the intersection of the line and its normal line
        h = p1[1] / p2[1]  # propotion from decoding
        square_dist_diff = -2 * (self.stdev ** 2) * tf.math.log(h)  # dist_1^2 - dist_2^2
        diff_dist = square_dist_diff / dist  # dist_1-dist_2
        dist_1 = (diff_dist + dist) / 2
        dist_2 = dist - dist_1
        # dist_1 = tf.concat([tf.expand_dims(dist_1, axis=2), tf.expand_dims(dist_1, axis=2)], axis=2)
        position = p1[0] - tf.math.multiply(tf.expand_dims(dist_1, axis=2), v)  # (10,100,2)

        # normal line equation
        slope1 = ((p2[0])[:, :, 1] - p1[0][:, :, 1]) / ((p2[0])[:, :, 0] - p1[0][:, :, 0])  # (y2-y1)/(x2-x1)
        slope_n1 = -1 / slope1
        # y = slope_n1*(x-alpha[0]) + alpha[1] # line function of normal vector for v1
        return slope_n1, position


class HeadDirectionCellEnsemble(CellEnsemble):
    """Calculates the dist over HD cells given an absolute angle."""

    def __init__(self, n_cells, concentration=20, seed=None,
                 soft_targets=None, soft_init=None):
        super(HeadDirectionCellEnsemble, self).__init__(n_cells,
                                                        soft_targets,
                                                        soft_init)
        # Create a random Von Mises with fixed cov over the position
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(-np.pi, np.pi, (n_cells))
        self.kappa = np.ones_like(self.means) * concentration

    def unnor_logpdf(self, x):
        return self.kappa * tf.cos(x - self.means[np.newaxis, np.newaxis, :])


# if __name__ == '__main__':
#
#     def test(m):
#         log_posteriors = m - tf.reduce_logsumexp(m, keep_dims=True)
#         init = softmax(log_posteriors)
#         return init
#
#     m = np.array([0.3, 0.7, 0.9])
#     log_posteriors = m - tf.reduce_logsumexp(m, keep_dims=True)
#     s = softmax(log_posteriors)
#     # s = test(p)
#     with tf.Session() as sess:
#         array = sess.run({'s': s,
#                           'lp': log_posteriors})
#     ob = np.exp(array['lp'])
#     diff = array['s'] - ob
#     print(diff)
