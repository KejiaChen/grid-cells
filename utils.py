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

"""Helper functions for creating the training graph and plotting.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import ensembles    # pylint: disable=g-bad-import-order


np.seterr(invalid="ignore")


def get_place_cell_ensembles(
        env_size, neurons_seed, targets_type, lstm_init_type, n_pc, pc_scale):
    """Create the ensembles for the Place cells."""
    place_cell_ensembles = [
            ensembles.PlaceCellEnsemble(
                    n,
                    stdev=s,
                    pos_min=-env_size / 2.0,
                    pos_max=env_size / 2.0,
                    seed=neurons_seed,
                    soft_targets=targets_type,
                    soft_init=lstm_init_type)
            for n, s in zip(n_pc, pc_scale)
    ]
    return place_cell_ensembles


def get_head_direction_ensembles(
        neurons_seed, targets_type, lstm_init_type, n_hdc, hdc_concentration):
    """Create the ensembles for the Head direction cells."""
    head_direction_ensembles = [
            ensembles.HeadDirectionCellEnsemble(
                    n,
                    concentration=con,
                    seed=neurons_seed,
                    soft_targets=targets_type,
                    soft_init=lstm_init_type)
            for n, con in zip(n_hdc, hdc_concentration)
    ]
    return head_direction_ensembles


def encode_initial_conditions(init_pos, init_hd, place_cell_ensembles,
                              head_direction_ensembles):
    initial_conds = []
    for ens in place_cell_ensembles:
        initial_conds.append(
                tf.squeeze(ens.get_init(init_pos[:, tf.newaxis, :]), axis=1))
    for ens in head_direction_ensembles:
        initial_conds.append(
                tf.squeeze(ens.get_init(init_hd[:, tf.newaxis, :]), axis=1))
    return initial_conds


def encode_targets(target_pos, target_hd, place_cell_ensembles,
                   head_direction_ensembles):
    ensembles_targets = []
    for ens in place_cell_ensembles:  # list_length:1
        ensembles_targets.append(ens.get_targets(target_pos))
    for ens in head_direction_ensembles:
        ensembles_targets.append(ens.get_targets(target_hd))
    return ensembles_targets


def decode_targets(ensembles_targets, place_cell_ensembles,
                   head_direction_ensembles):
    euclidean_targets = []
    for ens in place_cell_ensembles:
        euclidean_targets.append(ens.decode_position(ensembles_targets[0]))
    return euclidean_targets


def clip_all_gradients(g, var, limit):
    # print(var.name)
    return (tf.clip_by_value(g, -limit, limit), var)


def clip_bottleneck_gradient(g, var, limit):
    if ("bottleneck" in var.name or "pc_logits" in var.name):
        return (tf.clip_by_value(g, -limit, limit), var)
    else:
        return (g, var)


def no_clipping(g, var):
    return (g, var)


def concat_dict(acc, new_data):
    """Dictionary concatenation function."""

    def to_array(kk):
        if isinstance(kk, np.ndarray):
            return kk
        else:
            return np.asarray([kk])

    for k, v in new_data.iteritems():
        if isinstance(v, dict):
            if k in acc:
                acc[k] = concat_dict(acc[k], v)
            else:
                acc[k] = concat_dict(dict(), v)
        else:
            v = to_array(v)
            if k in acc:
                acc[k] = np.concatenate([acc[k], v])
            else:
                acc[k] = np.copy(v)
    return acc


def get_scores_and_plot(scorer,
                        data_abs_xy,
                        activations,
                        directory,
                        filename,
                        plot_graphs=True,    # pylint: disable=unused-argument
                        nbins=20,    # pylint: disable=unused-argument
                        cm="jet",
                        sort_by_score_60=True):
    """Plotting function."""

    # Concatenate all trajectories
    xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
    act = activations.reshape(-1, activations.shape[-1])
    n_units = act.shape[1]
    # Get the rate-map for each unit

    # for i in range(n_units):
    #     temp = scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
    #     print(temp)

    s = [
            scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
            for i in xrange(n_units)
    ]
    # Get the scores
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(
            *[scorer.get_scores(rate_map) for rate_map in s])
    # Separations
    # separations = map(np.mean, max_60_mask)
    # Sort by score if desired
    if sort_by_score_60:
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = range(n_units)
    # Plot
    cols = 16
    rows = int(np.ceil(n_units / cols))
    fig = plt.figure(figsize=(24, rows * 4))
    for i in xrange(n_units):
        rf = plt.subplot(rows * 2, cols, i + 1)
        acr = plt.subplot(rows * 2, cols, n_units + i + 1)
        if i < n_units:
            index = ordering[i]
            title = "%d (%.2f)" % (index, score_60[index])
            # Plot the activation maps
            scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
            # Plot the autocorrelation of the activation maps
            scorer.plot_sac(
                    sac[index],
                    mask_params=max_60_mask[index],
                    ax=acr,
                    title=title,
                    cmap=cm)
    # Save
    if not os.path.exists(directory):
        os.makedirs(directory)
    with PdfPages(os.path.join(directory, filename), "w") as f:
        plt.savefig(f, format="pdf")
    plt.close(fig)
    return (np.asarray(score_60), np.asarray(score_90),
            np.asarray(map(np.mean, max_60_mask)),
            np.asarray(map(np.mean, max_90_mask)))


def plot_trajectories(target_trajectory, decode_trajectory, num, directory, filename):
    size = np.size(target_trajectory, 0) - 1
    # id = int(np.ceil(size*np.random.rand(num)))  # select 10 trajctories randomly
    # size = 10
    cols = 5
    rows = int(np.ceil(num / cols))
    # fig = plt.figure()
    fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(15, 6))
    for i in range(rows):
        for j in range(cols):
            # fig, axis = plt.subplots(rows, cols, i+1, sharex='col', sharey='row')
            index = (i-1)*cols + j
            target_x = target_trajectory[index, :, 0]
            target_y = target_trajectory[index, :, 1]
            decode_x = decode_trajectory[index, :, 0]
            decode_y = decode_trajectory[index, :, 1]
            ax[i][j].plot(target_x, target_y, color='blue', linewidth=3.0)
            ax[i][j].plot(decode_x, decode_y, color='green', linewidth=3.0)
            ax[i][j].set_xlim((-1.1, 1.1))
            ax[i][j].set_ylim((-1.1, 1.1))
    fig.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    with PdfPages(os.path.join(directory, filename), "w") as f:
        plt.savefig(f, format="pdf")
    plt.close(fig)


def get_session(sess):
    """tf.train.MonitoredTrainingSession(...) doesn't support tf.train.Saver().save
       use this function to circumvent this issue """
    session = sess
    while type(session).__name__ != 'Session':
        # pylint: disable=W0212
        session = session._sess
    return session
