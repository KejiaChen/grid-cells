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

"""Supervised training for the Grid cell network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import os
import numpy as np
import tensorflow as tf
from absl import flags
import sys
import tensorflow_probability as tfp
import _tkinter
# import Tkinter    # pylint: disable=unused-import
import logging
import time

matplotlib.use('Agg')

import dmlab_dataset_reader as dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import model_new as model    # pylint: disable=g-bad-import
import scores_new as scores    # pylint: disable=g-bad-import-order
import utils_new as utils   # pylint: disable=g-bad-import-order


# Task config
flags.DEFINE_string("task_dataset_info", "square_room",
                    "Name of the room in which the experiment is performed.")
flags.DEFINE_string("task_root",
                    # "/home/learning/Documents/kejia/grid-cells",
                    None,
                    "Dataset path.")
flags.DEFINE_integer("use_data_files", 100,
                     "Number of files to read")
flags.DEFINE_float("task_env_size", 2.5,
                   "Environment size (meters).")
flags.DEFINE_list("task_n_pc", [256],
                  "Number of target place cells.")
flags.DEFINE_list("task_pc_scale", [0.01],
                  "Place cell standard deviation parameter (meters).")
flags.DEFINE_list("task_n_hdc", [12],
                  "Number of target head direction cells.")
flags.DEFINE_list("task_hdc_concentration", [20.],
                  "Head direction concentration parameter.")
flags.DEFINE_integer("task_neurons_seed", 8341,
                     "Seeds.")
flags.DEFINE_string("task_targets_type", "softmax",
                    "Type of target, soft or hard.")
flags.DEFINE_string("task_lstm_init_type", "softmax",
                    "Type of LSTM initialisation, soft or hard.")
flags.DEFINE_bool("task_velocity_inputs", True,
                  "Input velocity.")
flags.DEFINE_list("task_velocity_noise", [0.0, 0.0, 0.0],
                  "Add noise to velocity.")

# Model config
flags.DEFINE_integer("model_nh_lstm", 128, "Number of hidden units in LSTM.")
flags.DEFINE_integer("model_nh_bottleneck", 256,
                     "Number of hidden units in linear bottleneck.")
flags.DEFINE_list("model_dropout_rates", [0.5],
                  "List of floats with dropout rates.")
flags.DEFINE_float("model_weight_decay", 1e-5,
                   "Weight decay regularisation")
flags.DEFINE_bool("model_bottleneck_has_bias", False,
                  "Whether to include a bias in linear bottleneck")
flags.DEFINE_float("model_init_weight_disp", 0.0,
                   "Initial weight displacement.")

# Training config
flags.DEFINE_integer("training_epochs", 1000, "Number of training epochs.")
flags.DEFINE_integer("training_steps_per_epoch", 1000,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("training_minibatch_size", 10,
                     "Size of the training minibatch.")
flags.DEFINE_integer("training_evaluation_minibatch_size", 4000,
                     "Size of the minibatch during evaluation.")
flags.DEFINE_string("training_clipping_function", "utils.new_clip_all_gradients",
                    "Function for gradient clipping.")
flags.DEFINE_float("training_clipping", 1e-5,
                   "The absolute value to clip by.")
flags.DEFINE_string("training_optimizer_class",
                    "tf.compat.v1.train.RMSPropOptimizer",
                    # "tf.keras.optimizers.RMSprop",
                    "The optimizer used for training.")
flags.DEFINE_string("training_optimizer_options",
                    "{'learning_rate': 1e-5, 'momentum': 0.9}",
                    "Defines a dict with opts passed to the optimizer.")
flags.DEFINE_bool("train_with_vision", False,
                  "Train with visual inputs from dmlab.")

# Store
flags.DEFINE_string("saver_results_directory",
                    # "/home/learning/Documents/kejia/grid-cells/",
                    None,
                    "Path to directory for saving results.")
flags.DEFINE_integer("saver_eval_time", 2,
                     "Frequency at which results are saved.")
flags.DEFINE_integer("saver_pdf_time", 50,
                     "frequency to save a new pdf result")

# Switch mode: training or test
# flags.DEFINE_boolean("test",
#                      # "/home/learning/Documents/kejia/grid-cells/result",
#                      False,
#                      "choose 'train' or 'test'")

# Require flags from keyboard input
flags.mark_flag_as_required("task_root")
flags.mark_flag_as_required("saver_results_directory")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

FILE_PATH = os.path.realpath(__file__)
FILE_DIR, _ = os.path.split(FILE_PATH)

def train():
    """Training loop."""

    # tf.compat.v1.reset_default_graph()

    # Create the motion models for training and evaluation
    data_root = FLAGS.saver_results_directory + '/dm_lab_data'
    data_reader = dataset_reader.DataReader(
            FLAGS.task_dataset_info, root=data_root, num_threads=4, vision=FLAGS.train_with_vision)
    # train_batch = data_reader.read_batch(batch_size=FLAGS.training_minibatch_size)


    # Create the ensembles that provide targets during training
    place_cell_ensembles = utils.get_place_cell_ensembles(
            env_size=FLAGS.task_env_size,
            neurons_seed=FLAGS.task_neurons_seed,
            targets_type=FLAGS.task_targets_type,
            lstm_init_type=FLAGS.task_lstm_init_type,
            n_pc=FLAGS.task_n_pc,
            pc_scale=FLAGS.task_pc_scale)

    # print("place cell ensembles", place_cell_ensembles)

    head_direction_ensembles = utils.get_head_direction_ensembles(
            neurons_seed=FLAGS.task_neurons_seed,
            targets_type=FLAGS.task_targets_type,
            lstm_init_type=FLAGS.task_lstm_init_type,
            n_hdc=FLAGS.task_n_hdc,
            hdc_concentration=FLAGS.task_hdc_concentration)
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    # Model creation
    rnn_core = model.GridCellsRNNCell(
            target_ensembles=target_ensembles,
            nh_lstm=FLAGS.model_nh_lstm,
            nh_bottleneck=FLAGS.model_nh_bottleneck,
            dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
            bottleneck_weight_decay=FLAGS.model_weight_decay,
            bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
            init_weight_disp=FLAGS.model_init_weight_disp)
    rnn = model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

    # init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj

    def prepare_data(traj):
        if FLAGS.train_with_vision:
            init_pos, init_hd, ego_vel, target_pos, target_hd,   = traj
        else:
            init_pos, init_hd, ego_vel, target_pos, target_hd= traj
        # inputs
        input_tensors = []
        if FLAGS.task_velocity_inputs:
            # Add the required amount of noise to the velocities
            vel_noise = tfp.distributions.Normal(0.0, 1.0).sample(
                sample_shape=ego_vel.get_shape()) * FLAGS.task_velocity_noise
            input_tensors = [ego_vel + vel_noise] + input_tensors
        # Concatenate all inputs
        concat_inputs = tf.concat(input_tensors, axis=2)  # shape=(10*100*3)

        # encode initial and target
        initial_to_cells = utils.encode_initial_conditions(
            init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
        targets_to_cells = utils.encode_targets(
            target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)
        return concat_inputs, initial_to_cells, targets_to_cells

    # Replace euclidean positions and angles by encoding of place and hd ensembles
    # Note that the initial_conds will be zeros if the ensembles were configured
    # to provide that type of initialization
    # !!change in loops
    # initial_conds = utils.encode_initial_conditions(
    #         init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    # # print(initial_conds)
    #
    # # Encode targets as well
    # # !!change in loops
    # ensembles_targets = utils.encode_targets(
    #         target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

    # Estimate future encoding of place and hd ensembles inputing egocentric vels? to initialize three outputs?
    # outputs, _ = rnn(initial_conds, inputs, training=True)
    # ensembles_logits, bottleneck, lstm_output = outputs

    # Training loss
    @tf.function
    def manual_regularization(parameter):
        regularization = ((tf.nn.l2_loss(parameter) * 2) ** 0.5) * 0.5
        return regularization

    @tf.function
    def loss_object(targets, logits, l2_regularization=False):
        pc_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets[0], logits=logits[0], name='pc_loss')
        hd_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets[1], logits=logits[1], name='hd_loss')
        bi_loss = pc_loss + hd_loss
        training_loss = tf.reduce_mean(input_tensor=bi_loss, name='train_loss')
        # If use sonnet 2.0.0, add l2_regularization in loss since
        # sonnet has no inner regularizer in snt.Linear.
        # Disable l2_regularization if use keras instead
        if l2_regularization:
            loss_regularization = []
            # add regularization for bottleneck_w, hd_logits_w, pc_logits_w
            loss_regularization.append(manual_regularization(rnn.trainable_variables[3]))  # bottleneck_w
            loss_regularization.append(manual_regularization(rnn.trainable_variables[5]))  # hd_logits_w
            loss_regularization.append(manual_regularization(rnn.trainable_variables[7]))  # pc_logits_w
            # for p in rnn.trainable_variables:
            #     loss_regularization.append(tf.nn.l2_loss(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss = training_loss + FLAGS.model_weight_decay*loss_regularization
        return loss

    # Optimisation ops
    optimizer_class = eval(FLAGS.training_optimizer_class)    # pylint: disable=eval-used
    optimizer = optimizer_class(**eval(FLAGS.training_optimizer_options))    # pylint: disable=eval-used
    # grad = optimizer.compute_gradients(train_loss)
    # clip_gradient = eval(FLAGS.training_clipping_function)    # pylint: disable=eval-used
    # clipped_grad = [
    #         clip_gradient(g, var, FLAGS.training_clipping) for g, var in grad
    # ]
    # train_op = optimizer.apply_gradients(clipped_grad)

    # Store the grid scores
    grid_scores = dict()
    grid_scores['btln_60'] = np.zeros((FLAGS.model_nh_bottleneck,))
    grid_scores['btln_90'] = np.zeros((FLAGS.model_nh_bottleneck,))
    grid_scores['btln_60_separation'] = np.zeros((FLAGS.model_nh_bottleneck,))
    grid_scores['btln_90_separation'] = np.zeros((FLAGS.model_nh_bottleneck,))
    grid_scores['lstm_60'] = np.zeros((FLAGS.model_nh_lstm,))
    grid_scores['lstm_90'] = np.zeros((FLAGS.model_nh_lstm,))

    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(),
                                            masks_parameters)

    # with tf.compat.v1.train.SingularMonitoredSession() as sess:
    @tf.function
    def train_step(targets, inputs, init):
        # print("start tf function")
        with tf.GradientTape() as tape:
            outputs, _ = rnn(init, inputs, training=True)
            # print("trainable variables", rnn.trainable_variables)
            ensembles_logits, bottleneck, lstm_output = outputs
            loss = loss_object(targets, ensembles_logits, l2_regularization=True)
            grad = tape.gradient(loss, rnn.trainable_variables)
            grad_var = []
            # for i in range(12):
            #     temp = tf.tuple([grad[i], rnn.trainable_variables[i]])
            #     grad_var.append(temp)
            # grad = optimizer.compute_gradients(train_loss)
            clip_gradient = eval(FLAGS.training_clipping_function)  # pylint: disable=eval-used
            # clipped_grad = [
            #     clip_gradient(g, var, FLAGS.training_clipping) for g, var in grad
            # ]
            clipped_grad = [
                clip_gradient(g, FLAGS.training_clipping) for g in grad
            ]
            optimizer.apply_gradients(zip(clipped_grad, rnn.trainable_variables))
            return loss, grad

    @tf.function
    def eval_step(targets, inputs, init):
        outputs, _ = rnn(init, inputs, training=False)
        ensembles_logits, bottleneck, lstm_output = outputs
        # loss = loss_object(targets, ensembles_logits)
        return ensembles_logits, bottleneck, lstm_output

    # logging
    log_name = 'tensorflow_py3.7_dmlab_' + time.strftime("%m-%d_%H:%M", time.localtime())
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=FLAGS.task_root + '/log/' + log_name + '.log',
                        filemode='w')
    # logging.info('please log something')
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    # formatter
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    # handler
    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=rnn, iterator=iterator)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=rnn)
    check_dir = FLAGS.saver_results_directory + "/result/model/ckpt_dmlab" + time.strftime("%m-%d_%H:%M", time.localtime())

    manager = tf.train.CheckpointManager(checkpoint, directory=check_dir, max_to_keep=20,
                                         checkpoint_name='model_dmlab.ckpt')

    # uncomment this line to run in Eager mode for debugging
    # tf.config.run_functions_eagerly(True)
    for epoch in range(FLAGS.training_epochs):
        loss_acc = list()
        if FLAGS.model_nh_bottleneck:
            log.info("Adding dropout layers")
        for _ in range(FLAGS.training_steps_per_epoch):
            train_traj = data_reader.read(batch_size=FLAGS.training_minibatch_size)
            # init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
            conc_inputs, initial_conds, ensembles_targets = prepare_data(train_traj)
            train_loss, grad = train_step(ensembles_targets, conc_inputs, initial_conds)
            loss_acc.append(train_loss)
            # print(_)

        log.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
                 np.mean(loss_acc), np.std(loss_acc))
        # tf.compat.v1.logging.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
        #                           np.mean(loss_acc), np.std(loss_acc))
        if epoch % FLAGS.saver_eval_time == 0:
            res = dict()
            for _ in range(FLAGS.training_evaluation_minibatch_size //
                           FLAGS.training_minibatch_size):
                train_traj = data_reader.read(batch_size=FLAGS.training_minibatch_size)
                init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
                conc_inputs, initial_conds, ensembles_targets = prepare_data(train_traj)
                eval_ensembles_logits, eval_bottleneck, eval_lstm_output = eval_step(ensembles_targets, conc_inputs,
                                                                                     initial_conds)
                mb_res = {
                    'bottleneck': eval_bottleneck,
                    'lstm': eval_lstm_output,
                    'pos_xy': target_pos
                }
                res = utils.new_concat_dict(res, mb_res)  # evaluation output
                # print(_)

            # Store at the end of validation
            if epoch % FLAGS.saver_pdf_time == 0:
                filename = 'rates_and_sac_latest_hd_dmlab_' + time.strftime("%m-%d_%H:%M", time.localtime()) + '.pdf'
                plotname = 'trajectory_py2.7_' + time.strftime("%m-%d_%H:%M", time.localtime()) + '.pdf'
                manager.save()

            utils.plot_trajectories(res['pos_xy'], res['pos_xy'], 10, FLAGS.saver_results_directory+'/result', plotname, axis_min=-1.25, axis_max=1.25)

            grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
                'btln_60_separation'], grid_scores[
                'btln_90_separation'] = utils.get_scores_and_plot(
                latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                FLAGS.saver_results_directory+'/result', filename)
            grid_scores_60 = grid_scores['btln_60']
            grid_mask = np.zeros_like(grid_scores_60)
            grid_mask[grid_scores_60 >= 0.37] = 1
            num_grid_cells = np.sum(grid_mask)
            log.info('Epoch %i, number of grid-cell like cells %f', epoch, num_grid_cells)


if __name__ == '__main__':
    train()
