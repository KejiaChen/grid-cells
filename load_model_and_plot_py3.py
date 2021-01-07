"""Load the trained Gridcell Network"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
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

import dataset_reader_new as dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import model_new as model    # pylint: disable=g-bad-import-order
import scores_new as scores    # pylint: disable=g-bad-import-order
import utils_new as utils   # pylint: disable=g-bad-import-order


# Task config
flags.DEFINE_string("task_dataset_info", "square_room",
                    "Name of the room in which the experiment is performed.")
flags.DEFINE_string("task_root",
                    "/home/learning/Documents/kejia/grid-cells",
                    # None,
                    "Dataset path.")
flags.DEFINE_integer("use_data_files", 100,
                     "Number of files to read")
flags.DEFINE_float("task_env_size", 2.2,
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
flags.DEFINE_integer("training_epochs", 10, "Number of training epochs.")
flags.DEFINE_integer("training_steps_per_epoch", 10,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("training_minibatch_size", 10,
                     "Size of the training minibatch.")
flags.DEFINE_integer("training_evaluation_minibatch_size", 40,
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

# Store
flags.DEFINE_string("saver_results_directory",
                    "/home/learning/Documents/kejia/grid-cells/result",
                    # None,
                    "Path to directory for saving results.")
flags.DEFINE_integer("saver_eval_time", 2,
                     "Frequency at which results are saved.")
flags.DEFINE_integer("saver_pdf_time", 5,
                     "frequency to save a new pdf result")

# Require flags from keyboard input
flags.mark_flag_as_required("task_root")
flags.mark_flag_as_required("saver_results_directory")
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def load_model_and_plot(path):
    # Create the motion models for training and evaluation
    data_root = FLAGS.task_root + '/data'
    data_reader = dataset_reader.DataReader(
        FLAGS.task_dataset_info, root=data_root, num_threads=4)

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

    rnn_core = model.GridCellsRNNCell(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS.model_nh_lstm,
        nh_bottleneck=FLAGS.model_nh_bottleneck,
        dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
        bottleneck_weight_decay=FLAGS.model_weight_decay,
        bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
        init_weight_disp=FLAGS.model_init_weight_disp)
    rnn = model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

    # Optimisation ops
    optimizer_class = eval(FLAGS.training_optimizer_class)  # pylint: disable=eval-used
    optimizer = optimizer_class(**eval(FLAGS.training_optimizer_options))  # pylint: disable=eval-used

    def prepare_data(traj):
        init_pos, init_hd, ego_vel, target_pos, target_hd = traj
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

    @tf.function
    def eval_step(targets, inputs, init):
        outputs, _ = rnn(init, inputs, training=False)
        ensembles_logits, bottleneck, lstm_output = outputs
        # loss = loss_object(targets, ensembles_logits)
        return ensembles_logits, bottleneck, lstm_output

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=rnn)
    # path = '/home/learning/Documents/kejia/grid-cells/result/model/ckpt_py311-27_13:09/model_py3.ckpt-20'
    checkpoint.restore(save_path=path)
    print('load checkpoint')

    # evaluation
    res = dict()
    eval_loss = list()
    for _ in range(5):
        traj = data_reader.read(batch_size=FLAGS.training_minibatch_size)
        init_pos, init_hd, ego_vel, target_pos, target_hd = traj
        conc_inputs, initial_conds, ensembles_targets = prepare_data(traj)
        eval_ensembles_logits, eval_bottleneck, eval_lstm_output = eval_step(ensembles_targets, conc_inputs, initial_conds)
        eval_loss.append(loss_object(ensembles_targets, eval_ensembles_logits, l2_regularization=True))

        mb_res = {
            'bottleneck': eval_bottleneck,
            'lstm': eval_lstm_output,
            'pos_xy': target_pos
        }
        res = utils.new_concat_dict(res, mb_res)  # evaluation output

    print('mean loss %.5f, std loss %.5f', np.mean(eval_loss), np.std(eval_loss))


if __name__ == '__main__':
    ckpt_path = '/home/learning/Documents/kejia/grid-cells/result/model/ckpt_py311-27_13:09/model_py3.ckpt-20'
    load_model_and_plot(path=ckpt_path)
