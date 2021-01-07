"""load the saved grid cell NN and plot the predicted trajectory"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import numpy as np
import tensorflow as tf
import logging
import time
import _tkinter
# import Tkinter    # pylint: disable=unused-import

matplotlib.use('Agg')

import dataset_reader   # pylint: disable=g-bad-import-order, g-import-not-at-top
import model    # pylint: disable=g-bad-import-order
import scores    # pylint: disable=g-bad-import-order
import utils    # pylint: disable=g-bad-import-order

# tf.flags.DEFINE_string('saver_ckpt_directory',
#                        # "/home/learning/Documents/kejia/grid-cells/result/model/ckpt/py2-11-16/ckpt_py211-18_16:19",
#                        # None,
#                        'Path to directory for saving results.')
# FLAGS = tf.flags.FLAGS

# Task config
tf.flags.DEFINE_string('task_dataset_info', 'square_room',
                       'Name of the room in which the experiment is performed.')
tf.flags.DEFINE_string('task_root',
                       '/home/learning/Documents/kejia/grid-cells',
                       # None,
                       'Dataset path.')
tf.flags.DEFINE_float('task_env_size', 2.2,
                      'Environment size (meters).')
tf.flags.DEFINE_list('task_n_pc', [256],
                     'Number of target place cells.')
tf.flags.DEFINE_list('task_pc_scale', [0.01],
                     'Place cell standard deviation parameter (meters).')
tf.flags.DEFINE_list('task_n_hdc', [12],
                     'Number of target head direction cells.')
tf.flags.DEFINE_list('task_hdc_concentration', [20.],
                     'Head direction concentration parameter.')
tf.flags.DEFINE_integer('task_neurons_seed', 8341,
                        'Seeds.')
tf.flags.DEFINE_string('task_targets_type', 'softmax',
                       'Type of target, soft or hard.')
tf.flags.DEFINE_string('task_lstm_init_type', 'softmax',
                       'Type of LSTM initialisation, soft or hard.')
tf.flags.DEFINE_bool('task_velocity_inputs', True,
                     'Input velocity.')
tf.flags.DEFINE_list('task_velocity_noise', [0.0, 0.0, 0.0],
                     'Add noise to velocity.')

# Model config
tf.flags.DEFINE_integer('model_nh_lstm', 128, 'Number of hidden units in LSTM.')
tf.flags.DEFINE_integer('model_nh_bottleneck', 256,
                        'Number of hidden units in linear bottleneck.')
tf.flags.DEFINE_list('model_dropout_rates', [0.5],
                     'List of floats with dropout rates.')
tf.flags.DEFINE_float('model_weight_decay', 1e-5,
                      'Weight decay regularisation')
tf.flags.DEFINE_bool('model_bottleneck_has_bias', False,
                     'Whether to include a bias in linear bottleneck')
tf.flags.DEFINE_float('model_init_weight_disp', 0.0,
                      'Initial weight displacement.')

# Training config
tf.flags.DEFINE_integer('training_epochs', 1000, 'Number of training epochs.')
tf.flags.DEFINE_integer('training_steps_per_epoch', 1000,
                        'Number of optimization steps per epoch.')
tf.flags.DEFINE_integer('training_minibatch_size', 10,
                        'Size of the training minibatch.')
tf.flags.DEFINE_integer('training_evaluation_minibatch_size', 4000,
                        'Size of the minibatch during evaluation.')
tf.flags.DEFINE_string('training_clipping_function', 'utils.clip_all_gradients',
                       'Function for gradient clipping.')
tf.flags.DEFINE_float('training_clipping', 1e-5,
                      'The absolute value to clip by.')

tf.flags.DEFINE_string('training_optimizer_class', 'tf.train.RMSPropOptimizer',
                       'The optimizer used for training.')
tf.flags.DEFINE_string('training_optimizer_options',
                       '{"learning_rate": 1e-5, "momentum": 0.9}',
                       'Defines a dict with opts passed to the optimizer.')

# Store
tf.flags.DEFINE_string('saver_results_directory',
                       "/home/learning/Documents/kejia/grid-cells/result",
                       # None,
                       'Path to directory for saving results.')
tf.flags.DEFINE_integer('saver_eval_time', 2,
                        'Frequency at which results are saved.')
tf.flags.DEFINE_integer("saver_pdf_time", 50,
                        "frequency to save a new pdf result")


# Require flags from keyboard input
# tf.flags.mark_flag_as_required('task_root')
# tf.flags.mark_flag_as_required('saver_results_directory')
FLAGS = tf.flags.FLAGS


def load_model_and_plot():
    model_dir = './result/model/ckpt_py211-18_16:19'
    meta_name = model_dir + '/model_py2.ckpt.meta'

    tf.reset_default_graph()

    # Create the motion models for training and evaluation
    data_root = FLAGS.task_root + '/data'
    data_reader = dataset_reader.DataReader(
        FLAGS.task_dataset_info, root=data_root, num_threads=4)
    train_traj = data_reader.read(batch_size=FLAGS.training_minibatch_size)
    # read_ops = data_reader.read_ops
    # read_temp0 = read_ops[0]
    # read_temp1 = read_ops[1]
    # read_temp2 = read_ops[2]

    # Create the ensembles that provide targets during training
    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=FLAGS.task_env_size,
        neurons_seed=FLAGS.task_neurons_seed,
        targets_type=FLAGS.task_targets_type,
        lstm_init_type=FLAGS.task_lstm_init_type,
        n_pc=FLAGS.task_n_pc,
        pc_scale=FLAGS.task_pc_scale)

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

    input_tensors = []
    init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
    if FLAGS.task_velocity_inputs:
        # Add the required amount of noise to the velocities
        vel_noise = tf.distributions.Normal(0.0, 1.0).sample(
            sample_shape=ego_vel.get_shape()) * FLAGS.task_velocity_noise
        input_tensors = [ego_vel + vel_noise] + input_tensors
    # Concatenate all inputs
    rnn_inputs = tf.concat(input_tensors, axis=2)

    # Replace euclidean positions and angles by encoding of place and hd ensembles
    # Note that the initial_conds will be zeros if the ensembles were configured
    # to provide that type of initialization
    initial_conds = utils.encode_initial_conditions(
        init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)

    # Encode targets as well
    ensembles_targets = utils.encode_targets(
        target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

    # Estimate future encoding of place and hd ensembles inputing egocentric vels to initialize three outputs?
    outputs, _ = rnn(initial_conds, rnn_inputs, training=True)
    ensembles_logits, bottleneck, lstm_output = outputs

    # Training loss
    pc_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ensembles_targets[0], logits=ensembles_logits[0], name='pc_loss')
    hd_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ensembles_targets[1], logits=ensembles_logits[1], name='hd_loss')
    total_loss = pc_loss + hd_loss
    train_loss = tf.reduce_mean(total_loss, name='train_loss')

    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.import_meta_graph(meta_name)  # load the saver where graph is saved
        saver.restore(sess, tf.train.latest_checkpoint('./result/model/ckpt_py211-18_16:19'))  # restore the weights
        graph = tf.get_default_graph()  # load the saved graph
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')
        gv = [v for v in tf.global_variables()]

        # input = sess.graph.get_tensor_by_name('inputs:0')
        # output = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
        print('load saved model')
        output = sess.run(outputs)
        res_total_loss = sess.run(train_loss)
        res = sess.run({'total_loss': train_loss,
                        'init': initial_conds,
                        'target': target_pos
                        # 'euclidean_position': euclidean_pos,
                        # 'decoding_diff': diff
                        },
                       feed_dict={})

        # gv = [v for v in tf.global_variables()]
        # for v in gv:
        #     print(v.name, '\n')
        #
        # cell_init_b = graph.get_tensor_by_name('grid_cell_supervised/cell_init/b:0')
        # cell_init_w = graph.get_tensor_by_name('grid_cell_supervised/cell_init/w:0')
        # state_init_b = graph.get_tensor_by_name('grid_cell_supervised/state_init/b:0')
        # state_init_w = graph.get_tensor_by_name('grid_cell_supervised/state_init/w:0')
        # lstm_w = graph.get_tensor_by_name('grid_cells_core/lstm/w_gates:0')
        # lstm_b = graph.get_tensor_by_name('grid_cells_core/lstm/b_gates:0')
        # bottleneck_w = graph.get_tensor_by_name('grid_cells_core/bottleneck/w:0')
        # pc_w = graph.get_tensor_by_name('grid_cells_core/pc_logits/w:0')
        # pc_b = graph.get_tensor_by_name('grid_cells_core/pc_logits/b:0')
        # hdc_w = graph.get_tensor_by_name('grid_cells_core/pc_logits_1/w:0')
        # hdc_b = graph.get_tensor_by_name('grid_cells_core/pc_logits_1/b:0')
        #
        #
        # feed_dict = {""}
        #
        # # op_to_restore = graph.get_operations()
        #
        # print("bottleneck_w:  ", bottleneck_w.eval())
        #
        # result = sess.run(outputs, feed_dict)

        return res


if __name__ == '__main__':
    result = load_model_and_plot()
    # outputs, _ = rnn(initial_conds, inputs, training=True)
    # ensembles_logits, bottleneck, lstm_output = outputs
