import threading
import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
import logging
import time
import numpy as np

import supervised_training.dmlab_dataset_reader as dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import supervised_training.model_new as grid_model    # pylint: disable=g-bad-import
import supervised_training.scores_new as scores    # pylint: disable=g-bad-import-order
import supervised_training.utils_new as utils   # pylint: disable=g-bad-import-order

from Flags import *


class GridModel:
    def __init__(self, optimizer, ckpt_path, load_model=False):
        """Initialize."""
        super(GridModel, self).__init__()
        # Create the ensembles that provide targets during training
        self.place_cell_ensembles = utils.get_place_cell_ensembles(
            env_size=FLAGS.task_env_size,
            neurons_seed=FLAGS.task_neurons_seed,
            targets_type=FLAGS.task_targets_type,
            lstm_init_type=FLAGS.task_lstm_init_type,
            n_pc=FLAGS.task_n_pc,
            pc_scale=FLAGS.task_pc_scale)
        self.head_direction_ensembles = utils.get_head_direction_ensembles(
            neurons_seed=FLAGS.task_neurons_seed,
            targets_type=FLAGS.task_targets_type,
            lstm_init_type=FLAGS.task_lstm_init_type,
            n_hdc=FLAGS.task_n_hdc,
            hdc_concentration=FLAGS.task_hdc_concentration)
        self.target_ensembles = self.place_cell_ensembles + self.head_direction_ensembles

        # Model creation
        self.rnn_core = grid_model.GridCellsRNNCell(
            target_ensembles=self.target_ensembles,
            nh_lstm=FLAGS.model_nh_lstm,
            nh_bottleneck=FLAGS.model_nh_bottleneck,
            dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
            bottleneck_weight_decay=FLAGS.model_weight_decay,
            bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
            init_weight_disp=FLAGS.model_init_weight_disp)
        self.rnn = grid_model.GridCellsRNN(self.rnn_core, FLAGS.model_nh_lstm)

        self.opt = optimizer

        if load_model:
            checkpoint = tf.train.Checkpoint(optimizer=self.opt, net=self.rnn)
            # path = '/home/learning/Documents/kejia/grid-cells/result/model/ckpt_py311-27_13:09/model_py3.ckpt-20'
            checkpoint.restore(save_path=ckpt_path)
            print('load checkpoint')

    def forward_grid_code(self, prev_pose, vels):
        """forward pass of rnn with no gradients"""
        outputs, _ = self.rnn(prev_pose, vels, training=False)

        ens_targets, bottleneck, lstm_output = outputs
        return tf.stop_gradient(bottleneck)


class GridAgent(threading.Thread):
    def __init__(self, replay_buffer, grid_model, logger, ckpt_manger, condition):
        threading.Thread.__init__(self)
        self.replay_buffer = replay_buffer
        self.lock = threading.Lock()
        self.thread_name = "grid_cell"
        # data_root = FLAGS.saver_results_directory + '/dm_lab_data'
        # self.data_reader = dataset_reader.DataReader(
        #     FLAGS.task_dataset_info, root=data_root, num_threads=4, vision=FLAGS.dataset_with_vision)

        self.model = grid_model
        self.COND = condition
        self.manager = ckpt_manger

        # self.place_cell_ensembles = utils.get_place_cell_ensembles(
        #     env_size=FLAGS.task_env_size,
        #     neurons_seed=FLAGS.task_neurons_seed,
        #     targets_type=FLAGS.task_targets_type,
        #     lstm_init_type=FLAGS.task_lstm_init_type,
        #     n_pc=FLAGS.task_n_pc,
        #     pc_scale=FLAGS.task_pc_scale)
        # self.head_direction_ensembles = utils.get_head_direction_ensembles(
        #     neurons_seed=FLAGS.task_neurons_seed,
        #     targets_type=FLAGS.task_targets_type,
        #     lstm_init_type=FLAGS.task_lstm_init_type,
        #     n_hdc=FLAGS.task_n_hdc,
        #     hdc_concentration=FLAGS.task_hdc_concentration)
        # self.target_ensembles = self.place_cell_ensembles + self.head_direction_ensembles
        #
        # # Model creation
        # self.rnn_core = grid_model.GridCellsRNNCell(
        #     target_ensembles=self.target_ensembles,
        #     nh_lstm=FLAGS.model_nh_lstm,
        #     nh_bottleneck=FLAGS.model_nh_bottleneck,
        #     dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
        #     bottleneck_weight_decay=FLAGS.model_weight_decay,
        #     bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
        #     init_weight_disp=FLAGS.model_init_weight_disp)
        # self.rnn = grid_model.GridCellsRNN(self.rnn_core, FLAGS.model_nh_lstm)
        #
        # self.opt = optimizer

        # self.log = self.setup_logger()
        self.log = logger
        print("initialize on thread", self.thread_name)

    def prepare_data(self, traj):
        if FLAGS.dataset_with_vision:
            init_pos, init_hd, ego_vel, target_pos, target_hd, frame = traj
        else:
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
            init_pos, init_hd, self.model.place_cell_ensembles, self.model.head_direction_ensembles)
        targets_to_cells = utils.encode_targets(
            target_pos, target_hd, self.model.place_cell_ensembles, self.model.head_direction_ensembles)
        return concat_inputs, initial_to_cells, targets_to_cells

    def setup_logger(self):
        log_name = 'tensorflow_py3.7_dmlab_A3C' + time.strftime("%m-%d_%H:%M", time.localtime())
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=FLAGS.task_root + '/log/' + log_name + '.log',
                            # filename='/log/' + log_name + '.log',
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
        return log

    @tf.function
    def manual_regularization(self, parameter):
        regularization = ((tf.nn.l2_loss(parameter) * 2) ** 0.5) * 0.5
        return regularization

    @tf.function
    def loss_object(self, targets, logits, l2_regularization=False):
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
            loss_regularization.append(self.manual_regularization(self.model.rnn.trainable_variables[3]))  # bottleneck_w
            loss_regularization.append(self.manual_regularization(self.model.rnn.trainable_variables[5]))  # hd_logits_w
            loss_regularization.append(self.manual_regularization(self.model.rnn.trainable_variables[7]))  # pc_logits_w
            # for p in rnn.trainable_variables:
            #     loss_regularization.append(tf.nn.l2_loss(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss = training_loss + FLAGS.model_weight_decay * loss_regularization
        return loss

    @tf.function
    def train_step(self, targets, inputs, init):
        # print("start tf function")
        with tf.GradientTape() as tape:
            outputs, _ = self.model.rnn(init, inputs, training=True)
            # print("trainable variables", rnn.trainable_variables)
            ensembles_logits, bottleneck, lstm_output = outputs
            loss = self.loss_object(targets, ensembles_logits, l2_regularization=True)
            grad = tape.gradient(loss, self.model.rnn.trainable_variables)
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
            self.model.opt.apply_gradients(zip(clipped_grad, self.model.rnn.trainable_variables))
            return loss, grad

    @tf.function
    def eval_step(self, targets, inputs, init):
        outputs, _ = self.model.rnn(init, inputs, training=False)
        ensembles_logits, bottleneck, lstm_output = outputs
        # loss = loss_object(targets, ensembles_logits)
        return ensembles_logits, bottleneck, lstm_output

    def train(self):
        self.log.info("initialize on thread %s", self.thread_name)

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
        latest_epoch_scorer = scores.GridScorer(20, self.replay_buffer.get_coord_range(FLAGS.task_env_size),
                                                masks_parameters)

        # checkpoint = tf.train.Checkpoint(optimizer=self.model.opt, net=self.model.rnn)
        # check_dir = FLAGS.saver_results_directory + "/result/model/ckpt_dmlab" + time.strftime("%m-%d_%H:%M",
        #                                                                                        time.localtime())
        #
        # manager = tf.train.CheckpointManager(checkpoint, directory=check_dir, max_to_keep=20,
        #                                      checkpoint_name='model_dmlab_A3C.ckpt')

        for epoch in range(FLAGS.training_epochs):
            loss_acc = list()
            if FLAGS.model_nh_bottleneck:
                self.log.info("Adding dropout layers")
            for _ in range(FLAGS.training_steps_per_epoch):
                # train_traj = self.data_reader.read(batch_size=FLAGS.training_minibatch_size)
                train_traj = self.replay_buffer.sample(batch_size=FLAGS.training_minibatch_size,
                                                       sequence_length=FLAGS.sequence_length)
                # init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
                conc_inputs, initial_conds, ensembles_targets = self.prepare_data(train_traj)
                train_loss, grad = self.train_step(ensembles_targets, conc_inputs, initial_conds)
                loss_acc.append(train_loss)
                # print(_)

            self.log.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
                          np.mean(loss_acc), np.std(loss_acc))
            # tf.compat.v1.logging.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
            #                           np.mean(loss_acc), np.std(loss_acc))
            if epoch % FLAGS.saver_eval_time == 0:
                res = dict()
                for _ in range(FLAGS.training_evaluation_minibatch_size //
                               FLAGS.training_minibatch_size):
                    # train_traj = self.data_reader.read(batch_size=FLAGS.training_minibatch_size)
                    train_traj = self.replay_buffer.sample(batch_size=FLAGS.training_minibatch_size,
                                                           sequence_length=FLAGS.sequence_length)
                    if FLAGS.dataset_with_vision:
                        init_pos, init_hd, ego_vel, target_pos, target_hd, frame = train_traj
                    else:
                        init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
                    conc_inputs, initial_conds, ensembles_targets = self.prepare_data(train_traj)
                    eval_ensembles_logits, eval_bottleneck, eval_lstm_output = self.eval_step(ensembles_targets,
                                                                                              conc_inputs,
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
                    filename = 'rates_and_sac_latest_hd_dmlab_A3C' + time.strftime("%m-%d_%H:%M",
                                                                                time.localtime()) + '.pdf'
                    # plotname = 'trajectory_py2.7_' + time.strftime("%m-%d_%H:%M", time.localtime()) + '.pdf'
                    self.manager.save()

                # utils.plot_trajectories(res['pos_xy'], res['pos_xy'], 10, FLAGS.saver_results_directory + '/result',
                #                         plotname, axis_min=-1.25, axis_max=1.25)

                grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
                    'btln_60_separation'], grid_scores[
                    'btln_90_separation'] = utils.get_scores_and_plot(
                    latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                    FLAGS.saver_results_directory + '/result', filename, plot_graphs=False)
                grid_scores_60 = grid_scores['btln_60']
                grid_mask = np.zeros_like(grid_scores_60)
                grid_mask[grid_scores_60 >= 0.37] = 1
                num_grid_cells = np.sum(grid_mask)
                self.log.info('Epoch %i, number of grid-cell like cells %f', epoch, num_grid_cells)

    def run(self):
        # while True:
        #     memory_length = self.replay_buffer.get_memory_length()
        #     if memory_length > FLAGS.training_minibatch_size:
        #         print("start supervised training")
        #         break
        self.COND.acquire()
        self.COND.wait()
        print("start supervised training")
        self.train()
        self.COND.release()


def load_grid_cell(path, optimizer):
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

    rnn_core = grid_model.GridCellsRNNCell(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS.model_nh_lstm,
        nh_bottleneck=FLAGS.model_nh_bottleneck,
        dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
        bottleneck_weight_decay=FLAGS.model_weight_decay,
        bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
        init_weight_disp=FLAGS.model_init_weight_disp)
    rnn = grid_model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=rnn)
    # path = '/home/learning/Documents/kejia/grid-cells/result/model/ckpt_py311-27_13:09/model_py3.ckpt-20'
    checkpoint.restore(save_path=path)
    print('load checkpoint')

    return rnn
