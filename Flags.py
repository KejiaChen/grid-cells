from absl import flags
import sys
# Task config
flags.DEFINE_string("task_root",
                    # "/home/learning/Documents/kejia/grid-cells",
                    None,
                    "Dataset path.")
flags.DEFINE_string("data_root",
                    "/home/learning/Documents/kejia/grid-cells/dm_lab_data/",
                    "path of the dataset folder to store data")
flags.DEFINE_string("map_name",
                    "map_10_0.txt",
                    "name of the txt map")
flags.DEFINE_string("saver_results_directory",
                    # "/home/learning/Documents/kejia/grid-cells/",
                    None,
                    "Path to directory for saving results.")
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

# A3C Training config
flags.DEFINE_string("A3C_training_optimizer_options",
                    "{'learning_rate': 1e-6, 'momentum': 0.99}",  # lr [1e-6, 2e-4]
                    "Defines a dict with opts passed to the optimizer.")
flags.DEFINE_float("alpha",
                   0.50,  # [0.48, 0.52]
                   "baseline cost")
flags.DEFINE_float("beta",
                   8e-5,  # [6e-5, 1e-4]
                   "entropy regularization")
flags.DEFINE_float("gamma",
                   0.99,
                   "discount factor in the value function")
flags.DEFINE_integer("backprop_len",
                     100,  # 100
                     "backpropagation steps in actor-critic learner")
flags.DEFINE_integer("save_interval",
                     50,
                     "backpropagation steps in actor-critic learner")
flags.DEFINE_integer("action_repeat",
                     4,
                     "repeat each action selected by the actor")
flags.DEFINE_integer("num_worker",
                     32,
                     "number of workers each running on one thread")
flags.DEFINE_integer("max_episode_length",
                     5400,  # 5400
                     "Number of maximum training steps in one episode.")
flags.DEFINE_integer("episode_num",
                     1000,
                     "Number of episodes.")
flags.DEFINE_string("A3C_training_optimizer_class",
                    "tf.keras.optimizers.RMSprop",
                    "The optimizer used for training.")

# Grid Training config
flags.DEFINE_integer("training_epochs", 1000, "Number of training epochs.")
flags.DEFINE_integer("training_steps_per_epoch", 1000,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("training_minibatch_size", 10,
                     "Size of the training minibatch.")
flags.DEFINE_integer("sequence_length",
                     100,
                     "number of steps in each trajectory used for grid training")
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
flags.DEFINE_bool("dataset_with_vision", False,
                  "Load vision inputs from dataset.")
flags.DEFINE_bool("train_with_vision", False,
                  "Train with visual inputs from dmlab.")
flags.DEFINE_bool("load_grid_cell", False,
                  "Load pretrained grid cell model")
flags.DEFINE_integer("saver_eval_time", 2,
                     "Frequency at which results are saved.")
flags.DEFINE_integer("saver_pdf_time", 50,
                     "frequency to save a new pdf result")

# A3C Model config
flags.DEFINE_integer("policy_nh_lstm",
                     256,
                     "Number of hidden units in policy LSTM.")

# Grid Model Config
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

# Environment config
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

# Debug
flags.DEFINE_string("task_dataset_info", "square_room",
                    "Name of the room in which the experiment is performed.")

flags.mark_flag_as_required("task_root")
flags.mark_flag_as_required("saver_results_directory")
FLAGS = flags.FLAGS
FLAGS(sys.argv)