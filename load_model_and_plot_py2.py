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

tf.flags.DEFINE_string('saver_ckpt_directory',
                       "/home/learning/Documents/kejia/grid-cells/result/model/ckpt/",
                       # None,
                       'Path to directory for saving results.')
FLAGS = tf.flags.FLAGS


def loadm_model_and_plot():
    modelname = 'model_py2.7' + 'time'

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(FLAGS.saver_results_directory + modelname + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()

        cell_init_b = graph.get_tensor_by_name('grid_cell_supervised/cell_init/b:0')
        cell_init_w = graph.get_tensor_by_name('grid_cell_supervised/cell_init/w:0')
        state_init_b = graph.get_tensor_by_name('grid_cell_supervised/state_init/b:0')
        state_init_w = graph.get_tensor_by_name('grid_cell_supervised/state_init/w:0')

        # tbd
