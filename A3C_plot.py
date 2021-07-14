import json
import matplotlib.pyplot as plot
from absl import flags
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

# Task config
flags.DEFINE_string("task_root",
                    "/home/learning/Documents/kejia/grid-cells",
                    # None,
                    "Dataset path.")
flags.DEFINE_string("data_root",
                    "/home/learning/Documents/kejia/grid-cells/dm_lab_data/",
                    "path of the dataset folder to store data")
flags.DEFINE_string("map_name",
                    "map_10_0.txt",
                    "name of the txt map")
flags.DEFINE_string("saver_results_directory",
                    "/home/learning/Documents/kejia/grid-cells/",
                    # None,
                    "Path to directory for saving results.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)


def running_mean(list, run_len=100):
    avg = []
    if len(list) < 100:
        print("Short of data!")
    else:
        for i in range(run_len, len(list)):
            avg.append(sum(list[i-run_len:i])/run_len)
    return avg


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, "result/graph/reward07-12_14:27")
    supplement_list = [0]*100
    stats_file = FLAGS.saver_results_directory + "log/drl_log/" + 'stats07-12_14:27.json'

    with open(stats_file, 'r') as fp:
        stats_dict = json.load(fp)

    epsiode_reward = supplement_list + stats_dict['episode_reward']
    plot_list = running_mean(epsiode_reward, run_len=100)

    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 1, len(stats_dict['episode_reward']))
    ax.plot(plot_list)
    plt.xlabel("episode", fontsize=10)
    plt.ylabel("reward", fontsize=10)
    # plt.show()
    plt.savefig(fname)

    print("data loaded")
