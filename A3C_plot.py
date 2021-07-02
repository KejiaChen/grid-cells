import json
import matplotlib.pyplot as plot
from absl import flags
import time
import sys

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


if __name__ == "__main__":
    stats_file = FLAGS.saver_results_directory + "log/drl_log/" + 'stats07-02_14:10.json'

    with open(stats_file, 'r') as fp:
        stats_dict = json.load(fp)

    print("data loaded")
