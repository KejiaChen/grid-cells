# Vector-based navigation using grid-like representations in artificial agents

This package provides an implementation of the supervised learning experiments
in Vector-based navigation using grid-like representations in artificial agents,
as [published in Nature](https://www.nature.com/articles/s41586-018-0102-6)

Any publication that discloses findings arising from using this source code must
cite "Banino et al. "Vector-based navigation using grid-like representations in
artificial agents." Nature 557.7705 (2018): 429."

## Introduction

The grid-cell network is a recurrent deep neural network (LSTM). This network
learns to path integrate within a square arena, using simulated trajectories
modelled on those of foraging rodents. The network is required to update its
estimate of location and head direction using translational and angular velocity
signals which are provided as input. The output of the LSTM projects to place
and head direction units via a linear layer which is subject to regularization.
The vector of activities in the place and head direction units, corresponding to
the current position, was provided as a supervised training signal at each time
step.

The dataset needed to run this code can be downloaded from
[here](https://console.cloud.google.com/storage/browser/grid-cells-datasets).

The files contained in the repository are the following:

*   `train.py` is where the training and logging loop happen; The file comes
    with the flags defined in Table 1 of the paper. In order to run this file
    you will need to specify where the dataset is stored and where you want to
    save the results. The results are saved in PDF format and they contains the
    ratemaps and the spatial autocorrelagram order by grid score. The units are
    ordered from higher to lower grid score. Only the last evaluation is saved.
    Please note that given random seeds results can vary between runs.

*   `data_reader.py` read the TFRecord and returns a ready to use batch, which
    is already shuffled.

*   `model.py` contains the grid-cells network

*   `scores.py` contains all the function for calculating the grid scores and
    doing the plotting.

*   `ensembles.py` contains the classes to generate the targets for training of
    the grid-cell networks.

## Train

The implementation requires an installation of
[Python] version 3.7, and
[TensorFlow](https://www.tensorflow.org/) version 2.3.1, and
[Sonnet](https://github.com/deepmind/sonnet) version 2.0.0.

```shell
$ pip install tensorflow-gpu==2.3.1
$ pip install --upgrade tensorflow-probability==0.11.1
$ pip install dm-sonnet==2.0.0
$ pip install keras==2.4.3
$ pip install --upgrade numpy==1.18.5
$ pip install --upgrade tensorflow==1.12.0 ??
$ pip install --upgrade dm-sonnet==1.27 ??
$ pip install --upgrade scipy==1.4.1
($ sudo apt install pkg-config)
($ sudo apt-get install libpng-dev)
($ sudo apt-get install libfreetype6-dev)
$ pip install --upgrade matplotlib==3.3.2
$ pip install --upgrade tensorflow-probability==0.5.0 ??
$ pip install --upgrade 1.12.1
$ pip install dm-tree==0.1.5
$ pip install absl-py==1.15.0
```

An example training script can be executed from a python interpreter:

```shell
$  python train.py --task_root='/home/kejia/grid-cells' --saver_results_directory='/home/kejia/grid-cells/result'
```

## Deepmind Lab
1. Build [DeepMind Lab](https://github.com/deepmind/lab.git) following [official build instructions](https://github.com/deepmind/lab/blob/master/docs/users/build.md). 
Note that you may also need to install [bazel](https://docs.bazel.build/versions/master/install-ubuntu.html) (use the binary installer if google services are not available) and a specific version of python dev if it is not included in ```python3-dev``` :
```
sudo apt-get install python3.8-dev
```

2. Create datasets from deepmind lab
```
python dmlab_demo.py --map_name='map_10_0.txt' --data_root='/home/learning/Documents/kejia/grid-cells/dm_lab_data/' --dataset_size=100 --file_length=100 --eps_length=100 --coord_range=2.5
```
This will create a dataset from a dmlab enviroment define by the txt file. The dataset contains 100 files with 100 trajectories in each file. Each trajectory consists of 100 steps. The range of coordinates is (-1.25:1.25, -1.25:1.25). The dataset is stored in the ```data_root```.

Note that the first and last file in the dataset may be incomplete, and thus need to be replaced.

On the server, alternatively:
```
python dmlab_demo.py --map_name='map_10_0.txt' --data_root='/data/bing/kejia/dm_lab_data/' --dataset_size=100 --file_length=100 --eps_length=100 --coord_range=2.5
```

3. Train with dmlab
```
python dmlab_train.py --task_root='/home/learning/Documents/kejia/grid-cells' --saver_results_directory='/home/learning/Documents/kejia/grid-cells/' --dataset_with_vision=False --train_with_vision=False
```
On the server, alternatively:
```
python dmlab_train.py --task_root='home/bing_TUM/kejia/grid-cells' --saver_results_directory='/data/bing/kejia/' --dataset_with_vision=True --train_with_vision=False
```

Disclaimer: This is not an official Google product.

