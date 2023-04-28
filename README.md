# Latent Score Generative Model for Fast Calorimeter Simulation

In this repo is the implementation of an adpted version of [Score-based Generative Modeling in Latent Space](https://arxiv.org/abs/2106.05931) using Tensorflow to generate calorimeter images.

## Requirements:
* [Tensorflow 2.9](https://www.tensorflow.org/)
* [Tensorflow addons](https://www.tensorflow.org/addons)
* [Tensorflow Probability](https://www.tensorflow.org/probability)
* [Horovod](https://horovod.ai/) for multi-GPU training

## Data
The implementation is based on the [Fast Calorimeter Simulation Challenge 2022](https://calochallenge.github.io/homepage/), in particular, only tested for Dataset 2, while most of the implementation required for datasets 1 and 3 are also available (but likely not functional).

Download the data [here](https://zenodo.org/record/6366271)

## Training

To train a new model do:

```bash
cd scripts
python train.py --data_folder PATH/TO/FILES
```

## Sampling and Plotting

To sample new events from the trained model do:

```bash
cd scripts
python plot.py --sample --nevts 1000
```

to sample 1000 events. After the sampling is done, a new file with the same structure as the initial training files will be created in the ```--data_folder``` path. From there, you can get some plots by running:

```bash
mkdir ../plots
python plot.py --plot_folder ../plots
```

plots created should be available at the folder ```../plots```