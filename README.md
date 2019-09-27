# Building-Classification

This is the code for the project of *soft-story building classification*.

### Prerequisites

Tensorflow 1.0.

Python 2

CPU or NVIDIA GPU + CUDA CuDNN

### Getting Started
step 1: Install latest version of TF-slim following the instruction [here](https://github.com/tensorflow/models/tree/master/research/slim)

step 2: Put this repo in the foloder /models/research/slim/

step 3: Download [pre-trained models](https://github.com/tensorflow/models/tree/master/research/slim) (ResNet50/152 or InceptionV3/V4) and put them in the folder /models/research/slim/pretrained/; download the data via this link and put them in /models/research/slim//tfrecords/

step 4: Run the code.

### Training a Model

cd /models/research/slim/

./finetune_resnet_50_on_buildings.sh

### Evaluating a Model

cd /models/research/slim/

./finetune_resnet_50_on_buildings_eval.sh

### Datasets and Results


