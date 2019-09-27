#!/bin/bash
#
# This script performs the following operations:
#
# 1. Evaluate ResNet50 model on the buildings validation set.
#
#
# Usage:
# cd slim
# ./finetune_resnet_50_on_buildings_eval.sh
set -e

# Where the pre-trained ResNet50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=./pretrained/

# Where the checkpoint to be evaluated.
TRAIN_DIR=/PATH/TO/TRAINED/MODEL/

# Where the dataset is saved to.
DATASET_DIR=./tfrecords/Santa_Monica/
DATASET_NAME=buildings_smbv2_SM_random

# set model
MODEL_NAME=resnet_v1_50

# set subset
SET=validation

# Run evaluation.

echo "Extracting ${SET} Features..."
/usr/bin/python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${SET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}

