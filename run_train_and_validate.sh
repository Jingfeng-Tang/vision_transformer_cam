#!/bin/bash

set -e
set -x

python train_and_validate.py --model_name 'vit_base'\
                             --num_classes 20\
                             --weights ''\
                             --freeze_layers False\
                             --epochs 500\
                             --batch_size 32\
                             --lr 0.001\
                             --lrf 0.01\
                             --dataset_path '/data/c425/tjf/datasets/VOC2012/'\
                             --train_img_name_path '/data/c425/tjf/vit/voc12/train.txt'\
                             --val_img_name_path '/data/c425/tjf/vit/voc12/val.txt'\
                             --ori_cam_path '/data/c425/tjf/vit/origincams/'\
                             --device 'cuda:0'

