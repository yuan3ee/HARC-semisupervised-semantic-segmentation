#!/bin/bash

FILENAME="/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/lib/models/backbones/pretrained/3x3resnet50-imagenet.pth"

# mkdir -p backbones/pretrained
wget https://github.com/yassouali/CCT/releases/download/v0.1/3x3resnet50-imagenet.pth -O $FILENAME
