#!/bin/sh
 
python3 train.py --image_generator cgan --n_epochs 100 --dataset_size 10000
python3 train.py --image_generator ddpm --n_epochs 10 --dataset_size 10000