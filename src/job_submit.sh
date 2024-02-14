#!/bin/sh

# version 0 :
python3 train.py --n_epoch 5 --batch_size 16 --soft_label_type gaussian

# version 1 :
python3 train.py --n_epoch 5 --batch_size 16

# version 2:
python3 train.py --n_epoch 5 --batch_size 16 --soft_label_type gaussian --without_decay

# version 3:
python3 train.py --n_epoch 5 --batch_size 16 --without_decay

# version 4;
python3 train.py --n_epoch 5 --batch_size 16 --use_all_images --soft_label_type gaussian

# version 5;
python3 train.py --n_epoch 5 --batch_size 16 --use_all_images --soft_label_type gaussian --resume_training --resume_version 4 # previous experiment killed 