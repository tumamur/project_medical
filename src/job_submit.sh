#!/bin/sh

# version 1 :
# python3 --n_epochs 5 --batch_size 16 --soft_label_type gaussian

# version 2 :
python3 --n_epochs 5 --batch_size 16

# version 3:
python3 --n_epochs 5 --batch_size 16 --soft_label_type gaussian --without_decay

# version 4:
python3 --n_epochs 5 --batch_size 16 --without_decay

# version 5;
python3 --n_epochs 5 --batch_size 16 --use_all_images --soft_label_type gaussian