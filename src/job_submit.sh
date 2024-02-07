#!/bin/sh

# version 3 : python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --adaptive_threshold_disc 
# version 4: python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --adaptive_threshold_gen --adaptive_threshold_disc
# version : python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --use_float_reports --adaptive_threshold_gen --adaptive_threshold_disc


python3 train.py --n_epoch 15 --batch_size 1 --update_freq 1000 --adaptive_threshold_gen --adaptive_threshold_disc # version 6: 
python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --adaptive_threshold_gen --adaptive_threshold_disc  # version 7: 
python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --adaptive_threshold_gen --adaptive_threshold_disc # version 8: 
python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --adaptive_threshold_gen --adaptive_threshold_disc --img_gen_lr 0.001 --img_disc_lr 0.00001 --report_gen_lr 0.0001 --report_disc_lr 0.00001 #version 9
python3 train.py --n_epoch 15 --batch_size 16 --update_freq 50 --adaptive_threshold_gen --adaptive_threshold_disc --lambda_cycle_loss 20 --img_gen_lr 0.001 --img_disc_lr 0.00001 --report_gen_lr 0.0001 --report_disc_lr 0.00001 #version 10
