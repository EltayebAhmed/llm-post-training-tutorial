#!/bin/bash
rm -rf rl_model_ft 
python3.11 train.py --save-dir=rl_model_ft --num_train_steps=700 --batch_size=4096 --base_model_dir=base_model_train/checkpoint_700