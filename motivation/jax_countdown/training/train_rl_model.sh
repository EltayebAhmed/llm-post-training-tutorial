#!/bin/bash
rm -rf rl_model_ft 
python3.11 train_rl.py --save-dir=rl_model_ft --num_train_steps=5 --batch_size=4096 --base_model_path=base_model_train/checkpoint_699