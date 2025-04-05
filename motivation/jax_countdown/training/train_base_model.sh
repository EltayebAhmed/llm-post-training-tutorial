#!/bin/bash
rm -rf base_model_train
python3.11 train.py --save-dir=base_model_train --num_train_steps=700 --batch_size=4096