#!/bin/bash
python generate_data.py --n_samples=150000 --n_operands=4 --min_operand=0 \
    --max_operand=9 --alpha=3 --seed=42 --output_filepath=data.csv
    head -n 140000 data.csv > data_train.csv
    tail -n 10000 data.csv > data_test.csv