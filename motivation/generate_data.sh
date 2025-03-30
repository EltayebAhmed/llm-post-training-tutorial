#!/bin/bash
python3 generate_data.py --n_samples=5000 --n_operands=4 --min_operand=0 \
    --max_operand=9 --alpha=0.14 --seed=42 --output_filepath=data.csv
    head -n 4500 data.csv > data_train.csv
    tail -n 500 data.csv > data_test.csv