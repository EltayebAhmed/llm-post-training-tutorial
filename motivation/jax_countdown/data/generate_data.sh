#!/bin/bash
python3 generate_data.py --n_samples=50000 --n_operands=4 --min_operand=0 \
    --max_operand=9 --alpha=3 --seed=42 --output_filepath=data.csv
    head -n 45000 data.csv > data_train.csv
    tail -n 5000 data.csv > data_test.csv