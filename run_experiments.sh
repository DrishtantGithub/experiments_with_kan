#!/usr/bin/env bash
# quick script to run essential experiments
set -e
python src/train/train_toy.py --quick --activation kan --save_dir ./results/toy_kan
python src/train/train_toy.py --quick --activation relu --save_dir ./results/toy_relu

python src/train/train_cifar.py --quick --use_kan --save_dir ./results/cifar_kan
python src/train/train_cifar.py --quick --save_dir ./results/cifar_relu
