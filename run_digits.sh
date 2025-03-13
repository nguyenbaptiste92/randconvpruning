#!/usr/bin/env bash
trap "exit" INT
base_command="uv run train_digits.py -g 0 --net pruned_resnet20 --verbose --source mnist10k --metric accuracy --save_model -chs 3 "

exp_name="test_experiment_digit"
id_job=10
rand_seed=1

training_settings=("--n_epoch 1 -bs 32 -lr 0.0001")
rand_conv_settings=("-ks 1 3 5 7 -rc -idp 0.5 -cl -clw 5 -db kaiming_normal")
pruning_settings=("-pruning --rewinding --rewinding_epoch 1 -ps 30 -p_algs L1")


$base_command --exp_name $exp_name -jb $id_job -rs $rand_seed $training_settings $rand_conv_settings $pruning_settings
