#!/usr/bin/env bash
trap "exit" INT
base_command="uv run train_har.py -g 0 --net pruned_temportal_cnn1d --source realworld --source_localisation chest --metric accuracy f1score --save_model -chs 3 "

exp_name="test_experiment_har"
id_job=10
rand_seed=1

training_settings=("--n_epoch 70 -bs 32 -lr 0.001 --scheduler MultiStepLR --milestones 60 90 -gm 0.1")
rand_conv_settings=("-ks 1 3 5 7 -rc -idp 0.5 -cl -clw 5 -db kaiming_normal")
pruning_settings=("-pruning -p_algs Synflow -ps 70 -p_iter 100")


$base_command --exp_name $exp_name -jb $id_job -rs $rand_seed $training_settings $rand_conv_settings $pruning_settings
