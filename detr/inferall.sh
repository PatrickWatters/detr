#!/bin/bash

THRESH=0.5
MODEL_DIR=~/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp6_randummy/exp6_eosp03_q20
DUMMY=4

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/0_non/ --mode 0 --device cuda:0  --thresh $THRESH --dummy $DUMMY

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_smoke/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY


# grep -B 1 "dummy" /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/annotations/annotation_density_train.json
