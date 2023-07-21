#!/bin/bash
PY_ARGS=${@:1}
THRESH=0.5
MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/SingleClass_March2022/Added_Empties/exp-scam-q10-eosp2
DUMMY=0

#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/0_non/ --mode 0 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS} --nmsup 0.2 --disp 1

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_smoke/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS} --nmsup 0.2 --disp 1


# grep -B 1 "dummy" /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/annotations/annotation_density_train.json
# --nmsup 0.2 --disp 1

#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}


# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/EmptyDummy_March22/dummy-oct7/exp-doct_q20-lr/temp
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp5_randummy/exp5rd_eos-p01_q20_lr/temp2
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp6_randummy/exp6_q20/chosen
