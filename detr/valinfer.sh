#!/bin/bash
PY_ARGS=${@:1}
THRESH=0.5
#DATA_DIR=/home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/
#DATA_DIR=/home/yazdi/NEMO_DENSE/DATSETS/data1/val_images/
DATA_DIR=/home/yazdi/NEMO_DENSE/DATSETS/single_class/val_images/
#MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp1-2_regular/Exp1-March2022
#MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Sep23_exp
MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/SingleClass_March2022/Regular

DUMMY=0


MODEL_DIR1=${MODEL_DIR}/exp-sc-q10-eosp8/temp
#MODEL_DIR2=${MODEL_DIR}/exp_g1_q50
#MODEL_DIR3=${MODEL_DIR}/exp_g1_q50_p05
#MODEL_DIR4=${MODEL_DIR}/exp_g1_q50_p5
#MODEL_DIR5=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp6_randummy/exp6_eosp008_q24
#MODEL_DIR6=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp6_randummy/exp6_eosp3_q20




#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/ --mode 1 --device cuda:1  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/0_non/ --mode 0 --device cuda:1  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}


# grep -B 1 "dummy" /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/annotations/annotation_density_train.json
# --nmsup 0.2 --disp 1

./inferloop.sh ${MODEL_DIR1} --data_path ${DATA_DIR} --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

#./valinferloop.sh ${MODEL_DIR2} --data_path ${DATA_DIR} --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

#./valinferloop.sh ${MODEL_DIR3} --data_path ${DATA_DIR} --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

#./valinferloop.sh ${MODEL_DIR4} --data_path ${DATA_DIR} --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

#./valinferloop.sh ${MODEL_DIR5} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/ --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

#./valinferloop.sh ${MODEL_DIR6} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/ --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}



#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}


# ./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data11_Oct7/val_images/ --mode 1 --device cuda:1  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}
