#!/bin/bash
PY_ARGS=${@:1}
THRESH=0.5
DISP=1
MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/SingleClass_March2022/Regular/exp-sc-q10-r101/temp
DUMMY=0
 #!!!!!------------------------------->>> DONT FORGET TO SET DUMMY IF AVAILABLE <<<--------------------------!!!!!

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/1_xs --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY --disp $DISP ${PY_ARGS}

# --nmsup 0.2 --disp 1

#MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/EmptyDummy_March22/dummy-oct7/exp-doct_q20-eosp05-lr

#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/1_xs --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY --disp $DISP ${PY_ARGS}


#MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/March-2022_collage-dummy/exp-march-q16-eosp3-lr

#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/1_xs --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY --disp $DISP ${PY_ARGS}













#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp5_randummy/exp5rd_eos-p01_q20_lr/temp2
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp5_randummy/exp5rd_q30_eosp01_lr
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/EmptyDummy_March22/dummy3-empty/exp-d3e_q30_eosp01_lr
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/SingleClass_March2022/Added_Empties/exp-scam-q10-eosp2
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp6_randummy/exp6_eosp03_q20

