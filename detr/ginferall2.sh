#!/bin/bash
PY_ARGS=${@:1}
DISP=0
THRESH=0.5
MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Perfect_overlap/Dummy/pol_q20
DUMMY=4
 #!!!!!------------------------------->>> DONT FORGET TO SET DUMMY IF AVAILABLE <<<--------------------------!!!!!

###./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/0_non/ --mode 0 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}

./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_smoke/ --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}


./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/val_sets/1_xs --mode 1 --device cpu  --thresh $THRESH --dummy $DUMMY --disp $DISP ${PY_ARGS}


# grep -B 1 "dummy" /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/annotations/annotation_density_train.json
# --nmsup 0.2 --disp 1

#./inferloop.sh ${MODEL_DIR} --data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/val_images/ --mode 1 --device cuda:0  --thresh $THRESH --dummy $DUMMY ${PY_ARGS}
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp5_randummy/exp5rd_eos-p01_q20_lr/temp2
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/March-2022_collage-dummy/exp-march-q16-eosp3-lr
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/AWS-exps-2022/exp6_aws_eosp07_q20_dc5_lr9
# MODEL_DIR=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Sep23_exp/exp_g1_q50_p05

cut -d, -f1 ${MODEL_DIR}/infer_logs/test-1_smoke-$THRESH.txt > ${MODEL_DIR}/temp1
cut -d, -f1 ${MODEL_DIR}/infer_logs/test-0_non-$THRESH.txt > ${MODEL_DIR}/temp0
cut -d, -f2-3 ${MODEL_DIR}/infer_logs/test-1_smoke-$THRESH.txt > ${MODEL_DIR}/temp2

paste ${MODEL_DIR}/temp0 ${MODEL_DIR}/temp1 | awk '{$0=$2/($1+$2)}1' > ${MODEL_DIR}/temprecision
paste ${MODEL_DIR}/temprecision ${MODEL_DIR}/temp1 | awk '{$0=2*(($1*($2/100))/($1+($2/100)))}1' > ${MODEL_DIR}/tempf1


paste ${MODEL_DIR}/temp1 ${MODEL_DIR}/temp0 ${MODEL_DIR}/tempf1 ${MODEL_DIR}/temp2 > ${MODEL_DIR}/infer_logs/combined-$THRESH.txt

rm ${MODEL_DIR}/temp1
rm ${MODEL_DIR}/temp0
rm ${MODEL_DIR}/temp2
rm ${MODEL_DIR}/temprecision
rm ${MODEL_DIR}/tempf1

