#!/bin/bash
# NOTE: runs DETR test.py for all the checkpoints and stores the result in a separete file

FILES=$1 #path to the experiments e.g. dexperiments/collage_g1/exp_g1_gamma3/
PY_ARGS=${@:2} 

for f in $FILES/*.pth; do
	echo "Inferring on $f model"
	python test.py  \
	--resume $f \
	${PY_ARGS}

done


#Infdef 
#python dtest3.py --images_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/ 
#--resume dexperiments/collage_g1/exp_g1_gamma3/checkpoint0099.pth --num_queries 300 --num_cl 91 --mode 1 --device cuda:0  --thresh 0.5

#./inferloop.sh dexperiments/collage_g1/exp_g1_gamma3 --images_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/ --mode 1 --device cuda:0  --thresh 0.5

#Inf2

# Sample full test script
#node02:~/NEmo/detr$ python test.py
# --data_path /home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_small/
# --resume ~/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp5_randummy/exp5rd_eos-p01_q20_lr/checkpoint0299.pth
# --num_queries 20 --thresh 0.5 --nmsup 0
# --num_cl 8 --dummy 7 --mode 1 --device cpu
