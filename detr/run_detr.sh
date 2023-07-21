#!/bin/bash


EXP_DIR=/home/ec2-user/detr/experiments/dummies/exp6_aws_q20_r101dc5_lrd350s800
#RESUME_PATH=dexperiments/collage_g1/exp_g1_gamma3/checkpoint.pth
#WEIGHTS_PATH=weigths/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth
DATA_DIR=/home/ec2-user/deformable/data/data11_Oct7/
LABEL_PATH=/home/ec2-user/deformable/data/data11_Oct7/annotations/
#LABEL_PATH=/home/ec2-user/deformable/data/data_g1
#./run_defdetr.sh --two_stage --with_box_refine --epoch 150 --resume /home/ec2-user/deformable/Deformable-DETR/weights/r50_deformable_detr-checkpoint.pth --device cuda:0 --focal_alpha 0.4


PY_ARGS=${@:1}

python -u main.py \
--dataset_file coco --data_path ${DATA_DIR} \
--label_path ${LABEL_PATH} \
--output_dir ${EXP_DIR}  \
--num_queries 20 --num_cl 5 --batch_size 1 \
${PY_ARGS}



#python main.py --dataset_file coco --coco_path /home/ec2-user/deformable/data/data_g1/
# --output_dir /home/ec2-user/deformable/experiments/data_g1/exp_g1_q16_eosp5
# --resume weigths/r50_deformable_detr-checkpoint.pth --epochs 100 --num_queries 300
# --num_cl 91 --batch_size 2 --lr 8e-5 --dropout 0.2 --device cuda:0

#--with_box_refine --two_stage --focal_alpha 0.4 --focal_gamma 2
#r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth





python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --dataset_file smoke \
--data_path /home/yazdi/NEMO_DENSE/DATSETS/data11_Oct7 \
--output_dir /home/yazdi/NEMO_DENSE/EXPERIMENTS/EmptyDummy_March22/dummy-oct7/exp-doct_q20-lr \
--resume weights/detr-r50-e632da11.pth \
--epochs 650 --num_queries 20 --batch_size 2 \
--dropout 0.2 --num_cl 6 --eos_coef 0.1 \
--lr 8e-5 --lr_drop 200 \
--label_path /home/yazdi/NEMO_DENSE/DATSETS/data11_Oct7/emptydummyOct/



python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --dataset_file smoke \
--data_path /home/yazdi/NEMO_DENSE/DATSETS/data_collage_dummy \
--output_dir /home/yazdi/NEMO_DENSE/EXPERIMENTS/March-2022_collage-dummy/exp-march-q20-eosp1-lr-lrd \
--resume weights/detr-r50-e632da11.pth \
--epochs 650 --num_queries 20 --batch_size 2 \
--dropout 0.2 --eos_coef 0.01 --num_cl 6 \
--lr 8e-5 --lr_drop 150 \
--label_path /home/yazdi/NEMO_DENSE/DATSETS/data_collage_dummy/annotations/


python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --dataset_file smoke \
--data_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy \
--output_dir /home/yazdi/NEMO_DENSE/EXPERIMENTS/EmptyDummy_March22/dummy3-empty/exp-d3e_q20_lrd25-wd \
--resume weights/detr-r50-e632da11.pth \
--epochs 650 --num_queries 20 --batch_size 2 \
--num_cl 6 --eos_coef 0.02 \
--lr_drop 25 --weight_decay 1e-5 \
--label_path /home/yazdi/NEMO_DENSE/DATSETS/data3dummy/emptydummy3/
