
#!/bin/bash
OUTRES=/home/yazdi/NEMO_DENSE/EXPERIMENTS/Exp_randummy/Exp6_randummy/exp6_q20/

python main.py --dataset_file smoke \
 --data_path /home/yazdi/NEMO_DENSE/val_sets/xtra-small/ \
 --label_path /home/yazdi/NEMO_DENSE/val_sets/xtra-small/labels/single_class/ \
 --output_dir ${OUTP}/eval_xs/ \
 --resume ${OUTP}/checkpoint_max_mAP.pth \
 --custom --eval
