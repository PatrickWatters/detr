# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str) #** set argument to --data_path
    parser.add_argument('--label_path', type=str)

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--custom', action='store_true',
                        help="If true, the classification head will not be erased, suitable when evaluating or resuming your own custom model and want to preserve the weights.")
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--num_cl', default=4, type=int) #number of classes, only works with smoke dataset
    #parser.add_argument('--dummy', default=0, type=int)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    if args.label_path is None:
        args.label_path = args.data_path #distinguishing annotation root from image root 

    if args.custom and args.resume: # revising args based on checkpoint to avoid mistakes
        checkpointx = torch.load(args.resume, map_location='cpu')
        cargs = checkpointx['args']
        print("updating args:\n** num_cl [{} --> {}]\n** num_queries [{} --> {}]\n** batch_size [{} --> {}]".format(args.num_cl,cargs.num_cl,args.num_queries,cargs.num_queries, args.batch_size, cargs.batch_size))
        args.num_cl, args.num_queries, args.batch_size = cargs.num_cl, cargs.num_queries, cargs.batch_size

    #args.batch_size = 2

    print(args)
    if args.output_dir and utils.is_main_process():
        with (Path(args.output_dir) / "args.txt").open("w+") as fl:
            fl.write(str(args).replace(" ","\n"))  


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if not args.custom: # if we are resuming from our custom model we want to continue its training and thus preserve the weights (classification head) [March 23,2022]
            print("!!!! Removing classification head !!!!")
            del checkpoint["model"]["class_embed.weight"] #** deletes model weights to use a pretrained model weights
            del checkpoint["model"]["class_embed.bias"]
            del checkpoint["model"]["query_embed.weight"]

        #del checkpoint["model"]["class_embed.weight"] #** deletes model weights to use a pretrained model weights
        #del checkpoint["model"]["class_embed.bias"] #** deletes model weights to use a pretrained model weights
        #del checkpoint["model"]["query_embed.weight"]   #** deletes model weights to use a pretrained model weights


        model_without_ddp.load_state_dict(checkpoint['model'], strict = False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    maxAP = maxAR = maxAPsmall = maxAP50 = maxAP75 = maxComboS = maxComboM = maxAPm = maxAPlow = maxARlow = 0
    minLoss = minLoss_bbox = 100
    auxheader={'events':'mAP ,AP50, AP75, APsmall, mAR, loss, loss_bbox, ComboS, ComboM, APm, APlow, ARlow'} #[Added] March 19, 2022
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        # validation COCO
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        # Monitor on validation criteria 0:mAP , 1:mAPsmall, 2:mAR, 3:loss, 4:loss_bbox, 5: AP50, 6: AP75, 7:Combo
        # Monitor on validation criteria 0:mAP , 1:AP50, 2:AP75, 3:APsmall, 4:mAR, 5: loss, 6: loss_bbox, 7:ComboS 8:ComboM 9:APm 10:APlow, 11:ARlow --> [UPDATED] March 26, 2022 
        event=[0,0,0,0,0,0,0,0,0,0,0,0] # flags for checkpoint events
        multevent=ideal = lrd = 0
        saveAP, saveLoss = 50, 30
        epoch_AP = test_stats['coco_eval_bbox'][0]
        epoch_AP50 = test_stats['coco_eval_bbox'][1]
        epoch_AP75 = test_stats['coco_eval_bbox'][2]
        epoch_APsmall = test_stats['coco_eval_bbox'][3]
        epoch_APm = test_stats['coco_eval_bbox'][4]
        epoch_APlarge = test_stats['coco_eval_bbox'][5]
        epoch_AR = test_stats['coco_eval_bbox'][6]
        #epoch_combo = (test_stats['coco_eval_bbox'][7] + 2*test_stats['coco_eval_bbox'][9] + 3*epoch_APsmall + epoch_AP50 + epoch_AP) / 8 #March 18: adding weighted average combo metric
        epoch_comboS = (test_stats['coco_eval_bbox'][10] + 2*test_stats['coco_eval_bbox'][9] + 2*epoch_APsmall) / 5 #March 20,3:17 AM: updating weighted average combo metric
        epoch_comboM = (2*test_stats['coco_eval_bbox'][10] + test_stats['coco_eval_bbox'][9] + 2*epoch_APm) / 5 #March 23: adding weighted average combo metric
        epoch_loss = test_stats['loss']
        epoch_loss_bbox = test_stats['loss_bbox']
        epoch_APlow = epoch_APsmall + epoch_APm
        epoch_ARlow = test_stats['coco_eval_bbox'][9] + test_stats['coco_eval_bbox'][10]


        if(epoch_AP > maxAP):
            maxAP, event[0] = epoch_AP, 1
        if(epoch_AP50 > maxAP50):
            maxAP50, event[1] = epoch_AP50, 1
        if(epoch_AP75 > maxAP75):
            maxAP75, event[2] = epoch_AP75, 1
        if(epoch_APsmall > maxAPsmall):
            maxAPsmall, event[3] = epoch_APsmall, 1
        if(epoch_AR > maxAR):
            maxAR, event[4] = epoch_AR, 1
        if (epoch_loss < minLoss):
            minLoss, event[5] = epoch_loss, 1
        if (epoch_loss_bbox < minLoss_bbox):
            minLoss_bbox, event[6] = epoch_loss_bbox, 1
        if(epoch_comboS > maxComboS):
            maxComboS, event[7] = epoch_comboS, 1
        if(epoch_comboM > maxComboM):
            maxComboM, event[8] = epoch_comboM, 1
        if(epoch_APm > maxAPm):
            maxAPm, event[9] = epoch_APm, 1
        if(epoch_APlow > maxAPlow):
            maxAPlow, event[10] = epoch_APlow, 1
        if(epoch_ARlow > maxARlow):
            maxARlow, event[11] = epoch_ARlow, 1
        if (sum(event[:-2])>=4):
            multevent=1
        if (sum(event[-2:])==2):
            ideal=1

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                lrd = 1
                print("!saving checkpoint!")
            # save best checkpoint on monitored values
            if any(event) or lrd:
                print("*** Nemo saves best ***")
                auxLog={'epoch':epoch,'event':event,
                        'mAP':np.round(epoch_AP*100,2),'AP50':np.round(epoch_AP50*100,2),'APsmall':np.round(epoch_APsmall*100,2),'APm':np.round(epoch_APm*100,2),'APlarge':np.round(epoch_APlarge*100,2),
                        'mAR':np.round(epoch_AR*100,2),'AP75':np.round(epoch_AP75*100,2),'loss':np.round(epoch_loss,4),'loss_bbox':np.round(epoch_loss_bbox,4),
                        'class_error':np.round(test_stats['class_error'],4),'loss_ce':np.round(test_stats['loss_ce'],4),
                        'coco_eval_bbox':test_stats['coco_eval_bbox']}
                #auxLog = {key: np.round(auxLog[key],4) for key in auxLog}
                if (epoch + 1) > saveAP and event[0] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_mAP.pth')
                    #auxLog.update({'saved':1})
                if (epoch + 1) > saveAP and event[3] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_APsmall.pth')
                if (epoch + 1) > saveAP and event[4] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_mAR.pth')
                if (epoch + 1) > saveAP and event[1] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_AP50.pth')
                if (epoch + 1) > saveAP and event[2] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_AP75.pth')
                if (epoch + 1) > saveLoss and event[5] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_min_loss.pth')
                if (epoch + 1) > saveLoss and event[6] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_min_loss_bbox.pth')
                if (epoch + 1) > 50 and multevent == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_multi_event.pth')
                if (epoch + 1) > 20 and event[7] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_comboS.pth')
                if (epoch + 1) > 20 and event[8] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_comboM.pth')
                if (epoch + 1) > saveAP and event[9] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_APm.pth')
                if (epoch + 1) > saveAP and event[10] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_APlow.pth')
                if (epoch + 1) > saveAP and event[11] == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_ARlow.pth')
                if (epoch + 1) > 50 and ideal == 1:
                    checkpoint_paths.append(output_dir / f'checkpoint_ideal_APARlow.pth')
            #
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process() and epoch==0:
            with (output_dir / "auxlog.txt").open("a") as f2:
                f2.write(json.dumps(auxheader) + "\n")


        if args.output_dir and utils.is_main_process() and any(event):
            with (output_dir / "auxlog.txt").open("a") as f2:
                f2.write(json.dumps(auxLog) + "\n")

        if args.output_dir and utils.is_main_process() and (epoch==args.epochs-1 or (epoch+1)%100==0):
            with (output_dir / "auxlog.txt").open("a") as f2:
                f2.write(json.dumps(auxLog) + "\n")
                maxstats={'maxAP':np.round(maxAP*100,2),'maxAP50':np.round(maxAP50*100,2),'maxAP75':np.round(maxAP75*100,2),'maxAPs':np.round(maxAPsmall*100,2),'maxAR':np.round(maxAR*100,2),
                          'minLoss':np.round(minLoss,4),'minLoss_bbox':np.round(minLoss_bbox,4),'maxComboS':np.round(maxComboS*100,2),'maxComboM':np.round(maxComboM*100,2),'maxAPm':np.round(maxAPm*100,2)}
                f2.write(json.dumps(maxstats) + "\n")



        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
