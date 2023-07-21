# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import csv
#import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from torchvision.ops import nms #adding nms to refine overlapping bboxes

import torch

import util.misc as utils

from models import build_model
from datasets.face import make_face_transforms

import matplotlib.pyplot as plt
import time


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        filenames.sort()
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm': #[put for video]
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
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

    # # Loss
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
    parser.add_argument('--dataset_file', default='face')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_base', action='store_true')

    parser.add_argument('--thresh', default=0.5, type=float)
    parser.add_argument('--num_cl', default=4, type=int)
    parser.add_argument('--disp', default=0, type=int) #set 1 to plot and save
    parser.add_argument('--disp_attn', default=0, type=int) #set 1 to plot encoder-decoder attention weights
    parser.add_argument('--disp_sattn', default=0, type=int) #set 1 to plot self attention weights


    parser.add_argument('--nmsup', default=0,
                        help='apply non-max suppression to bounding boxes', type=float)

    parser.add_argument('--iou_thresh', default=0.2, help='threshold for non-max suppression', type=float)
    parser.add_argument('--dummy', default=0, help='dummy class category_id to be ignored', type=float)
    parser.add_argument('--mode', default=2, type=int,
                        help='0:print only if smoke detected, 1:print only if no smoke, 2:print all')
                        

    return parser

# plot_sattn(image, orig_image, sattn, figsave_path_sattn)
def plot_sattn(img,im,sattn,img_name):
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32
    COLORS=['c','c','r','r']

    # let's select 4 reference points for visualization
    idxs = [(130, 160), (555, 200), (100,1250), (415, 685),]
    #idxs = [(350, 690), (230,860),]

    #import pdb; pdb.set_trace()
    print(img.shape)

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='viridis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}',fontsize=20)

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(im)
    for (y, x), c in zip(idxs,COLORS):
        scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), 30, color=c, alpha=0.77))
        fcenter_ax.axis('off')
    fig.savefig(img_name)
    plt.close()

#plot_attn(h,w,bboxes_scaled,probas,img,dec_attn_weights, keep, figsave_path) # probas[keep] --> probask

def plot_attn(h,w, bboxes_scaled, probas, im, dec_attn_weights, keep,img_name):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    #CLASSES= [ 'N/A', 'smoke']

    if args.num_cl == 2:
        CLASSES = [
        'N/A','smoke'
        ]
        #title1 = 'NEmo' +maxIOU
        title1 = 'Nemo-detr-sc'
    elif args.num_cl == 4:
        CLASSES = [
        'N/A', 'low', 'mid', 'high'
        ]
        title1 = 'Nemo-detr-d'
    elif args.num_cl == 5 and args.dummy==4 :
        CLASSES = [
        'N/A', 'low', 'mid', 'high', 'no-smoke', 'N/A'
        ]
        title1 = 'Nemo-detr-dda'
    elif args.num_cl == 8 and args.dummy==7 :
        CLASSES = [
        'N/A', 'low', 'N/A', 'mid', 'N/A' , 'high', 'N/A' ,'no-smoke'
        ]
        title1 = 'Nemo-detr-dda'
    elif args.num_cl == 6:
        CLASSES = [
        'N/A', 'low', 'N/A', 'mid', 'N/A' , 'high'
        ]
        title1 = 'Nemo-detr-dg'

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7),squeeze=False)
    colors = COLORS * 100
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    fig.tight_layout()

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(img_name,bbox_inches = 'tight', pad_inches = 0)
    plt.close()



def plot_results(pil_img, prob, boxes,img_name):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    title1 = 'Nemo'

    # TODO: After Python 3.10 : We can use match case instead of lot of nested ifs
    if args.num_cl == 2:
        CLASSES = [
        'N/A','smoke'
        ]
        title1 = 'Nemo-detr-sc'
    elif args.num_cl == 4:
        CLASSES = [
        'N/A', 'low', 'mid', 'high'
        ]
        title1 = 'Nemo-detr-d'
    elif args.num_cl == 5 and args.dummy==4 :
        CLASSES = [
        'N/A', 'low', 'mid', 'high', 'no-smoke', 'N/A'
        ]
        title1 = 'Nemo-detr-dda'
    elif args.num_cl == 8 and args.dummy==7 :
        CLASSES = [
        'N/A', 'low', 'N/A', 'mid', 'N/A' , 'high', 'N/A' ,'no-smoke'
        ]
        title1 = 'Nemo-detr-dda'
    elif args.num_cl == 6:
        CLASSES = [
        'N/A', 'low', 'N/A', 'mid', 'N/A' , 'high'
        ]
        title1 = 'Nemo-detr-dg'
    else:
        CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        title1 = 'DETR'

    title2 = os.path.basename(img_name)
    fs = 36 #font size


    h,w,_ = pil_img.shape    
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=16,
                bbox=dict(facecolor='yellow', alpha=0.5))
    ax.text(w*0.993,h*0.01, title1,fontsize=fs,bbox=dict(facecolor='cyan',alpha=0.85),horizontalalignment='right',verticalalignment='top')
    #ax.set_title(title2,fontsize=16,y=-0.05)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.axis('off')
    plt.savefig(img_name,bbox_inches = 'tight', pad_inches = 0)
    #plt.show()
    plt.close()

    
@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    dummy_class=args.dummy   # a new class to include non-smoke images. The training for this class is 
                    #positioned at bbox [0,0,1,1] xmin,ymin,w,h

    #loaded_model = os.path.basename(args.resume)
    loaded_images = os.path.basename(os.path.normpath(args.data_path))
    if args.resume:
        loaded_model = os.path.basename(args.resume)

    if args.disp: #save inferences
        (output_path / 'Inferences').mkdir(exist_ok=True)

    if args.disp_attn: #save inferences
        (output_path / 'Attn_viz').mkdir(exist_ok=True)

    if args.disp_sattn: #save inferences
        (output_path / 'sAttn_viz').mkdir(exist_ok=True)

    duration = 0
    has_smoke = 0
    counter = 0
    v = args.mode
    print("Inferring {} images.".format(len(images_path)))
    for img_sample in images_path:
        counter +=1
        filename = os.path.basename(img_sample)
        if v==2:
            print("processing...{}".format(filename))

        filename = filename.replace('jpeg','').replace('.jpg','').replace(';','-').replace(' ','_')
        if args.disp:
            figsave_path = output_path / "Inferences" / ('nemo_'+filename+'.png')
        if args.disp_attn:
            figsave_path_attn = output_path / "Attn_viz" / ('attn_nemo_'+filename+'.png')
        if args.disp_sattn:
            figsave_path_sattn = output_path / "sAttn_viz" / ('sattn_nemo_'+filename+'.png')
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        transform = make_face_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)

        if len(bboxes_scaled) == 0:
            if v>=1:
                print("no-smoke:  {}".format(filename))
            #print("no-smoke:  {}".format(filename))
            continue

        #import pdb; pdb.set_trace();

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        #import pdb; pdb.set_trace();
        
        # output of the CNN
        f_map = conv_features['0']

        # get the HxW shape of the feature maps of the CNN
        shape = f_map.tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        sattn = enc_attn_weights[0].reshape(shape + shape)
        
        if args.disp_sattn:
            print("Encoder attention:      ", enc_attn_weights[0].shape)
            print("Feature map:            ", f_map.tensors.shape)
            print("-"*10)
            print("Reshaped self-attention:", sattn.shape)
            plot_sattn(image, orig_image, sattn, figsave_path_sattn)

        if args.disp_attn:
            plot_attn(h,w,bboxes_scaled,probas, orig_image ,dec_attn_weights, keep, figsave_path_attn)

        if dummy_class: # if prediction is the dummy class it means no-smoke, we remove the dummy boxes
            #print("dummy class: %d",dummy_class)
            dummy_catid = dummy_class
            keep2= probas.max(-1).indices != dummy_catid #
            keep1 = torch.logical_and(keep,keep2)
            if(len(keep)!=len(keep2)):
                print("remvoing {} dummy bboxes".format(len(keep)-len(keep2)))
            keep = keep1
        
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        #probas = probas[keep].cpu().data.numpy()
        probask = probas[keep]

        if len(bboxes_scaled) == 0:
            if v>=1:
                print("no-smoke:  {}".format(filename))
            #print("no-smoke:  {}".format(filename))
            continue


        #import pdb; pdb.set_trace();
        #there was a detection
        has_smoke += 1
        if v!=1: #display if v=2 or when we are predictint non-smoke images
            print("[{}/{}] [{:.2f}%]".format(has_smoke,counter,(has_smoke/counter)*100))
        #print("[{}/{}] [{:.2f}%]".format(has_smoke,counter,(has_smoke/counter)*100))

        img = np.array(orig_image)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #NEmo: apply nms to discard highly overlapping bboxes
        #probask = probas[keep]
        #probask = probas

        if args.nmsup:
            #import pdb; pdb.set_trace();
            k=torch.max(probask,1)
            scores = k.values
            keepnms = nms(boxes=bboxes_scaled, scores=scores, iou_threshold=args.iou_thresh)
            bboxes_scaled = bboxes_scaled[keepnms]
            probask = probask[keepnms]

        for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            #cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
        #if args.disp_attn:
            #plot_attn(h,w,bboxes_scaled,probask, img,dec_attn_weights, keepnms, figsave_path_attn) # probas[keep] --> probask

        if args.disp:
            plot_results(img, probask, bboxes_scaled,figsave_path) # probas[keep] --> probask
            

        # img_save_path = os.path.join(output_path, filename)
        # cv2.imwrite(img_save_path, img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow("img", img)
        #cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        if v!=1:
            print("Processing...{} ({:.3f}s)".format(filename, infer_time))
        #print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))
    print("*DONE* smoke has been detected in {}/{} images".format(has_smoke,len(images_path)))
    test_filename="test-{}-{}.txt".format(loaded_images,args.thresh)
    infer_data = [has_smoke,cepoch,loaded_model,np.round(avg_duration,4)]


    #import pdb;pdb.set_trace()
    if output_dir:
        (output_dir / 'infer_logs').mkdir(exist_ok=True)
        with (output_dir / "infer_logs" / test_filename).open("a") as f:
            writer = csv.writer(f)
            #f.write(json.dumps(testLog) + "\n")
            #writer.writerow(infer_header)
            writer.writerow(infer_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.resume and not args.detr_base:
        checkpoint = torch.load(args.resume, map_location='cpu')
        cargs = checkpoint['args']
        cepoch = checkpoint['epoch']
        args.num_cl, args.num_queries, args.batch_size = cargs.num_cl, cargs.num_queries, cargs.batch_size

    model, _, postprocessors = build_model(args)
    if args.resume and not args.detr_base:
        model.load_state_dict(checkpoint['model'])
    if args.output_dir: # only needed if resuming from a barebone checkpoint 
        output_dir = Path(args.output_dir)
    else:
        output_dir = os.path.dirname(args.resume)
        output_dir = Path(output_dir)

    if output_dir and not args.detr_base: # This is where to save the test logs which is same as the experiment output_dir
        with (output_dir / "testargs.txt").open("w+") as fl:
            fl.write(str(cargs).replace(" ","\n"))
    if args.resume and args.detr_base:
        detrdict = torch.load(args.resume)
        model = detrdict["model"]

    model.to(device)
    image_paths = get_images(args.data_path)

    infer(image_paths, model, postprocessors, device, output_dir)
