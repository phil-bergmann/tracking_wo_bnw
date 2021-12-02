#!/usr/bin/env python
#SBATCH --job-name=tracktor
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=50GB
#SBATCH --output=output/%j.out
#SBATCH --time=4320
#SBATCH --exclude=node2
#SBATCH --gres=gpu:1

import argparse
import os
import os.path as osp
import random
import sys
import copy

import matplotlib.pyplot as plt
import numpy as np
import tensorboardX as tb
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('src/obj_det'))

import transforms as T
import utils
from engine import evaluate, train_one_epoch
from mot_data import MOTObjDetect
import models

# torchvision.models.detection.__dict__['retinanet_resnet50_fpn'] = retinanet_resnet50_fpn
# torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'] = fasterrcnn_resnet50_fpn


seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.set_deterministic(True)

parser = argparse.ArgumentParser()

parser.add_argument('--split', required=True)
parser.add_argument('--train_mot_dir', required=True)
parser.add_argument('--test_mot_dir', required=True)
parser.add_argument('--test_mot_set', default='train')
parser.add_argument('--data_root', default='data')
parser.add_argument('--name', required=True)
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--eval_train', action='store_true')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_drop', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--train_vis_threshold', type=float, default=0.25)
parser.add_argument('--test_vis_threshold', type=float, default=0.0)
parser.add_argument('--no_coco_pretraining', action='store_true')
# parser.add_argument('--write_result_files', action='store_true')
parser.add_argument('--arch', type=str, default='fasterrcnn_resnet50_fpn')
parser.add_argument('--resume_model_path', type=str, default=None)
parser.add_argument('--trainable_backbone_layers', type=int, default=3)
parser.add_argument('--nms_thresh', type=float, default=0.5)
parser.add_argument('--score_thresh', type=float, default=0.05)
parser.add_argument('--save_model_interval', type=int, default=0)

parser.add_argument('--frame_range_test_start', type=float, default=0.0)
parser.add_argument('--frame_range_test_end', type=float, default=1.0)
parser.add_argument('--frame_range_train_start', type=float, default=0.0)
parser.add_argument('--frame_range_train_end', type=float, default=1.0)
parser.add_argument('--eval_interval', type=int, default=1)

args = parser.parse_args()

start_epoch = 1
output_dir = f"output/obj_det/{args.name}"
tb_dir = f"tensorboard/obj_det/{args.name}"

if not osp.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

train_data_dir = osp.join(args.data_root, args.train_mot_dir, 'train')
test_data_dir = osp.join(args.data_root, args.test_mot_dir, args.test_mot_set)

if '3_fold' in args.split:
    if '1' == args.split[-1]:
        train_split_seqs = ['MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-11']
        test_split_seqs = ['MOT17-02', 'MOT17-10', 'MOT17-13']
    elif '2' == args.split[-1]:
        train_split_seqs = ['MOT17-02', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-13']
        test_split_seqs = ['MOT17-04', 'MOT17-11']
    elif '3' == args.split[-1]:
        train_split_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_split_seqs = ['MOT17-05', 'MOT17-09']
    else:
        raise NotImplementedError
else:
    # train_split_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    train_split_seqs = ['MOT17-02', 'MOT17-05', 'MOT17-09', 'MOT17-11']
    # train_split_seqs = [
    #     s for s in os.listdir(train_data_dir)
    #     if 'MOT17' in s and os.path.isdir(os.path.join(train_data_dir, s))]

if 'MOT20' in train_data_dir:
    train_split_seqs = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
if 'MOT20' in test_data_dir:
    test_split_seqs = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

# dataset = MOTObjDetect(test_data_dir, split_seqs=test_split_seqs)
# dataset = MOTObjDetect(train_data_dir, split_seqs=train_split_seqs)
# img, target = dataset[247]
# img, target = T.ToTensor()(img, target)
# plot(img, target['boxes'])
# exit()

#
# DATASETS
#

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# use our dataset and defined transformations
dataset = MOTObjDetect(
    train_data_dir,
    get_transform(train=True),
    split_seqs=train_split_seqs,
    vis_threshold=args.train_vis_threshold,
    frame_range_start=args.frame_range_train_start,
    frame_range_end=args.frame_range_train_end)
dataset_no_random = MOTObjDetect(
    train_data_dir,
    get_transform(train=False),
    split_seqs=train_split_seqs,
    vis_threshold=args.train_vis_threshold,
    frame_range_start=args.frame_range_train_start,
    frame_range_end=args.frame_range_train_end)

dataset_test = MOTObjDetect(
    test_data_dir,
    get_transform(train=False),
    split_seqs=test_split_seqs,
    vis_threshold=args.test_vis_threshold,
    frame_range_start=args.frame_range_test_start,
    frame_range_end=args.frame_range_test_end)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_no_random = torch.utils.data.DataLoader(
    dataset_no_random, batch_size=args.batch_size, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# INIT MODEL AND OPTIM

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"DEVICE: {device}")

def get_detection_model(num_classes, arch):
    # load an instance segmentation model pre-trained on COCO
    model_func = getattr(models, arch)

    kwargs = {}
    if 'rcnn' in arch:
        kwargs['box_nms_thresh'] = args.nms_thresh
        kwargs['box_score_thresh'] = args.score_thresh
    elif 'retinanet' in arch:
        kwargs['nms_thresh'] = args.nms_thresh
        kwargs['score_thresh'] = args.score_thresh
    else:
        raise NotImplementedError

    model = model_func(
        num_classes=num_classes,
        pretrained=not args.no_coco_pretraining,
        trainable_backbone_layers=args.trainable_backbone_layers,
        **kwargs)

    return model

# get the model using our helper function
model = get_detection_model(dataset.num_classes, args.arch)

# move model to the right device
model.to(device)

if args.resume_model_path is not None:
    print(f"LOAD MODEL: {args.resume_model_path}")
    model_state_dict = torch.load(args.resume_model_path)
    if 'model' in model_state_dict:
        model_state_dict = model_state_dict['model']
    model.load_state_dict(model_state_dict)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[args.lr_drop], gamma=0.1)


# TRAINING

def evaluate_and_write_result_files(model, data_loader):
    print(f'EVAL {data_loader.dataset}.')

    model.eval()

    # COCO eval
    iou_types = ["bbox"]
    # if data_loader.dataset.has_masks:
    # if args.arch == 'maskrcnn_resnet50_fpn':
    #     iou_types.append("segm")
    # if args.arch == 'keypointrcnn_resnet50_fpn':
    #     iou_types.append("keypoints")
    coco_eval, results, loss_dicts = evaluate(model, data_loader, device, iou_types)
    evaluation_metrics = {'AP': coco_eval.coco_eval['bbox'].stats[0]}

    # MOT17Det eval
    # if args.write_result_files:
    # filelist = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
    # for f in filelist:
    #     os.remove(os.path.join(output_dir, f))

    data_loader.dataset.write_results_files(results, output_dir)

    # if 'MOT17' in args.test_mot_dir and args.test_mot_set == 'train':
    #     evaluator = DET_evaluator()
    #     overall_results, _ = evaluator.run(
    #         benchmark_name='MOT17Det',
    #         gt_dir=osp.join(args.data_root, args.test_mot_dir),
    #         res_dir=output_dir,
    #         eval_mode='train',
    #         seqmaps_dir='src/MOTChallengeEvalKit/seqmaps')

    #     evaluation_metrics['AP_MOT17Det'] = 0.0
    #     evaluation_metrics['MODA_MOT17Det'] = 0.0
    #     if overall_results is not None:
    #         evaluation_metrics['AP_MOT17Det'] = overall_results.AP
    #         evaluation_metrics['MODA_MOT17Det'] = overall_results.MODA

    return evaluation_metrics, loss_dicts



tb_writer = tb.SummaryWriter(tb_dir)

if args.eval_interval > 0:
    evaluation_metrics, loss_dicts = evaluate_and_write_result_files(model, data_loader_test)

    for metric, metric_value in evaluation_metrics.items():
        tb_writer.add_scalar(f'VAL/{metric}', metric_value, 0)
    for loss_key in loss_dicts[0].keys():
        loss = torch.tensor([loss_dict[loss_key] for loss_dict in loss_dicts])
        tb_writer.add_scalar(f'VAL/{loss_key}', loss.mean(), 0)

    best_eval_metrics = copy.deepcopy(evaluation_metrics)
    for metric, metric_value in evaluation_metrics.items():
        best_eval_metrics[metric] = metric_value

        print(f'Save best {metric} ({metric_value:.2f}) model at epoch: {0}')
        torch.save({'model': model.state_dict()}, osp.join(output_dir, f"best_{metric}.model"))

if args.only_eval:
    exit()

for epoch in range(start_epoch, args.num_epochs + 1):
    print(f'TRAIN {data_loader.dataset}')
    loss_dicts = train_one_epoch(
        model, optimizer, data_loader, device, epoch, print_freq=50)

    # update the learning rate
    lr_scheduler.step()
    tb_writer.add_scalar('TRAIN/LR', lr_scheduler.get_last_lr(), epoch)
    for loss_key in loss_dicts[0].keys():
        loss = torch.tensor([loss_dict[loss_key] for loss_dict in loss_dicts])
        tb_writer.add_scalar(f'TRAIN/{loss_key}', loss.mean(), epoch)

    # evaluate
    if args.eval_interval > 0 and (epoch % args.eval_interval == 0 or epoch in [1, args.num_epochs]):
        if args.eval_train:
            evaluation_metrics = evaluate_and_write_result_files(model, data_loader_no_random)
            for metric, metric_value in evaluation_metrics.items():
                tb_writer.add_scalar(f'TRAIN/{metric}', metric_value, epoch)

        evaluation_metrics, loss_dicts = evaluate_and_write_result_files(model, data_loader_test)

        for metric, metric_value in evaluation_metrics.items():
            tb_writer.add_scalar(f'VAL/{metric}', metric_value, epoch)
        for loss_key in loss_dicts[0].keys():
            loss = torch.tensor([loss_dict[loss_key] for loss_dict in loss_dicts])
            tb_writer.add_scalar(f'VAL/{loss_key}', loss.mean(), epoch)

        for metric, metric_value in evaluation_metrics.items():
            if metric_value > best_eval_metrics[metric]:
                best_eval_metrics[metric] = metric_value

                print(f'Save best {metric} ({metric_value:.2f}) model at epoch: {epoch}')
                torch.save({'model': model.state_dict()}, osp.join(output_dir, f"best_{metric}.model"))

    if args.save_model_interval > 0 and epoch % args.save_model_interval == 0:
        torch.save({'model': model.state_dict()}, osp.join(output_dir, f"epoch_{epoch}.model"))
    torch.save({'model': model.state_dict()}, osp.join(output_dir, "last.model"))
