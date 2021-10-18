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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('src/obj_det'))

import transforms as T
import utils
from engine import evaluate, train_one_epoch
from mot_data import MOTObjDetect


seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

parser.add_argument('--test_split', required=True)
parser.add_argument('--train_mot_dir', required=True)
parser.add_argument('--test_mot_dir', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--eval_train', action='store_true')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--lr_drop', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--train_vis_threshold', type=float, default=0.25)
parser.add_argument('--test_vis_threshold', type=float, default=0.25)
parser.add_argument('--no_coco_pretraining', action='store_true')
parser.add_argument('--arch', type=str, default='fasterrcnn_resnet50_fpn')


args = parser.parse_args()

start_epoch = 1
lr = 0.001
eval_nth_epoch = 1
output_dir = f"output/obj_det/{args.name}"
tb_dir = f"tensorboard/obj_det/{args.name}"
resume_model_path = None

train_split_seqs = test_split_seqs = None
train_data_dir = osp.join(f'data/{args.train_mot_dir}', 'train')
test_data_dir = osp.join(f'data/{args.test_mot_dir}', 'train')

if '3_fold' in args.test_split:
    if '1' == args.test_split[-1]:
        train_split_seqs = ['MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-11']
        test_split_seqs = ['MOT17-02', 'MOT17-10', 'MOT17-13']
    elif '2' == args.test_split[-1]:
        train_split_seqs = ['MOT17-02', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-13']
        test_split_seqs = ['MOT17-04', 'MOT17-11']
    elif '3' == args.test_split[-1]:
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

    test_split_seqs = [f'MOT17-{args.test_split}']
    for seq in test_split_seqs:
        train_split_seqs.remove(seq)


if not osp.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

tb_writer = tb.SummaryWriter(tb_dir)

# Image.open(osp.join(f'data/{args.train_mot_dir}', 'train/MOT17-02/img1/000001.jpg'))



def plot(img, boxes):
    fig, ax = plt.subplots(1, dpi=96)

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    width, height, _ = img.shape

    ax.imshow(img, cmap='gray')
    fig.set_size_inches(width / 80, height / 80)

    for box in boxes:
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=1.0)
        ax.add_patch(rect)

    plt.axis('off')
    fig.savefig('temp.png', dpi=fig.dpi)
    plt.show()

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
    vis_threshold=args.train_vis_threshold)
dataset_no_random = MOTObjDetect(
    train_data_dir,
    get_transform(train=False),
    split_seqs=train_split_seqs,
    vis_threshold=args.train_vis_threshold)
# dataset_test = MOTObjDetect(
#     osp.join(f'data/{args.train_mot_dir}', 'test'),
#     get_transform(train=False)))
dataset_test = MOTObjDetect(
    test_data_dir,
    get_transform(train=False),
    split_seqs=test_split_seqs,
    vis_threshold=args.test_vis_threshold)

# dataset_test_blur = MOTObjDetect(
#     osp.join('data/MOT17_Anonymous_BodyBWBlur', 'train'),
#     get_transform(train=False),
#     split_seqs=test_split_seqs,)
# dataset_test_cia = MOTObjDetect(
#     osp.join('data/MOT17_Anonymous_BodyCIA', 'train'),
#     get_transform(train=False),
#     split_seqs=test_split_seqs,)

# split the dataset in train and test set
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

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
# data_loader_test_blur = torch.utils.data.DataLoader(
#     dataset_test_blur, batch_size=args.batch_size, shuffle=False, num_workers=4,
#     collate_fn=utils.collate_fn)
# data_loader_test_cia = torch.utils.data.DataLoader(
#     dataset_test_cia, batch_size=args.batch_size, shuffle=False, num_workers=4,
#     collate_fn=utils.collate_fn)


# INIT MODEL AND OPTIM

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_detection_model(num_classes, arch):
    # load an instance segmentation model pre-trained on COCO
    model_func = getattr(torchvision.models.detection, arch)

    kwargs = {}
    if arch == 'fasterrcnn_resnet50_fpn':
        kwargs['box_nms_thresh'] = 0.3
    elif arch == 'retinanet_resnet50_fpn':
        kwargs['nms_thresh'] = 0.3

    model = model_func(
        num_classes=num_classes,
        pretrained=not args.no_coco_pretraining,
        # trainable_backbone_layers=5,
        **kwargs)

    return model

# get the model using our helper function
model = get_detection_model(dataset.num_classes, args.arch)
# move model to the right device
model.to(device)

if resume_model_path is not None:
    model_state_dict = torch.load(resume_model_path)
    model.load_state_dict(model_state_dict)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10 every 10 epochs
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[args.lr_drop], gamma=0.1)


# TRAINING

def evaluate_and_write_result_files(model, data_loader):
    print(f'EVAL {data_loader.dataset}.')

    model.eval()
    results = {}
    for imgs, targets in tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                  'scores': pred['scores'].cpu()}

    data_loader.dataset.write_results_files(results, output_dir)
    evaluation_metrics = data_loader.dataset.print_eval(results)

    return evaluation_metrics

# evaluate_and_write_result_files(model, data_loader_no_random)
if args.only_eval:
    evaluate_and_write_result_files(model, data_loader_test)
    # evaluate_and_write_result_files(model, data_loader_test_blur)
    # evaluate_and_write_result_files(model, data_loader_test_cia)
    exit()

best_AP = 0.0
for epoch in range(start_epoch, args.num_epochs + 1):
    print(f'TRAIN {data_loader.dataset}')
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)

    # update the learning rate
    lr_scheduler.step()
    tb_writer.add_scalar('TRAIN/LR', lr_scheduler.get_last_lr(), epoch)

    # evaluate on the test dataset
    if epoch % eval_nth_epoch == 0 or epoch in [1, args.num_epochs]:
        if args.eval_train:
            evaluation_metrics = evaluate_and_write_result_files(model, data_loader_no_random)
            tb_writer.add_scalar('TRAIN/AP', evaluation_metrics['AP'], epoch)

        evaluation_metrics = evaluate_and_write_result_files(model, data_loader_test)
        # print(evaluation_metrics)
        # evaluate(model, data_loader_test, device)
        # # exit()

        tb_writer.add_scalar('VAL/AP', evaluation_metrics['AP'], epoch)

        if evaluation_metrics['AP'] > best_AP:
            best_AP = evaluation_metrics['AP']

            print(f'Save best AP ({best_AP:.2f}) model at epoch: {epoch}')
            torch.save(model.state_dict(), osp.join(output_dir, "best_AP.model"))

    #   evaluate_and_write_result_files(model, data_loader_test_blur)
    #   evaluate_and_write_result_files(model, data_loader_test_cia)

    torch.save(model.state_dict(), osp.join(output_dir, "last.model"))


# pick one image from the test set
data_loader = torch.utils.data.DataLoader(
    dataset_no_random, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

for imgs, target in data_loader:
    print(dataset._img_paths[0])

    model.eval()
    with torch.no_grad():
        prediction = model([imgs[0].to(device)])[0]

    plot(imgs[0], prediction['boxes'])
    plot(imgs[0], target[0]['boxes'])
    break
