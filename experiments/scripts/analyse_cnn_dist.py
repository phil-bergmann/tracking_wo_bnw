import _init_paths

import os.path as osp
import os
import numpy as np
from sacred import Experiment
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from PIL import Image

import matplotlib.pyplot as plt

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
#from tracker.alex import alex
from tracker.resnet import resnet50
from tracker.datasets.factory import Datasets
from tracker.triplet_loss import _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask

from torchvision.transforms import CenterCrop, Normalize, ToTensor, Compose, Resize, ToPILImage
from torch.autograd import Variable

ex = Experiment()
ex.add_config('output/tracker/pretrain_cnn/res50-bh4-all/sacred_config.yaml')
weights = 'output/tracker/pretrain_cnn/res50-bh4-all/ResNet_iter_25245.pth'
#ex.add_config('output/tracker/pretrain_cnn/bh4-smallTrain/sacred_config.yaml')
#weights = 'output/tracker/pretrain_cnn/bh4-smallTrain/ResNet_iter_25254.pth'
#ex.add_config('output/tracker/pretrain_cnn/marcuhmot_small/sacred_config.yaml')
#weights = 'output/tracker/pretrain_cnn/marcuhmot_small/ResNet_iter_26496.pth'
#ex.add_config('output/tracker/pretrain_cnn/marcuhmot_small/sacred_config.yaml')
#weights = 'output/tracker/pretrain_cnn/marcuhmot/ResNet_iter_27200.pth'
#ex.add_config('output/tracker/pretrain_cnn/kitti_bh_Car_1_2/sacred_config.yaml')
#weights =  'output/tracker/pretrain_cnn/kitti_bh_Car_1_2/ResNet_iters_25065.pth'
#ex.add_config('output/tracker/pretrain_cnn/kitti_small_bh_Car_1_2/sacred_config.yaml')
#weights = 'output/tracker/pretrain_cnn/kitti_small_bh_Car_1_2/ResNet_iters_24624.pth'

def build_crop(im_path, gt):
    im = cv2.imread(im_path)
    height, width, channels = im.shape
    
    gt[0] = np.clip(gt[0], 0, width-1)
    gt[1] = np.clip(gt[1], 0, height-1)
    gt[2] = np.clip(gt[2], 0, width-1)
    gt[3] = np.clip(gt[3], 0, height-1)

    im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

    im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_LINEAR)

    return im

def build_samples(data):
    """Builds the samples out of the sequence."""

    tracks = {}

    for t, sample in enumerate(data):
        im_path = sample['im_path']
        gt = sample['gt']

        for k,v in tracks.items():
            if k in gt.keys():
                v.append({'t':t, 'id':k, 'im_path':im_path, 'gt':gt[k]})
                del gt[k]

        # For all remaining BB in gt new tracks are created
        for k,v in gt.items():
            tracks[k] = [{'t':t, 'id':k, 'im_path':im_path, 'gt':v}]

    # sample max_per_person images and filter out tracks smaller than 4 samples
    #outdir = get_output_dir("siamese_test")
    res = []
    for k,v in tracks.items():
        l = len(v)
        if l >= 2:
            pers = []
            for i in range(l):
                pers.append([v[i]['t'], build_crop(v[i]['im_path'], v[i]['gt'])])

            res.append(pers)
    return res

@ex.automain
def my_main(_config, cnn):
    print(_config)

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")
    
    network = resnet50(pretrained=True, **cnn['cnn'])
    network.load_state_dict(torch.load(weights))
    network.eval()
    network.cuda()
    

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")

    output_dir = osp.join(get_output_dir('MOT_analysis'), 'siamese_dist')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for db in Datasets("mot_train_", {'vis_threshold':0.5}):
        print("[*] Evaluating {}".format(db))
        data = db.data
        data = build_samples(data)

        results_seq = []

        for person in data:

            images = []
            times = []

            transformation = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            for sample in person:
                im = cv2.cvtColor(sample[1], cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im)
                im = transformation(im)
                images.append(im)
                times.append(sample[0])
            images = torch.stack(images, 0)

            embeddings = network(Variable(images.cuda(), volatile=True)).data.cpu()

            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)

            x = embeddings.unsqueeze(1).expand(n, m, d)
            y = embeddings.unsqueeze(0).expand(n, m, d)

            dist = torch.sqrt(torch.pow(x - y, 2).sum(2))

            res = []

            for i in range(n):
                for j in range(n):
                    if i < j:
                        res_x = times[j] - times[i]
                        res_y = dist[i,j]
                        if res_x <= 100:
                            res.append([res_x, res_y])
            results_seq += res

        results += results_seq
        #r = np.array(results_seq)

    # build values for plot
    r = np.array(results)
    x_max = 100
    x_val = np.arange(1,x_max+1)
    y_val = np.zeros(x_max)
    y_std = np.zeros(x_max)

    for x in x_val:
        vals = r[r[:,0] == x, 1]
        mean = np.mean(vals)
        y_val[x-1] = mean
        y_std[x-1] = np.sqrt(np.mean((vals-mean)**2))
    #plt.scatter(x_val, y_val, s=1**2)
    plt.errorbar(x_val, y_val, yerr=y_std, fmt='o')
    plt.xlabel('frames distance')
    plt.ylabel('feature distance')
    plt.xlim((0, 100))

    # calculate variance
    #var_step = 10
    #x_var = np.arange(var_step/2, x_max, 10)
    #y_var = np.zeros(x_max//var_step)
    #for x in x_var:
    #    vals = r[(r[:,0] > x-var_step/2) * (r[:,0] <= x+var_step/2), 1]


    #    y_val[x-1] = y

    #plt.errorbar(x, y, yerr=yerr, fmt='o')
    #plt.ylim((0,10))
    #plt.savefig(osp.join(output_dir, "{}-{}.pdf".format(t, detections)), format='pdf')
    #plt.close()

    #plt.legend()
    plt.savefig(osp.join(output_dir, "dist_err.pdf"), format='pdf')
    plt.close()