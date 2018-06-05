
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

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
#from tracker.alex import alex
from tracker.resnet import resnet50
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper
from tracker.mot_siamese import MOT_Siamese
from tracker.triplet_loss import _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask

from torchvision.transforms import CenterCrop, Normalize, ToTensor, Compose, Resize, ToPILImage
from torch.autograd import Variable

ex = Experiment()
#ex.add_config('output/tracker/pretrain_cnn/res50-bh4-all/sacred_config.yaml')
#weights = 'output/tracker/pretrain_cnn/res50-bh4-all/ResNet_iter_25245.pth'
ex.add_config('output/tracker/pretrain_cnn/res50-bh3-small_train/sacred_config.yaml')
weights = 'output/tracker/pretrain_cnn/res50-bh3-small_train/ResNet_iter_25137.pth'
#thresholds = [6.0, 6.5, 7.0, 7.5, 8.0]
train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = ["train"]
#sequences = ["MOT17-09"]

def calcScores(network, data):
    # calculate labels
    ind = 0
    meta = []
    for d in data:
        meta += [ind]*len(d)
        ind += 1
    labels = torch.LongTensor(meta)

    # images have to be center cropped to right size from (288, 144) to (256, 128)
    images = []
    transformation = Compose([CenterCrop((256, 128)), ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for d in data:
        tens = []
        for im in d:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            im = transformation(im)
            tens.append(im)
        images.append(torch.stack(tens, 0))

    embeddings = torch.cat([network(Variable(im.cuda(), volatile=True)).data for im in images],0).cpu()

    pos_mask = _get_anchor_positive_triplet_mask(labels)
    neg_mask = _get_anchor_negative_triplet_mask(labels)

    # compute pariwise square distance matrix
    n = embeddings.size(0)
    m = embeddings.size(0)
    d = embeddings.size(1)

    x = embeddings.unsqueeze(1).expand(n, m, d)
    y = embeddings.unsqueeze(0).expand(n, m, d)

    dist = torch.sqrt(torch.pow(x - y, 2).sum(2))

    dist_diag = dist + torch.eye(dist.size(0)) * 10000

    pos = 0
    ges = dist_diag.size(0)
    pos_float_mask = pos_mask.float()
    for i in range(dist_diag.size(0)):
        value, index = dist_diag[i].min(0)
        pos += pos_float_mask[i,index[0]]
    print("Result top 1 acc: {}".format(pos/ges))
    print(pos)
    print(ges)

    """
    pos_distances = dist * pos_mask.float()
    neg_distances = dist * neg_mask.float()
    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()
    # calculate the right classifications
    for t in thresholds:
        # every 0 entry is also le t so filter with mask here
        pos_right = torch.le(pos_distances, t) * pos_mask
        pos_right = pos_right.sum()
        neg_right = torch.gt(neg_distances, t).sum()
        
        print("[*] Threshold set to: {}".format(t))
        print("Positive right classifications: {:.2f}% {}/{}".format(pos_right/num_pos*100, pos_right, num_pos))
        print("Negative right classifications: {:.2f}% {}/{}".format(neg_right/num_neg*100, neg_right, num_neg))
        print("All right classifications: {:.2f}% {}/{}".format((pos_right+neg_right)/(num_pos+num_neg)*100,
                                                                pos_right+neg_right, num_pos+num_neg))
    """

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

    dataloader = {'P':18, 'K':4, 'vis_threshold':0.3, 'max_per_person':40, 'crop_H':256, 'crop_W':128,
                    'transform': 'center', 'split':'small_val'}
    for s in sequences:
        if s == "train":
            db_train = MOT_Siamese_Wrapper('train', dataloader)
            data = db_train._dataloader.data
            print("[*] Evaluating whole train set...")
            calcScores(network, data)
        else:
            db_train = MOT_Siamese(s, **dataloader)
            data = db_train.data
            print("[*] Evaluating {}...".format(s))
            calcScores(network, data)