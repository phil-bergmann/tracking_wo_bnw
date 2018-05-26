
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
from tracker.resnet_ce import resnet50
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper
from tracker.mot_siamese import MOT_Siamese
from tracker.triplet_loss import _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask

from torchvision.transforms import CenterCrop, Normalize, ToTensor, Compose, Resize, ToPILImage
from torch.autograd import Variable

ex = Experiment()
ex.add_config('output/tracker/pretrain_cnn/res50-ce3-small_train/sacred_config.yaml')
weights = 'output/tracker/pretrain_cnn/res50-ce3-small_train/ResNet_iter_14763.pth'
pth = 'output/tracker/cnn_demo/'

train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = train
#sequences = train[3:4]

def plotImages(network, data):
    # calculate labels
    ind = 0
    meta = []
    for d in data:
        meta += [ind]*len(d)
        ind += 1
    labels = torch.LongTensor(meta)

    # images have to be center cropped to right size from (288, 144) to (256, 128)
    images = []
    plot_images = []
    transformation = Compose([CenterCrop((256, 128)), ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for d in data:
        tens = []
        for im in d:
            plot_images.append(im)
            imm = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imm = Image.fromarray(imm)
            imm = transformation(imm)
            tens.append(imm)
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

    pos_distances = dist * pos_mask.float()
    neg_distances = dist * neg_mask.float()

    
    # select a anchor
    anchors = [i for i in range(dist.size(0))]
    anchors = np.random.choice([i for i in anchors], 30, replace=False)

    """
    for a in anchors:
        d = dist[a]

        pos_ind = pos_mask[a].nonzero()[:,0]
        pos_ind = np.random.choice([i for i in pos_ind], 3, replace=False)
        neg_ind = neg_mask[a].nonzero()[:,0]
        neg_ind = np.random.choice([i for i in neg_ind], 4, replace=False)

        im_path = os.path.join(pth, str(a)+".jpg")
        fig, ax = plt.subplots(2,4,figsize=(48, 48))

        for i in range(8):
            if i == 0:
                im = plot_images[a]
                sc = 0
            elif i < 4:
                im = plot_images[pos_ind[i-1]]
                sc = d[pos_ind[i-1]]
            else:
                im = plot_images[neg_ind[i-4]]
                sc = d[neg_ind[i-4]]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            ax[i//4,i%4].imshow(np.asarray(im), aspect='equal')
            ax[i//4,i%4].set_title("Euclidean Distance: {}".format(sc), size='x-large')
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(im_path)
        plt.close()
    """

    # TODO plot K nearest targets
    _, indices = torch.sort(dist, dim=1)
    for a in anchors:
        ind = indices[a]
        d = dist[a]

        im_path = os.path.join(pth, str(a)+".jpg")
        fig, ax = plt.subplots(2,5,figsize=(48, 48))

        for i in range(10):
            if i == 0:
                im = plot_images[a]
                sc = 0
            else:
                im = plot_images[ind[i]]
                sc = d[ind[i]]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            ax[i//5,i%5].imshow(np.asarray(im), aspect='equal')
            ax[i//5,i%5].set_title("Euclidean Distance: {}".format(sc), size='x-large')

        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(im_path)
        plt.close()


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

    if not osp.exists(pth):
        os.makedirs(pth)
    

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")

    dataloader = {'P':18, 'K':4, 'vis_threshold':0.1, 'max_per_person':8, 'crop_H':256, 'crop_W':128,
                    'transform': 'center', 'split':'small_val'}
    for s in sequences:
        if s == "train":
            db_train = MOT_Siamese_Wrapper('train', dataloader)
            data = db_train._dataloader.data
            print("[*] Evaluating whole train set...")
            plotImages(network, data)
        else:
            db_train = MOT_Siamese(s, **dataloader)
            data = db_train.data
            print("[*] Evaluating {}...".format(s))
            plotImages(network, data)