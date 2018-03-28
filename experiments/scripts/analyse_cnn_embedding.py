
import _init_paths

import os.path as osp
import os
import numpy as np
from sacred import Experiment
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import tensorboardX as tb

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
from tracker.alex import alex
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper

ex = Experiment()
ex.add_config('output/tracker/pretrain_cnn/alex13/sacred_config.yaml')
weights = 'output/tracker/pretrain_cnn/alex13/alex_iter_4540.pth'

@ex.automain
def my_main(_config, cnn):
    print(_config)

    #tb_dir = osp.join(get_tb_dir(cnn['module_name']), cnn['name'])
    tb_dir = osp.join(get_tb_dir(cnn['module_name']), cnn['name'])
    

    if not osp.exists(tb_dir):
        os.makedirs(tb_dir)

    writer = tb.SummaryWriter(tb_dir)

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")
    
    network = alex(pretrained=True)
    network.load_state_dict(torch.load(weights))
    network.train()
    network.cuda()
    

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")

    dataloader = {'P':18, 'K':4, 'vis_threshold':0.5, 'max_per_person':40, 'crop_H':224, 'crop_W':112}
    db_train = MOT_Siamese_Wrapper('train', dataloader)
    train_data = db_train._dataloader.data

    ind = 0
    meta = []
    for d in train_data:
        meta += [ind]*len(d)
        ind += 1

    images = [torch.from_numpy(d).cuda() for d in train_data]

    embeddings = torch.cat([network.test_images(im).data for im in images],0)

    #embeddings = network.test_images(images).data

    #emb1 = embeddings[:1,:]
    #im1 = images[:1,:].view(1,-1)
    #for i in range(200):
    #    cos = F.cosine_similarity(emb1, embeddings[i+1:i+2,:])
    #    #im_cos = F.cosine_similarity(im1, images[i+1:i+2,:].view(1,-1))
    #    im_cos = 0
    #    print("im_cos: {}, cos: {}".format(im_cos, cos))


    thumb = []
    for im in torch.cat(images,0).cpu().numpy():
        im += frcnn_cfg.PIXEL_MEANS
        thumb.append(torch.from_numpy(cv2.resize(im, (16, 32), interpolation=cv2.INTER_LINEAR)))
    
    thumb = torch.stack(thumb, 0)
    # permute from NxHxWxC to NxCxHxW and divide by 255
    thumb = thumb.permute(0,3,1,2) / 255
    # change form BGR to RGB
    thumb[:,:,:,:] = thumb[:,[2,1,0],:,:]
    
    writer.add_embedding(embeddings, label_img=thumb, tag='0', global_step=4540, metadata=meta)

