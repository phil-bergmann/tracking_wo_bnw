
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
from PIL import Image

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
from tracker.alex import alex
from tracker.resnet import resnet50
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper

from torchvision.transforms import CenterCrop, Normalize, ToTensor, Compose, Resize, ToPILImage
from torch.autograd import Variable

ex = Experiment()
ex.add_config('output/tracker/pretrain_cnn/res50-wt-small_train/sacred_config.yaml')
weights = 'output/tracker/pretrain_cnn/res50-wt-small_train/ResNet_iter_25254.pth'

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
    
    network = resnet50(pretrained=True, **cnn['cnn'])
    network.load_state_dict(torch.load(weights))
    network.eval()
    network.cuda()
    

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")

    dataloader = {'P':18, 'K':4, 'vis_threshold':0.5, 'max_per_person':40, 'crop_H':256, 'crop_W':128,
                    'transform': 'center', 'split':'small_val'}
    db_train = MOT_Siamese_Wrapper('train', dataloader)
    train_data = db_train._dataloader.data

    ind = 0
    meta = []
    for d in train_data:
        meta += [ind]*len(d)
        ind += 1

    # images have to be center cropped to right size from (288, 144) to (256, 128)
    images = []
    transformation = Compose([CenterCrop((256, 128)), ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for d in train_data:
        tens = []
        for im in d:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            im = transformation(im)
            tens.append(im)
        images.append(torch.stack(tens, 0))

    embeddings = torch.cat([network(Variable(im.cuda(), volatile=True)).data for im in images],0)

    #embeddings = network.test_images(images).data

    #emb1 = embeddings[:1,:]
    #im1 = images[:1,:].view(1,-1)
    #for i in range(200):
    #    cos = F.cosine_similarity(emb1, embeddings[i+1:i+2,:])
    #    #im_cos = F.cosine_similarity(im1, images[i+1:i+2,:].view(1,-1))
    #    im_cos = 0
    #    print("im_cos: {}, cos: {}".format(im_cos, cos))


    #numpy_images = np.concatenate([d[:, 8:136, 16:272, :] for d in train_data],0)
    res = Compose([Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                    Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                    ToPILImage(), Resize((32,16)), ToTensor()])
    thumb = []
    for im in torch.cat(images, 0):
        thumb.append(res(im))
        #thumb.append(torch.from_numpy(cv2.resize(im, (16, 32), interpolation=cv2.INTER_LINEAR)))
    #thumb[0].save('/usr/stud/bergmanp/test.jpg')
    
    thumb = torch.stack(thumb, 0)
    # permute from NxHxWxC to NxCxHxW and divide by 255
    #thumb = thumb.permute(0,3,1,2) / 255
    # change form BGR to RGB
    #thumb[:,:,:,:] = thumb[:,[2,1,0],:,:]
    
    writer.add_embedding(embeddings, label_img=thumb, tag=cnn['name'], global_step=25254, metadata=meta)

