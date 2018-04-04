
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.utils.model_zoo as model_zoo

import numpy as np
import random
import cv2

from .triplet_loss import batch_hard_triplet_loss, batch_all_triplet_loss, _get_triplet_mask

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class Alex(models.AlexNet):
    def __init__(self, output_dim=1000):
        super(Alex, self).__init__(output_dim)
        self.name = "alex"
        self.output_dim = output_dim
        # remove last max pool layer
        #self.features = nn.Sequential(*list(self.features._modules.values())[:-1])
        # resize classifier first layer for smaller input
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        #assert x.size(0) == 1, "[!] Only supports one image at a time"
        x = self.features(x)
        # input 224x112 => 256 * 6 * 2
        x = x.view(x.size(0), 256 * 6 * 2)
        x = self.classifier(x)
        return x

    def test_images(self, images):
        images = self.prepare_images(images)
        images = Variable(images).cuda()

        return self.forward(images)

    def prepare_images(self, images):
        """Prepares image for this model."""
        # permute from NxHxWxC to NxCxHxW and divide by 255
        images = images.permute(0,3,1,2) / 255
        # change form BGR to RGB
        images[:,:,:,:] = images[:,[2,1,0],:,:]
        # probably mean and std should be changed, but mean is nearly the same
        return images

    def test_rois(self, image, rois):
        """Tests the rois on a particular image. Should be inside image."""
        x = self.build_crops(image, rois).cuda()
        #print(x.size())
        x = Variable(self.prepare_images(x))
        
        return self.forward(x)

    def build_crops(self, image, rois):
        np_image = image.cpu().numpy()
        res = []
        for r in rois:
            x0 = int(r[0])
            y0 = int(r[1])
            x1 = int(r[2])
            y1 = int(r[3])
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1

            im = np_image[0,y0:y1, x0:x1]
            im = cv2.resize(im, (112, 224), interpolation=cv2.INTER_LINEAR)
            res.append(torch.from_numpy(im))
        res = torch.stack(res, 0)
        if image.is_cuda:
            res = res.cuda()
        return res

    def _crop_pool_layer(self, bottom, rois, max_pool=True):
        # from pytorch-faster-rcnn
        # implement it using stn
        # box to affine
        # input (x1,y1,x2,y2)
        """
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1      ]
        """
        rois = rois.detach()

        x1 = rois[:, 0::4] / 17.2307692308
        y1 = rois[:, 1::4] / 17.2307692308
        x2 = rois[:, 2::4] / 17.2307692308
        y2 = rois[:, 3::4] / 17.2307692308

        height = bottom.size(2)
        width = bottom.size(3)

        # affine theta
        theta = Variable(rois.data.new(rois.size(0), 2, 3).zero_())
        theta[:, 0, 0] = ((x2 - x1) / (width - 1)).view(-1)
        theta[:, 0 ,2] = ((x1 + x2 - width + 1) / (width - 1)).view(-1)
        theta[:, 1, 1] = ((y2 - y1) / (height - 1)).view(-1)
        theta[:, 1, 2] = ((y1 + y2 - height + 1) / (height - 1)).view(-1)

        POOLING_SIZE = 6 #alexnet parameters
        #POOLING_SIZE = 7 ORIGINAL

        pre_pool_size = 13 if max_pool else POOLING_SIZE
        #pre_pool_size = POOLING_SIZE*2 if max_pool else POOLING_SIZE ORIGINAL
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
        torch.backends.cudnn.enabled = False
        crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
        torch.backends.cudnn.enabled = True
        if max_pool:
            crops = F.max_pool2d(crops, 3, 2)
            #crops = F.max_pool2d(crops, 2, 2) ORIGINAL
        return crops

    def load_pretrained_cnn(self, state_dict):
        pretrained_state_dict = {k: v for k,v in state_dict.items() for kk,vv in self.state_dict().items() if k==kk and v.size() == vv.size()}
        updated_state_dict = self.state_dict()
        updated_state_dict.update(pretrained_state_dict)
        self.load_state_dict(updated_state_dict)

    def sum_losses(self, batch, loss, margin):
        """For Pretraining

        Function for preatrainind this CNN with the triplet loss. Takes a sample of N=PK images, P different
        persons, K images of each. K=4 is a normal parameter

        Args:
            batch (list): [images, labels], images are Tensor of size (N,H,W,C), H=224, W=112, labels Tensor of
            size (N)
        """

        inp = self.prepare_images(batch[0][0])
        inp = Variable(inp).cuda()

        labels = batch[1][0]
        labels = labels.cuda()

        embeddings = self.forward(inp)

        if loss == 'hard':
            # not functional, converges to margin because it makes all output vectors the same
            triplet_loss = batch_hard_triplet_loss(labels, embeddings, margin)
        elif loss == 'all':
            # not functional error explodes
            triplet_loss, fraction_positive_triplets = batch_all_triplet_loss(labels, embeddings, margin)
            print("Frac-Pos: {}".format(fraction_positive_triplets.data[0]))
        elif loss == 'triplet':
            # works, batch all strategy
            m = _get_triplet_mask(labels).nonzero()
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)
            triplet_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=1)
        elif loss == 'cosine':
            # suboptimal but seems to work
            x1 = embeddings[:-4]
            x2 = embeddings[2:-2]
            l = np.array(17*[1,1,-1,-1])
            l = Variable(torch.from_numpy(l)).cuda()
            triplet_loss = F.cosine_embedding_loss(x1,x2,l,margin=margin)
        else:
            raise NotImplementedError("Loss: {}".format(loss))

        losses = {}
        losses['total_loss'] = triplet_loss

        return losses

def alex(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Alex(**kwargs)
    if pretrained:
        model.load_pretrained_cnn(model_zoo.load_url(model_urls['alexnet']))
    return model