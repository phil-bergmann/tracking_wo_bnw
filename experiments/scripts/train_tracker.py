
import _init_paths

import os.path as osp
import os
import numpy as np
from sacred import Experiment
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import subprocess

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
from tracker.resnet_ce import resnet50
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper
from tracker.mot_siamese import MOT_Siamese
from tracker.triplet_loss import _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask

from torchvision.transforms import CenterCrop, Normalize, ToTensor, Compose, Resize, ToPILImage
from torch.autograd import Variable

ex = Experiment()
GPU = 1
calc = True
offset = 8 + 3*GPU

@ex.automain
def my_main(_config):

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")
    
    network = resnet50(pretrained=True, output_dim=128)
    #network.load_state_dict(torch.load(weights))
    network.eval()
    #network.cuda()
    print("")

    s = "┈▔▔╲╱▔▔▔╲╱▔▔▏\n" \
        "┈┈╲＿╱╰╮┈╭╯╲＿╱\n" \
        "┈┈┈╱▏▉╮┈╭▉▕\n" \
        "┈╱▔╰╲╰╰┊╯╯╱╲\n" \
        "┈▏╰╰▕╰╰┳╯╯▏╯▏\n" \
        "┈▏╰╰╰╲╰┻╯╱╯╯▏\n" \
        "┈▏╰╰╰╰▔▔▔╯╯╯▏\n" \
        "┈╲╰╰╭╮╯╯╯╭╮╱\n" \
        "┈┈┃┳┫┣━┳┳┫┃\n" \
        "┈┈┃┃┃┃┈┃┃┃┃\n" \
        "┈┈┗┛┗┛┈┗┛┗┛\n"
    print(s)
    print("[*] Doing nothing, only block GPU that it does not get stolen again :)")

    captured = False

    data = Variable(torch.rand(1,3,256,128), volatile=True).cuda()

    while True:
        if not captured:
            x = subprocess.check_output(['nvidia-smi'])
            mb = x.decode('ascii').split("\n")[offset].split("|")[2].strip(" ").split("MiB")[0]
            mb = int(mb)
            if mb < 3000:
                network.cuda()
                captured = True
                print("[*] Captured!")
            time.sleep(5)
        elif calc:
            network(data)
        else:
            time.sleep(5)
