Sequential Tracking

PR: rename CUDAHOME to CUDA_HOME

PR: add "pip install cffi" to install manual

is_cuda as command line argument
    add switch to FasterRCNN.anchor_target_layer

sacred - experiment framework


roi or crop polling mode?!
plot metrics in tensorboard
how is train and val loss averaged?
how many epochs?
Overfitting:
    train with 1/2 of samples, skip 1/4 and valid with 1/4 per sequence
    VGG16
    L1/2 regularization
    train MOTdet with pretrained FRCNN (trained on VOC)

make installable (use pip install -e for development):
- test with copied trainval_net.py script (change imports)
- add python imports to setup.py (https://github.com/comp-imaging/ProxImaL/blob/master/setup.py). test import with "pytorch-faster-rcnn" package name
- write setup which compiles nvcc (https://github.com/timmeinhardt/pybm3d_gpu/blob/master/setup.py)
- add ffi build.py files to setup.py. change paths

change bounding box handling to add padding


Devkit:
https://bitbucket.org/amilan/motchallenge-devkit
Put into data/motchallenge-devkit
Build with compile.m
