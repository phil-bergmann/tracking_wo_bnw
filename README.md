Sequential Tracking

PR: rename CUDAHOME to CUDA_HOME
PR: add "pip install cffi" to install manual

is_cuda as command line argument
    add switch to FasterRCNN.anchor_target_layer

sacred - experiment framework

Pretrain Faster RCNN on VOC
remove solver.py and demo_CPU
fix gitignore data/*


tensorboard
change bounding box handeling to ignore if it extends the image (for now)
write score function (ratio of TP / (TP+FP).)

make installable (use pip install -e for development)
- test with copied trainval_net.py script (change imports)
- add python imports to setup.py (https://github.com/comp-imaging/ProxImaL/blob/master/setup.py). test import with "pytorch-faster-rcnn" package name
- write setup which compiles nvcc (https://github.com/timmeinhardt/pybm3d_gpu/blob/master/setup.py)
- add ffi build.py files to setup.py. change paths
    