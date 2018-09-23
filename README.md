Sequential Tracking


FRCNN: clipping boxes or not after bb regression has no effect (why??)


how many epochs?
Overfitting:
    train with 1/2 of samples, skip 1/4 and valid with 1/4 per sequence

change bounding box handling to add padding


Devkit:
https://bitbucket.org/amilan/motchallenge-devkit
Put into data/motchallenge-devkit
Build with compile.m


train frcnn:
python experiments/scripts/train_faster_rcnn.py with vgg16 mot tag='test'

pretrained weights should be in data/imagenet_weights/{network_name}.pth
e.g.: data/imagenet_weights/vgg16.pth

test frcnn:
python experiments/scripts/test_faster_rcnn.py with {path_to_sacred_config}


Requirements:
install pyfrcnn
install coco into pyfrcnn/data not sequential_tracking/data
tensorboardX
sacred
pyyaml
tensorboard+tensorflow


Normalize not whole image but crops for person reid!
wrong: do not de normalize crops directly to print, they change the image.
Copy crops and then denormalize for printing!
