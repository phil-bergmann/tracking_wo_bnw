Sequential Tracking

### Installation
1. Clone the Faster R-CNN fork and follow the instructions under "Installation"
  ```Shell
  git clone https://github.com/phil-bergmann/pytorch-faster-rcnn
  ´´´

2. Switch pytorch-faster-rcnn to "dev" branch

3. Clone this repository
  ```Shell
  git clone https://github.com/timmeinhardt/sequential_tracking
  ´´´

4. Make a symbolic link from sequential_tracking/tracker/frcnn pointing to pytorch-faster-rcnn/lib. For example if pytorch-faster-rcnn and sequential_tracking are in the same folder:
  ```Shell
  cd sequential_tracking/src
  ln -s ../../pytorch-faster-rcnn/lib/ frcnn
  ´´´

### Training Faster R-CNN
1. Download MOT17Det dataset and paste the "MOT17Det" folder in sequential_tracking/data/

2. Download pretrained models for VGG16 or Res101 as described in https://github.com/phil-bergmann/pytorch-faster-rcnn under "Train your own model" and paste them inside sequential_tracking/data/imagenet_weights/ and name them "vgg16.pth" or "res101.pth"

3. Train the Faster R-CNN by running
  ```Shell
  python experiments/scripts/train_faster_rcnn.py with {vgg16, res101} {mot, small_mot} tag={%name} description={%description}
  ´´´
  inside sequential_tracking. The "name" and "description" parameters are optional. If no "tag" is provided a timestamp will be used.


### Training the Siamese CNN
For reidentification a siamese CNN has to be trained. Follow these instructions to do so:
1. If not done before download MOT17Det dataset and paste the "MOT17Det" folder in sequential_tracking/data/

2. Modify sequential_tracking/experiments/cfgs/pretrain_cnn.yaml to your needs.

3. From within sequential_tracking run
  ```Shell
  python experiments/pretrain_cnn.py
  ´´´


### Evaluating the Tracker
1. Download the MOT16Labels and MOT16-det-dpm-raw and paste them into sequential_tracking/data/

2. Modify sequential_tracking/experiments/cfgs/tracker.yaml to your needs, especially frcnn_weights, cnn_weights and cnn_config need to point to the right places.

3. Run the tracker from within sequential_tracking folder:
  ```Shell
  python experiments/scripts/test_tracker.py
  ´´´


### Devkit
To evaluate the results install the official MOTChallenge devkit from https://bitbucket.org/amilan/motchallenge-devkit. Paste it into sequential_tracking/data/ and build it with compile.m.

test frcnn:
python experiments/scripts/test_faster_rcnn.py with {path_to_sacred_config}

Requirements:
install pyfrcnn
install coco into pyfrcnn/data not sequential_tracking/data
tensorboardX
sacred
pyyaml
tensorboard+tensorflow