# Tracking without Bells and Whistles

## Installation
1. Clone this repository
  ```
  git clone https://github.com/phil-bergmann/tracking_wo_BnW
  ```

2. Setup a anaconda environment with the environment.yml in the repository
  ```
  conda env create -f environment.yml
  ```
  The name of the environment will be "tracking_wo_BnW". Activate it!
  ```
  conda activate tracking_wo_BnW
  ```

3. Clone the Faster R-CNN fork, switch to dev branch.
  ```
  git clone https://github.com/phil-bergmann/pytorch-faster-rcnn
  git checkout dev
  ```
  Now follow the instructions in Readme.md under "Installation". If run into any problems to run the compiled modules try to use nvcc tool of version 9.0 as the whole project is based on an old PyTorch version.

4. Make a symbolic link from sequential_tracking/tracker/frcnn pointing to pytorch-faster-rcnn/lib. For example if pytorch-faster-rcnn and tracking_wo_BnW are in the same folder:
  ```
  cd tracking_wo_BnW/src
  ln -s ../../pytorch-faster-rcnn/lib/ frcnn
  ```

5. Download [MOT17Det dataset](https://motchallenge.net/data/MOT17Det.zip) and paste the "MOT17Det" folder in data/ folder. As the images are the same for MOT17Det, MOT17 and MOT16 the dataloader always uses the images in the MOT17Det folder and only uses the addionaly provided label and detection files in order to avoid redundancy of image data in the folder. Download the according label and/or detection files from the [benchmark website](https://motchallenge.net/) and extract them in the data/ folder (e.g. MOT16Labels, MOT16-det-dpm-raw, MOT17Labels). If needed download the [2DMOT15 dataset](https://motchallenge.net/data/2DMOT2015.zip) and extract it in the data/ folder.

## Using pretrained weights
Weights for the tracker that were used to prduce the results in the corresponding paper are provided in a [google drive folder](https://drive.google.com/open?id=1tnM3ap7NaYY00cEn5i2S2Zheq4lpyc4i). Just add them to the according folders in the repository. Faster R-CNN weights trained on MOT17Det and weights for the siamese network also trained on MOT17Det are provided. Additionaly the weights pretrained on imagenet needed to retrain the Faster R-CNN linked to in the Readme.md under "Train your own model" are provided if the original ones should disappear for whatever reason.

## Training Faster R-CNN
1. Download pretrained models for VGG16 or Res101 as described in the Readme.md under "Train your own model" and paste them inside data/imagenet_weights/ and name them "vgg16.pth" or "res101.pth". Alternatively the same files are provided in the [google drive folder](https://drive.google.com/open?id=1tnM3ap7NaYY00cEn5i2S2Zheq4lpyc4i).

2. Train the Faster R-CNN by running
  ```
  python experiments/scripts/train_faster_rcnn.py with {vgg16, res101} {mot, small_mot} tag={%name} description={%description}
  ```
  inside the root folder. The "name" and "description" parameters are optional. If no "tag" is provided a timestamp will be used.

## Training the Siamese CNN
For reidentification a siamese CNN has to be trained. Follow these instructions to do so:

1. Modify experiments/cfgs/siamese.yaml to your needs.

2. From within sequential_tracking run
  ```
  python experiments/train_siamese.py
  ```

## Evaluating the Tracker

1. Modify experiments/cfgs/tracker.yaml to your needs, especially frcnn_weights, frcnn_config, siamese_weights and siamese_config need to point to the right places.

2. Run the tracker from within the root folder:
  ```
  python experiments/scripts/test_tracker.py
  ```

## Devkit
To evaluate the results install the official [MOTChallenge devkit](https://bitbucket.org/amilan/motchallenge-devkit).
