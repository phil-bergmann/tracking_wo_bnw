# Tracking without Bells and Whistles

## Installation

1. Clone and enter this repository:
  ```
  git clone https://github.com/phil-bergmann/tracking_wo_BnW
  cd tracking_wo_BnW
  ```

2. Setup an [Anaconda](https://conda.io/docs/user-guide/install/index.html) environment with the given `environment.yml`:
  ```
  conda env create -f environment.yml
  ```
  The name of the environment will be *tracking_wo_BnW*. Activate it:
  ```
  conda activate tracking_wo_BnW
  ```
  Manually install the following packages:
  ```
  pip install easydict sacred pyyaml tensorboardX opencv-python h5py
  ```

3. Clone the `dev` branch of our Faster R-CNN fork:
  ```
  git clone -b dev https://github.com/phil-bergmann/pytorch-faster-rcnn
  ```

4. Make a symbolic link from `pytorch-faster-rcnn/lib` to `src/frcnn`:
  ```
  cd src
  ln -s pytorch-faster-rcnn/lib/ frcnn
  ```

5. Compile the Faster R-CNN code by following the installation instructions in the `README.md` of the `pytorch-faster-rcnn` repository. The compilation requires CUDA 9.0 and its corresponding `nvcc` compiler.

6. Download MOT Challenge data:
    1. Download [MOT17Det dataset](https://motchallenge.net/data/MOT17Det.zip) and extract the `MOT17Det` folder into the `data` folder. As the images are the same for MOT17Det, MOT17 and MOT16 we only need one set of images for all three benchmarks.
    2. Download the benchmark label and/or detection files for [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16Labels.zip), [MOT16Labels](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) and extract them in the `data` folder. 
    3. If needed download the [2DMOT15 dataset](https://motchallenge.net/data/2DMOT2015.zip) and extract it in the `data` folder.

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

2. Run from the root folder:
  ```
  python experiments/scripts/train_siamese.py
  ```

## Evaluating the Tracker

1. Modify experiments/cfgs/tracker.yaml to your needs, especially frcnn_weights, frcnn_config, siamese_weights and siamese_config need to point to the right places.

2. Run the tracker from the root folder:
  ```
  python experiments/scripts/test_tracker.py
  ```
To evaluate the results install the official [MOTChallenge devkit](https://bitbucket.org/amilan/motchallenge-devkit).
