# Tracking without Bells and Whistles

## Installation

1. Clone includind object detector submodules and enter this repository:
  ```
  git clone --recurse-submodules https://github.com/phil-bergmann/tracking_wo_BnW
  cd tracking_wo_BnW
  ```

2. Install packages for Python 3.6:
    `pip3 install -r requirements.txt`

3. Install the Faster R-CNN + FPN object detector::
  ```
  pip install https://github.com/timmeinhardt/FPN_Pytorch/archive/tracking_wo_bnw.zip
  ```

4. Compile the Faster R-CNN code by following the installation instructions in the `README.md` of the `pytorch-faster-rcnn` repository. The compilation requires CUDA 9.0 and its corresponding `nvcc` compiler.

5. Download MOT Challenge data:
    1. Download [MOT17Det dataset](https://motchallenge.net/data/MOT17Det.zip) and extract the `MOT17Det` folder into the `data` folder. As the images are the same for MOT17Det, MOT17 and MOT16 we only need one set of images for all three benchmarks.
    2. Download the benchmark label and/or detection files for [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16Labels.zip), [MOT16Labels](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) and extract them in the `data` folder.
    3. If needed download the [2DMOT15 dataset](https://motchallenge.net/data/2DMOT2015.zip) and extract it in the `data` folder.

## Pretrained weights
Weights for the tracker that were used to prduce the results in the corresponding paper are provided in a [google drive folder](https://drive.google.com/open?id=1tnM3ap7NaYY00cEn5i2S2Zheq4lpyc4i). Just add them to the according folders in the repository. Faster R-CNN weights trained on MOT17Det and weights for the siamese network also trained on MOT17Det are provided. Additionaly the weights pretrained on imagenet needed to retrain the Faster R-CNN linked to in the Readme.md under "Train your own model" are provided if the original ones should disappear for whatever reason.

## Train and test object detector (Faster R-CNN)
1. Download pretrained models for VGG16 or Res101 as described in the Readme.md under "Train your own model" and paste them inside data/imagenet_weights/ and name them "vgg16.pth" or "res101.pth". Alternatively the same files are provided in the [google drive folder](https://drive.google.com/open?id=1tnM3ap7NaYY00cEn5i2S2Zheq4lpyc4i).

2. Train the Faster R-CNN with:
  ```
  python experiments/scripts/train_faster_rcnn.py with {vgg16, res101} {mot, small_mot} tag={%name} description={%description}
  ```
  inside the root folder. The "name" and "description" parameters are optional. If no "tag" is provided a timestamp will be used.

3. Test the Faster R-CNN with:
  ```
  python experiments/scripts/test_faster_rcnn.py with {mot, small_mot} {link to sacred_config.yaml of model}
  ```
  inside the root folder. Link to config file is in the respective `output` subdirectory, e.g., `output/frcnn/res101/mot_2017_train/180k/sacred_config.yaml`.

## Training the Siamese CNN
For reidentification a siamese CNN has to be trained. Follow these instructions to do so:

1. Modify experiments/cfgs/siamese.yaml to your needs.

2. Run from the root folder:
  ```
  python experiments/scripts/train_siamese.py
  ```

## Evaluating our Tracktor++

1. Modify experiments/cfgs/tracker.yaml to your needs, especially frcnn_weights, frcnn_config, siamese_weights and siamese_config need to point to the right places.

2. Run the tracker from the root folder:
  ```
  python experiments/scripts/test_tracktor.py
  ```
To evaluate the results install the official [MOTChallenge devkit](https://bitbucket.org/amilan/motchallenge-devkit).
