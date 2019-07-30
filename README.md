# Tracking without bells and whistles

This repository provides the implementation of our paper **Tracking without bells and whistles** (Philipp Bergmann, Tim Meinhardt, Laura Leal-Taixe) [https://arxiv.org/abs/1903.05625]. All results presented in our work were produced with this code.

In addition to our supplementary document, we provide an illustrative [web-video-collection](https://drive.google.com/open?id=1cl2jkY2s_Uq5CQYPZYA_dsY-DAbV4xS-). The collection includes examplary Tracktor++ tracking results and multiple video examples to accompany our analysis of state-of-the-art tracking methods.

![Visualization of Tracktor](data/method_vis_standalone.png)

## Installation

1. Clone and enter this repository:
  ```
  git clone --recurse-submodules https://github.com/phil-bergmann/tracking_wo_bnw
  cd tracking_wo_bnw
  ```

2. Install packages for Python 3.6:
    1. `pip3 install -r requirements.txt`
    2. Faster R-CNN + FPN: `pip3 install -e src/fpn`
    3. (**Optional**) Faster R-CNN: `pip3 install -e src/frcnn`
    4. Tracktor: `pip3 install -e .`
    5. PyTorch 0.3.1 for CUDA 9.0: `pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl`

3. Compile Faster R-CNN + FPN and Faster R-CNN:
    1. Make sure the `nvcc` compiler with CUDA 9.0 is working and all CUDA paths are set (in particular `export CPATH=/usr/local/cuda-9.0/include`).
    2. `sh src/fpn/fpn/make.sh`
    3. (**Optional**) `sh src/frcnn/frcnn/make.sh`

4. MOTChallenge data:
    1. Download [MOT17Det dataset](https://motchallenge.net/data/MOT17Det.zip) and extract the `MOT17Det` folder into the `data` folder. As the images are the same for MOT17Det, MOT17 and MOT16 we only need one set of images for all three benchmarks.
    2. Download the benchmark label and/or detection files for [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16Labels.zip), [MOT16Labels](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) and extract them in the `data` folder.
    3. (**Optional**) Download the [2DMOT15 dataset](https://motchallenge.net/data/2DMOT2015.zip) and extract it in the `data` folder.

5. Download object detector and re-identifiaction Siamese network weights and MOTChallenge result files for ICCV 2019:
    1. Download zip file from [here](https://drive.google.com/open?id=1E0seC4zSdAsKUNScv4M0eAu7fG_v65_Q).
    2. Extract in `output` directory.

## Evaluate Tracktor++
In order to configure, organize, log and reproduce our computational experiments we structured our code with the [Sacred](http://sacred.readthedocs.io/en/latest/index.html) framework. For a detailed explanation of the Sacred interface please read its documentation.

1. Our Tracktor can be configured by changing the corresponding `experiments/cfgs/tracker.yaml` config file. The default configuration runs Tracktor++ with the FPN object detector as described in the paper.

2. Run Tracktor by executing:

  ```
  python experiments/scripts/test_tracktor.py
  ```

3. The results will be logged in the corresponding `output` directory. To evaluate the results download and run the official [MOTChallenge devkit](https://bitbucket.org/amilan/motchallenge-devkit).



## Train and test object detector
# Faster R-CNN + FPN


## Train and test object detector
# Faster R-CNN

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

## Training the re-identifaction Siamese network

1. The training config file is located at `experiments/cfgs/siamese.yaml`.

2. Start training by executing:
  ```
  python experiments/scripts/train_siamese.py
  ```

