# Tracking without bells and whistles

This repository provides the implementation of our paper **Tracking without bells and whistles** (Philipp Bergmann, Tim Meinhardt, Laura Leal-Taixe) [https://arxiv.org/abs/1903.05625]. All results presented in our work were produced with this code.

In addition to our supplementary document, we provide an illustrative [web-video-collection](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-supp_video_collection.zip). The collection includes examplary Tracktor++ tracking results and multiple video examples to accompany our analysis of state-of-the-art tracking methods.

![Visualization of Tracktor](data/method_vis_standalone.png)

## Installation

1. Clone and enter this repository:
  ```
  git clone --recurse-submodules https://github.com/phil-bergmann/tracking_wo_bnw
  cd tracking_wo_bnw
  ```

2. Install packages for Python 3.6 in [virtualenv](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/):
    1. `pip3 install -r requirements.txt`
    2. Faster R-CNN + FPN: `pip3 install -e src/fpn`
    3. Faster R-CNN: `pip3 install -e src/frcnn`
    4. Tracktor: `pip3 install -e .`
    5. PyTorch 0.3.1 for CUDA 9.0: `pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl`

3. Compile Faster R-CNN + FPN and Faster R-CNN:
    1. Make sure the `nvcc` compiler with CUDA 9.0 is working and all CUDA paths are set (in particular `export CPATH=/usr/local/cuda-9.0/include`).
    2. Compile with: `sh src/fpn/fpn/make.sh`
    3. Compile with: `sh src/frcnn/frcnn/make.sh`
    4. If compilation was not successful, check README.md and issues of official Faster-RCNN [repository](https://github.com/jwyang/faster-rcnn.pytorch/) for help.

4. MOTChallenge data:
    1. Download [MOT17Det](https://motchallenge.net/data/MOT17Det.zip), [MOT16Labels](https://motchallenge.net/data/MOT16Labels.zip), [2DMOT2015](https://motchallenge.net/data/2DMOT2015.zip), [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) and place them in the `data` folder. As the images are the same for MOT17Det, MOT17 and MOT16 we only need one set of images for all three benchmarks.
    2. Unzip all the data by executing:
    ```
    unzip -d MOT17Det MOT17Det.zip
    unzip -d MOT16Labels MOT16Labels.zip
    unzip -d 2DMOT2015 2DMOT2015.zip
    unzip -d MOT16-det-dpm-raw MOT16-det-dpm-raw.zip
    unzip -d MOT17Labels MOT17Labels.zip
    ```

5. Download object detector and re-identifiaction Siamese network weights and MOTChallenge result files for ICCV 2019:
    1. Download zip file from [here](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output.zip).
    2. Extract in `output` directory.

## Evaluate Tracktor++
In order to configure, organize, log and reproduce our computational experiments we structured our code with the [Sacred](http://sacred.readthedocs.io/en/latest/index.html) framework. For a detailed explanation of the Sacred interface please read its documentation.

1. Our Tracktor can be configured by changing the corresponding `experiments/cfgs/tracktor.yaml` config file. The default configuration runs Tracktor++ with the FPN object detector as described in the paper.

2. Run Tracktor by executing:

  ```
  python experiments/scripts/test_tracktor.py
  ```

3. The results are logged in the corresponding `output` directory. To evaluate the results download and run the official [MOTChallenge devkit](https://bitbucket.org/amilan/motchallenge-devkit).

## Train and test object detector (Faster-RCNN + FPN)

We pretrained the object detector on PASCAL VOC and did an extensive hyperparameter cross-validation. The resulting training command is:
  ```
  python trainval_net.py voc_init_iccv19 --dataset mot_2017_train --net res101 --bs 2 --nw 4 --epochs 38 --save_dir weights --cuda --use_tfboard True --lr_decay_step 20 --pre_checkpoint weights/res101/pascal_voc_0712/v2/fpn_1_12.pth --pre_file weights/res101/pascal_voc_0712/v2/config.yaml
  ```

Test the provided object detector by executing:
  ```
  python experiments/scripts/test_fpn.py voc_init_iccv19 --cuda --net res101 --dataset mot_2017_train --imdbval_name mot_2017_train --checkepoch 27
  ```

## Training the re-identifaction Siamese network

1. The training config file is located at `experiments/cfgs/siamese.yaml`.

2. Start training by executing:
  ```
  python experiments/scripts/train_siamese.py
  ```

## Publication
 If you use this software in your research, please cite our publication:
 
```
@article{DBLP:journals/corr/abs-1903-05625,
    author    = {Philipp Bergmann and
                Tim Meinhardt and
                Laura Leal{-}Taix{\'{e}}},
    title     = {Tracking without bells and whistles},
    journal   = {CoRR},
    volume    = {abs/1903.05625},
    year      = {2019},
    url       = {http://arxiv.org/abs/1903.05625},
    archivePrefix = {arXiv},
    eprint    = {1903.05625},
    timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
    biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-05625},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
