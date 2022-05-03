![Test Image 4](https://motchallenge.net/img/header-bg/mot_bannerthin.png)
# Official MOTChallenge Evaluation Kit
This repository contains the evaluation scripts for all challenges, available at www.MOTChallenge.net.
This devkit replaces the previous version that used to be accessible over https://bitbucket.org/amilan/motchallenge-devkit and is no longer maintained.
```diff
- IMPORTANT!
- The MOT evaluation code is not any longer maintained. 
```
Please visit the [new official python evaluation code](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md). 

## Requirements 
* Python (3.6.9 or newer)
* MATLAB (tested with R2020a, other versions should work too)


## Directories
All evaluation scripts assume the following directory structure: 

### ./res
This directory contains the tracking results (for each sequence); result files should be placed to subfolders
### ./seqmaps
Sequence lists for all supported different benchmarks
 
### ./data
This directory contains the ground truth data (for several different sequences/challenges)

### ./vid 
This directory is for the visualization of results or annotations


## Evaluation Scripts
This repo provides the evaluation scripts for the following challenges of www.motchallenge.net:

### MOT - Multi-Object Tracking - MOT15, MOT16, MOT17, MOT20
```diff
- IMPORTANT!
- The MOT evaluation code is not any longer maintained. 
```
Please visit the [new official python evaluation code](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md). 


[MOT Evaluation](MOT/README.md)

Challenge Name | Data url |
|----- | ----------- |
|2D MOT 15| https://motchallenge.net/data/2D_MOT_2015/ |
|MOT 16| https://motchallenge.net/data/MOT16/       |
|MOT 17| https://motchallenge.net/data/MOT17/       |
|MOT 20| https://motchallenge.net/data/MOT20/       |

### DET - Multi-Object Detection - MOT16Det, MOT20Det
[DET Evaluation](DET/README.md)
|Challenge Name | Data url |
|----- | ------------- | 
|MOT17Det| https://motchallenge.net/data/MOT17Det/ |
|MOT20Det| https://motchallenge.net/data/MOT20Det/ |

### MOTS - Multi-Object Tracking and Segmentation - MOTS20
[MOTS Evaluation](MOTS/README.md)
|Challenge Name | Data url | 
|----- | ---------------|
|MOTS | https://motchallenge.net/data/MOTS/ |

### 3D-ZeF20 
[ZF3D Evaluation](ZF3D/README.md)
|Challenge Name | Data url |
|----- | ---------------------- |
|3D-ZeF20 | https://motchallenge.net/data/3D-ZeF20/ |

### TAO - Tracking Any Object Challenge 
[TAO Evaluation](https://github.com/TAO-Dataset/tao)
|Challenge Name | Data url |
|----- | ---------------------- |
|TAO | https://github.com/TAO-Dataset/tao |

## Feedback and contact
We are constantly working on improving our benchmark to provide the best performance to the community.
You can help us to make the benchmark better to open issues in the repo and report bugs to the author:
```
Patrick Dendorfer - patrick.dendorfer@tum.de
```

