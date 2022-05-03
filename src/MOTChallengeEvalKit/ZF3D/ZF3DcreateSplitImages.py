
import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from Visualize import Visualizer
from ZF3D.Camera import Camera
import glob
import os
import cv2
import numpy as np
import pandas as pd



def genSplitViewImages(image_dir):
    """
    Create image folder and generate split-view images
    """




    imgF_dir = os.path.join(image_dir, 'imgF')
    imgT_dir = os.path.join(image_dir, 'imgT')
    output_dir = os.path.join(image_dir, 'img1')
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for frame in os.listdir(imgF_dir):
        if frame.endswith(".jpg"):
            f = os.path.split(frame)[-1]

            output_file = os.path.join(output_dir,f)
            
            if os.path.exists(output_file):
                    continue


            imgF = cv2.imread(os.path.join(imgF_dir,f))
            imgT = cv2.imread(os.path.join(imgT_dir,f))

            splitImg = np.hstack((imgF,imgT))

            cv2.imwrite(output_file,splitImg)


if __name__ == "__main__":
	image_dir = "data/3DZeF20/train/ZebraFish-02"
	genSplitViewImages(image_dir)
