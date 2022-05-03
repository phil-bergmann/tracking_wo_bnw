import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from Visualize import Visualizer
from ZF3D.Camera import Camera
import glob
import os
import cv2
import numpy as np
import pandas as pd

class ZF3DVisualizer(Visualizer):


        def load(self, FilePath):
            """
            Load results and camera specifications for a given sequence
            """
            root_dir = os.path.join(os.path.dirname(self.FilePath), os.pardir)

            data = pd.read_csv(FilePath, sep=",|;| ", header=None, usecols=[0,1,2,3,4], names=['frame','id','3d_x','3d_y','3d_z'], engine='python')

            # Remove NaN elements
            data = data.dropna()

            self.camT, self.camF = self.getCam()
            return data



        def getCam(self):

            # Top camera parameters
            top_int = os.path.join(self.metaInfoDir,'camT_intrinsic.json')
            top_ext = os.path.join(self.metaInfoDir,'camT_references.json')
            camT = Camera(intrinsicPath = top_int,
              extrinsicPath = top_ext)

            # Front camera parameters
            front_int = os.path.join(self.metaInfoDir,'camF_intrinsic.json')
            front_ext = os.path.join(self.metaInfoDir,'camF_references.json')
            camF = Camera(intrinsicPath = front_int,
              extrinsicPath = front_ext)
            return camT, camF
        def drawResults(self, im = None, t = 0):


                curr_frame = self.resFile[self.resFile["frame"] == t]

                targets = list(curr_frame["id"].unique())
                for targetID in targets:
                        IDstr = "%d" % targetID
                        curr_target = curr_frame[curr_frame["id"] == targetID]

                        # Project 3D detections into the camera planes (2D)
                        x_world = np.float(curr_target["3d_x"])
                        y_world = np.float(curr_target["3d_y"])
                        z_world = np.float(curr_target["3d_z"])
                        camT_pt = self.camT.forwardprojectPoint(x_world, y_world, z_world)
                        camF_pt = self.camF.forwardprojectPoint(x_world, y_world, z_world)

                        # Shift camF point (front-view camera) to account for side-by-side view
                        camT_pt[0] += im.shape[1]/2

                        color = self.colors[int(targetID % len(self.colors))]
                        color = tuple([int(c*255) for c in color])

                        camT_pt = (int(camT_pt[0]), int(camT_pt[1]))
                        camF_pt = (int(camF_pt[0]), int(camF_pt[1]))

                        cv2.circle(im,camT_pt,10,color,4)
                        cv2.circle(im,camF_pt,10,color,4)
                        cv2.putText(im,IDstr,camT_pt,cv2.FONT_HERSHEY_SIMPLEX,2, color = color, thickness = 3)
                        cv2.putText(im,IDstr,camF_pt,cv2.FONT_HERSHEY_SIMPLEX,2, color = color, thickness = 3)

                return im

if __name__ == "__main__":

	visualizer = ZF3DVisualizer(
	    seqName = "ZebraFish-04",
	    FilePath ="data/3DZeF20/train/ZebraFish-02/gt/gt.txt",
	    image_dir = "data/3DZeF20/train/ZebraFish-02/img1",
	    mode = "gt",
	    output_dir  = "vid",
	    metaInfoDir = "data/3DZeF20/train/ZebraFish-02")

	visualizer.generateVideo(
	        displayTime = True,
	        displayName = "ZF",
	        showOccluder = True,
	        fps = 25 )
