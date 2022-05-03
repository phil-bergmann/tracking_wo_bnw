import sys
from Visualize import Visualizer
from mots_common.io import load_sequences, load_seqmap, load_txt
import pycocotools.mask as rletools
import glob
import os
import cv2
import colorsys
import numpy as np

def apply_mask(image, mask, color, alpha=0.5):
	"""
	 Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c],image[:, :, c])
	return image


class MOTSVisualizer(Visualizer):

	def load(self, FilePath):
		return load_txt(FilePath)



	def drawResults(self, im = None, t = 0):
		self.draw_boxes = False


		for obj in self.resFile[t]:

			color = self.colors[obj.track_id % len(self.colors)]

			color = tuple([int(c*255) for c in color])


			if obj.class_id == 1:
				category_name = "Car"
			elif obj.class_id == 2:
				category_name = "Ped"
			else:
				category_name = "Ignore"
				color = (0.7*255, 0.7*255, 0.7*255)

			if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
				x, y, w, h = rletools.toBbox(obj.mask)

				pt1=(int(x),int(y))
				pt2=(int(x+w),int(y+h))

				category_name += ":" + str(obj.track_id)
				cv2.putText(im, category_name, (int(x + 0.5 * w), int( y + 0.5 * h)), cv2.FONT_HERSHEY_TRIPLEX,self.imScale,color,thickness =2)
				if self.draw_boxes:
					cv2.rectangle(im,pt1,pt2,color,2)


			binary_mask = rletools.decode(obj.mask)

			im = apply_mask(im, binary_mask, color)
		return im

if __name__ == "__main__":
	visualizer = MOTSVisualizer(
	seqName = "MOTS20-11",
	FilePath ="data/MOTS/train/MOTS20-11/gt/gt.txt",
	image_dir = "data/MOTS/train/MOTS20-11/img1",
	mode = "gt",
	output_dir  = "vid")

	visualizer.generateVideo(
	        displayTime = True,
	        displayName = "seg",
	        showOccluder = True,
	        fps = 25 )
