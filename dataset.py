from os.path import isfile
import cv2 as cv
import numpy as np
import glob

def read_dataset():
	images = [cv.imread(file) for file in glob.glob("./CRCHistoPhenotypes_2016_04_28/Detection/*/*.bmp")]
	return images