import sys
import numpy as np
from skimage import io,transform
import csv
import cv2
import os

os.chdir('/home/shenling/inpainting/imagenet_test_data')
name = os.listdir(os.getcwd())
print(name)

id = 0
with open('/home/shenling/inpainting/imagenet_test.csv','wb') as csvfile:
	w = csv.writer(csvfile)
	for x in name:
		w.writerow(['/home/shenling/inpainting/imagenet_test_data/'+x, id])
		id += 1