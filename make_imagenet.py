import sys
import numpy as np
from skimage import io, transform
import csv
#import cv2
import os

os.chdir('/disk/0/storage/shenling/ILSVRC2012_img_train/')
name = os.listdir(os.getcwd())
name.sort()
print(name)

id = 0
with open('/home/shenling/inpainting/imagenet.csv','w') as csvfile:    
    w = csv.writer(csvfile)
    for x in name:
        w.writerow(['/disk/0/storage/shenling/ILSVRC2012_img_train/'+x, id])
        id+=1
    

