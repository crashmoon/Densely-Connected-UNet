import sys
import numpy as np
from skimage import io, transform
import csv
#import cv2
import os

os.chdir('/home/shenling/inpainting/paris/paris_train/paris_train_original/')
name = os.listdir(os.getcwd())
name.sort()
print(name)

id = 0
with open('/home/shenling/inpainting/train.csv','w') as csvfile:    
    w = csv.writer(csvfile)
    for x in name:
        w.writerow(['/home/shenling/inpainting/paris/paris_train/paris_train_original/'+x, id])
        id+=1
    

