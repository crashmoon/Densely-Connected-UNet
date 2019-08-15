import sys
import numpy as np
from skimage import io, transform
import csv
#import cv2
import os

os.chdir('/home/shenling/inpainting/paris/paris_eval/paris_eval_corrupted/')
name = os.listdir(os.getcwd())
name.sort()
print(name)

id = 0
with open('/home/shenling/inpainting/test.csv','w') as csvfile:    
    w = csv.writer(csvfile)
    for x in name:
        w.writerow(['/home/shenling/inpainting/paris/paris_eval/paris_eval_corrupted/'+x, id])
        id+=1
    

