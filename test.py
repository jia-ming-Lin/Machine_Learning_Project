# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:44:48 2018

@author: jiaminglin
"""
import sys
import numpy as np
from skimage.feature import hog 
from sklearn.metrics import accuracy_score 
from train import model_generate
import matplotlib.pyplot as plt
import cv2
def processd_data_with_hog_train(data):
    datalist = []
    for j in range (0,data.size):
        a = cv2.resize(data[j],(28,28))
        datalist.append(a)
    datalist=np.array(datalist)
    test_hogimagelist = [] 
    test_hoglist = [] 
    for c in range (0,datalist.shape[0]): 
        fd1, hog_image1 = hog(datalist[c], orientations = 8,pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualize=True, multichannel=False) 
        test_hoglist.append(fd1.T) 
        test_hogimagelist.append(hog_image1)
    test_hog = np.array(test_hoglist)
    return test_hog
def output(data,out):
    clf=model_generate(Train=True)
    y=clf.predict(processd_data_with_hog_train(data))
    y=y.reshape(len(y),1)
    return np.save(str(out),y)    

if __name__ == "__main__":
    output(np.load(sys.argv[1]),sys.argv[2])

    

