#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 20:31:26 2021

@author: gama
"""
import os
import numpy as np
import cv2
from math import isnan
from image_path_selector import rmse


def rmseee(img_a, img_b, blackwhite=False):
    if not blackwhite:
        return 1 - (np.sqrt(np.sum((img_a - img_b) ** 2)) / np.sqrt(255 ** 2 * len(img_a[0]) * len(img_a) * 3))
    else:
        return 1 - (np.sqrt(np.sum((img_a - img_b) ** 2)) / np.sqrt(1 ** 2 * len(img_a[0]) * len(img_a)))

def norm_cross_correlation(img_a, img_b, blackwhite=False):
    if not blackwhite:
        img_a = img_a / 255
        img_b = img_b / 255

    num = np.sum(img_a * img_b)
    den = np.sqrt(np.sum(img_a ** 2) * np.sum(img_b ** 2))

    try:
        val = num / den
        if isnan(val):
            return -1
        return val
    except:
        return -1

#from image_path_selector import rmse
pathimg_prio = "/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/Try1/run_Training0_Engine0/gen00000_seedrun1897954337_rank0_indiv00006.jpg"
imgprio = cv2.imread(pathimg_prio)



path_folder = "/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/Try1/run_Training0_Engine0"



cpt=0
L=[]
for img in os.listdir(path_folder):
    
    arrayimg =os.path.join(path_folder,img)
    if arrayimg == pathimg_prio:
        continue
    age = cv2.imread(arrayimg)
    
    print()
    print(arrayimg)
    #print(imgprio)
    if rmse(imgprio,age)==1:
        print("equal gg",img)
        cpt=cpt+1
        L.append(img)
    else:
        print("not equel",img)
    
print("total equal)",cpt)
print(L)