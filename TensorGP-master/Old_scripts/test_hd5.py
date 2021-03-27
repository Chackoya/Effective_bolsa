#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:03:52 2021

@author: gama
"""
import numpy as np
from pathlib import Path
import h5py
import cv2

import os

hdf5_dir = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/darrjjj"
os.mkdir(hdf5_dir)
pathimgs= "/home/gama/BolsaStuff/EML_mainRep/Art_datasets/old_backups/art_test_nima"#ext/External_0.jpg"


###
#%%
from PIL import Image
save_path = hdf5_dir+'/numpy.hdf5'

"""
print('image size: %d bytes'%os.path.getsize(pathimgs))
hf = h5py.File(save_path, 'a') # open a hdf5 file

with open(pathimgs, 'rb') as img_f:
     binary_data = img_f.read()      # read the image as python binary

#img_np = np.array(Image.open(pathimgs))

binary_data_np = np.asarray(binary_data)

#hf.create_group('group1')
hf.create_dataset('default', data=binary_data_np)  # write the data to hdf5 file



#print(g1.items())
hf.close()  # close the hdf5 file
print('hdf5 file size: %d bytes'%os.path.getsize(save_path))




#%%read
import io
read_path = hdf5_dir+'/numpy.hdf5'


hf = h5py.File(read_path, 'r') # open a hdf5 file
key = list(hf.keys())[0]
print("Keys: %s" % key)
print(list(hf.keys()))


print(hf[key])
data = np.array(hf[key])   # write the data to hdf5 file
#print(data)
img = Image.open(io.BytesIO(data))
print('image size:', img.size)
hf.close()  # close the hdf5 file
img.show()

#%%



"""
"""
listpath = os.listdir(pathimgs)
print(listpath)
hf = h5py.File(save_path, 'a') # open a hdf5 file
cpt=0
for p in listpath:
    pathcomplete = os.path.join(pathimgs,p)
    
    newg = hf.create_group("Evol"+str(cpt))
    
    listimgs= os.listdir(pathcomplete)
    for img in listimgs:
        print(img)
        pathtmpimg= os.path.join(pathcomplete,img)
        with open(pathtmpimg, 'rb') as img_f:
            binary_data = img_f.read()      # read the image as python binary
        #img_np = np.array(Image.open(pathimgs))
        binary_data_np = np.asarray(binary_data)
        
        newg.create_dataset(img,data=binary_data_np)#write data to hdf5 file
        print("newg:",list(newg))
    
    
    cpt+=1


print("Items",hf.keys())
    
print(hf["Evol0"])

print(hf["Evol0"].keys())
    

#%%
import io
read_path = hdf5_dir+'/numpy.hdf5'


hfread = h5py.File(read_path, 'r') # open a hdf5 file
print("Here",list(hfread.keys()))

key = list(hfread.keys())[0]
print("Keys: %s" % key)


#print(list(hfread[key].keys()))


#print()

listtmp=list(hfread[key].keys())
print("LIST:",listtmp)

#print()
#print(hfread[key].items())
#data = np.array(hf[key])
#print(data)
print()
data = np.array(listtmp[0])   # write the data to hdf5 file

print(data)

G1=hf.get(key)
G1_items=list(G1.items())
print(G1_items[0])

d = np.array(G1_items[0][1])

img = Image.open(io.BytesIO(d))
print('image size:', img.size)
hf.close()  # close the hdf5 file
img.show()
"""
#%%
import io
read_path="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/teuchdfffzz/run_Training0/hdFILE.hdf5"

hfread = h5py.File(read_path, 'r') # open a hdf5 file

print("Here",list(hfread.keys()))

key = list(hfread.keys())[0]
print("Keys: %s" % key)

g1= hfread.get(key)

G1_items = list(g1.items())
print(G1_items)
    
d = np.array(G1_items[0][1])

img = Image.fromarray((d))

print('image size:', img.size)
hfread.close()  # close the hdf5 file
img.show()
