#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate internal dataset 
"""
import os
import cv2
import timeit
from kerasApp_engine_caller import init_engine_kerasApp
from gen_jsonfile_engine import writejson_kerasApp_engine
from training_models_keras import train_model_for_TENSORGP,test_model_for_TENSORGP
from image_path_selector import rmse,norm_cross_correlation,zero_mean_norm_cross_correlation
start = timeit.default_timer()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
steps_pop = 150#to create 25k images we make 250*100 runs


#path_efective = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/kerasApp_example.py "
#path_efective_params = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/params_kerasApp_engine.json"

#ENGINE Gen JSON
script_gen_jsonFile_engine = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/gen_jsonfile_engine.py"


internal_folder="Internalnew"
dstpath_folderSave="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/1_NEW_INTERNALS/Internalnew_depth12_dim128x128tmpFORSAVING/"#"/home/gama/BolsaStuff/EML_mainRep/Art_datasets/"+internal_folder

dstpath_folderSaveTRUE="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/1_NEW_INTERNALS/Internalnew_depth12_dim128x128/"
if(not(os.path.isdir(dstpath_folderSave))):
    os.mkdir(dstpath_folderSave)

if(not(os.path.isdir(dstpath_folderSaveTRUE))):
    os.mkdir(dstpath_folderSaveTRUE)

threshold_NCC=0.9525
for i in range(steps_pop):

    #os.system("python "+script_gen_jsonFile_engine+" --nameFolderImgs "+dstpath_folderSave+"/internal_"+str(i)) #Generate other json file with different seed => saved on path_efective_params
    
    dataJson = writejson_kerasApp_engine(dstpath_folderSave)

    dataJson["nameFolderImgs"]=dataJson["nameFolderImgs"]+str(dataJson["seed"])+"/"

    init_engine_kerasApp(dataJson,50)

    list_imgs = os.listdir(dataJson["nameFolderImgs"]) 
    
    
    
    path_foldertmp =dataJson["nameFolderImgs"]
    
    
    #os.mkdir(archpath)
    archive = []
    

    if len(archive)==0:
        tmpimg = os.path.join(path_foldertmp, list_imgs[0])
        #read_aim = cv2.imread(os.path.join(path_foldertmp, img))
        archive.append(tmpimg)
        #print(archive)
        
    if len(archive)!=0:
        #print("SIZE:",len(archive))
        for current_img in list_imgs[1:]:
            path_img_current = os.path.join(path_foldertmp,current_img)
            
            #print("PATHICURRENT:",path_img_current)
            read_current_img = cv2.imread(path_img_current)
            isTheSame = False
            
            for archimg in archive:
                #print("ARCHIIMG:",archimg)
                #path_archimg = os.path.join(archimg[0],archimg[1])
                read_archimg = cv2.imread(archimg)
                verif=norm_cross_correlation(read_current_img,read_archimg)
                print(verif)
                if verif >= threshold_NCC and verif!=-1:

                    isTheSame = True
                    break
            #When there is no image in the archive that is the same as the one we're currently checking, we add it to the archive
            if isTheSame==False:
                print()
                print("ADDED IMAGE>>>>>>>>>>>")
                archive.append(path_img_current)
            
            
        

    cpt=0
    for aim in archive:

        read_aim = cv2.imread(aim)
        tmpname= dstpath_folderSaveTRUE+str(dataJson["seed"])+"_arch_"+str(cpt)+".jpg"

        cv2.imwrite(tmpname, read_aim,[int(cv2.IMWRITE_JPEG_QUALITY), 100])#Add images to the archive to analyze later
        #cv2.imwrite(folderImgs_Retraining_Art+"/"+aim[1], read_aim, [int(cv2.IMWRITE_JPEG_QUALITY), 100])#Add to the dataset
        cpt=cpt+1
    
    
    
        
        
        
        
        
    #os.system("python "+path_efective+" --jsonFile "+path_efective_params)#+"--ID_run"+str(j))




stop = timeit.default_timer()
print('Time: ', stop - start)  