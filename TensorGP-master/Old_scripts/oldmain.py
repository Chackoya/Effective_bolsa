#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to launch framework effective...



"""
from image_path_selector import rmse,norm_cross_correlation,zero_mean_norm_cross_correlation
import os
import cv2
import random
import timeit
from gen_jsonfile_engine import writejson_kerasApp_engine
from kerasApp_engine_caller import init_engine_kerasApp
start = timeit.default_timer()

"""
paper
get fitness each file

rename seed gen rank indv fitness

comparar runs gens do msm run

https://cdv.dei.uc.pt/wp-content/uploads/2020/06/correia2019breaking.pdf 
https://cdv.dei.uc.pt/publications/ 
"""

    
###############################################################################


nb_training_runs =1#9 #Number of retraining steps 
nb_generator_runs = 2#4#Number of runs (generator) before Retraining

experiment_folder_name="archivestest"  #Change here to specify name of the folder to save the new imgs

#Folder to filter the images(supervisor) before giving it to the retrain folder
dstpath="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/"+experiment_folder_name
#retrain folder:
folderImgs_Retraining_Art ="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/Art_256new_depth15_retrain/Internalnew"#"#newTesting" #FOLDER FOR RETRAIN WHERE WE ADD THE NEW IMAGES AFTER SUPERVISOR STEP


folder_chosenImgs ="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/"+experiment_folder_name



########PATH OF SCRIPTS / JSON FILES:
path_kerasApp ="/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/mainKerasApp.py"
json_kerasApp_params_train="/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/params_train.json" #JSON FILE FOR PRE-LAUNCH (NO RETRAIN)
json_kerasApp_params_retraining="/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/params_retraining.json" #JSON FILE FOR RETRAINING
#---
path_efective = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/kerasApp_example.py "
jsonparams_efective = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/params_kerasApp_engine.json"
########

#KERAS APP JSON
script_gen_jsonFile_kerasApp = "/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/gen_jsonfile_kerasapp.py"

#ENGINE Gen JSON
script_gen_jsonFile_engine = "/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/gen_jsonfile_engine.py"

#---

#dissim_min_threshold = 0.01#0.75
#adequacy_treshold= 0.0001
#maxArch =15

threshold_NCC = 0.85
if(not(os.path.isdir(dstpath))):
    os.mkdir(dstpath)
    
for i in range(nb_training_runs+1):#Start with a model already trained on the dataset Art;
    list_imgs_allruns=[]
    path_archive=dstpath+"/ARCHIVES_runTraining"+str(i)
    os.mkdir(path_archive)
    
    suffix_dstpath_train = dstpath+"/run_Training"+str(i)
    os.mkdir(suffix_dstpath_train)
    
    
    
    for j in range(nb_generator_runs):
        suffix_dstpath_engine =suffix_dstpath_train +"/run_Engine"+str(j)
        
        #Generation of images with the model trained (and retrained..)
        print("Generating imgs to the internal folder")
        
        #cmd_for_json = "python "+script_gen_jsonFile_engine+" --nameFolderImgs "+suffix_dstpath_engine#dstpath+suffix_dstpath
        #os.system(cmd_for_json) #Generate other json file with different seed => saved on path_efective_params
        
        #----Call generator of parameters engine:
        writejson_kerasApp_engine(suffix_dstpath_engine)
        
        
        os.system("python "+path_efective+" --jsonFile "+jsonparams_efective)#+"--ID_run"+str(j))
        
        
        
        list_imgs = os.listdir(suffix_dstpath_engine) #List of imgs of the current run(n top best)
        for img_tmp in list_imgs:
            list_imgs_allruns.append( (suffix_dstpath_engine,img_tmp ) )

        print()
        
        #list_imgs.sort()
        """
        list_imgs = os.listdir(dstpath+suffix_dstpath)
        list_imgs.sort()
        cpt = 0
        path_best_img = os.path.join(dstpath+suffix_dstpath,list_imgs[-2]) #same number as n best imgs 
        best_img = cv2.imread(path_best_img)
        #print("BEST:",path_best_img)
        for img in list_imgs:
            path_current_img =os.path.join(dstpath+suffix_dstpath,img)
            tmp_img = cv2.imread(path_current_img)
            
            #verify = norm_cross_correlation(best_img, tmp_img) 
            #print("verif:",verify,"     ",rmse(best_img,tmp_img))
            
            #if rmse(best_img,tmp_img)==1 or path_current_img == path_best_img:
            if norm_cross_correlation(best_img, tmp_img) <= threshold or path_current_img == path_best_img:
                #print(">:",path_current_img)
                cv2.imwrite(folderImgs_Retraining_Art+"/"+img, tmp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cpt+=1
        print(cpt)
        ###
        """
    archive_imgs = os.listdir(path_archive) #List of imgs that are present in the archive(could be zero)
    #print(list_imgs_allruns)
    if len(archive_imgs)==0:
        
        first_fpath,first_imgname = list_imgs_allruns[0]
        """
        loadedimg = cv2.imread(os.path.join(fpath,imgname))
        cv2.imwrite(path_archive+"/"+imgname+".jpg",loadedimg, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        """
        archive_imgs.append((first_fpath,first_imgname))
        print()
        print(">>>>>>>>>>>>>Saving the img :",first_fpath+"/"+first_imgname)
        print()
    if len(archive_imgs)!=0:
        for fpath,imgname in list_imgs_allruns:
            path_img_current = os.path.join(fpath,imgname)
            read_current_img = cv2.imread(path_img_current)
            
            isTheSame = False

            for archimg in archive_imgs:
                
                print(len(archive_imgs))
                path_archimg = os.path.join(archimg[0],archimg[1])
                read_archimg = cv2.imread(path_archimg)
                

                verif=norm_cross_correlation(read_current_img,read_archimg)
                
                print()
                if verif >= threshold_NCC:
                    print()
                    print("---------IMG NOT ADDED",verif)
                    print("Current img:",imgname)
                    print("Versus: ",archimg[1])
                    print()
                    isTheSame = True
                    break
            #When there is no image in the archive that is the same as the one we're currently checking, we add it to the archive
            if isTheSame==False:
                print()
                print("img:",imgname)
                archive_imgs.append((fpath,imgname))
                
                
    
    for aim in archive_imgs:
        read_aim = cv2.imread(os.path.join(aim[0], aim[1]))
        cv2.imwrite(path_archive+"/"+aim[1], read_aim,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
        
    if i == nb_training_runs:#avoid last training process...
        break
    #Retraining on the new dataset:
    #os.system("python "+path_kerasApp+" train --jsonFile "+json_kerasApp_params_train)
    os.system("python "+path_kerasApp+" train --jsonFile "+json_kerasApp_params_retraining)  #as we progress in the runs we train on the extra images from the generations
    
    
    

stop = timeit.default_timer()
print('TIME TO FINISH ALL THE EXPERIMENT: ', stop - start)  



    
"""
for fpath,imgname in list_imgs_allruns:
            #img => contains complete path of the image
            #print(imgname)
            fitness_current_img = float(imgname[-24:-4])
            #print(fitness_current_img)
            if fitness_current_img > adequacy_treshold:
                img_read = cv2.imread(os.path.join(fpath,imgname))
                #print(img_read)
                cv2.imwrite(path_archive+"/"+imgname+".jpg",img_read, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                break
            else:
                list_imgs_allruns.remove((fpath,imgname))

    lenArch = len(archive_imgs)
    if lenArch!=0:
        if lenArch < maxArch:
            for fpath,imgname in list_imgs_allruns:
                img_current = os.path.join(fpath,imgname)
                
                if lenArch<maxArch:
                    diss = dissimility(img_current, archive_imgs,lenArch)
                else:
                    diss = dissimility(img_current, archive_imgs,maxArch)
                if diss>=dissim_min_threshold:
                    img_read = cv2.imread(img_current)
                    #print(img_read)
                    cv2.imwrite(path_archive+"/"+imgname+".jpg",img_read, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else: 
   
            """
    