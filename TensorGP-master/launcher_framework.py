#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to launch framework effective...



"""
from image_path_selector import rmse,norm_cross_correlation,zero_mean_norm_cross_correlation
import os
import cv2
import csv
import random
import timeit
import h5py
from gen_jsonfile_engine import writejson_kerasApp_engine
from kerasApp_engine_caller import init_engine_kerasApp
from training_models_keras import train_model_for_TENSORGP,test_model_for_TENSORGP
from utils_plots_perf import plotLOSS,plotACCURACY
start = timeit.default_timer()

"""
paper
get fitness each file

rename seed gen rank indv fitness

comparar runs gens do msm run

https://cdv.dei.uc.pt/wp-content/uploads/2020/06/correia2019breaking.pdf
https://cdv.dei.uc.pt/publications/ 

gen stats clasifier + paper (MISSING SOME)
git NOT OK 

128 OK
dataset OK 

base dados save imgs ( kinda ok)

treino delete imgs ; save deleted imgs to a folder (OK)

"""

###############################################################################


nb_training_runs =2#9 #Number of retraining steps 
nb_generator_runs =2 #4#Number of runs (generator) before Retraining

experiment_folder_name="minirunfffgrffffffun"#Change here to specify name of the folder to save the new imgs
threshold_NCC = 0.85
n_best_fitted_for_supervisor =2
using_train_until_perfection=True
savingPlots=True
usingHDF5=True
#Folder to filter the images(supervisor) before giving it to the retrain folder
dstpath="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/"+experiment_folder_name

#retrain folder:
folderImgs_Retraining_Art ="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/Art_128new_depth12_retrain/Internalnew"#"#newTesting" #FOLDER FOR RETRAIN WHERE WE ADD THE NEW IMAGES AFTER SUPERVISOR STEP
folderImgs_TestingExternals_Art ="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/Art_128new_depth12_retrain/External"#after training we test on the external to check if there's any missclassfication

folder_chosenImgs ="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/backupRuns_Engine/"+experiment_folder_name

########PATH OF SCRIPTS / JSON FILES:
path_kerasApp ="/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/mainKerasApp.py"
json_kerasApp_params_train="/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/params_train.json" #JSON FILE FOR PRE-LAUNCH (NO RETRAIN)
json_kerasApp_params_retraining="/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/params_retraining.json" #JSON FILE FOR RETRAINING

#---
#TRAINING BEFORE STARTING EFECTIVE ENGINE (to be verified if working properly )
#Check json files but usually it's 10 epochs atm for basic training +transfer learning+ finetunning; 
#and 3 epochs for retraining during engine
train_on_normal_dataset=False
if train_on_normal_dataset:
    os.system("python "+path_kerasApp+" train --jsonFile "+json_kerasApp_params_train)

counter_removed=0
###
### BEGIN EFECTIVE ENGINE
###
counter_images_added_ds=[]#nb of images added to the dataset
counter_images_total=[]
if(not(os.path.isdir(dstpath))):
    os.mkdir(dstpath)
    
for i in range(nb_training_runs+1):#Start with a model already trained on the dataset Art;
    list_imgs_allruns=[]
    ###### CREATING FOLDERSS AND INIT PATHS
    path_archive=dstpath+"/ARCHIVES_runTraining"+str(i)
    os.mkdir(path_archive)
    
    suffix_dstpath_train = dstpath+"/run_Training"+str(i)
    os.mkdir(suffix_dstpath_train)
    
    suffix_dstpath_deleted_imgs = dstpath+"/deleted_imgs_runTraining"+str(i)
    os.mkdir(suffix_dstpath_deleted_imgs)
    ######
    
    #if usingHDF5:
    #    hf = h5py.File(suffix_dstpath_train, 'a') # open a hdf5 file
    #LOOP FOR EC runs
    for j in range(nb_generator_runs):
        suffix_dstpath_engine =suffix_dstpath_train +"/run_Engine"+str(j)
        
        #Generation of images with the model trained (and retrained..)
        print("Generating imgs to the internal folder")
        #----Call generator of parameters engine:
        if usingHDF5:
            dataJson = writejson_kerasApp_engine(suffix_dstpath_engine,suffix_dstpath_train)
        else:
            dataJson = writejson_kerasApp_engine(suffix_dstpath_engine)
            
        init_engine_kerasApp(dataJson,n_best_fitted_for_supervisor)

        list_imgs = os.listdir(suffix_dstpath_engine) #List of imgs of the current run(n top best)
        for img_tmp in list_imgs:
            list_imgs_allruns.append( (suffix_dstpath_engine,img_tmp ) )
        #list_imgs.sort()

    ##END LOOP
    ###
    archive_imgs = os.listdir(path_archive) #List of imgs that are present in the archive(could be zero)
    if len(archive_imgs)==0: #if archive is empty just add first one and filter the rest by comparing
        first_fpath,first_imgname = list_imgs_allruns[0]
        """
        loadedimg = cv2.imread(os.path.join(fpath,imgname))
        cv2.imwrite(path_archive+"/"+imgname+".jpg",loadedimg, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        """
        archive_imgs.append((first_fpath,first_imgname))
    #Filtering; objective: add different images to the dataset
    if len(archive_imgs)!=0:
        for fpath,imgname in list_imgs_allruns:
            path_img_current = os.path.join(fpath,imgname)
            read_current_img = cv2.imread(path_img_current)
            isTheSame = False
            for archimg in archive_imgs:
                path_archimg = os.path.join(archimg[0],archimg[1])
                read_archimg = cv2.imread(path_archimg)
                verif=norm_cross_correlation(read_current_img,read_archimg)
                if verif >= threshold_NCC:
                    isTheSame = True#similar img so dont save it
                    break
            #When there is no image in the archive that is the same as the one we're currently checking, we add it to the archive
            if isTheSame==False:
                archive_imgs.append((fpath,imgname))
                
                
    #ADD IMGS TO DATASET+ARCHIVE
    for aim in archive_imgs:
        read_aim = cv2.imread(os.path.join(aim[0], aim[1]))
        cv2.imwrite(path_archive+"/"+aim[1], read_aim,[int(cv2.IMWRITE_JPEG_QUALITY), 100])#Add images to the archive to analyze later
        cv2.imwrite(folderImgs_Retraining_Art+"/"+aim[1], read_aim, [int(cv2.IMWRITE_JPEG_QUALITY), 100])#Add to the dataset

    #update counters stats
    counter_images_added_ds.append(len(archive_imgs))#archive_imgs contains all images that were filtered and added to the dataset 
    counter_images_total.append(len(list_imgs_allruns))#List_imgs_all_runs contains all imgs from the ec loop
    
    
    
    
    if i == nb_training_runs:#avoid last training process...
        break
    
    #Retraining on the new dataset:
        
    if using_train_until_perfection: #Training until perfection and deleting failures in the way...
        perfect_pred=False
        while not(perfect_pred):
            #os.system("python "+path_kerasApp+" train --jsonFile "+json_kerasApp_params_retraining)  #as we progress in the runs we train on the extra images from the generations
            trained_model,history_model = train_model_for_TENSORGP(jsondata=json_kerasApp_params_retraining)
            
            if(savingPlots):
                print(path_archive)
                print("SVING PLOT")
                #print("2BND",trained_model.history,trained_model.history["loss"])
                plotLOSS(history_model,suffix_dstpath_train,"classifier_iter"+str(i))
                plotACCURACY(history_model,suffix_dstpath_train,"classifier_iter"+str(i))
            
            
            list_fails_external = test_model_for_TENSORGP(trained_model, folderImgs_TestingExternals_Art)
            print("size:",len(list_fails_external))
            if (len(list_fails_external)> 50):
                print("deleting files from ext perf")
                
                #fail is a tuple : fail[0] is the complete path of the img missclassified and fail[1] is only the suffix (the name of the img)
                for fail in list_fails_external: #Read each image with a failed classification (external classified as internal) and cut and paste them in a folder to check
                    tmp_path=os.path.join(fail[0],fail[1])
                    read_fail = cv2.imread(tmp_path)
                    cv2.imwrite(suffix_dstpath_deleted_imgs+"/"+fail[1], read_fail, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    os.remove(tmp_path)
                    counter_removed+=1
            else:
                perfect_pred=True
    else:
        trained_model,history_model = train_model_for_TENSORGP(jsondata=json_kerasApp_params_retraining)
    
    f = open(dstpath+"/csv_runs_stats.csv", "a",newline="")
    #f.write("ITER_FRAMEWORK_"+str(i)+"||    TOTAL EXT IMGS EVOLVED: "+str(len(list_imgs_allruns))+"  ||    NB IMGS ADDED TO DS/ARCHIVE: "+str(len(archive_imgs)))
    row_to_add=["Iter_"+str(i), str(len(list_imgs_allruns)),str(len(archive_imgs))]
    writer= csv.writer(f)
    writer.writerow(row_to_add)
    f.close()
        


#SOME EXTRA STATS
total_evolved_ext = sum(counter_images_total)
print("TOTAL",total_evolved_ext)

only_added = sum(counter_images_added_ds)
print("Only added:",only_added)


stop = timeit.default_timer()
print('TIME TO FINISH ALL THE EXPERIMENT: ', stop - start)

print()
print("nb of removed imgs",counter_removed)


f = open(dstpath+"/overall_stats.txt", "a")
f.write("\n TOTAL EVOLVED IMGS:"+str(total_evolved_ext)+" .....  IMAGES ADDED:"+str(only_added)+"         PERCENTAGE ADDED: "+str(only_added/total_evolved_ext))
f.write("\n TIME ELAPSED TOTAL:"+str(stop-start))
f.close()





