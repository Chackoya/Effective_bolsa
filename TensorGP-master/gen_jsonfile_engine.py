#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:14:45 2021

@author: gama
"""
import json
import random

import argparse



##############################################################################
###
### WRITE JSON FOR kerasApp_example.py (engine effective)
##############################################################################


def writejson_kerasApp_engine(nameFolder=None,namehdf5=None):
    data = {
    #'modelWeights_path':"/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/Pretrained_models/newinternal_MobileNet_FineTunning_10epochs.h5",#Xception_FineTunning_10epochs.h5",#MobileNet_FineTunning_15epochs.h5",
    #'modelWeights_path':"/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/Pretrained_models/newinternal_MobileNet_FineTunning_10epochs.h5",#feb_Xception5epochs_12depth.h5",#feb_mobileNet5epochs_15depth.h5",
    'modelWeights_path':"/home/gama/BolsaStuff/EML_mainRep/Bolsa_main/Pretrained_models/mars_MobileNET128_5epochs_12depth.h5",
    'seed':random.randint(0, 2147483647),
    'resolution': [128, 128, 3],#[256, 256, 3],
    'useGPU':True,
    'number_generations':200,#50#500
    'population_size':20,
    'tournament_size':3,
    'mutation_rate':0.05,
    'crossover_rate': 0.8,
    'max_tree_depth':12,#15;22
    'method':'ramped half-and-half',
    'objective':'maximizing',#'minimizing',
    'stop_criteria':'generation',
    'immigration':10000,
    'debug':0,
    'save_to_file':10,
    'save_graphics':True,
    'show_graphics':False,
    'read_init_pop_from_file':None,
    'nameFolderImgs':nameFolder,
    'nameHDF5_file':namehdf5,    
    }
    with open('/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/params_kerasApp_engine.json', 'w') as outfile:
        json.dump(data, outfile,indent=4)

    print("GENERATE NEW JSON ENGINE PARAMS /SEEDS")
    return data


##
"""
parser = argparse.ArgumentParser(prog='gen_jsonfile_engine.py')
parser.add_argument("--nameFolderImgs", type=str, 
                              required=False,
                              default = None,
                              help = "Enter the path of the json File with the parameters")
    
    #parser.add_argument("--id_run",type=int,required=False,default=None,help="Id run")
args = parser.parse_args()
nameFolder = args.nameFolderImgs




###
print(">> AppEngine json File Generated")
writejson_kerasApp_engine(nameFolder)
"""