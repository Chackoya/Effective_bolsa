from engine import *
import tensorflow as tf
from tensorflow import keras
import argparse
import json
#from utils.score_utils import mean_score, std_score
import numpy as np
import os
# Example:
# fitness function for kerasApps classifier
def kerasApp_classifier(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    objective = kwargs.get('objective')
    _resolution = kwargs.get('resolution')
    _stf = kwargs.get('stf')

    images = True #set true to save best individual
    
    #---- Retrain 
    _nameFolderImgs=kwargs.get("nameFolderImgs")
    #print("folder",_nameFolderImgs)
    _id_run=kwargs.get('id_run_seed') #it's the seed (which is kinda of unique between runs)
    
    creating_dataset =True #put this to true if the goal is to generate new internal and "savingImgs_Retrain to false"
    savingImgs_Retrain=False
    n_best_fitted = 2
    print(">>>>>",_nameFolderImgs)
    print()
    if _nameFolderImgs is not None:
        pathImgs_Retrain=_nameFolderImgs#+"/gen" + str(generation).zfill(5) #modify here path for the folder
    
    #pathImgs_Retraining_Art ="/home/gama/BolsaStuff/EML_mainRep/Art_datasets/Art256new_retrain/Internalnew/gen"+ str(generation).zfill(5)
    #pathImgs_Retraining_Art = "/home/gama/BolsaStuff/EML_mainRep/Art_datasets/1_NEW_INTERNALS/Internalnew_depth15/gen"+ str(generation).zfill(5)
    #------------
    
    fn = f_path + "gen" + str(generation).zfill(5)
    fitness = []
    best_ind = 0

    # set objective function according to min/max
    fit = 0
    if objective == 'minimizing':
        condition = lambda: (fit < max_fit)  # minimizing
        max_fit = float('inf')
    else:
        condition = lambda: (fit > max_fit) # maximizing
        max_fit = float('-inf')


    number_tensors = len(tensors)
    with tf.device('/CPU:0'):

        # classifier
        x = np.stack([tensors[index].numpy() for index in range(number_tensors)], axis = 0)
        #x = keras.applications.mobilenet.preprocess_input_mob(x)
        
        #Modify here if using mobile net or the xpceiton..
        #x = keras.applications.mobilenet.preprocess_input(x)
        x = keras.applications.xception.preprocess_input(x)
        
        
        #print("here:",x.shape)
        
        scores = model.predict(x, batch_size = number_tensors, verbose=0)
        #print("SCORES:  ",scores)
        # scores
        #print(number_tensors)
        for index in range(number_tensors):

            if generation % _stf == 0:
                save_image(tensors[index], index, fn) # save image
                if creating_dataset:
                    save_image(tensors[index],index,_nameFolderImgs) #this line is for creating internal dataset, remove it after
            ##
            fit = scores[index][0]
            
            #fit = scores[index][0] #sigmoid
            #print(">>fit:",fit,index)
            #print(fit)
            #print("Fit:",fit)
            if condition():
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            
            population[index]['fitness'] = fit


    
    # save best indiv
    if images:
        save_image(tensors[best_ind], best_ind, fn, addon='_best')
        
    
    if savingImgs_Retrain: #saving on a folder to retrain the model
        if(not(os.path.isdir(_nameFolderImgs))):
            os.mkdir(_nameFolderImgs)
        sort_index = sorted(range(len(fitness)), key=fitness.__getitem__)

        sort_index.reverse()
        #First index of sort_index is the best one, 2nd index is the 2nd best etc..
        counter_tmp =0
        for i in sort_index:
            #print(i,fitness[i])
            if counter_tmp==n_best_fitted:
                break
            save_imageBIS(tensors[i],i, pathImgs_Retrain+"/gen" + str(generation).zfill(5)+"_seedrun"+str(_id_run)+"_rank"+str(counter_tmp),"_fitness"+str(fitness[i]).ljust(20,"0"))
            #save_image(tensors[i],i,pathImgs_Retraining_Art+"_seedrun"+str(_id_run)+"_rank"+str(counter_tmp))
            
            counter_tmp+=1

    return population, population[best_ind]


##main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='kerasApp_example.py')
    parser.add_argument("--jsonFile", metavar = "FILE", 
                              required=True,
                              default = None,
                              help = "Enter the path of the json File with the parameters")
    
    #parser.add_argument("--id_run",type=int,required=False,default=None,help="Id run")
    args = parser.parse_args()
    jsondata = args.jsonFile
    #id_run = args.id_run
    
    generate_dataset=False
    
    #json File is loaded, now get parameters (# GP params)
    with open(jsondata) as json_file:
        dataJson = json.load(json_file)
        #init:
        modelName = dataJson['modelWeights_path']
        seed = dataJson['seed']
        resolution = dataJson['resolution']
        useGPU = dataJson['useGPU']
        number_generations = dataJson['number_generations']
        population_size = dataJson['population_size']
        tournament_size = dataJson['tournament_size']
        mutation_rate = dataJson['mutation_rate']
        crossover_rate = dataJson['crossover_rate']
        max_tree_depth = dataJson['max_tree_depth']
        method = dataJson['method']
        objective = dataJson['objective']
        stop_criteria = dataJson['stop_criteria']
        immigration = dataJson['immigration']
        debug = dataJson['debug']
        save_to_file = dataJson['save_to_file']
        save_graphics = dataJson['save_graphics']
        show_graphics = dataJson['show_graphics']
        read_init_pop_from_file = dataJson['read_init_pop_from_file']
        name_folder_Imgs= dataJson['nameFolderImgs']

    if useGPU:
        dev = '/gpu:0' # device to run, write '/cpu_0' to tun on cpu
    else:
        dev = '/cpu_0'
        
    
    # build function and terminal sets according to resolution
    dim = len(resolution)
    build_function_set(function_set)
    build_terminal_set(dim, resolution, dev)

    #Mobile classifier
    
    #Modify path according to your config... also change line 49 or 50 (uncomment the useless one)
    #modelName = '/home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/weights/mobilenet_art5epochs.h5' #json 
    model= keras.models.load_model(modelName)
    model.summary()    


    # create engine from json data
    engine = Engine(fitness_func = kerasApp_classifier,
                    population_size = population_size,#100
                    tournament_size = tournament_size,#3
                    mutation_rate = mutation_rate,#0.1
                    crossover_rate = crossover_rate,#0.9
                    max_tree_depth = max_tree_depth,#20
                    target_dims= resolution,
                    method=method,#'ramped half-and-half',
                    objective=objective,#minimizing',#'maximizing',
                    device=dev,
                    stop_criteria=stop_criteria,#'generation'
                    stop_value=number_generations,
                    immigration=immigration,
                    seed = seed,
                    debug=debug,
                    save_to_file=save_to_file,#10
                    save_graphics=save_graphics,#True,
                    show_graphics=show_graphics,#False,
                    read_init_pop_from_file=read_init_pop_from_file,#None
                    nameFolderImgs=name_folder_Imgs
                    )
    
    
    print("\n >>>STARTING EVOLUTIONARY RUN\n")
    # run evolutionary process
    engine.run()
    # A file can be loaded by specifing the "read_init_pop_from_file" variable:
    # read_init_pop_from_file = 'population_example.txt'
    
    
    #####
    #SAVE PARAMS USED:
    """
    with open(engine.experiment.working_directory+'params_Engine.json', 'w') as outfile:
        json.dump(dataJson, outfile,indent=4)
    
    
    if (name_folder_Imgs is not None) and generate_dataset==False:
        with open(name_folder_Imgs+'_params_used_Engine.json', 'w') as outfile:
            json.dump(dataJson, outfile,indent=4)
    """
    
