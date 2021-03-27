import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
import json
import os
import utils_plots_perf


def test_model_for_TENSORGP(model,path_dataset_ext):
    """
    model: the model trained
    path: path of the external folder 

    -> Checks if all imgs are perfectly classified as external (external is 0 label)
    
    """
    list_failures_preds = []
    
    list_of_imgs = os.listdir(path_dataset_ext)
    for img in list_of_imgs:
        path_img = os.path.join(path_dataset_ext,img)
        #print(path_img)
        pred_list = classifyImg_general(model , path_img)
        score =  np.argmax(pred_list[0])
        #print(score)        
        if score == 1:
            list_failures_preds.append((path_dataset_ext,img))
            
        
    return list_failures_preds
    
def classifyImg_general(loaded_model,imgsPath):
    image_size = (150, 150)
    img = keras.preprocessing.image.load_img(imgsPath, target_size=image_size)
    plt.imshow(img)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    predictions = loaded_model.predict(img_array)

    return predictions
def train_model_for_TENSORGP(jsondata):
    """

    Parameters
    ----------

    jsondata: string => path of the json file with the parameters;hyperparameters
    
    """
    showPlots=False
    """
    if jsondata==None: # If we dont have data parameters from a json file, use default stuff...
        thekerasAPP="Xception"  #VGG16 
        image_size = (150, 150)
        batch_size = 32
        val_split= 0.2
        actfunc="sigmoid"
        useFineTuning=True
        useCheckPoints=False
        #fromScratch=False
        usingRetraining_EvolutionaryEngine=False   
    else:
    """
    with open(jsondata) as json_file:
        data = json.load(json_file)
        thekerasAPP=data["model"]
        saveFileName=data["save_name"]
        inputPATH_DATA=data["inputPath"]
        image_size=data["image_size"]
        batch_size=data["batch_size"]
        val_split = data["validation_split"]
        actfunc=data["activation_function"]
        epochs=data["epochs"]
        useFineTuning=data["useFineTuning"]
        useCheckPoints=data["useCheckpoints"]
        path_Checkpoints=data["path_folder_Checkpoints"]
        usingRetraining_EvolutionaryEngine=data["usingRetraining_EvolutionaryEngine"]
    #####
    
    
    #Generate the datasets, there are some requirements for the function that follows in terms of hierarchy:
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputPATH_DATA,#"FaceOrNoFace",
        validation_split=val_split, #0.2 usually
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        
        label_mode='categorical'
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputPATH_DATA,#"FaceOrNoFace",
        validation_split=val_split, #0.2 usually
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        
        label_mode= 'categorical'
    )
    #map a function of resizing
    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, image_size), y))
    val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, image_size), y))

    train_ds = train_ds.prefetch(buffer_size=32) #useful for systems pipeline based
    val_ds = val_ds.prefetch(buffer_size=32)#it starts fetching the next data while classifying the current one


    #Introducing "sample diversity" by applying random transformations. 
    #Improve classifier by showing new aspects and by reducing overfitting.
    
    
    if usingRetraining_EvolutionaryEngine==False:
        data_augmentation = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
        )
        
        #TRANSFER LEARNING WORKFLOW:
            #1) instantiate a base model and load pre-trained weights into it;
            #2) Freeze all layers in the base model by setting;
            #3) Create a new model on top of the output of one (or several) layers from the base model.
            #4) train your new model on your new dataset.
               
        weights="imagenet"
        include_top=False
        base_model= eval("keras.applications."+thekerasAPP+"(weights=\"imagenet\",include_top=False)")#"(weights=None,include_top=False)")
        #(weights=\"imagenet\",include_top=False)")
    
        #base_model = keras.applications.mobilenet
        """
        base_model = keras.applications.VGG16( #Xception
        #weights="imagenet",  # Load weights pre-trained on ImageNet.
        
        #input_shape=(150, 150, 3),
        
        include_top=False,
        )  # Do not include the ImageNet classifier at the top.
        """
        # Freeze the base_model
        base_model.trainable = False
                
        #Creation of new model ;
        inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
        x = data_augmentation(inputs)  # Apply random data augmentation
        

        # Pre-trained Xception weights requires that input be normalized
        # from (0, 255) to a range (-1., +1.), the normalization layer
        # does the following, outputs = (inputs - mean) / sqrt(var)
        if(thekerasAPP=="Xception"):
            norm_layer = keras.layers.experimental.preprocessing.Normalization()
            mean = np.array([127.5] * 3)
            var = mean ** 2
            # Scale inputs to [-1, +1]
            x = norm_layer(x)
            norm_layer.set_weights([mean, var])
            #This code is specific for Xception check sources for details
            
        else:#Rescale for others models
            x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
        
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
        #outputs = keras.layers.Dense(1,activation=actfunc)(x)#sigmoid
        outputs = keras.layers.Dense(2,activation=actfunc)(x)#softmax
        model = keras.Model(inputs, outputs)
        model.summary()
    
        #Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(),
            #loss=keras.losses.categorical_crossentropy(from_logits=(True)),#softmax
            #loss=keras.losses.BinaryCrossentropy(from_logits=True),#sigmoid
            #metrics=[keras.metrics.BinaryAccuracy()],
            #metrics=[keras.metrics.categorical_accuracy(y_true, y_pred)]
            loss='categorical_crossentropy',#sparse_
            metrics=['accuracy']
        )
        
        
        
    #Else use a already trained model by loading him and retrain with the increased dataset
    else:
        print()
        print(">>>>>>>>>>>>>LOADING EXISTING MODEL AND RETRAINING...")
        print()
        model= keras.models.load_model(saveFileName+".h5") # contains the path of the previously trained model (keras APP)
        model.summary()
        """
        print()
        #base_model.layers[-1].trainable=False
        for layer in base_model.layers:
            print(layer.trainable)
            #layer.trainable = False
            #print(layer.trainable)
        base_model.summary()
        """
        
        
    print("LOADED MODEL")
    #### USE CHECKPOINTS DURING TRAINING.(each epoch)
    if useCheckPoints:
        print(">>>>>>>>>>>>>>>>>USING CHECKPOINTS")
        filepath = path_Checkpoints+thekerasAPP+ "-best_in_validationacc_{epoch:04d}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min')#save_best_only=True)
        history_model= model.fit(train_ds, epochs=epochs, validation_data=val_ds,callbacks=[checkpoint])
        """
        usage ModelCheckPoint monitor values: mode min for loss; max for acc
            loss
            val_loss
        
            
        """
    else:
        #Training the new top layer
        #epochs = 5
        history_model = model.fit(train_ds, epochs=epochs, validation_data=val_ds)



    ###############
    #NEXT STEP(optional in practice)
    #FINE TUNING: unfreeze the model and train it entirely with a low learning rate;
    #running in inference mode => batch norm layers dont update their batch statistics
    if useFineTuning:
        
        base_model.trainable = True
        model.summary()
    
        model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        #loss="binary_crossentropy",
        metrics=["accuracy"],
        )
        
        
        nbepochs = 3
        #print()
        #print("STARTING FINETUNING FOR EPOCHS:",nbepochs)
        history_model = model.fit(train_ds, epochs=nbepochs, validation_data=val_ds)
        
        #base_model.trainable = False
    
    
    print()
    print("History:")
    print(history_model.history)
    print()
    
    if showPlots:
        utils_plots_perf.plotLOSS(history_model)
        utils_plots_perf.plotACCURACY(history_model)
        plt.show() #block main thread until all blocks are closed
    #print()
    if saveFileName!=None:
        model.save(saveFileName+'.h5')
        print("TRAINING OF THE MODEL & SAVING PROCESS IS OVER...")

###SAVE CHECKPOINts
    return model,history_model