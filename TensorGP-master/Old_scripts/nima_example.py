from engine import *
import tensorflow as tf
from tensorflow import keras
#from keras.models import Model
#from keras.layers import Dense, Dropout

#from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenet import preprocess_input as preprocess_input_mob

from utils.score_utils import mean_score, std_score


# Example:
# fitness function for nima classifier
def nima_classifier(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    objective = kwargs.get('objective')
    _resolution = kwargs.get('resolution')
    _stf = kwargs.get('stf')
    
    images = True

    

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

        # NIMA classifier
        x = np.stack([tensors[index].numpy() for index in range(number_tensors)], axis = 0)
        #x = keras.applications.mobilenet.preprocess_input_mob(x)
        x = keras.applications.mobilenet.preprocess_input(x)
        scores = model.predict(x, batch_size = number_tensors, verbose=0)
        #scores = model.predict()
    
        # scores
        for index in range(number_tensors):

            if generation % _stf == 0:
                save_image(tensors[index], index, fn) # save image

            mean = mean_score(scores[index])
            std = std_score(scores[index])
            # fit = mean - std
            fit = mean

            if condition():
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            population[index]['fitness'] = fit

    # save best indiv
    if images:
        save_image(tensors[best_ind], best_ind, fn, addon='_best')
    return population, population[best_ind]


if __name__ == "__main__":

    # GP params
    seed = random.randint(0, 2147483647)
    resolution = [128, 128, 3]
    dev = '/gpu:0' # device to run, write '/cpu_0' to tun on cpu
    number_generations = 20

    # build function and terminal sets according to resolution
    dim = len(resolution)
    build_function_set(function_set)
    build_terminal_set(dim, resolution, dev)

    # NIMA example
    base_model = keras.applications.MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = keras.layers.Dropout(0.75)(base_model.output)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.models.Model(base_model.input, x)
    model.load_weights('weights/weights_mobilenet_aesthetic_0.07.hdf5')


    # create engine
    engine = Engine(fitness_func = nima_classifier,
                    population_size = 30,
                    tournament_size = 3,
                    mutation_rate = 0.1,
                    crossover_rate = 0.9,
                    max_tree_depth = 20,
                    target_dims=resolution,
                    method='ramped half-and-half',
                    objective='maximizing',
                    device=dev,
                    stop_criteria='generation',
                    stop_value=number_generations,
                    immigration=10000,
                    seed = seed,
                    debug=0,
                    save_to_file=10,
                    save_graphics=True,
                    show_graphics=False,
                    read_init_pop_from_file=None)
    # run evolutionary process
    engine.run()

    # A file can be loaded by specifing the "read_init_pop_from_file" variable:
    # read_init_pop_from_file = 'population_example.txt'
