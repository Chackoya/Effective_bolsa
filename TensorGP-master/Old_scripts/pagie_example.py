from engine import *

# Example:
# fitness function for the pagie polynomial classifier
def calc_fit(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    _stf = kwargs.get('stf')
    display_images = False

    #print("Current directory to save images: " + f_path)

    fn = f_path + "gen" + str(generation).zfill(5)
    fitness = []
    times = []
    best_ind = 0

    # set objective function according to min/max
    fit = 0
    condition = lambda: (fit < max_fit) # minimizing
    max_fit = float('inf')

    for i in range(len(tensors)):

        # save individual (not for
        if (generation % _stf) == 0 and display_images:
            save_image(tensors[i], i, fn)

        # time fitness
        start_ind = time.time()
        fit = tf_rmse(tensors[i], target)

        if condition():
            max_fit = fit
            best_ind = i

        times.append((time.time() - start_ind) * 1000.0)
        fitness.append(fit)
        population[i]['fitness'] = fit

    # save best
    if display_images:
        save_image(tensors[best_ind], best_ind, fn)
    return population, population[best_ind]


def pagie_poly(terminal_set, res):
    #global terminal_set
    x4 = tf.square(tf.square(terminal_set['x']))
    y4 = tf.square(tf.square(terminal_set['y']))
    c1 = tf.constant(1.0, dtype=tf.float32, shape=res)
    t1 = resolve_div_node(c1, tf.math.add(c1, resolve_div_node(c1, x4)))
    t2 = resolve_div_node(c1, tf.math.add(c1, resolve_div_node(c1, y4)))
    return tf.cast(tf.scalar_mul(127.5, tf.math.add(t1, t2)), tf.uint8)


if __name__ == "__main__":

    # GP params
    seed = random.randint(0, 2147483647)
    resolution = [224, 224, 3]
    dev = '/gpu:0' # device to run, write '/cpu_0' to tun on cpu
    number_generations = 20

    # build function and terminal sets according to resolution
    dim = len(resolution)
    build_function_set(function_set)

    #print(terminal_set)
    ts = build_terminal_set(dim, resolution, dev)
    #print(terminal_set)

    target = pagie_poly(ts, resolution)

    # create engine
    engine = Engine(fitness_func = calc_fit,
                    population_size = 50,
                    tournament_size = 3,
                    mutation_rate = 0.1,
                    crossover_rate = 0.8,
                    max_tree_depth = 15,
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
