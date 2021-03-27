import csv
import sys
from collections import Counter
import copy
import datetime
import math
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pylab
import time
import random

from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import numpy as np


## ====================== tensorflow operators ====================== ##
def resolve_var_node(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = tf.reshape(tf.range(dimensions[n], dtype=tf.float32), temp)
    _resolution = dimensions[n]
    dimensions[n] = 1
    res = tf.scalar_mul(((1.0 / (_resolution - 1)) * domain_delta), res)
    res = tf.math.add(res, tf.constant(min_domain, tf.float32, res.shape))
    res = tf.tile(res, dimensions)
    return res


def resolve_abs_node(child1, dims=[]):
    return tf.math.abs(child1)

def resolve_add_node(child1, child2, dims=[]):
    return tf.math.add(child1, child2)

def resolve_and_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_and(left_child_tensor, right_child_tensor), tf.float32))

def resolve_xor_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_xor(left_child_tensor, right_child_tensor), tf.float32))

def resolve_or_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_or(left_child_tensor, right_child_tensor), tf.float32))

def resolve_cos_node(child1, dims=[]):
    return tf.math.cos(tf.scalar_mul(math.pi, child1))

def resolve_div_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(child1, tf.float32)
    right_child_tensor = tf.cast(child2, tf.float32)
    return tf.math.divide_no_nan(left_child_tensor, right_child_tensor)

def resolve_exp_node(child1, dims=[]):
    return tf.math.exp(child1)

def resolve_if_node(child1, child2, child3, dims=[]):
    return tf.where(child3 < 0, child1, child2)

def resolve_log_node(child1, dims=[]):
    res = tf.where(child1 > 0, tf.math.log(child1), tf.constant(-1, tf.float32, dims))
    return res

def resolve_max_node(child1, child2, dims=[]):
    return tf.math.maximum(child1, child2)

def resolve_mdist_node(child1, child2, dims=[]):
    return tf.scalar_mul(0.5, tf.add(child1, child2))

def resolve_min_node(child1, child2, dims=[]):
    return tf.math.minimum(child1, child2)

def resolve_mod_node(child1, child2, dims=[]):
    return tf.math.mod(child1, child2)

def resolve_mult_node(child1, child2, dims=[]):
    return tf.math.multiply(child1, child2)

def resolve_neg_node(child1, dims=[]):
    return tf.math.negative(child1)


def resolve_pow_node(child1, child2, dims=[]):
    return tf.where(child1 == 0, tf.constant(0, tf.float32, dims), tf.math.pow(tf.math.abs(child1), tf.math.abs(child2)));

def resolve_sign_node(child1, dims=[]):
    #return tf.math.divide_no_nan(tf.math.abs(child1),child1)
    return tf.math.sign(child1)

def resolve_sin_node(child1, dims=[]):
    return tf.math.sin(tf.scalar_mul(math.pi, child1))

def resolve_sqrt_node(child1, dims=[]):
    return tf.where(child1 > 0, tf.math.sqrt(child1), tf.constant(0, tf.float32, dims))

def resolve_sub_node(child1, child2, dims=[]):
    return tf.math.subtract(child1, child2)

def resolve_tan_node(child1, dims=[]):
    return tf.where(child1 != (math.pi * 0.5), tf.math.tan(tf.scalar_mul(math.pi, child1)), tf.constant(0, tf.float32, dims))

def resolve_warp_node(tensors, image, dimensions):
    n = len(dimensions)
    #print("[DEBUG]:\tWarp dimensions: " + str(dimensions))
    #print("[DEBUG]:\tWarp log(y): ")
    #print(tensors[1].numpy())

    tensors = [tf.where(tf.math.is_nan(t), tf.zeros_like(t), t) for t in tensors]


    indices = tf.stack([
        tf.clip_by_value(
            tf.round(tf.multiply(
                tf.constant((dimensions[k] - 1) * 0.5, tf.float32, shape=dimensions),
                tf.math.add(tensors[k], tf.constant(1.0, tf.float32, shape=dimensions))
            )),
            clip_value_min=0.0,
            clip_value_max=(dimensions[k] - 1)
        ) for k in range(n)],
        axis=n
    )

    indices = tf.cast(indices, tf.int32)
    #print("[DEBUG]:\tWarp Indices: ")
    #print(indices.numpy())
    return tf.gather_nd(image, indices)

# return x * x * x * (x * (x * 6 - 15) + 10);
# simplified to x2*(6x3 - (15x2 - 10x)) to minimize TF operations
def resolve_sstepp_node(child1, dims=[]):
    x = resolve_clamp(child1)
    x2 = tf.square(x)
    return tf.add(tf.multiply(x2,
                              tf.subtract(tf.scalar_mul(6.0 * domain_delta, tf.multiply(x, x2)),
                                          tf.subtract(tf.scalar_mul(15.0 * domain_delta, x2),
                                                      tf.scalar_mul(10.0 * domain_delta, x)))),
                  tf.constant(min_domain, dtype=tf.float32, shape=dims))

# return x * x * (3 - 2 * x);
# simplified to (6x2 - 4x3) to minimize TF operations
def resolve_sstep_node(child1, dims=[]):
    x = resolve_clamp(child1)
    x2 = tf.square(x)
    return tf.add(tf.subtract(tf.scalar_mul(3.0 * domain_delta, x2),
                              tf.scalar_mul(2.0 * domain_delta, tf.multiply(x2, x))),
                  tf.constant(min_domain, dtype=tf.float32, shape=dims))

def resolve_step_node(child1, dims=[]):
    return tf.where(child1 < 0.0,
                    tf.constant(-1.0, dtype=tf.float32, shape=dims),
                    tf.constant(1.0, dtype=tf.float32, shape=dims))

def resolve_clamp(tensor):
    return tf.clip_by_value(tensor, clip_value_min=min_domain, clip_value_max=max_domain)
    #return tf.cast(tf.cast(tensor, tf.uint8), tf.float32)

def resolve_frac_node(child1, dims=[]):
    return tf.subtract(child1, tf.floor(child1))

def resolve_clip_node(child1, child2, child3, dims=[]): # a < n < b
    return tf.minimum(tf.maximum(child1, child2), child3)

def resolve_len_node(child1, child2, dims=[]):
    return tf.sqrt(tf.add(tf.square(child1), tf.square(child2)))

def resolve_lerp_node(child1, child2, child3, dims=[]):
    child3 = resolve_frac_node(child3, dims)
    t_dist = tf.subtract(child2, child1)
    t_dist = tf.multiply(child3, t_dist)
    return tf.math.add(child1, t_dist)

def tf_rmse(child1, child2):
    child1 = tf.cast(child1, tf.float32)
    child2 = tf.cast(child2, tf.float32)
    elements = np.prod(child1.shape.as_list())
    sdiff = tf.math.squared_difference(child1, child2)
    mse = tf.math.reduce_sum(sdiff).numpy() / elements
    mse = math.sqrt(mse)
    return mse

## ====================== tensorflow node class ====================== ##

class Node:
    def __init__(self, value, children, terminal):
        self.value = value
        self.children = children
        self.terminal = terminal

    def node_tensor(self, tens, dims):

        dimensionality = function_set[self.value][0] # should be called arity
        if self.value != 'warp':
            arg_list = tens[:dimensionality]
        else:
            arg_list = [tens[1:dimensionality] + [z_terminal], tens[0]]

        arg_list += [dims]

        return function_set[self.value][1](*arg_list)


    def get_tensor(self, dims):

        if self.terminal:
            if self.value == 'scalar':
                args = len(self.children)
                if args == 1:
                    return tf.constant(self.children[0], tf.float32, dims)
                else:
                    last_dim = dims[-1]
                    extend_children = self.children + ([self.children[-1]] * (last_dim - args))
                    return tf.stack(
                        [tf.constant(float(c), tf.float32, dims[:-1]) for c in extend_children[:last_dim]],
                        axis = len(dims) - 1
                    )
                # both cases do the same if args == 1, this condition is here for speed concerns if
                # the last dimension is very big (e.g. 2d(1024, 1024) we don't want color in that case)
            else:
                return terminal_set[self.value]
        else:
            tens_list = []
            for c in self.children:
                tens_list.append(c.get_tensor(dims))

            return self.node_tensor(tens_list, dims)

    def get_str(self):
        if self.terminal and self.value != 'scalar':
            return str(self.value)
        else:
            string_to_use = self.value
            strings_to_differ = ['and', 'or', 'if']
            if self.value in strings_to_differ:
                string_to_use = '_' + self.value

            string_to_use += '('
            c = 0
            size = len(self.children)
            while True:
                if self.terminal:
                    string_to_use += str(self.children[c])
                else:
                    string_to_use += self.children[c].get_str()
                c += 1
                if c > size - 1: break
                string_to_use += ', '

            return string_to_use + ')'

    def get_depth(self, depth=0):
        if self.terminal:
            return depth
        else:
            max_d = 0
            for i in self.children:
                child_depth = i.get_depth(depth + 1)
                if max_d < child_depth:
                    max_d = child_depth
            return max_d

    def get_graph(self, dot):
        global node_index
        node_label = str(node_index)
        dot.node(node_label, str(self.value))
        node_index += 1
        if self.value != 'scalar':
            for i in self.children:
                dot.edge(node_label, i.get_graph(dot))
        else:
            for i in self.children:
                new_node = dot.node(str(node_index), str(i))
                dot.edge(node_label, str(node_index))
                node_index += 1
        return node_label


def constrain(n, a, b):
    return min(max(n, a), b)

def map_8bit_to_domain(n):
    return min_domain + (n / 255.0 * domain_delta)


def str_to_tree(stree, terminal_set, number_nodes=0):
    if stree in terminal_set:
        return number_nodes, Node(value=stree, terminal=True, children=[])
    elif stree[:6] == 'scalar':
        numbers = [constrain(float(x), min_domain, max_domain) for x in re.split('\(|\)|,', stree)[1:-1]]

        return number_nodes, Node(value='scalar', terminal=True, children=numbers)
    else:
        x = stree[:-1].split("(", 1)
        value = x[0]
        if x[0][0] == '_':
            value = x[0][1::]
        x = x[1]
        pc = 0
        last_pos = 0
        children = []

        for i in range(len(x)):
            c = x[i]
            if c == '(':
                pc += 1
            elif c == ')':
                pc -= 1
            elif c == ',' and pc == 0:
                number_nodes, tree = str_to_tree(x[last_pos:i], terminal_set, number_nodes)
                children.append(tree)
                last_pos = i + 2

        number_nodes, tree = str_to_tree(x[last_pos:], terminal_set, number_nodes)
        children.append(tree)
        if value == "if":
            children = [children[1], children[2], children[0]]
        return number_nodes + 1, Node(value=value, terminal=False, children=children)


def set_device(device = '/gpu:0', debug_lvl = 1):
    cuda_build = tf.test.is_built_with_cuda()
    gpus_available = len(tf.config.list_physical_devices('GPU'))

    # because these won't give an error but will be unspecified
    if (device is None) or (device == '') or (device == ':'):
        device = '0'
    try:
        result_device = device

        # just to verify errors
        with(tf.device(device)):
            a = tf.constant(2, dtype=tf.float32)
            if debug_lvl > 0:
                if a == 2:
                    print("Device " + device + " successfully tested, using this device. ")
                else:
                    print("Device " + device + " not working.")
    except RuntimeError or ValueError:
        if cuda_build and gpus_available > 0:
            result_device = '/gpu:0'
            print("[WARNING]:\tCould not find the specified device, reverting to GPU.")
        else:
            result_device = '/cpu:0'
            print("[WARNING]:\tCould not find the specified device, reverting to CPU.")
    return result_device


# TODO: should this be an inner class of Engine()?
class Experiment:

    def set_experiment_filename(self):
        date = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-3]
        return "run__" + date + "__" + str(self.ID) + "__images"

    def set_generation_directory(self, generation):
        try:
            self.current_directory = self.working_directory + "generation_" + str(generation).zfill(5) + "/"
            os.makedirs(self.current_directory)
            #print("[DEBUG]:\tSet current directory to: " + self.current_directory)
        except FileExistsError:
            self.current_directory = self.working_directory
            print("Could not create directory for generation" + str(generation) + " as it already exists")

    def set_experiment_ID(self):
        return int(time.time() * 1000.0) << 16

    def __init__(self, previous_state, seed = None, wd = "/output_images/"):

        self.ID = self.set_experiment_ID() if (previous_state is None) else previous_state['ID']
        self.seed = self.ID if (seed is None) else seed
        self.filename = self.set_experiment_filename()

        try:
            self.working_directory = os.getcwd() + wd + self.filename + "/"
            os.makedirs(self.working_directory)
        except FileExistsError:
            print("[WARNING]:\tExperiment directory already exists, saving files to current directory.")
            print("[WARNING]:\tFilename: " + self.working_directory)
            self.working_directory = ''

        self.current_directory = self.working_directory
        self.immigration_directory = self.working_directory + "immigration/"
        try:
            os.makedirs(self.immigration_directory)
        except FileExistsError:
            print("[WARNING]:\tImmigration directory already exists.")

def save_image(tensor, index, fn, addon=''):
    path = fn + addon + "_indiv" + str(index).zfill(5) + ".jpg"#".png"
    aux = np.array(tensor, dtype='uint8')
    #Image.fromarray(aux, mode="L").save(path) # no color
    Image.fromarray(aux, mode="RGB").save(path) # color
    
    return path


def save_imageBIS(tensor, index, fn, addon=''): #new fct slighty modified 
    path = fn + "_indiv" + str(index).zfill(5)+ addon  + ".jpg"#".png"
    aux = np.array(tensor, dtype='uint8')
    #Image.fromarray(aux, mode="L").save(path) # no color
    Image.fromarray(aux, mode="RGB").save(path) # color
    return path


class Engine:

    ## ====================== genetic operators ====================== ##

    def crossover(self, parent_1, parent_2):
        crossover_node = None
        if self.engine_rng.random() < 0.9:  # TODO: review, this is Koza's rule
            # function crossover
            parent_1_candidates = self.get_candidates(parent_1, True)
            parent_1_chosen_node = self.engine_rng.choice(list(parent_1_candidates.elements()))
            possible_children = []
            for i in range(len(parent_1_chosen_node.children)):
                if not parent_1_chosen_node.children[i].terminal:
                    possible_children.append(i)
            if possible_children != []:
                crossover_node = copy.deepcopy(
                    parent_1_chosen_node.children[self.engine_rng.sample(possible_children, 1)[0]])
            else:
                crossover_node = copy.deepcopy(parent_1_chosen_node)
        else:
            parent_1_terminals = self.get_terminals(parent_1)
            crossover_node = self.engine_rng.choice(list(parent_1_terminals.elements()))
        if crossover_node is None:
            print("[ERROR]: Did not select a crossover node")
        new_individual = copy.deepcopy(parent_2)
        parent_2_candidates = self.get_candidates(new_individual, True)
        parent_2_chosen_node = self.engine_rng.choice(list(parent_2_candidates.elements()))

        #print("Node (val, children, terminal) : " + str(parent_2_chosen_node.value) + ", " + str(len(parent_2_chosen_node.children)) + ", " + str(parent_2_chosen_node.terminal))

        parent_2_chosen_node.children[self.engine_rng.randint(0, len(parent_2_chosen_node.children) - 1)] = crossover_node
        return new_individual

    def tournament_selection(self):
        if self.objective == 'minimizing':
            _st = float('inf')
        else:
            _st = -float('inf')

        winner = {'fitness': _st}
        while winner['fitness'] == _st:
            tournament_population = self.engine_rng.sample(self.population, self.tournament_size)
            for i in tournament_population:
                if ((self.objective == 'minimizing' and (i['fitness'] < winner['fitness'])) or
                        (self.objective != 'minimizing' and (i['fitness'] > winner['fitness']))):
                    winner = i
        return winner

    def mutation(self, parent, method):
        number_funcs = 4
        random_n = self.engine_rng.random()
        func = int(random_n / (1/ number_funcs))

        if func == 0:
            return self.subtree_mutation(parent, method)
        elif func == 1:
            return self.point_mutation(parent)
        elif func == 2:
            return self.promotion_mutation(parent)
        else:
            return self.demotion_mutation(parent)

    def random_terminal(self):
        _l = []
        if self.engine_rng.random() < self.scalar_prob:
            _v = 'scalar'

            #color
            if self.engine_rng.random() < self.uniform_scalar_prob:
                _l = [self.engine_rng.uniform(0, 1) for i in range(self.target_dims[-1])]
            else:
                _l = [self.engine_rng.uniform(0, 1)] * self.target_dims[-1]


        else:
            _v = self.engine_rng.sample(list(terminal_set), 1)[0]
        return Node(terminal = True, children = _l, value = _v)


    def copy_node(self, n):
        return Node(value=n.value, terminal=n.terminal, children=n.children)

    def promotion_mutation(self, parent):
        new_individual = copy.deepcopy(parent)

        # every node except last depth (terminals)
        candidates = self.list_nodes(new_individual, True, True, False, False)

        if len(candidates) > 0:
            chosen_node = self.engine_rng.choice(candidates) # parent = root

            # random child of chosen
            chosen_child = chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)]
            chosen_node.value =  chosen_child.value
            chosen_node.children = chosen_child.children
            chosen_node.terminal = chosen_child.terminal

        return new_individual


    def demotion_mutation(self, parent):
        new_individual = copy.deepcopy(parent)
        #print("[DEBUG D] Before:\t" + new_individual.get_str())

        # every node except last depth (terminals)
        candidates = self.list_nodes(new_individual, True, True, False, True)
        chosen_node = self.engine_rng.choice(candidates)

        #insert node between choosen and choosen's child
        #random child of chosen
        chosen_child = chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)]

        _v = chosen_child.value
        _c = chosen_child.children
        _t = chosen_child.terminal
        child_temp = Node(value=_v, children=_c, terminal=_t)


        chosen_child.value = self.engine_rng.choice(list(function_set))
        chosen_child.terminal = False
        chosen_child.children = []

        nchildren = function_set[chosen_child.value][0]
        new_child_position = self.engine_rng.randint(0, nchildren - 1)

        for i in range(nchildren):
            if i == new_child_position:
                chosen_child.children.append(child_temp)
            else:
                chosen_child.children.append(self.random_terminal())
        return new_individual


    def list_nodes(self, node, root = False, add_funcs = True, add_terms = True, add_root = False):
        res = []
        if (node.terminal and add_terms) or ((not node.terminal) and add_funcs and ((not root) or add_root)):
            res.append(node)
        if not node.terminal:
            for c in node.children:
                res += self.list_nodes(c, False, add_funcs, add_terms, add_root)
        return res


    def subtree_mutation(self, parent, method):
        new_individual = copy.deepcopy(parent)
        candidates = self.get_candidates(new_individual, True)

        chosen_node = self.engine_rng.choice(list(candidates.elements()))

        _, mutation_node = self.generate_program(method, -1, chosen_node.get_depth(), root=True)

        chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)] = mutation_node
        return new_individual

    def replace_nodes(self, node):
        if node.terminal:
            if self.engine_rng.random() < self.scalar_prob:
                node.value = 'scalar'
                # node.children = [self.engine_rng.uniform(0, 1)] # no color
                # color
                if self.engine_rng.random() < self.uniform_scalar_prob:
                    node.children = [self.engine_rng.uniform(0, 1)] * self.target_dims[-1]
                else:
                    node.children = [self.engine_rng.uniform(0, 1) for i in range(self.target_dims[-1])]
            else:
                if node.value == 'scalar':
                    node.value = self.engine_rng.choice(list(terminal_set))[0]
                else:
                    temp_tset = terminal_set.copy()
                    del temp_tset[node.value]
                    node.value = self.engine_rng.choice(list(temp_tset))[0]
        else:
            arity_to_search = function_set[node.value][0]
            set_of_same_arities = arity_fset[arity_to_search][:]
            set_of_same_arities.remove(node.value)

            if len(set_of_same_arities) > 0:
                node.value = self.engine_rng.choice(set_of_same_arities)


        if not node.terminal:
            for i in node.children:
                if self.engine_rng.random() < 0.05:
                    self.replace_nodes(i)


    def point_mutation(self, parent):
        new_individual = copy.deepcopy(parent)
        candidates = self.list_nodes(new_individual, True, True, True, True)
        chosen_node = self.engine_rng.choice(candidates)
        self.replace_nodes(chosen_node)
        return new_individual

    def get_candidates(self, node, root):
        candidates = Counter()
        for i in node.children:
            if (i is not None) and (not i.terminal):
                candidates.update([node])
                candidates.update(self.get_candidates(i, False))
        if root and candidates == Counter():
            candidates.update([node])
        return candidates

    def get_terminals(self, node):
        candidates = Counter()
        if node.terminal:
            candidates.update([node])
        else:
            for i in node.children:
                if i is not None:
                    candidates.update(self.get_terminals(i))
        return candidates

    ## ====================== init class ====================== ##

    def __init__(self,
                 fitness_func,
                 population_size = 100,
                 tournament_size = 3,
                 mutation_rate = 0.15,
                 crossover_rate = 0.9,
                 min_tree_depth = -1,
                 max_tree_depth = 8,
                 max_init_depth = None,
                 method = 'ramped half-and-half',
                 terminal_prob = 0.2,
                 scalar_prob = 0.55,
                 uniform_scalar_prob = 0.7,
                 stop_criteria = 'generation',
                 stop_value = 10,
                 objective = 'minimizing',
                 immigration = float('inf'),
                 target_dims = None,
                 max_nodes = -1,
                 seed = None,
                 debug = 0,
                 save_graphics = True,
                 show_graphics = True,
                 warp_mode = None,
                 device = '/cpu:0',
                 save_to_file = 10,
                 previous_state=None,
                 read_init_pop_from_file=None,
                 nameFolderImgs=None,#new
                 kerasModel=None,#new
                 n_best_imgs_for_supervisor=None,
                 nameHDF5_file=None):#new

        print("\n\n" + ("=" * 84))
        self.last_engine_time = time.time()
        start_init = self.last_engine_time

        self.recent_fitness_time = 0
        self.recent_tensor_time = 0
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_tree_depth = max_tree_depth
        self.min_tree_depth = min_tree_depth
        self.stop_criteria = stop_criteria

        self.save_graphics = save_graphics
        self.show_graphics = show_graphics

        #new params for retraining tracking------
        self.id_run_seed=seed
        self.nameFolderImgs=nameFolderImgs# Folder for retraining purposes
        self.kerasModel = kerasModel
        self.n_best_imgs_for_supervisor=n_best_imgs_for_supervisor
        self.nameHDF5_file=nameHDF5_file
        #----
        self.immigration = immigration
        self.debug = debug
        self.save_to_file = save_to_file
        self.max_nodes = max_nodes
        self.objective = objective
        self.warp_mode = 'image' if (warp_mode is None) else 'full'

        # probabilities
        self.terminal_prob = terminal_prob
        self.scalar_prob = scalar_prob
        self.uniform_scalar_prob = uniform_scalar_prob


        self.max_init_depth = self.max_tree_depth if (max_init_depth is None) else max_init_depth
        self.target_dims = [128, 128] if (target_dims is None) else target_dims

        self.dimensionality = len(self.target_dims)
        self.device = set_device(device=device) # Check for GPU

        self.fitness_func = fitness_func
        if fitness_func is None:
            print("[ERROR]:\tFitness function must be defined.")
            return

        self.experiment = Experiment(previous_state, seed)

        if method in ['ramped half-and-half', 'grow', 'full']:
            self.method = method
        else:
            self.method = 'ramped half-and-half'


        if previous_state is not None:
            self.current_generation = previous_state['generations']
            self.experiment.seed += self.current_generation

            self.experiment.set_generation_directory(self.current_generation)
            self.engine_rng = random.Random(self.experiment.seed)

            # time counters
            self.elapsed_init_time = previous_state['elapsed_init_time']
            self.elapsed_fitness_time = previous_state['elapsed_fitness_time']
            self.elapsed_tensor_time = previous_state['elapsed_tensor_time']
            self.elapsed_engine_time = previous_state['elapsed_engine_time']

            self.population = previous_state['population']
            self.best = previous_state['best']

        else:
            self.current_generation = 0

            # time counters
            self.elapsed_init_time = 0
            self.elapsed_fitness_time = 0
            self.elapsed_tensor_time = 0
            self.elapsed_engine_time = 0

            self.experiment.set_generation_directory(self.current_generation)
            self.engine_rng = random.Random(self.experiment.seed)

            _s = time.time()
            self.population, self.best = self.initialize_population(self.max_init_depth,
                                                                self.population_size,
                                                                self.method,
                                                                self.max_nodes,
                                                                immigration = False, read_from_file = read_init_pop_from_file)

            timetaken = time.time() - _s
            print("Init time taken: ", timetaken)




        # for now:
        # fitness - stop evolution if best individual is close enought to target (or given value)
        # generation - stop evolution after a given number of generations
        if self.objective == 'minimizing':
            self.objective_condition = lambda x: (x < self.best['fitness'])
        else:
            self.objective_condition = lambda x: (x > self.best['fitness'])

        if stop_criteria == 'fitness': # while confition
            self.stop_value = stop_value

            if self.objective == 'minimizing':
                self.condition = lambda: (self.best > self.stop_value)
            else:
                self.condition = lambda: (self.best < self.stop_value)
        else:
            self.stop_value = int(stop_value)
            self.condition = lambda: (self.current_generation <= self.stop_value)

        pops = self.population_stats(self.population)


        self.write_pop_to_csv()

        self.elapsed_init_time += time.time() - start_init
        self.update_engine_time()
        self.save_state_to_file(self.experiment.current_directory)

        print("Elapsed init time: ", self.elapsed_init_time)
        if self.debug > 0:
            self.print_engine_state(force_print=True)
        else:
            print("Resolution:\t", self.target_dims)
            print("Seed:\t", self.experiment.seed)
            print("(Avg, Std) fitness:\t" + str(pops['fitness'][0]) + ", " + str(pops['fitness'][1]))
            print("(Avg, Std) depth:\t" + str(pops['depth'][0]) + ", " + str(pops['depth'][1]))

        self.current_generation += 1


    def generate_program(self, method, max_nodes, max_depth, min_depth = -1, root=True):
        terminal = False
        children = []

        # set primitive node
        # (max_depth * max_nodes) == 0 means if we either achieved maximum depth ( = 0) or there are no more
        # nodes allowed on this tree then we have to force the terminal placement
        # to ignore max_nodes, we can set this to -1
        #if (max_depth * max_nodes) == 0 or (
        #        (not root) and method == 'grow' and self.engine_rng.random() < (len(terminal_set) / (len(terminal_set) + len(function_set)))):
        # TODO: review ((self.max_init_depth - max_depth) >= min_depth), works if we call initialize pop with max_init_depth

        if (max_depth * max_nodes) == 0 or (
                (not root) and
                (method == 'grow') and
                ((self.max_init_depth - max_depth) >= min_depth) and
                (self.engine_rng.random() < self.terminal_prob)
        ):
            if self.engine_rng.random() < self.scalar_prob:
                primitive = 'scalar'
                if self.engine_rng.random() < self.uniform_scalar_prob:
                    children = [self.engine_rng.uniform(0, 1) for i in range(self.dimensionality)]
                else:
                    children = [self.engine_rng.uniform(0, 1)]
            else:
                primitive = self.engine_rng.sample(list(terminal_set), 1)[0] # TODO: what is this???
            terminal = True
        else:
            primitive = self.engine_rng.sample(list(function_set), 1)[0]
            if max_nodes > 0: max_nodes -= 1

        # set children
        if not terminal:
            for i in range(function_set[primitive][0]):
                max_nodes, child = self.generate_program(method, max_nodes, max_depth - 1, min_depth = min_depth, root=False)
                children.append(child)

        return max_nodes, Node(value=primitive, terminal=terminal, children=children)

    def generate_population(self, individuals, method, max_nodes, max_depth, min_depth = -1):
        if max_depth < 2:
            if max_depth <= 0:
                return 0, []
            if max_depth == 1:
                method = 'full'
        population = []
        pop_nodes = 0
        if method == 'full' or method == 'grow':
            for i in range(individuals):

                # TODO: why was this without min_depth? TEST
                _n, t = self.generate_program(method, max_nodes, max_depth, min_depth=min_depth)

                tree_nodes = (max_nodes - _n)
                pop_nodes += tree_nodes
                population.append({'tree': t, 'fitness': 0, 'depth': t.get_depth()})
        else:

            # -1 means no minimun depth, but the ramped 5050 default should be 2 levels
            _min_depth = 2 if (min_depth <= 0) else min_depth

            divisions = max_depth - (_min_depth - 1)
            parts = math.floor(individuals / divisions)
            last_part = individuals - (divisions - 1) * parts
            #load_balance_index = (max_depth - 1) - (last_part - parts)
            load_balance_index = (max_depth + 1) - (last_part - parts)
            part_array = []

            num_parts = parts
            mfull = math.floor(num_parts / 2)
            for i in range(_min_depth, max_depth + 1):

                # TODO: shouldn't i - 2 be i - min_depth? TEST or just i
                #if i - 2 == load_balance_index:
                if i == load_balance_index:
                    num_parts += 1
                    mfull += 1
                part_array.append(num_parts)
                met = 'full'
                for j in range(num_parts):

                    if j >= mfull:
                        met = 'grow'
                    _n, t = self.generate_program(met, max_nodes, i, min_depth = min_depth)
                    tree_nodes = (max_nodes - _n)
                    #print("Tree nodes: " + str(tree_nodes))
                    pop_nodes += tree_nodes
                    population.append({'tree': t, 'fitness': 0, 'depth': t.get_depth()})

        if len(population) != individuals:
            print("[ERROR]:\tWrong number of individuals generated: " + str(len(population)) + "/" + str(individuals))

        return pop_nodes, population

    def final_transform(self, final_tensor):
        final_tensor = tf.clip_by_value(final_tensor, clip_value_min=-1, clip_value_max=1)
        final_tensor = tf.math.add(final_tensor, tf.constant(1.0, tf.float32, self.target_dims))
        final_tensor = tf.scalar_mul(127.5, final_tensor)
        final_tensor = tf.cast(final_tensor, tf.uint8)
        return final_tensor

    def final_transform_domain(self, final_tensor):
        final_tensor = tf.clip_by_value(final_tensor, clip_value_min=min_domain, clip_value_max=max_domain)
        final_tensor = tf.math.subtract(final_tensor, tf.constant(min_domain, tf.float32, self.target_dims))
        final_tensor = tf.scalar_mul(255 / domain_delta, final_tensor)
        final_tensor = tf.cast(final_tensor, tf.uint8)
        return final_tensor

    def calculate_tensors(self, population):
        tensors = []

        with tf.device(self.device):
            start = time.time()

            for p in population:
                test_tens = p['tree'].get_tensor(self.target_dims)

                tens = self.final_transform_domain(test_tens)
                tensors.append(tens)


            time_tensor = time.time() - start

        self.elapsed_tensor_time += time_tensor
        self.recent_tensor_time = time_tensor
        return tensors, time_tensor


    def fitness_func_wrap(self, population, f_path):

        # calculate tensors
        if self.debug > 4: print("\nEvaluating generation: " + str(self.current_generation))
        tensors, time_taken = self.calculate_tensors(population)
        if self.debug > 4: print("Calculated " + str(len(tensors)) + " tensors in (s): " + str(time_taken))

        # calculate fitness
        if self.debug > 4: print("Assessing fitness of individuals...")
        _s = time.time()
        # Notes: measuring time should not be up to the fit function writer. We should provide as much info as possible
        # Maybe best pop shouldnt be required
        population, best_pop = self.fitness_func(generation = self.current_generation,
                                                 population = population,
                                                 tensors = tensors,
                                                 f_path = f_path,
                                                 rng = self.engine_rng,
                                                 objective = self.objective,
                                                 resolution = self.target_dims,
                                                 stf = self.save_to_file,
                                                 id_run_seed=self.id_run_seed,#new
                                                 debug = False if (self.debug == 0) else True,
                                                 nameFolderImgs=self.nameFolderImgs,
                                                 model=self.kerasModel,
                                                 n_best_imgs_for_supervisor=self.n_best_imgs_for_supervisor,
                                                 nameHDF5_file=self.nameHDF5_file)
        fitness_time = time.time() - _s

        self.elapsed_fitness_time += fitness_time
        self.recent_fitness_time = fitness_time
        if self.debug > 4: print("Assessed " + str(len(tensors)) + " fitness tensors in (s): " + str(fitness_time))

        return population, best_pop


    def initialize_population(self, max_depth, individuals, method, max_nodes, immigration = False, read_from_file = None):
        #generate trees

        start_init_population = time.time()

        if read_from_file is None:
            nodes_generated, population = self.generate_population(individuals, method, max_nodes, max_depth, self.min_tree_depth)
        else:
            if ".txt" not in read_from_file:
                print("[ERROR]:\tCould not read from file: " + str(read_from_file) + ", randomly generating population instead.")
                read_from_file = None
                nodes_generated, population = self.generate_population(individuals, method, max_nodes, max_depth, self.min_tree_depth)
            else: # read from population files

                # open population files
                strs = []
                with open(read_from_file) as fp:
                    line = fp.readline()
                    cnt = 0
                    while line and cnt < self.population_size:
                        line = line[:-1]
                        strs.append(line)
                        line = fp.readline()
                        cnt += 1
                    if cnt < self.population_size:
                        print("[WARNING]:\tCould only read " + str(cnt) + " expressions from population file " + str(read_from_file) +
                              " instead of specified population size of " + str(self.population_size));

                # convert population to graph
                population = []
                nodes_generated = 0
                maxpopd = -1
                for p in strs:
                    t, node = str_to_tree(p, terminal_set)

                    thisdep = node.get_depth()
                    if thisdep > maxpopd:
                        maxpopd = thisdep

                    population.append({'tree': node, 'fitness': 0, 'depth': node.get_depth()})
                    if self.debug > 0:
                        print("Number of nodes:\t:" + str(t))
                        print(node.get_str())
                    nodes_generated += t
                if self.debug > 0:
                    print("Total number of nodes:\t" + str(nodes_generated))

                if maxpopd > self.max_tree_depth:
                    newmaxpopd = max(maxpopd, self.max_tree_depth)
                    print("[WARNING]:\tMax depth of input trees (" + str(maxpopd) + ") is higher than defined max tree depth (" +
                    str(self.max_tree_depth) + "), clipping max tree depth value to " + str(newmaxpopd))
                    self.max_tree_depth = newmaxpopd

                if self.debug > 0:
                    for p in population:
                        print(p['tree'].get_str())

        tree_generation_time = time.time() - start_init_population

        if self.debug > 0: # 1
            print("Generated Population: ")
            self.print_population(population)

        #calculate fitness
        f_fitness_path = self.experiment.immigration_directory if immigration else self.experiment.current_directory
        population, best_pop = self.fitness_func_wrap(population=population,
                                                      f_path=f_fitness_path)

        total_time = self.recent_fitness_time + self.recent_tensor_time

        if self.debug > 0:  # print info
            print("\nInitial Population: ")
            print("Generation method: " + ("Ramdom expression" if (read_from_file is None) else "Read from file"))
            print("Generated trees in: " + str(tree_generation_time) + " (s)")
            print("Evaluated Individuals in: " + str(total_time) + " (s)")
            print("\nIndividual(s): ")
            print("Nodes generated: " + str(nodes_generated))
            for i in range(individuals):
                print("\nIndiv " + str(i) + ":\nExpr: " + population[i]['tree'].get_str())
                print("Fitness: " + str(population[i]['fitness']))
                print("Depth: " + str(population[i]['depth']))

        return population, best_pop

    def write_pop_to_csv(self):
        fn = self.experiment.current_directory + "gen_" + str(self.current_generation).zfill(5) + "stats.csv"
        with open(fn, mode='w', newline='') as file:
            fwriter = csv.writer(file, delimiter='|')
            for p in self.population:
                fwriter.writerow([p['fitness'], p['depth'], p['tree'].get_str()])

    def population_stats(self, population, fitness = True, depth = True):
        keys = []
        res = {}
        if fitness: keys.append('fitness')
        if depth: keys.append('depth')
        for k in keys:
            res[k] = []
            for p in population:
                res[k].append(p[k])
            _avg = np.average(res[k])
            _std = np.std(res[k])
            _best = self.best[k]
            res[k] = [_avg, _std, _best]
        return res

    def run(self):

        # statistics
        data = []
        pops = self.population_stats(self.population)
        data.append([pops['fitness'][0], pops['fitness'][1], pops['fitness'][2], pops['depth'][0], pops['depth'][1], pops['depth'][2]])
        print("\n[generation, fitness avg, fitness std, fitness best, depth avg, depth std, depth best]\n")
        print("[", self.current_generation - 1, ", ", data[-1][0], ", ", data[-1][1], ", ", data[-1][2], ", ", data[-1][3], ", ", data[-1][4], ", ", data[-1][5], "]")

        while self.condition():

            # Update seed according to generation
            self.engine_rng = random.Random(self.experiment.seed)

            # Set directory to save engine state in this generation
            self.experiment.set_generation_directory(self.current_generation)

            # immigrate individuals
            if self.current_generation % self.immigration == 0:
                immigrants, _ = self.initialize_population(self.max_init_depth,
                                                           self.population_size,
                                                           self.method,
                                                           self.max_nodes,
                                                           immigration = True, read_from_file = None)

                self.population.extend(immigrants)
                self.population = self.engine_rng.sample(self.population, self.population_size)

            # create new population of individuals
            new_population = [self.best]
            max_tree_depth = self.best['depth']

            retrie_cnt = []

            # for each individual, build new population
            for current_individual in range(self.population_size - 1):

                member_depth = float('inf')

                # generate new individual with acceptable depth

                rcnt = 0
                while member_depth > self.max_tree_depth:

                    parent = self.tournament_selection()
                    random_n = self.engine_rng.random()
                    if random_n < self.crossover_rate:
                        parent_2 = self.tournament_selection()
                        indiv_temp = self.crossover(parent['tree'], parent_2['tree'])
                    elif (random_n >= self.crossover_rate) and (random_n < self.crossover_rate + self.mutation_rate):
                        indiv_temp = self.mutation(parent['tree'], self.method)
                    else:
                        indiv_temp = parent['tree']

                    member_depth = indiv_temp.get_depth()
                    rcnt+=1
                retrie_cnt.append(rcnt)

                # add newly formed child
                new_population.append({'tree': indiv_temp, 'fitness': 0, 'depth': member_depth})
                if member_depth > max_tree_depth:
                    max_tree_depth = member_depth

                # print individual
                if self.debug > 10: print("Individual " + str(current_individual) + ": " + indiv_temp.get_str())

            if self.debug >= 4:
                rstd = np.average(np.array(retrie_cnt))
                print("[DEBUG]:\tAverage evolutionary ops retries for generation " + str(self.current_generation) + ": " + str(rstd))


            # calculate fitness of the new population
            new_population, _new_best = self.fitness_func_wrap(population=new_population,
                                                               f_path=self.experiment.current_directory)
            # update best and population
            if self.objective_condition(_new_best['fitness']): self.best = _new_best
            self.population = new_population

            # update engine time
            self.update_engine_time()

            # save engine state
            #if self.save_to_file != 0 and (self.current_generation % self.save_to_file) == 0:
            self.save_state_to_file(self.experiment.current_directory)

            # add population data to statistics
            pops = self.population_stats(self.population)
            data.append([pops['fitness'][0], pops['fitness'][1], pops['fitness'][2], pops['depth'][0], pops['depth'][1], pops['depth'][2]])
            print("[", self.current_generation, ", ", data[-1][0], ", ", data[-1][1], ", ", data[-1][2], ", ", data[-1][3], ", ", data[-1][4], ", ", data[-1][5], "]")

            self.write_pop_to_csv()

            # print engine state
            self.print_engine_state(force_print=False)

            # advance generation
            self.current_generation += 1
            self.experiment.seed += 1

        # write statistics(data) to csv
        self.write_stats_to_csv(data)

        # print final stats
        if self.debug < 0:
            self.print_engine_state(force_print=True)
        else:
            print("\nElapse Engine Time: \t" + str(self.elapsed_engine_time) + " sec.")
            print("Elapse Tensor Time: \t" + str(self.elapsed_tensor_time) + " sec.")
            print("Elapse Fitness Time:\t" + str(self.elapsed_fitness_time) + " sec.")
            #Print into a file
            fileopen = open(self.experiment.working_directory+"TimeElapsedTrack.txt","w+")
            fileopen.write("Elapse Engine Time: \t" + str(self.elapsed_engine_time) + " sec.")
            fileopen.write("\nElapse Tensor Time: \t" + str(self.elapsed_tensor_time) + " sec.")
            fileopen.write("\nElapse Fitness Time:\t" + str(self.elapsed_fitness_time) + " sec.")
            fileopen.close()
            
            
            
            

        if self.save_graphics: self.graph_statistics()
        timings = [self.elapsed_engine_time, self.elapsed_tensor_time, self.elapsed_fitness_time]
        fitn = [[data[k][0] for k in range(len(data))], [data[k][2] for k in range(len(data))]]
        depn = [[data[k][3] for k in range(len(data))], [data[k][5] for k in range(len(data))]]
        print("=" * 84, "\n\n")
        return data, timings, fitn, depn

    def graph_statistics(self):

        if not self.show_graphics:
            matplotlib.use('Agg')

        matplotlib.rcParams.update({'font.size': 16})

        avg_fit = []
        std_fit = []
        best_fit = []
        avg_dep = []
        std_dep = []
        best_dep = []
        lcnt = 0

        with open(self.experiment.working_directory + 'overall_stats.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if self.debug > 10: print(row)
                lcnt += 1
                avg_fit.append(float(row[0]))
                std_fit.append(float(row[1]))
                best_fit.append(float(row[2]))
                avg_dep.append(float(row[3]))
                std_dep.append(float(row[4]))
                best_dep.append(float(row[5]))

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(lcnt), avg_fit, linestyle='-', label="AVG")
        ax.plot(range(lcnt), best_fit, linestyle='-', label="BEST")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Fitness across generations')
        fig.set_size_inches(12, 8)
        plt.savefig(self.experiment.working_directory + 'Fitness.png')
        #plt.savefig(self.nameFolderImgs+'_FitnessTrack.png') #NEW LINE 
        if self.show_graphics: plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(lcnt), avg_dep, linestyle='-', label="AVG")
        ax.plot(range(lcnt), best_dep, linestyle='-', label="BEST")
        ax.set_xlabel('Generations')
        ax.set_ylabel('Depth')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Avg depth across generations')
        fig.set_size_inches(12, 8)
        plt.savefig(self.experiment.working_directory + 'Depth.png')
        #plt.savefig(self.nameFolderImgs+'_DepthTrack.png') #NEW LINE 
        if self.show_graphics: plt.show()
        plt.close(fig)


    def write_stats_to_csv(self, data):
        fn = self.experiment.working_directory + "overall_stats.csv"
        with open(fn, mode='w', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            for d in data:
                fwriter.writerow(d)

    def update_engine_time(self):
        t_ = time.time()
        self.elapsed_engine_time += t_ - self.last_engine_time
        self.last_engine_time = t_

    def save_state_to_file(self, filename):
        #print("[DEBUG]:\t" + filename)
        with open(filename + "log.txt", "w") as text_file:
            try:
                # general info

                text_file.write("Engine state information:\n")
                text_file.write("Population Size: " + str(self.population_size) + "\n")
                text_file.write("Tournament Size: " + str(self.tournament_size) + "\n")
                text_file.write("Mutation Rate: " + str(self.mutation_rate) + "\n")
                text_file.write("Crossover Rate: " + str(self.crossover_rate) + "\n")
                text_file.write("Minimun Tree Depth: " + str(self.max_tree_depth) + "\n")
                text_file.write("Maximun Tree Depth: " + str(self.min_tree_depth) + "\n")
                text_file.write("Initial Tree Depth: " + str(self.max_init_depth) + "\n")
                text_file.write("Population method: " + str(self.method) + "\n")
                text_file.write("Terminal Probability: " + str(self.terminal_prob) + "\n")
                text_file.write("Scalar Probability (from terminals): " + str(self.scalar_prob) + "\n")
                text_file.write("Uniform Scalar (scalarT) Probability (from terminals): " + str(self.uniform_scalar_prob) + "\n")
                text_file.write("Stop Criteria: " + str(self.stop_criteria) + "\n")
                text_file.write("Stop Value: " + str(self.stop_value) + "\n")
                text_file.write("Objective: " + str(self.objective) + "\n")
                text_file.write("Generations per immigration: " + str(self.immigration) + "\n")
                text_file.write("Dimensions: " + str(self.target_dims) + "\n")
                text_file.write("Max nodes: " + str(self.max_nodes) + "\n")
                text_file.write("Debug Level: " + str(self.debug) + "\n")
                text_file.write("Warp mode: " + str(self.warp_mode) + "\n")
                text_file.write("Device: " + str(self.device) + "\n")
                text_file.write("Save to file: " + str(self.save_to_file) + "\n")
                # previous state and others
                text_file.write("Generation: " + str(self.current_generation) + "\n")
                text_file.write("Engine Seed : " + str(self.experiment.seed) + "\n") # redundancy
                text_file.write("Engine ID : " + str(self.experiment.ID) + "\n")     # to check while loading
                text_file.write("Elapse Engine Time: " + str(self.elapsed_engine_time) + "\n")
                text_file.write("Elapse Initiation Time: " + str(self.elapsed_init_time) + "\n")
                text_file.write("Elapse Tensor Time: " + str(self.elapsed_tensor_time) + "\n")
                text_file.write("Elapse Fitness Time: " + str(self.elapsed_fitness_time) + "\n")

                # population
                text_file.write("\n\nPopulation: \n")

                text_file.write("\nBest individual:")
                text_file.write("\nExpression: " + str(self.best['tree'].get_str()))
                text_file.write("\nFitness: " + str(self.best['fitness']))
                text_file.write("\nDepth: " + str(self.best['depth']) + "\n")

                # 4 lines per indiv starting at 32
                text_file.write("\nCurrent Population:\n")
                for i in range(self.population_size):
                    ind = self.population[i]
                    text_file.write("\n\nIndividual " + str(i) + ":")
                    text_file.write("\nExpression: " + str(ind['tree'].get_str()))
                    text_file.write("\nFitness: " + str(ind['fitness']))
                    text_file.write("\nDepth: " + str(ind['depth']))


            except IOError as e:
                print("[ERROR]:\tI/O error while writing engine state ({0}): {1}".format(e.errno, e.strerror))

    def print_population(self, population):
        for i in range(len(population)):
            p = population[i]
            print("\nIndividual " + str(i) + ":")
            print("Fitness:\t" + str(p['fitness']))
            print("Depth:\t" + str(p['depth']))
            print("Expression:\t" + str(p['tree'].get_str()))

    def print_engine_state(self, force_print = False):
        if force_print and self.debug > 0:
            print("\n____________________Engine state____________________")
            if not self.condition(): print("The run is over!")

            print("\nGeneral Info:")
            print("Engine Seed:\t" + str(self.experiment.seed))
            print("Engine ID:\t" + str(self.experiment.ID))
            print("Generation:\t" + str(self.current_generation))

            print("\nBest Individual:")
            print("Fitness:\t" + str(self.best['fitness']))
            print("Depth:\t" + str(self.best['depth']))
            if self.debug > 2:
                print("Expression:\t" + str(self.best['tree'].get_str()))

            #print population
            if self.debug > 10:
                print("\nPopulation:")
                self.print_population(self.population)

            print("\nTimers:")
            print("Elapsed initial time:\t" + str(round(self.elapsed_init_time, 6)) + " s")
            print("Elapsed fitness time:\t" + str(round(self.elapsed_fitness_time, 6)) + " s")
            print("Elapsed tensor time:\t" + str(round(self.elapsed_tensor_time, 6)) + " s")
            print("Elapsed engine time:\t" + str(round(self.elapsed_engine_time, 6)) + " s")
            print("\n____________________________________________________\n")


def build_function_set(fset):
    global arity_fset
    global function_set

    fset = sorted(fset)
    result = {}
    operators_def = {  # arity, function
        'abs': [1, resolve_abs_node],
        'add': [2, resolve_add_node],
        'and': [2, resolve_and_node],
        'clip': [3, resolve_clip_node],
        'cos': [1, resolve_cos_node],
        'div': [2, resolve_div_node],
        'exp': [1, resolve_exp_node],
        'frac': [1, resolve_frac_node],
        'if': [3, resolve_if_node],
        'len': [2, resolve_len_node],
        'lerp': [3, resolve_lerp_node],
        'log': [1, resolve_log_node],
        'max': [2, resolve_max_node],
        'mdist': [2, resolve_mdist_node],
        'min': [2, resolve_min_node],
        'mod': [2, resolve_mod_node],
        'mult': [2, resolve_mult_node],
        'neg': [1, resolve_neg_node],
        'or': [2, resolve_or_node],
        'pow': [2, resolve_pow_node],
        'sign': [1, resolve_sign_node],
        'sin': [1, resolve_sin_node],
        'sqrt': [1, resolve_sqrt_node],
        'sstep': [1, resolve_sstep_node],
        'sstepp': [1, resolve_sstepp_node],
        'step': [1, resolve_step_node],
        'sub': [2, resolve_sub_node],
        'tan': [1, resolve_tan_node],
        'warp': [dim, resolve_warp_node],
        'xor': [2, resolve_xor_node],
    }

    arity_order = {}
    for e in fset:
        fset_list = operators_def.get(e)
        arity = fset_list[0]
        result[e] = fset_list
        if arity not in arity_order:
            arity_order[arity] = []
        arity_order[arity].append(e)

    #print(arity_order)

    arity_fset, function_set = arity_order, result
    #return arity_order, result

def build_terminal_set(dim, dims, dev):
    global terminal_set
    global z_terminal

    result = {}
    with tf.device(dev):
        for i in range(dim - 1):
            digit = i
            name = ""

            while True:
                n = digit % 26
                val = n if n <= 2 else n - 26
                name = chr(ord('x') + val) + name
                digit //= 26
                if digit <= 0:
                    break

            #print("[DEBUG]:\tAdded terminal " + str(name))

            vari = i
            if i < 2: vari = 1 - i
            result[name] = resolve_var_node(np.copy(dims), vari)

    z_terminal = resolve_var_node(np.copy(dims), 2)
    terminal_set = result
    #print(terminal_set)
    return result

weight_num = 1
node_index = 0
function_set = {'max','min','abs','add','and','or','mult','sub','xor','neg','cos','sin','tan','sqrt','div','exp','log', 'warp', 'if', 'pow', 'sign', 'mdist'}
terminal_set = {}
arity_fset = {}
operator_duration = 0
min_domain = -1
max_domain = 1
domain_delta = max_domain - min_domain
dim = 3
z_terminal = 0
