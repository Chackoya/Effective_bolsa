import copy
import datetime
import json
import os
import time
from glob import glob

import cv2
import numpy as np
import sys
import random

from image_path_selector import rmse

'''
Implements X-Faces Engine
'''


class EFECTIVE():

    def __init__(self):

        self.images_to_gather = None
        self.gathered_imgs = 0
        self.ec_run_setup = None
        self.ec_run_stagnation = 0
        self.current_instances = []
        self.current_instances_bbox = []

    def execute(self):
        print('Objective: gather ', self.images_to_gather)
        while self.termination_criteria():

            ec_run, ec_callback = self.perform_ec_run()

            ec_run_imgs = [ec_run.composites[key]['img'] for key in ec_run.composites]
            #ec_run_gt = [ec_run.composites[key]['ground_truth'] for key in ec_run.composites]
            #ec__run_names = [ec_run.composites[key]['img_path'].replace('/','-') for key in ec_run.composites]
            print('ec_run_imgs : ', len(ec_run_imgs))
            # print(ec_run_gt)
            supervised_indexes = self.supervise(ec_run_imgs)

            # update saved set
            list_imgs = [ec_run_imgs[i].copy() for i in supervised_indexes]
            print('list_imgs', len(list_imgs))
            list_bbox_str = [str(ec_run_gt[i]) for i in supervised_indexes]
            list_name_str = [ec__run_names[i] for i in supervised_indexes]

            # fixme fazer dict e colocar para json... e permitir exportar
            self.export_instances(list_name_str, list_imgs, list_bbox_str)

            for img in list_imgs:
                self.current_instances.append(img)
            #for bbox in list_bbox_str:
            #    self.current_instances_bbox.append(bbox)
            self.gathered_imgs += len(supervised_indexes)

            if len(supervised_indexes) == 0:
                print('didnt gather images.. ', self.ec_run_stagnation)
                self.ec_run_stagnation += 1
            else:
                self.ec_run_stagnation = 0

            print('Next seed...')
            print(self.gathered_imgs, self.images_to_gather)
            print('current instances ', len(self.current_instances), 'current bboxes ',
                  len(self.current_instances_bbox))
            # next random seed
            self.ec_run_setup.random_seed += 1
            #exit(0)
        print("execute terminated")

    def initialize(self, start_seed, classifier_weights):
        # fixme check...
        self.gathered_imgs = 0
        self.images_to_gather = 10
        self.classifier_weights = classifier_weights
        a_setup = Setup()
        a_setup.generations = 1
        a_setup.multiprocesses = 1
        a_setup.population_size = 10
        a_setup.random_seed = start_seed
        a_setup.torn_size = 2
        self.ec_run_setup = a_setup
        self.export_path = 'xfaces-supervised-' + str(timestamp)

    def load(self, import_path):
        self.export_path = import_path  # "xfaces-supervised-2019-02-05-023016"
        self.current_instances = [cv2.imread(im) for im in glob(import_path + '/*')]
        self.gathered_imgs += len(self.current_instances)
        print('loaded', len(self.current_instances))

    def termination_criteria(self):
        return (self.gathered_imgs < self.images_to_gather) or self.ec_run_stagnation == 3

    def perform_ec_run(self):

        ##############
        ##### Executa EC run  vai buscar as imagens novas!
        ##############
        #evo_engine = XfacesEngine()
        # os.makedirs(str(self.ec_run_setup.random_seed), exist_ok=True)
        # Configure XFaces
        #xfaces_cb = Xfaces2019()
        #xfaces_cb.set_setup(self.ec_run_setup)
        #xfaces_cb.set_model(evo_engine)
        #xfaces_cb.input_w = 416
        #xfaces_cb.input_h = 416
        #xfaces_cb.max_to_load = -1
        # 0 - is the base
        #xfaces_cb.parts = {0: 'face', 1: 'brow-left', 2: 'brow-right', 3: 'eye-left', 4: 'eye-right', 5: 'nose',
        #                   6: 'mouth'}
        #xfaces_cb.pairing_eyes = True
        #xfaces_cb.pairing_brows = True

        # 1  - initialize callback
        # path to images
        # yolo config
        # model .h5
        #xfaces_cb.init('celebA2000-data-20190201-165851', 'E0.json',
        #               self.classifier_weights)
        #
        # print(xfaces_cb.faces_data)
        #evo_engine.faces_data = xfaces_cb.faces_data

        # 2  - initialize engine and evolve
        #evo_engine.initialize(setup=self.ec_run_setup, individual_ref=XfacesIndividual2019, selection_ref=Tournament,
        #                      evaluation_ref=xfaces_cb.composite_eval, callbacks=[xfaces_cb])
        #evo_engine.evolve(callbacks=[xfaces_cb])

        return evo_engine, xfaces_cb

    def supervise(self, imgs, threshold=1.0):
        # check if they are different
        tmp_set = []
        if len(self.current_instances) == 0:
            return list(range(len(imgs)))

        # percorre todas as imagens e seleciona
        for i in range(len(imgs)): 
            already_in_set = False
            for curr_img in self.current_instances:
                # 
                if rmse(imgs[i], curr_img) >= threshold:
                    already_in_set = True
                    break
            if not already_in_set:
                tmp_set.append(i)

        return tmp_set

    def export_instances(self, ec_run_imgnames, ec_run_imgs, ec_run_gt):
        os.makedirs(self.export_path, exist_ok=True)
        ### Exportar para disco com uma descrição
        with open(self.export_path + '.txt', 'a') as f:
            for indiv in range(len(ec_run_imgs)):
                img = ec_run_imgs[indiv]
                #box = ec_run_gt[indiv] # nao existe bounding box neste caso
                ref_name = ec_run_imgnames[indiv]
                imname = os.path.join(self.export_path,
                                      str(len(self.current_instances) + indiv) + '-' + ref_name + '.jpg')
                cv2.imwrite(imname, img)
                # print(box)
                
                #box_s = box.split()
                #f.write('{0} 1 {1} {2} {3} {4}\n'.format(imname, box_s[0], box_s[1], box_s[2], box_s[3]))



global timestamp

if __name__ == '__main__':

    start = time.time()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')

    if len(sys.argv) != 1 and sys.argv[1] == 'ef':
        efective = EFECTIVE()
        efective.initialize(int(sys.argv[2]),sys.argv[3])
        if len(sys.argv) == 5:
            print('loading previous supervisor state',sys.argv[4])
            efective.load(sys.argv[4])
        efective.execute()
    
        
