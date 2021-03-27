#!/bin/bash
echo "SCRIPT SHELL STARTED..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow_gpuenv

echo "start python script..."
python /home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/kerasApp_example.py --jsonFile /home/gama/BolsaStuff/EML_mainRep/DeepEfective/TensorGP-master/params_kerasApp_engine.json





