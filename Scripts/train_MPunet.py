import os
import subprocess
import sys

global mp_path
global mp_path_from_modeldirectory
mp_path = "../.venv37/Lib/site-packages/mpunet/bin/mp.py"
mp_path_from_modeldirectory = "../../.venv37/Lib/site-packages/mpunet/bin/mp.py"

def train_mpunet(data_dir="../Data/RMIs/data_folder", root="../Models"):
    name = input("Enter the name of the model respecting the following naming rules\n"
                 "\tExample 1 : MPUNET_RAND_30TR_15TE_03 --> means that the model is a multiplanar model (MPUNET) trained on a random sample (RAND) of 30 patients (30TR) and tested on 15 others (15TE). Finally it's the third of its kind (03)\n"
                 "\tExample 2 : MPUNET_5CV_30TR_15TE --> means the model had been tested in a 5 group cross-validation process (5CV) on groups of 30 training data and 15 testing data"
                 "\n\t Enter name here : ")
    project_dir = f"{root}/{name}"
    # Run mp.py with separate arguments
    subprocess.run(f"{sys.executable} {mp_path} init_project --name {name} --data_dir {data_dir} --root {root}", check=True)
    os.chdir(project_dir)
    print(f"\nTraining the model {name}... This action may take a while...")
    subprocess.run(f"{sys.executable} {mp_path_from_modeldirectory} train --overwrite --num_GPUs=0", check=True,)
    subprocess.run(f"{sys.executable} {mp_path_from_modeldirectory} train_fusion --overwrite --num_GPUs=0", check=True,)
    print(f"{name}'s training is done.")
    #test_mpunet(name, project_dir)

def test_mpunet(name: str, project_dir: str):
    print(f"\nTesting the model {name}... This action may take a while...")
    subprocess.run(f"{sys.executable} {mp_path} predict --num_GPUs=0 --out_dir predictions", check=True)
    print(f"{name}'s testing is done. See 'predictions' file for the results")
    return

