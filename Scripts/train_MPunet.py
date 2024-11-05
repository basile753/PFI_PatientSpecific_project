import os
import subprocess
import sys

global mp_path
global script_path

#You might need to change the path if you virtual environment is set up differently
#On windows usually :
#mp_path = "../.venv37/Lib/site-packages/mpunet/bin/mp.py"
#On a linux venv :
mp_path = "../.venv37/lib/python3.7/site-packages/mpunet/bin/mp.py"
script_path = os.path.abspath(__file__)

def simple_train_mpunet(data_dir="../Data/RMIs/data_folder", root="../Models"):
    name = input("Enter the name of the model respecting the following naming rules\n"
                 "\tExample 1 : MPUNET_RAND_30TR_15TE_03 --> means that the model is a multiplanar model (MPUNET) trained on a random sample (RAND) of 30 patients (30TR) and tested on 15 others (15TE). Finally it's the third of its kind (03)\n"
                 "\tExample 2 : MPUNET_5CV_30TR_15TE --> means the model had been tested in a 5 group cross-validation process (5CV) on groups of 30 training data and 15 testing data"
                 "\n\t Enter name here : ")
    project_dir = f"{root}/{name}"
    mp_path_from_modeldirectory = f"../{mp_path}"
    # Run mp.py with separate arguments
    subprocess.run([
        sys.executable,  # Path to the Python interpreter
        mp_path,  # Path to the mp.py script
        "init_project",  # Command to initialize the project
        "--name", name,  # Name argument
        "--data_dir", data_dir,  # Data directory argument
        "--root", root  # Root directory argument
    ], check=True)
    # Setting the number of classes on 9 (9 differents segments + the background class)
    with open(f"{project_dir}/train_hparams.yaml", 'r') as file:
        content = file.read()
        file.close()
    # Gain of time (otherwise the algorythm will determine itself)
    updated_content = content.replace("n_classes: Null", "n_classes: 10")
    #
    updated_content = updated_content.replace('loss: "SparseCategoricalCrossentropy"', 'loss: "SparseGeneralizedDiceLoss"')
    updated_content = updated_content.replace('metrics: ["sparse_categorical_accuracy"]', 'metrics: ["sparse_categorical_accuracy"]')
    with open(f"{project_dir}/train_hparams.yaml", 'w') as file:
        file.write(updated_content)

    print(f"\nTraining the model {name}... This action may take a while...")
    os.chdir(project_dir)
    subprocess.run([
        sys.executable,
        mp_path_from_modeldirectory,
        "train",
        "--overwrite",
        "--num_GPUs=1",
        "--no_images",
        "--force_GPU", "0"
    ], check=True)
    subprocess.run([
        sys.executable,
        mp_path_from_modeldirectory,
        "train_fusion",
        "--overwrite",
        "--num_GPUs=1",
        "--no_images",
        "--force_GPU", "0"
    ], check=True)
    os.chdir(script_path)
    print(f"{name}'s training is done.")

def cv_train_mpunet(data_dir="../Data/RMIs/data_folder", root="../Models"):
    name = input("Enter the name of the model respecting the following naming rules\n"
                 "\tExample 1 : MPUNET_RAND_30TR_15TE_03 --> means that the model is a multiplanar model (MPUNET) trained on a random sample (RAND) of 30 patients (30TR) and tested on 15 others (15TE). Finally it's the third of its kind (03)\n"
                 "\tExample 2 : MPUNET_5CV_30TR_15TE --> means the model had been tested in a 5 group cross-validation process (5CV) on groups of 30 training data and 15 testing data"
                 "\n\t Enter name here : ")
    mp_path_from_modeldirectory = f"../../{mp_path}"
    for split in os.listdir(data_dir):
        data_dir_split = f"{data_dir}/{split}/data_folder"
        project_dir = f"{root}/{name}"
        os.makedirs(project_dir, exist_ok=True)
        split_dir = f"{project_dir}/{split}"
        subprocess.run([
            sys.executable,  # Path to the Python interpreter
            mp_path,  # Path to the mp.py script
            "init_project",  # Command to initialize the project
            "--name", split,  # Name argument
            "--data_dir", data_dir_split,  # Data directory argument
            "--root", project_dir  # Root directory argument
        ], check=True)
        # Setting the number of classes on 9 (9 differents segments + the background class)
        with open(f"{split_dir}/train_hparams.yaml", 'r') as file:
            content = file.read()
            file.close()
        updated_content = content.replace("n_classes: Null", "n_classes: 10")
        with open(f"{split_dir}/train_hparams.yaml", 'w') as file:
            file.write(updated_content)
        os.chdir(split_dir)
        print(f"\nTraining the model {name} using {split}... This action may take a while...")
        subprocess.run([
            sys.executable,
            mp_path_from_modeldirectory,
            "train",
            "--overwrite",
            "--num_GPUs=1",
            "--no_images"
        ], check=True)
        subprocess.run([
            sys.executable,
            mp_path_from_modeldirectory,
            "train_fusion",
            "--overwrite",
            "--num_GPUs=1",
            "--no_images"
        ], check=True)
        print(f"{name}'s training on {split} is done.")
        os.chdir(script_path)

def entry():
    if input("Do you want to train the model in cross-validation? (y/n) ") == "y":
        cv_train_mpunet()
    else:
        simple_train_mpunet()



