import os
import subprocess
import sys
import shutil

global mp_path
global script_path

#You might need to change the path if you virtual environment is set up differently
#On windows usually :
#mp_path = "../.venv37/Lib/site-packages/mpunet/bin/mp.py"
#On a linux venv :
mp_path = "../.venv37/lib/python3.7/site-packages/mpunet/bin/mp.py"
script_path = os.path.abspath(__file__)

def simple_train_mpunet(data_dir, root, name, project_dir):

    mp_path_from_modeldirectory = f"../{mp_path}"

    # --------INPUT FIXED HYPERPARAMETERS----------
    print("By default n_classes: 10, loss function : SparseGeneralizedDiceLoss, batch size: 16, modify it directly in "
          "Scripts/Training.py")
    n_epochs = input("Enter the number of epochs (default : 300) : ")
    if n_epochs == '':
        n_epochs = 300
    n_gpus = input("How many GPUs would you like to use (default : 1): ")
    if n_gpus == '':
        n_gpus = 1

    # ------------INITIALISATION-------------
    subprocess.run([
        sys.executable,  # Path to the Python interpreter
        mp_path,  # Path to the mp.py script
        "init_project",  # Command to initialize the project
        "--name", name,  # Name argument
        "--data_dir", data_dir,  # Data directory argument
        "--root", root  # Root directory argument
    ], check=True)

    # --------SETTING FIXED HYPERPARAMETERS--------------
    with open(f"{project_dir}/train_hparams.yaml", 'r') as file:
        content = file.read()
        file.close()
    # Gain of time (otherwise the algorythm will determine itself)
    updated_content = content.replace("n_classes: Null", "n_classes: 10")

    # Here change the loss function or the evaluation metric if you need.
    updated_content = updated_content.replace('loss: "SparseCategoricalCrossentropy"', 'loss: "SparseGeneralizedDiceLoss"')
    updated_content = updated_content.replace('metrics: ["sparse_categorical_accuracy"]', 'metrics: ["sparse_categorical_accuracy"]')

    # Here enter the number of batch size and epochs (WARNING significantly influence the calculation time)
        # batch_size refers to the amount of data processed at once for noise reduction purpose
    updated_content = updated_content.replace('batch_size: 16',
                                              'batch_size: 16')
        # n_epochs refers to the amount of time the entire dataset is passed through the model
            #Note: Because of shuffling, Augmenters and the generation of 6 different views, the actual amount of data
            #is actually way higher that the amount of sample originally provided (EX 28 images ==> 2528 data).
    updated_content = updated_content.replace('n_epochs: 500',
                                              f'n_epochs: {n_epochs}')
    with open(f"{project_dir}/train_hparams.yaml", 'w') as file:
        file.write(updated_content)
        file.close()

    # -----------MPUNET TRAINING (2 phases)-------------
    print(f"\nTraining the model {name}... This action may take a while...")
    os.chdir(project_dir)
    subprocess.run([
        sys.executable,
        mp_path_from_modeldirectory,
        "train",
        "--overwrite",
        f"--num_GPUs={n_gpus}"
    ], check=True)
    subprocess.run([
        sys.executable,
        mp_path_from_modeldirectory,
        "train_fusion",
        "--overwrite",
        f"--num_GPUs={n_gpus}"
    ], check=True)
    os.chdir(script_path)
    print(f"{name}'s training is done.")

def entry(root="../Models"):
    """
    This is the entry function of the training script
    """
    print("IMPORTANT NOTE : If you wish to perform cross-validation training, please perform each split separately as simple "
          "train/val processes")
    data_dir = input("Enter the data directory (default : ../Data/RMIs/data_folder): ")
    if data_dir == '':
        data_dir = "../Data/RMIs/data_folder"
    model = input("Enter the model you would like to train between these [MPunet: 1, NNunet: NOT READY] : ")
    if model == '1':
        model = "MPunet"
    train = len(os.listdir(data_dir+"/train/images"))
    val = len(os.listdir(data_dir + "/val/images"))
    type = input("Enter the split number or let it blank if this is a simple training without cross validation : ")
    if type == "":
        type = "simple"
    else:
        type = "split"+type
    name = f'{model}_{train}train_{val}val_{type}'
    project_dir = f"{root}/{name}"
    os.makedirs(project_dir, exist_ok=True)
    if os.path.exists("../Logs/data_preprocessing_log.txt"):
        shutil.copy("../Logs/data_preprocessing_log.txt", f'{project_dir}/data_preprocessing_log.txt')

    if model == "MPunet":
        simple_train_mpunet(data_dir, root, name, project_dir)








