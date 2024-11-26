import os
import subprocess
import sys

global mp_path

#You might need to change the path if you virtual environment is set up differently
#On windows usually :
#mp_path = "../.venv37/Lib/site-packages/mpunet/bin/mp.py"
#On a linux venv :
mp_path = "../.venv37/lib/python3.7/site-packages/mpunet/bin/mp.py"

def predict_mpunet(im_path: str, mod_path: str, n_gpus):
    """
    Thus function aims to launch the auto-segmentation process of a fusion Mpunet model
    :param im_path: Path to images folder to predict
    :param mod_path: Path to the model
    """
    with open(f"{mod_path}/train_hparams.yaml", 'r') as file:
        content = file.readlines()
        file.close()
    content[58]=f"  base_dir: {im_path}\n"
    with open(f"{mod_path}/train_hparams.yaml", 'w') as file:
        file.writelines(content)
        file.close()

    output = f"{im_path}/results"
    eval_flag = input("Would you like to evaluate auto-segmentation ? (y/n): ")
    if eval_flag == "y":
        print(f"Prediction of the images in {im_path} with {mod_path} with evaluation")
        subprocess.run([
            sys.executable,  # Path to the Python interpreter
            mp_path,  # Path to the mp.py script
            "predict",
            "--project_dir", mod_path,  # Name argument
            f"--num_GPUs={n_gpus}", #Amount of GPUs to use
            "--out_dir", output,
            "--overwrite"
        ], check=True)
    else:
        print(f"Prediction of the images in {im_path} with {mod_path} without evaluation")
        subprocess.run([
            sys.executable,  # Path to the Python interpreter
            mp_path,  # Path to the mp.py script
            "predict",
            "--project_dir", mod_path,  # Name argument
            f"--num_GPUs={n_gpus}",  # Amount of GPUs to use
            "--out_dir", output,
            "--no_eval",
            "--overwrite"
        ], check=True)


def entry():
    """
    This is the entry function of the auto-segmentation script. Please use it if you wish to perfom predictions.
    """
    #Paths to images and model
    images_path = input("Enter the ABSOLUTE path to the images you would like to auto-segment, Must contain sub-folder 'images'"
                        " with the .nii.gz images (default: /container/Data/RMIs/to_predict): ")
    if images_path == "":
        images_path = "/container/Data/RMIs/to_predict"
    #if not os.path.exists(images_path):
    #    raise FileExistsError("The path you entered does not exist")
    model = input("Enter the name of the model you would like to use (default: MPunet_41DATA_split1): ")
    if model == "":
        model = "MPunet_41DATA_split1"
    model_path = f'../Models/{model}'
    if not os.path.exists(model_path):
        raise FileExistsError("The model path you entered does not exist: "+model_path)

    #Parameters
    n_gpus = input("How many GPUs would you like to use? (default: 1): ")
    if n_gpus == "":
        n_gpus = 1
    if model.startswith("MPunet_"):
        predict_mpunet(images_path, model_path, n_gpus)

