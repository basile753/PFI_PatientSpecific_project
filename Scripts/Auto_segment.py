import os
import subprocess
import sys

global mp_path

#You might need to change the path if you virtual environment is set up differently
mp_path = "../.venv_conda/Lib/site-packages/mpunet/bin/mp.py"

def predict_mpunet_3dunet(im_path: str, mod_path: str, n_gpus, model_type: str):
    """
    Thus function aims to launch the auto-segmentation process of a fusion Mpunet or a 3Dunet model.
    :param im_path: Path to images folder to predict
    :param mod_path: Path to the model
    """
    #-----------Inputs & parameters
    with open(f"{mod_path}/train_hparams.yaml", 'r') as file:
        content = file.readlines()
        file.close()
    content[58]=f"  base_dir: {im_path}\n"
    with open(f"{mod_path}/train_hparams.yaml", 'w') as file:
        file.writelines(content)
        file.close()

    output = f"{im_path}/results"
    eval_flag = input("Would you like to evaluate auto-segmentation ? (y/n): ")
    if model_type == "Fusion":
        prediction_type = "predict"
    elif model_type == "3D":
        prediction_type = "predict_3D"

    #-------------Prediction---------------
    if eval_flag == "y":
        print(f"Prediction of the images in {im_path} with {mod_path} with evaluation")
        subprocess.run([
            sys.executable,  # Path to the Python interpreter
            mp_path,  # Path to the mp.py script
            prediction_type,
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
            prediction_type,
            "--project_dir", mod_path,  # Name argument
            f"--num_GPUs={n_gpus}",  # Amount of GPUs to use
            "--out_dir", output,
            "--no_eval",
            "--overwrite"
        ], check=True)


def entry():
    """
    This is the entry function of the auto-segmentation script. Please use it if you wish to perform predictions (an automated segmentation).
    """
    #Paths to images and model
    images_path = input("Enter the ABSOLUTE path to the images you would like to auto-segment, Must contain sub-folder 'images'"
                        " with the .nii.gz images (default: /container/Data/RMIs/to_predict): ")
    if images_path == "":
        images_path = "/container/Data/RMIs/to_predict"
    if not os.path.exists(images_path):
        raise FileExistsError("The path you entered does not exist")
    model = input("Enter the name of the model you would like to use (default: MPunet_41DATA_split2): ")
    if model == "":
        model = "MPunet_41DATA_split2"
    model_path = f'../Models/{model}'
    if not os.path.exists(model_path):
        raise FileExistsError("The model path you entered does not exist: "+model_path)

    model_type = input('What kind of model would you like to use for prediction: [1: 3D-Unet, 2: Fusion-MPUnet]: ')
    if model_type == "1":
        model_type = "3D"
        print("WARNING: to perform 3Dunet prediction, please REMOVE the .numpy() at the line 258 of the script "
              "mpunet/utils/fusion/fuse_and_predict.py, the module is not updated.")
    elif model_type == "2":
        model_type = "Fusion"
    else:
        raise ValueError("Model type input is incorrect")

    #Parameters
    n_gpus = input("How many GPUs would you like to use? (default: 1): ")
    if n_gpus == "":
        n_gpus = 1
    predict_mpunet_3dunet(images_path, model_path, n_gpus, model_type)

if __name__ == "__main__":
    entry()
