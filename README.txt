Last update : A. Basile, uOttawa VRS, 30_01_2025

----------------------------------------------------- CONTEXT

This Git repository aims to be the workingplace of the PFI (patellofemoral instability) automatic segmentation and simulation project, part of Annagh's Thesis proposal, directed by Allison clouthier for the uOttawa.

The goal is to develop a ready-to-use software that can predict the risks of PFI for a specific young person from its MRI data.

To do so, the software will perform an automatic-segmentation made by a Machine Learning model, trained on a specific dataset of 41 teenager's knee MRIs with background of knee injuries (PFI, ACL...). Then a few kinematic simulation will be done from the previously segmented musculoskeletal model in order to assess the risk of PFI and/or PFI-specific kinematics.

Read the following thesis for more information : INSERT ANNAGH'S THESIS

------------------------------------------------- HOW TO USE

IMPORTANT to use the project, create a CONDA VirtualEnvironment named ".venv_conda" in the root directory, based on Python3.7 with the pip requirement from the requirement.txt file. To summarize, run the Following commands in the root project directory: 
	> conda env create --prefix .venv_conda -f environment.yml
	> conda activate .\.venv_conda
	> pip install gias2 # Dependency trick, ignore error message.
	> conda install numpy==1.20.1  # Dependency trick, ignore error message.
	> cd Scripts
	> python main.py

ADDITIONALLY the trained MPUnet model's files and the jam plugin (necessary for the insertion process) can't be pushed on github, please contact me so I could forward you these files: antoine.basile753@outlook.fr

==> Compatible with CUDA10.1 Cudnn7, have all the necessary libraries and GPUs (nvidia) toolkits installed.

Then SIMPLY RUN THE main.py SCRIPT 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DOCKER SETTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You might need to use a Docker to set up your own version of CUDA10.1 Cudnn7 if your system doesn't allow another/several CUDA versions for the training/predicting phases (options 2 and 3), to do so process the Dockerfile present in the project root folder with the following commands:
	> docker build -t project_name
	> docker run --gpus all -it project_name 
Then run the main.py script within the running docker.
Also to eventually import/export Data inside the running Docker or export results outside, use the follwing command:
	> docker cp source_folder destination_folder 
	(ex:> docker cp PFI_Autosegmentation_project/Data/ 7b1d664921c7:container to import Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

------------------------------------------------- WHAT IT CAN DO

	1) Pre-process your dataset of manually segmented MRI (in order to train a model)
	2) Train a model on your data
	3) Perform auto-segmentations based on a choosen model
	4) Apply a morphing process to segmentations AND propose to switch the model's parts from volume to shell-type mesh
	5) Insert the segmented model within a generic .osim musculoskeletic model (NOT IMPLEMENTED)
	6) Perform simulations and generate outputs (NOT IMPLEMENTED)



Please feel free to contact me if you have any question on my email adress: antoine.basile753@outlook.fr
Cheers.