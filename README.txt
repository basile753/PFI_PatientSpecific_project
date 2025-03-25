Last update : A. Basile, uOttawa VRS, 14_02_2025

----------------------------------------------------- CONTEXT

This Git repository aims to be the workingplace of the PFI (patellofemoral instability) automatic segmentation and simulation project, part of Annagh's Thesis proposal, directed by Allison clouthier for the uOttawa.

This is a ready-to-use program that can generate a patient-specific MSK model from the patient's MRI CT-scan images.

To do so, the software can train, test and evaluate a deep-learning model for full knee segmentation or use an existing one, then perform a full automatic-segmentation of the knee made by the deep learning model. Finally after going through a few transformations and morphing process, the patient-specific knee mesh is inserted within the SMITH2019 generic model, ready for further simulations.

Read the following thesis for more information : INSERT ANNAGH'S THESIS

------------------------------------------------- HOW TO USE

IMPORTANT to use the project, you'll need to create a conda environment, to do so run the Following commands in the root project directory: 

	> conda env create --prefix .venv_conda -f environment_windows.yml
	> conda activate .\.venv_conda
	> pip install gias2 # Dependency trick, ignore error message.
	> conda install numpy==1.20.1  # Dependency trick, ignore error message.
	> cd Scripts

THEN there are 2 manual tasks to do:
	
	1- unzip the "jam-plugin" folder at "PFI_PatientSpecific_project\Scripts\Morphing_scripts\insertion"
	2- Replace the line 4 of the __init__.py script of the opensim library (at this path: PFI_PatientSpecific_project\.venv_conda\lib\opensim\__init__.py) by these ones:
			> dll_directory = os.path.dirname(os.path.realpath(__file__))
			> os.environ['PATH'] = dll_directory + os.pathsep + os.environ['PATH']


ADDITIONALLY for the auto-segmentation task, if you wish to use one of the model we trained on our PFI/ACL dataset of 41 youth patients, please contact me so I could forward you these files: antoine.basile753@outlook.fr.

FOR THE TRAINING/TESTING/AUTOSEGMENTATION (options 2 & 3 of the program) ==> Depends on CUDA10.1 Cudnn7, have these libraries and GPUs (nvidia) toolkits installed, and set up a docker if these are incompatible with your system (see below for docker instructions).

Then SIMPLY RUN THE main.py SCRIPT:
	
	> python main.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DOCKER SETTING / LINUX ONLY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You might need to use a Docker to set up your own version of CUDA10.1 Cudnn7 if your system doesn't allow another/several CUDA versions for the training/predicting phases (options 2 and 3).

To do so, download and activate Docker on your machine, then process the Dockerfile present in the project root directory with the following commands:

	> docker build -t project_name .
	> docker run --gpus all -it project_name 

Then enter the created docker and activate the conda venv.
Then run the main.py script within the running docker.
Also to eventually import/export Data inside the running Docker or export results outside, use the follwing command:

	> docker cp source_folder destination_folder 
	(ex:> docker cp PFI_Autosegmentation_project/Data/ 7b1d664921c7:container to import Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

------------------------------------------------- WHAT IT CAN DO

         1: Pre-process a dataset of manually segmented MRI."
         2: Train and evaluate a model on your data (REQUIRES '1')."
         3: Perform auto-segmentations on MRI-DICOM based on a choosen model (REQUIRES a model, either using '2' or downloading a trained one)."
         4: Apply a morphing process to segmentations."
         5: Insert a segmented model within a generic .osim musculoskeletic model (REQUIRES '4') (IN PROGRESS)."
         (6: Perform simulations and generate outputs (NOT IMPLEMENTED).")

Note that some options might require previous options to be processed before (see description)"



Please feel free to contact me if you have any question on my email adress: antoine.basile753@outlook.fr
Cheers.