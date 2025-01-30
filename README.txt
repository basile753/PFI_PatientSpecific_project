Last update : A. Basile, uOttawa VRS, 30_01_2025

----------------------------------------------------- CONTEXT

This Git repository aims to be the workingplace of the PFI (patellofemoral instability) automatic segmentation and simulation project, part of Annagh's Thesis proposal, directed by Allison clouthier for the uOttawa.

The goal is to develop a ready-to-use software that can predict the risks of PFI for a specific young person from its MRI data.

To do so, the software will perform an automatic-segmentation made by a Machine Learning model, trained on a specific dataset of 41 teenager's knee MRIs with background of knee injuries (PFI, ACL...). Then a few kinematic simulation will be done from the previously segmented musculoskeletal model in order to assess the risk of PFI and/or PFI-specific kinematics.

Read the following thesis for more information : INSERT ANNAGH'S THESIS

------------------------------------------------- HOW TO USE

IMPORTANT to use the project, create a VirtualEnvironment named ".venv37" right in the root directory based on Python3.7 with the requirement from the requirement.txt file.
==> Compatible with CUDA10.1 Cudnn7, have all the necessary libraries and GPUs (nvidia) toolkits installed.

Then SIMPLY RUN THE main.py SCRIPT

You might need to use Docker to set up your own version of CUDA10.1 Cudnn7, the Dockerfile present in the project in functionnal and ready-to-use.

------------------------------------------------- WHAT IT CAN DO

	1) Pre-process your dataset of manually segmented MRI (in order to train a model)
	2) Train a model on your data
	3) Perform auto-segmentations based on a choosen model
	4) Apply a morphing process to segmentations
	5) Insert the segmented model within a generic .osim musculoskeletic model (NOT IMPLEMENTED)
	6) Perform simulations and generate outputs (NOT IMPLEMENTED)


