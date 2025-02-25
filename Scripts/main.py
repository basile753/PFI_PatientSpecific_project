"""
This is the main script of the project, it aims to interact with the other scripts in order to pre-process the data,
train and evaluate models.
@author: Antoine Basile
"""
import os
import Data_preprocessing
import Training
import Auto_segment
import Auto_morphing
import Insertion_MSK_model

def main(root: str):
    #-----Initialisation---------------------
    flag = True
    print("\n\n\t\tWelcome in the PFI patient-specific project script!\n\n"
          "Here you can process your patient's MRI data to create a specific finite element model and perform rigid bodies simulations.\n"
          "The present auto-segmentation models (and mean shapes for the morphing) are trained on a dataset of teenagers with ACL or PFI, \noutside of these bounds you should"
          " train your own models using this program and change the reference files for the morphing\n")
    while flag == True:
        choice = input("\nChoose an action you would like to perform:"
                       "\n\t1: Pre-process a dataset of manually segmented MRI (in order to train a model)."
                       "\n\t2: Train and evaluate a model on your data."
                       "\n\t3: Perform auto-segmentations on MRI-DICOM based on a choosen model."
                       "\n\t4: Apply a morphing process to segmentations."
                       "\n\t5: Insert a segmented model within a generic .osim musculoskeletic model (IN PROGRESS)."
                       "\n\t6: Perform simulations and generate outputs (NOT IMPLEMENTED)."
                       "\n\n ---> ")



        #-----Pre-processing---------------------
        if choice == "1":
            Data_preprocessing.entry()
            os.chdir(root)
        #---------------------------------------

        #-----Training the model---------
        if choice == "2":
            Training.entry()
            os.chdir(root)
        # ---------------------------------------

        #-----Auto-segment---------
        if choice == "3":
            Auto_segment.entry()
            os.chdir(root)
        # ---------------------------------------

        #-----Auto-morphing-------
        if choice == "4":
            Auto_morphing.entry()
            os.chdir(root)
        # ---------------------------------------

        #-----Insertion/Simulation-------
        if choice == "5":
            Insertion_MSK_model.entry()
            os.chdir(root)
        # ---------------------------------------

        if input("Would you like to perform another action? (y/n): ") != "y":
            flag = False

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root) #Change the working directory to PFI_Autosegmentation_project/Scripts
    main(root)