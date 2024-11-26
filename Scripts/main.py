"""
This is the main script of the project, it aims to interact with the other scripts in order to pre-process the data,
train and evaluate models.
"""
import os
import Data_preprocessing
import Training
import Auto_segment

def main():
    #-----Pre-processing---------------------
    if input("Would you like to pre-process data ? (y/n) ") == "y":
        Data_preprocessing.entry()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #---------------------------------------

    #-----Training the model---------
    if input("Would you like to train a model on the sorted data ? (y/n) ") == "y":
        Training.entry()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # ---------------------------------------

    #-----Auto-segment---------
    if input("Would you like to perform auto-segmentation tasks ? (y/n) ") == "y":
        Auto_segment.entry()
    # ---------------------------------------

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__))) #Change the working directory to
    # PFI_Autosegmentation_project/Scripts
    main()