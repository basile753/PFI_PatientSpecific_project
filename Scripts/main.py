"""
This is the main script of the project, it aims to interact with the other scripts in order to pre-process the data,
train and evaluate models.
"""
import os
import Data_preprocessing
import Training

def main():
    #-----Pre-processing---------------------
    if input("Would you like to pre-process data ? (y/n) ") == "y":
        Data_preprocessing.entry()
    #---------------------------------------

    #-----Training the model---------
    if input("Would you like to train a model on the sorted data ? (y/n) ") == "y":
        Training.entry()
    # ---------------------------------------

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()