#This program aims to launch the MPUnet algorythm in different cases scenario, in order to assess its viability in
# automatic segmentation for person specific model in the context of the XXX project.

#The objective is to find the best setup for auto-segmentation of bones, cartilages and meniscis in order to later to
# some kinematic simulation analysis, for our dataset composed of children's MRI with PFI (patellafemoral instability).

#The output of this code will be the trained model that works the best for our dataset (children with PFI), according
# to the comparison of their Dice score using the two-sided Wilcoxon signed-rank statistics with p=0,05.

#Three case scenario will be tested through this code :
    #1-The use of some pre-trained models, included in the MPUnet plugin.
    #2-The creation of a new model trained on our dataset with the default hyperparameters.
    #3-The creation of a new model trained on our dataset with specific hyperparameters.

import mpunet
from mpunet import *



def pretrained_model(model, test_data) -> list:
    """
    Test of a pretrained model, included in the MPUnet plugin.
    :param model: name of the pre-trained model
    :return: The dice scores of the model for each automatic segmentation
    """

def new_model(train_data, test_data, **parameters) -> list:
    """
    Test a new model trained on the dataset
    :param train_data: path to the training dataset
    :param test_data: path to the test dataset
    :param parameters: dictionary with hyperparameters of the new model (optional)
    :return: The dice scores of the model for each automatic segmentation
    """

