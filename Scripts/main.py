"""
This is the main script on the project, it aims to interact with the other scripts in order to pre-process the data,
train or evaluate models.
"""
from Scripts import Data_preprocessing as dp
from Scripts import train_MPunet as trmp

def main():
    #-----Pre-processing---------------------
    if input("Would you like to pre-process data ? (y/n) ") == "y":
        path = input("Enter the path of the RMIs Data : ") #Enter the path to the manual segmentation's data
        if type(path) is not str:
            raise TypeError("The path must be a chain of characters")
        else:
            dp.normalize(path) #Normalizing the data
            flag, list_patients = dp.check(path) #check which data are ready
            if flag == "y":
                dp.sort(path, list_patients) #Randomly sort the data that are ready in train/validation folders
    #---------------------------------------

    #-----Training the MPUnet model---------
    trmp.train_mpunet()

if __name__ == '__main__':
    main()