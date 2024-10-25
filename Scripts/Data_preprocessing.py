import os
import shutil
import random


segmented_stl = {
        "Femur": ["Fem", "fem", "femur"],
        "Tibia": ["Tib", "tib", "tibia", "TIbia"],
        "Patella": ["patella", "Pat", "pat", "patel", "Patel"],
        "Femur_cartilage": ["Fem_CA", "Fem_AC", "FemAC", "Femur AC", "Femur CA", "Femur_AC", "Femur_CA", "FemurAC", "FemurCA", "Femur Cartilage", "Fem AC"],
        "Tibia_cartilage_medial": ["Tib_CA_med", "TibACmed", "Med Tib AC","Tib AC med", "Tib_AC_med", "Tib AC_med", "Tibia_AC_med", "Tibia AC med", "Tib AC Med", "Tibia AC_med"],
        "Tibia_cartilage_lateral": ["Tib_CA_lat", "TibAClat", "Lat Tib AC","Tib AC lat", "Tib_AC_lat", "Tib AC_lat", "Tibia_AC_lat", "Tibia AC lat", "Tib AC Lat", "Tibia AC_lat"],
        "Patella_cartilage": ["Patella_CA", "PatAC", "Patella AC", "Patella_AC", "Pat AC", "Pat_AC", "PatellaAC", "Patella Cartilage"],
        "Menisc_medial": ["menmed", "med men", "med_men", "Med men", "Med_men", "Menisci_med", "Menisci med", "Med Men", "Med Meniscus", "MedMeniscus", "Medial meniscus", "Med_meniscus", "men med", "men Med", "Men_med", "men_med"],
        "Menisc_lateral": ["menlat", "lat men", "lat_men", "Lat men", "Lat_men", "Menisci_lat", "Menisci lat", "Lat Men", "Lat Meniscus", "LatMeniscus", "Lateral meniscus", "Lat_meniscus", "men lat", "men Lat", "Men_lat", "men_lat"],}

def find_key_from_value(dict, val):
    """
    Find the key of a value in a dictionary where several values are associated with 1 key.
    :param dict: the dictionary to be searched
    :param val: the value
    :return:
    """
    if val not in dict.keys():
        for key, values in dict.items():
            if val in values:
                return key
        print(f'The name {val} is not in the list of names to refactor')

def normalize(path):
    """
    Normalizes the data's segmentation files names in order to be processed and have a consistant naming according to
    the segmented_stl dictionary.
    """
    print(f'\n\tNormalizing files in {path}')
    # Walk through every folder and file in the path
    individuals = os.listdir(path)
    for individual in individuals:
        if individual.isdigit(): #Verify that the folder is a patient's MRI folder
            print(f'\nNormalizing individual No° : {individual}')
            filenames = os.listdir(path+"/"+individual)
            #print(filenames)
            for filename in filenames:
                if filename.endswith('.stl'):
                    filename_bis = filename.replace('.stl', '')
                    if filename.startswith('Segmentation_'):
                        filename_bis = filename_bis.replace('Segmentation_', '')
                        os.rename(path + '/' + individual + '/' + filename, path + '/' + individual + '/' + filename_bis + '.stl')
                        filename = filename.replace('Segmentation_', '')
                    elif filename.startswith('_'):
                        filename_bis = filename_bis[1:]
                        os.rename(path + '/' + individual + '/' + filename, path + '/' + individual + '/' + filename_bis + '.stl')
                        filename = filename[1:]
                    new_name = find_key_from_value(segmented_stl, filename_bis) #gets the standard name
                    if new_name is not None: #if different name
                        new_name = new_name + ".stl"
                        os.rename(path+'/'+individual+'/'+filename, path+'/'+individual+'/'+new_name) #rename the file to be standard
                        print(f"Renamed: {filename} -> {new_name}")
    return
                
def check(path):
    """
    Verify if all the necesary STL and DICOM files exists for every patient.
    :return: List of the No° of the patients whose files are ready.
    """
    list_ready = []
    print(f'\n\tChecking files in {path}')
    individuals = os.listdir(path)
    n=0
    for individual in individuals:
        flag = True
        if individual.isdigit(): #Verify that the folder is a patient's MRI folder
            n+=1
            print(f'\nChecking individual No° : {individual}')
            for key in segmented_stl.keys():
                key = key+".stl"
                if key not in os.listdir(path+"/"+individual):
                    flag = False
                    print(f'!!!!WARNING!!!!!! {key} is missing from patient No° {individual}')
            if "DICOM" not in os.listdir(path+"/"+individual):
                flag = False
                print(f'!!!!WARNING!!!!!! The DICOM files are missing from patient No° {individual}')
            if flag == True:
                list_ready.append(int(individual))
    list_ready = sorted(list_ready)
    print(f'\nTotal of patients considered : {n}')
    print(f'Number of patients data ready (fully segmented with DICOM in the folder) : {len(list_ready)}')
    print(f'\tList of the patients ready : {list_ready}')
    return input(f'\nSort the files ? True/False : \n'), list_ready

def create_randomsplits(list_patients):
    """
    This function aims to randomly split a selected amount of patient's data within the data that are ready in
    train/test splits.
    :param list_patients: the list of verified/ready patient's data.
    :return: Lists of the No° of train and test data split.
    """
    print(f"Here the list of the {len(list_patients)} patient's data selected : {list_patients}")
    amount_data = input(f'Choose how many patient you want to use to train/test the algorythm (default : all)')
    if amount_data == '':
        amount_data = len(list_patients)
    elif amount_data.isdigit() == False:
        raise TypeError("The amount choosen is not a number.")
    elif int(amount_data) > len(list_patients):
        raise ValueError("The amount choosen exceeds the number of ready patient's data.")
    amount_test_data = input(f'Choose how many patient you want to test the algorythm (default : 15) :')
    if amount_test_data == "":
        amount_test_data = 15
    elif amount_test_data.isdigit() == False:
        raise TypeError("The amount choosen is not a number.")
    elif int(amount_test_data) > int(amount_data):
        raise ValueError("The amount choosen exceeds the number of selected patients.")
    test_data = random.sample(list_patients, int(amount_test_data))
    for val in test_data:
        list_patients.remove(val)
    train_data = random.sample(list_patients, int(amount_data)-int(amount_test_data))
    print(f"Here is the result of the random split  :\n\t{len(train_data)} training data {train_data}\n\t{len(test_data)}"
          f" test data {test_data}")
    train_data = sorted(train_data)
    test_data = sorted(test_data)
    return train_data, test_data

def sort(path, list_patients):
    """
    This function sort the data in different folders in order to be processed for training the model.
        -All the DICOM data goes to "dicom" folder.
        -All the segmentations goes to "manual_segmentations" folder.
    Only the patients from list_patients are considered (use check() tu get all the ready patient's data).
    2 folds are generated (train/test data) and it is possible to use a certain amount of the patient's list
    randomized (in order to observe to influence of the training data size on the accuracy).
    """
    train_data, test_data = create_randomsplits(list_patients)

    print(f'\n\tSorting files in {path} in the train/test data folders in {"PFI_Autosegmentation_project/Data/RMIs"}')
    #Delete the previous directories
    if os.path.exists("../Data/RMIs/train_data"):
        shutil.rmtree("../Data/RMIs/train_data")
    if os.path.exists("../Data/RMIs/test_data"):
        shutil.rmtree("../Data/RMIs/test_data")
    os.makedirs("../Data/RMIs/train_data")
    os.makedirs("../Data/RMIs/test_data")


    #Sorting the training data
    for individual in train_data:
        individual = str(individual)
        print(f'Sorting patient No° {individual} as a train data')
        os.makedirs("../Data/RMIs/train_data/manual_segmentations/"+individual)
        os.makedirs("../Data/RMIs/train_data/dicom/"+individual)
        for key in segmented_stl.keys():
            key = key + ".stl"
            if key not in os.listdir(path + "/" + individual):
                print(f'!!!!WARNING!!!!!! {key} is missing from patient No° {individual}')
            else:
                shutil.copy(path + "/" + individual + "/" + key, "../Data/RMIs/train_data/manual_segmentations/" + individual)
        if "DICOM" not in os.listdir(path + "/" + individual):
            print(f'!!!!WARNING!!!!!! The DICOM files are missing from patient No° {individual}')
        else:
            shutil.copytree(path + "/" + individual + "/DICOM", "../Data/RMIs/train_data/dicom/" + individual, dirs_exist_ok = True)
       #Might not be necessary ?
       """ if "DICOMDIR" not in os.listdir(path + "/" + individual):
            print(f'!!!!WARNING!!!!!! The DICOMDIR file is missing from patient No° {individual}')
        else:
            shutil.copy(path + "/" + individual + "/DICOMDIR", "../Data/RMIs/train_data/dicom/" + individual)"""

    # Sorting the testing data
    for individual in test_data:
        individual = str(individual)
        print(f'Sorting patient No° {individual} as a test data')
        os.makedirs("../Data/RMIs/test_data/manual_segmentations/" + individual)
        os.makedirs("../Data/RMIs/test_data/dicom/" + individual)
        for key in segmented_stl.keys():
            key = key + ".stl"
            if key not in os.listdir(path + "/" + individual):
                print(f'!!!!WARNING!!!!!! {key} is missing from patient No° {individual}')
            else:
                shutil.copy(path + "/" + individual + "/" + key,
                            "../Data/RMIs/test_data/manual_segmentations/" + individual)
        if "DICOM" not in os.listdir(path + "/" + individual):
            print(f'!!!!WARNING!!!!!! The DICOM files are missing from patient No° {individual}')
        else:
            shutil.copytree(path + "/" + individual + "/DICOM", "../Data/RMIs/test_data/dicom/" + individual,
                            dirs_exist_ok=True)
        # Might not be necessary ?
        """if "DICOMDIR" not in os.listdir(path + "/" + individual):
            print(f'!!!!WARNING!!!!!! The DICOMDIR file is missing from patient No° {individual}')
        else:
            shutil.copy(path + "/" + individual + "/DICOMDIR", "../Data/RMIs/test_data/dicom/" + individual)"""
    return


if __name__ == "__main__":
    flag = False

    # Enter the path to the manual segmentation's data
    path = input("Enter the path of the RMIs Data : ")
    if type(path) is not str:
        raise TypeError("The path must be a chain of characters")

    # Call the function to rename files to their conventional names
    normalize(path)

    #Call the function to check the data that are ready
    flag, list_patients = check(path)

    if flag == "True":
        #Call the function to randomly sort the ready data in train/test folders (DICOM & manual_segmentation) for process
        sort(path, list_patients)
    else:
        print("Program ended without sorting the files.")

