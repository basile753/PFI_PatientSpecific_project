import shutil
import random
import utils
import os
from sklearn.model_selection import KFold
from datetime import datetime

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
    print(f'\nVerifying that the NiFTII files are generated for the patients that are ready...')
    for individual in list_ready:
        if 'image.nii.gz' not in os.listdir(path+"/"+str(individual)):
            print(f'\timage.nii.gz is missing from patient No° {individual}, please generate before training the model.')
        if 'segmentation.nii.gz' not in os.listdir(path+"/"+str(individual)):
            print(f'\tsegmentation.nii.gz is missing from patient No° {individual}, please generate before training the model.'
                  f'\n\t WARNING : segmentation.nii.gz must be a binary labelmap that has the same dimensions as the '
                  f'image, use 3Dslicer DWI Volume to resample.')
    return list_ready

def create_randomsplits(list_patients):
    """
    This function aims to randomly split a selected amount of patient's data within the data that are ready in
    train/validation splits.
    :param list_patients: the list of verified/ready patient's data.
    :return: Lists of the No° of train and validation data split.
    """
    choice = input(f'Do you want to perform a simple train/test split (1) or a cross-validation split (2) ? (1 or 2) : ')
    if choice == "1":
        print(f"Here the list of the {len(list_patients)} patient's data selected : {list_patients}")
        amount_data = input(f'Choose how many patient you want to use to train/validation the algorythm (default : all)')
        if amount_data == '':
            amount_data = len(list_patients)
        elif amount_data.isdigit() == False:
            raise TypeError("The amount choosen is not a number.")
        elif int(amount_data) > len(list_patients):
            raise ValueError("The amount choosen exceeds the number of ready patient's data.")
        amount_validation_data = input(f'Choose how many patient you want to validation the algorythm (default : 15) :')
        if amount_validation_data == "":
            amount_validation_data = 15
        elif amount_validation_data.isdigit() == False:
            raise TypeError("The amount choosen is not a number.")
        elif int(amount_validation_data) > int(amount_data):
            raise ValueError("The amount choosen exceeds the number of selected patients.")
        validation_data = random.sample(list_patients, int(amount_validation_data))
        for val in validation_data:
            list_patients.remove(val)
        train_data = random.sample(list_patients, int(amount_data)-int(amount_validation_data))
        print(f"Here is the result of the random split  :\n\t{len(train_data)} training data {train_data}\n\t{len(validation_data)}"
              f" validation data {validation_data}")
        train_data = sorted(train_data)
        validation_data = sorted(validation_data)
        return [train_data, validation_data], False
    elif choice == "2":
        n_splits = input(f'How many splits do you want to perform (default : 4) : ')
        if n_splits == "":
            n_splits = 4
        else:
            n_splits = int(n_splits)

        # Ask the user if they want to use a subset of the data
        use_subset = input(f'Do you want to use a subset of the data? (y/n), default: no): ').strip().lower()
        if use_subset in ["yes", "y"]:
            subset_size = input(f'Enter the number of data you want in every split (default: use all data): ').strip()
            if subset_size == "":
                subset_size = len(list_patients)
            else:
                subset_size = int(subset_size)
                subset_size = min(subset_size, len(list_patients))  # Ensure subset size does not exceed available data

            # Randomly sample a subset of patients
            list_patients_subset = random.sample(list_patients, subset_size)
            print(f'Using a subset of {subset_size} patients out of {len(list_patients)}.')
        else:
            list_patients_subset = list_patients

        # Generate the KFold splits (train and val indices)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)  # Random state set to None for true randomness
        split = [(train_index, val_index) for train_index, val_index in kf.split(list_patients_subset)]

        # Replace indices with actual patient IDs
        split_with_ids = []
        for train_index, val_index in split:
            train_patients = [list_patients_subset[i] for i in train_index]  # Map train indices to patient IDs
            val_patients = [list_patients_subset[i] for i in val_index]  # Map test indices to patient IDs
            split_with_ids.append((train_patients, val_patients))

        for i in range(n_splits):
            print(f'\nSplit {i + 1} : \n\tTrain : {split_with_ids[i][0]}\n\tValidation : {split_with_ids[i][1]}')
        return split_with_ids, True


def sort(path, list_patients):
    """
    This function sort the data in different folders in order to be processed for training the model.
        -All the DICOM data goes to "dicom" folder.
        -All the segmentations goes to "manual_segmentations" folder.
    Only the patients from list_patients are considered (use check() tu get all the ready patient's data).
    2 folds are generated (train/validation data) and it is possible to use a certain amount of the patient's list
    randomized (in order to observe to influence of the training data size on the accuracy).
    """
    train_val_data, flag_crossval = create_randomsplits(list_patients)
    print(f'\n\tSorting the files...')

    #In case of cross-validation splits
    if flag_crossval == True:
        #Delete the previous directories
        if os.path.exists("../Data/RMIs/data_folder"):
            shutil.rmtree("../Data/RMIs/data_folder")
        #The tree is generated as written in the MPUnet datasheet
        for i in range(len(train_val_data)):
            os.makedirs(f"../Data/RMIs/data_folder/split_{i+1}/data_folder/train/images")
            os.makedirs(f"../Data/RMIs/data_folder/split_{i+1}/data_folder/train/labels")
            os.makedirs(f"../Data/RMIs/data_folder/split_{i+1}/data_folder/val/images")
            os.makedirs(f"../Data/RMIs/data_folder/split_{i+1}/data_folder/val/labels")
            os.makedirs(f"../Data/RMIs/data_folder/split_{i+1}/data_folder/test/images")
            os.makedirs(f"../Data/RMIs/data_folder/split_{i+1}/data_folder/test/labels")
            print(f"Sorting the split {i+1}")
            # Sorting the training data
            for individual in train_val_data[i][0]:
                individual = str(individual)
                if "image.nii.gz" not in os.listdir(path + "/" + individual):
                    raise FileNotFoundError(
                        f'"image.nii.gz" not found in {path + "/" + individual}, please convert the DICOM data in a unique .nii.gz file.')
                elif "segmentation.nii.gz" not in os.listdir(path + "/" + individual):
                    raise FileNotFoundError(
                        f'"segmentation.nii.gz" not found in {path + "/" + individual}, please convert the segmentations data in a unique .nii.gz file.')
                else:
                    shutil.copy(path + "/" + individual + "/" + "image.nii.gz",
                                f"../Data/RMIs/data_folder/split_{i+1}/data_folder/train/images/" + individual + ".nii.gz")
                    shutil.copy(path + "/" + individual + "/" + "segmentation.nii.gz",
                                f"../Data/RMIs/data_folder/split_{i+1}/data_folder/train/labels/" + individual + ".nii.gz")
            # Sorting the validation data
            for individual in train_val_data[i][1]:
                individual = str(individual)
                if "image.nii.gz" not in os.listdir(path + "/" + individual):
                    raise FileNotFoundError(
                        f'"image.nii.gz" not found in {path + "/" + individual}, please convert the DICOM data in a unique .nii.gz file.')
                elif "segmentation.nii.gz" not in os.listdir(path + "/" + individual):
                    raise FileNotFoundError(
                        f'"segmentation.nii.gz" not found in {path + "/" + individual}, please convert the segmentations data in a unique .nii.gz file.')
                else:
                    shutil.copy(path + "/" + individual + "/" + "image.nii.gz",
                                f"../Data/RMIs/data_folder/split_{i+1}/data_folder/val/images/" + individual + ".nii.gz")
                    shutil.copy(path + "/" + individual + "/" + "segmentation.nii.gz",
                                f"../Data/RMIs/data_folder/split_{i+1}/data_folder/val/labels/" + individual + ".nii.gz")

    #Standard train/val split.
    else:
        #Delete the previous directories
        if os.path.exists("../Data/RMIs/data_folder"):
            shutil.rmtree("../Data/RMIs/data_folder")
        #The tree is generated as written in the MPUnet datasheet
        os.makedirs("../Data/RMIs/data_folder/train/images")
        os.makedirs("../Data/RMIs/data_folder/train/labels")
        os.makedirs("../Data/RMIs/data_folder/val/images")
        os.makedirs("../Data/RMIs/data_folder/val/labels")
        os.makedirs("../Data/RMIs/data_folder/test/images")
        os.makedirs("../Data/RMIs/data_folder/test/labels")

        #Sorting the training data
        for individual in train_val_data[0]:
            individual = str(individual)
            print(f'Sorting patient No° {individual} as a train data')
            if "image.nii.gz" not in os.listdir(path + "/" + individual):
                raise FileNotFoundError(f'"image.nii.gz" not found in {path + "/" + individual}, please convert the DICOM data in a unique .nii.gz file.')
            elif "segmentation.nii.gz" not in os.listdir(path + "/" + individual):
                raise FileNotFoundError(f'"segmentation.nii.gz" not found in {path + "/" + individual}, please convert the segmentations data in a unique .nii.gz file.')
            else:
                shutil.copy(path + "/" + individual + "/" + "image.nii.gz", "../Data/RMIs/data_folder/train/images/"+individual+".nii.gz")
                shutil.copy(path + "/" + individual + "/" + "segmentation.nii.gz", "../Data/RMIs/data_folder/train/labels/"+individual+".nii.gz")

        # Sorting the validation data
        for individual in train_val_data[1]:
            individual = str(individual)
            print(f'Sorting patient No° {individual} as a validation data')
            if "image.nii.gz" not in os.listdir(path + "/" + individual):
                raise FileNotFoundError(
                    f'"image.nii.gz" not found in {path + "/" + individual}, please convert the DICOM data in a unique .nii.gz file.')
            elif "segmentation.nii.gz" not in os.listdir(path + "/" + individual):
                raise FileNotFoundError(
                    f'"segmentation.nii.gz" not found in {path + "/" + individual}, please convert the segmentations data in a unique .nii.gz file.')
            else:
                shutil.copy(path + "/" + individual +"/"+ "image.nii.gz",
                            "../Data/RMIs/data_folder/val/images/" + individual + ".nii.gz")
                shutil.copy(path + "/" + individual +"/"+ "segmentation.nii.gz",
                            "../Data/RMIs/data_folder/val/labels/" + individual + ".nii.gz")
    return train_val_data

def write_log(list, split):
    """
    This function writes down the results of the train/val splits with date and time.
    """
    if os.path.exists("../Logs/data_preprocessing_log.txt"):
        os.remove("../Logs/data_preprocessing_log.txt")
    with open("../Logs/data_preprocessing_log.txt", "w") as file:
        file.write(f'The following split was performed at date : {datetime.now()}\n\n{len(list)}data were processed : '
                   f'{list}\n\nThe following split was performed {split}\nend-----------------------------------\n')

def entry():
    """
    This is the entry function of the pre-processing script
    """
    path = input("Enter the path of the RMIs Data : ")  # Enter the path to the manual segmentation's data
    if type(path) is not str:
        raise TypeError("The path must be a chain of characters")
    if input("Would you like to normalize the Data ? (y/n) : ") == "y":
        normalize(path) #Normalizing the nomenclature of the .stl files
    if input("Would you like to check the missing files for each patient ? NECESSARY for further sorting process "
             "(y/n) : ") == "y":
        list_patients = check(path) #check which data are ready
        if input("Would you like to sort the Data in random train/val datasets ? (y/n) : ") == "y":
            result = sort(path, list_patients) #Randomly sort the data that are ready in train/validation folders
            write_log(list_patients, result)


