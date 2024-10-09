import os
import shutil


segmented_stl = {
        "Femur": ["Fem"],
        "Tibia": ["Tib", "tib", "tibia", "TIbia"],
        "Patella": ["Pat", "pat", "patel", "Patel"],
        "Femur_cartilage": ["Femur AC", "Femur CA", "Femur_AC", "Femur_CA", "FemurAC", "FemurCA", "Femur Cartilage", "Fem AC"],
        "Tibia_cartilage_medial": ["Tib AC med", "Tib_AC_med", "Tib AC_med", "Tibia_AC_med", "Tibia AC med", "Tib AC Med", "Tibia AC_med"],
        "Tibia_cartilage_lateral": ["Tib AC lat", "Tib_AC_lat", "Tib AC_lat", "Tibia_AC_lat", "Tibia AC lat", "Tib AC Lat", "Tibia AC_lat"],
        "Patella_cartilage": ["Patella AC", "Patella_AC", "Pat AC", "Pat_AC", "PatellaAC", "Patella Cartilage"],
        "Menisc_medial": ["med men", "med_men", "Med men", "Med_men", "Menisci_med", "Menisci med", "Med Men", "Med Meniscus", "MedMeniscus", "Medial meniscus", "Med_meniscus", "men med", "men Med", "Men_med"],
        "Menisc_lateral": ["lat men", "lat_men", "Lat men", "Lat_men", "Menisci_lat", "Menisci lat", "Lat Men", "Lat Meniscus", "LatMeniscus", "Lateral meniscus", "Lat_meniscus", "men lat", "men Lat", "Men_lat"],}

def find_key_from_value(dict, val):
    if val not in dict.keys():
        for key, values in dict.items():
            if val in values:
                return key
        print(f'The name {val} is not in the list of names to refactor')

def normalize(path):
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
                    new_name = find_key_from_value(segmented_stl, filename_bis) #gets the standard name
                    if new_name is not None: #if different name
                        new_name = new_name + ".stl"
                        os.rename(path+'/'+individual+'/'+filename, path+'/'+individual+'/'+new_name) #rename the file to be standard
                        print(f"Renamed: {filename} -> {new_name}")
    return
                
def check(path):
    """
    This program verify that all the necesary STL and DICOM files exists for every patient.
    """
    print(f'\n\tChecking files in {path}')
    individuals = os.listdir(path)
    n=0
    for individual in individuals:
        if individual.isdigit(): #Verify that the folder is a patient's MRI folder
            n+=1
            print(f'\nChecking individual No° : {individual}')
            for key in segmented_stl.keys():
                key = key+".stl"
                if key not in os.listdir(path+"/"+individual):
                    print(f'!!!!WARNING!!!!!! {key} is missing from patient No° {individual}')
            if "DICOM" not in os.listdir(path+"/"+individual):
                print(f'!!!!WARNING!!!!!! The DICOM files are missing from patient No° {individual}')
    print(f'\nTotal of patients considered : {n}')
    return input(f'\nSort the files ? True/False\n (verify first that no file is missing)\n')

def sort(path):
    """
    This function sort the data in different folders in order to be processed for training the model.
        -All the DICOM data goes to "dicom" folder.
        -All the segmentations goes to "manual_segmentations" folder.
    """
    print(f'\n\tSorting files in {path}')
    individuals = os.listdir(path)
    for individual in individuals:
        if individual.isdigit(): #Verify that the folder is a patient's MRI folder
            print(f'\nSorting individual No° : {individual}')
            if not os.path.exists("../Data/RMIs/manual_segmentations/"+individual):
                os.makedirs("../Data/RMIs/manual_segmentations/"+individual)
            if not os.path.exists("../Data/RMIs/dicom/"+individual):
                os.makedirs("../Data/RMIs/dicom/"+individual)
            for key in segmented_stl.keys():
                key = key + ".stl"
                if key not in os.listdir(path + "/" + individual):
                    print(f'!!!!WARNING!!!!!! {key} is missing from patient No° {individual}')
                else:
                    shutil.copy(path + "/" + individual + "/" + key, "../Data/RMIs/manual_segmentations/" + individual)
            if "DICOM" not in os.listdir(path + "/" + individual):
                print(f'!!!!WARNING!!!!!! The DICOM files are missing from patient No° {individual}')
            else:
                shutil.copytree(path + "/" + individual + "/DICOM", "../Data/RMIs/dicom/" + individual, dirs_exist_ok = True)
            if "DICOMDIR" not in os.listdir(path + "/" + individual):
                print(f'!!!!WARNING!!!!!! The DICOMDIR file is missing from patient No° {individual}')
            else:
                shutil.copy(path + "/" + individual + "/DICOMDIR", "../Data/RMIs/dicom/" + individual)
    return


if __name__ == "__main__":
    flag = False

    # Define the path to the data after manual segmentation
    path = input("Enter the path of the RMIs Data : ")
    if type(path) is not str:
        raise TypeError("The path must be a chain of characters")

    # Call the function to rename files to their standard names
    normalize(path)

    #Call the function to check if any data is missing and/or if useless data are present
    flag = check(path)

    if flag == "True":
        #Call the function to sort the data in folders (DICOM & manual_segmentation) for process
        sort(path)
    else:
        print("Program ended without sorting the files.")

