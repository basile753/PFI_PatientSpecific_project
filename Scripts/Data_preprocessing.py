import shutil
import random
import utils
import os


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
    if input(f'\n generate niftii files (if not already) for the data that are ready ? (y/n) : ') == 'y':
        generate_niftii(path, list_ready)
    return input(f'\nSort the files in "PFI_Autosegmentation_project/Data/RMIs/data_folder" ? (y/n) : \n'), list_ready

def generate_niftii(path, list_patients):
    """
    Generate the .nii files from the images and segmentations of every patients, format used to train the MPUnet algorythm.
    The conversion is made through the 3D slicer software that must be installed on the device.
    :param path: The path to the patient's data.
    :param list_patients: A list of the No° of the patient whose files are ready for niftii file generation.
    """
    # Path to the 3D Slicer executable
    slicer_path = input("\nEnter your path to 3Dslicer software (default : D:\Programmes\Slicer 5.6.2\Slicer.exe) :")
    print("\n\tGenerating niftii files...")
    for individual in list_patients:
        script_dicom_to_nii = f"""
import os
import slicer
from DICOMLib import DICOMUtils

# Path to the DICOM folder and the output NIfTI file
dicom_folder = "{path}/{individual}"
output_file = "{path}/{individual}/image.nii.gz"

# Step 1: Load DICOM images
def load_dicom(dicom_folder):
    # Initialize the DICOM database if it is not already open
    if not slicer.dicomDatabase.isOpen:
        slicer.dicomDatabase.initializeDatabase(os.path.join(slicer.app.temporaryPath, 'ctkDICOM.sql'))
    
    # Import the DICOM folder and load the series
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(dicom_folder, db)
        
        # Automatically load the first DICOM series found
        patient_uid = db.patients()[0]
        study_uid = db.studiesForPatient(patient_uid)[0]
        series_uid = db.seriesForStudy(study_uid)[0]
        
        loaded_node_ids = DICOMUtils.loadSeriesByUID([series_uid])
        volume_node = slicer.mrmlScene.GetNodeByID(loaded_node_ids[0]) if loaded_node_ids else None
        return volume_node

# Step 2: Export to .nii.gz
def export_volume_as_nifti(volume_node, output_file):
    if volume_node:
        slicer.util.saveNode(volume_node, output_file)

# Run the DICOM to NIfTI conversion
volume_node = load_dicom(dicom_folder)
export_volume_as_nifti(volume_node, output_file)

# Clean exit from Slicer
slicer.app.exit()
        """
        script_segments_to_nii = f"""
import os
import slicer
from DICOMLib import DICOMUtils

# Path to the folder containing STL files and the output NIfTI file
stl_folder = r"{path}\{individual}"  # Folder containing the .stl files
output_file = r"{path}\{individual}\segmentation.nii.gz"

# Step 1: Import STL files as segments of a single segmentation
def import_stl_as_segmentation(stl_folder):
    # Create a segmentation node
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()  # To visualize the segments

    # Add each STL file as a separate segment
    for stl_file in {list(segmented_stl.keys())}:
        stl_file += ".stl"
        # Load the STL file as a temporary model
        model_node = slicer.util.loadModel("{path}/{individual}/"+stl_file)
        # Add the model as a new segment in the segmentation node
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentation_node)
        # Remove the temporary model
        slicer.mrmlScene.RemoveNode(model_node)
    
    return segmentation_node

# Step 2: Export the segmentation to .nii.gz
def export_segmentation_as_nifti(segmentation_node, output_file):
    # Convert the segmentation to a binary volume
    export_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentation_node, export_volume_node)
    # Save the segmentation volume in NIfTI format
    slicer.util.saveNode(export_volume_node, output_file)
    # Remove the exported node
    slicer.mrmlScene.RemoveNode(export_volume_node)

# Execute the segmentation process
segmentation_node = import_stl_as_segmentation(stl_folder)
export_segmentation_as_nifti(segmentation_node, output_file)

# Clean exit from Slicer
slicer.app.exit()
        """
        if os.path.exists("convert"):
            shutil.rmtree("convert")
        os.makedirs("convert")
        with open("convert/dicom_to_nii.py", 'w', encoding="utf-8") as file:
            file.write(script_dicom_to_nii)
            file.close()
        with open("convert/segments_to_nii.py", 'w', encoding="utf-8") as file:
            file.write(script_segments_to_nii)
            file.close()
        if not os.path.exists(f"{path}/{individual}/image.nii.gz"):
            utils.execute_3dslicer("convert/dicom_to_nii.py", slicer_path)
            print(f"Image Niftii file generated for patient No° {individual}")
        if not os.path.exists(f"{path}/{individual}/segmentation.nii.gz"):
            utils.execute_3dslicer("convert/segments_to_nii.py", slicer_path)
            print(f"Segmentation Niftii file generated for patient No° {individual}")
        else:
            print(f"Niftii files already generated for patient No° {individual}\n")
    return

def create_randomsplits(list_patients):
    """
    This function aims to randomly split a selected amount of patient's data within the data that are ready in
    train/validation splits.
    :param list_patients: the list of verified/ready patient's data.
    :return: Lists of the No° of train and validation data split.
    """
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
    return train_data, validation_data

def sort(path, list_patients):
    """
    This function sort the data in different folders in order to be processed for training the model.
        -All the DICOM data goes to "dicom" folder.
        -All the segmentations goes to "manual_segmentations" folder.
    Only the patients from list_patients are considered (use check() tu get all the ready patient's data).
    2 folds are generated (train/validation data) and it is possible to use a certain amount of the patient's list
    randomized (in order to observe to influence of the training data size on the accuracy).
    """
    train_data, validation_data = create_randomsplits(list_patients)

    print(f'\n\tSorting the files...')
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
    for individual in train_data:
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
    for individual in validation_data:
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
    return

