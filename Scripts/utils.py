import os
import subprocess
import nibabel as nib
import numpy as np
import h5py
import shutil


def create_No_empty_folders(path, start_No, end_No):
    """
    This function creates indexed folders in a specific directory (path) from a start No° and an end No°
    """
    for i in range(start_No, end_No+1):
        if not os.path.exists(path + str(i)):
            os.makedirs(path + str(i))

def execute_3dslicer(script: str, slicer_path="D:\Programmes\Slicer 5.6.2\Slicer.exe"):
    """
    With this function you can run 3Dslicer executing the script of your choice.
    :param script: Enter the script you want to execute.
    """
    if os.path.exists(slicer_path):
        # Run Slicer with the specified script
        subprocess.run([slicer_path, "--no-splash", "--no-main-window", "--python-script", script])
    else:
        raise FileNotFoundError(f"Your Slicer.exe path is not valid: {slicer_path}")


def change_niftii_labels():
    """
    In case the label generation is not correct, use this function to re-label the .nii.gz files so that the labels
    corresponds to the same segments between files (otherwise not possible to correctly train algorythm).
    :return:
    """
    path = input('Enter path to the data : ')
    for individual in os.listdir(path):
        if individual.isdigit() and int(individual) in [2]:
            # Load the .nii file
            if os.path.exists(f'{path}/{individual}/segmentation.nii.gz'):
                print(f'Re-labeling patient n° {individual}')
                nifti_img = nib.load(f'{path}/{individual}/segmentation.nii.gz')

                # Access the data as a NumPy array
                data = nifti_img.get_fdata()

                #Make sure the parameters are correct
                data = np.round(data)
                data = data.astype(int)
                header = nifti_img.header
                header["scl_slope"] = 1
                header["scl_inter"] = 0
                header["cal_max"] = 9
                header["cal_min"] = 0

                #Manually enter the current segment's labels ID (from 1 to 9)
                # ==> Open the .nii file as a segmentation, the ID is written for each segment as their segment number
                Femur = int(input('Enter the Femur current label '))
                Fem_AC = int(input('Enter the Femur_AC current label '))
                Tib = int(input('Enter the Tibia current label '))
                Tib_lat = int(input('Enter the Lateral tibial AC current label '))
                Tib_med = int(input('Enter the Medial tibial AC current label '))
                Men_lat = int(input('Enter the Lateral Meniscus current label '))
                Men_med = int(input('Enter the Medial Meniscus current label '))
                Patella = int(input('Enter the Patella current label '))
                Pat_AC = int(input('Enter the Patella_AC current label '))

                input_labels = [Femur, Fem_AC, Tib, Tib_lat, Tib_med, Men_lat, Men_med, Patella, Pat_AC]

                # Check for duplicate labels
                if len(input_labels) != len(set(input_labels)):
                    print("Error: Duplicate labels detected!")
                    print(f"Duplicate labels: {[label for label in input_labels if input_labels.count(label) > 1]}")
                    return  # Exit the function if duplicates are found

                data[data == Femur] = 11
                data[data == Fem_AC] = 22
                data[data == Tib] = 33
                data[data == Tib_lat] = 44
                data[data == Tib_med] = 55
                data[data == Men_lat] = 66
                data[data == Men_med] = 77
                data[data == Patella] = 88
                data[data == Pat_AC] = 99

                data[data == 11] = 1
                data[data == 22] = 2
                data[data == 33] = 3
                data[data == 44] = 4
                data[data == 55] = 5
                data[data == 66] = 6
                data[data == 77] = 7
                data[data == 88] = 8
                data[data == 99] = 9

                # Save the modified data back to a new .nii file
                modified_img = nib.Nifti1Image(data, nifti_img.affine, header)
                nib.save(modified_img, f'{path}/{individual}/segmentation.nii.gz')

def show_niftii_info():
    path = input('Enter path to the data : ')
    for individual in os.listdir(path):
        if individual.isdigit() and int(individual) in [2]:
            # Load the .nii file
            if os.path.exists(f'{path}/{individual}/segmentation.nii.gz'):
                print(f'checking patient n° {individual}')
                nifti_img = nib.load(f'{path}/{individual}/segmentation.nii.gz')

                # Access the data as a NumPy array
                data = nifti_img.get_fdata()
                header = nifti_img.header
                print(header)
                input("\nPress to continue\n")
                print(np.unique(data)) #Show all the labels in the file

def read_H5_file(file_path):
    try:
        # Open the .h5 file
        with h5py.File(file_path, 'r') as h5_file:
            print(f"Successfully opened file: {file_path}\n")

            # Recursively explore the file structure
            def explore_h5_group(group, indent=0):
                for key in group.keys():
                    item = group[key]
                    print(" " * indent + f"📂 {key} - {'Group' if isinstance(item, h5py.Group) else 'Dataset'}")
                    if isinstance(item, h5py.Group):  # If it's a group, explore recursively
                        explore_h5_group(item, indent + 4)
                    elif isinstance(item, h5py.Dataset):  # If it's a dataset, show shape and dtype
                        print(" " * (indent + 4) + f"📊 Shape: {item.shape}, Type: {item.dtype}")

            # Start exploring the file from the root group
            explore_h5_group(h5_file)

            dataset_path = input("\nEnter the path to a dataset you want to open (or leave empty to skip): ").strip()
            if dataset_path:
                try:
                    dataset = h5_file[dataset_path]
                    print(f"\nDataset '{dataset_path}' opened successfully!")
                    print(f"Shape: {dataset.shape}")
                    print(f"Type: {dataset.dtype}")

                    # Optionally print dataset data (be careful with large datasets!)
                    view_data = input("Do you want to view the data? (yes/no): ").strip().lower()
                    if view_data in ["yes", "y"]:
                        print("\nData:")
                        print(dataset[...])  # Access the data in the dataset
                except KeyError:
                    print(f"Dataset '{dataset_path}' not found in the file.")

    except Exception as e:
        print(f"Error reading file: {e}")

def export_stl_from_niftii(file_path: str, slicer_path):
    """This function export every part from a segmentation stored in the .nii.gz format into separate .stl files"""
    segment_names = ["Femur", "Femur_cartilage", "Tibia", "Tibia_cartilage_lateral", "Tibia_cartilage_medial",
                     "Menisc_lateral", "Menisc_medial", "Patella", "Patella_cartilage", "Fibula"]
    output_folder = os.path.join(file_path, "../", file_path.split("_PRED")[0])
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    script= \
    'import slicer\n'\
    'import os\n'\
    '# Set paths\n'\
    f'input_nifti_path = r"{file_path}"\n'\
    f'output_folder = r"{output_folder}"\n'\
    '# Load the .nii.gz as a segmentation directly\n'\
    'segmentation_node = slicer.util.loadSegmentation(input_nifti_path)\n'\
    '# Get segment IDs\n'\
    'segmentation = segmentation_node.GetSegmentation()\n'\
    'segment_ids = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]\n'\
    '# Export each segment separately\n'\
    'for seg_id in segment_ids:\n'\
    '    temp_segment_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")\n'\
    '    temp_segment_node.GetSegmentation().CopySegmentFromSegmentation(segmentation, seg_id)\n'\
    '    slicer.modules.segmentations.logic().ExportSegmentsClosedSurfaceRepresentationToFiles(\n'\
    '        output_folder, temp_segment_node\n'\
    '    )\n'\
    '    slicer.app.quit() # quit the scene'

    with open("temp_export_stl_to_niftii.py", "w") as f:
        f.write(script)
        f.close()
    if slicer_path != "":
        execute_3dslicer("temp_export_stl_to_niftii.py", slicer_path=slicer_path)
    else:
        execute_3dslicer("temp_export_stl_to_niftii.py")
    os.remove("temp_export_stl_to_niftii.py")
    for file in os.listdir(output_folder):
        i = int(file.split("_")[-1].split(".")[0])
        os.rename(os.path.join(output_folder, file), os.path.join(output_folder, segment_names[i-1] + ".stl"))
    print(f'files exported at {output_folder}.')

