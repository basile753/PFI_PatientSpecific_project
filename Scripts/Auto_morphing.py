import os
import shutil
import pyvista as pv
import utils as ut
import subprocess
import sys

def entry():
    """This is the entry function of the auto-morphing script. Please use it if you wish to perform morphing on an existing segmentation."""

    # Set the segmentations and slicer software paths
    model_dir = input("Enter the segmentations directory, they must be in the .nii.gz format (default: ../Data/RMIs/to_predict/results): ")
    if model_dir == "":
        model_dir = os.path.join(os.getcwd(), "../Data/RMIs/to_predict/results")
    slicer_path = input("Enter the path to your 3Dslicer executable (default: D:\Programmes\Slicer 5.6.2\Slicer.exe): ")
    if slicer_path == "":
        slicer_path = "D:\Programmes\Slicer 5.6.2\Slicer.exe"
    root = os.getcwd()
    print("The following segmentations will be morphed: ")
    for file in os.listdir(model_dir):
        if file.endswith(".nii.gz"):
            print(f"\r{file}")

    # Export the segments from the .niifti file into separate .stl files for each segmentation.
    for file in os.listdir(model_dir):
        if file.endswith(".nii.gz"):
            patient_no = file.split("_")[0]
            print(f'Exporting segments from {file} as {patient_no}...')
            output_folder = os.path.join(model_dir, patient_no)
            ut.export_stl_from_niftii(os.path.join(model_dir, file), slicer_path)

            # Transform the cartilages into surface-type meshes for better morphing (except for the femur cartilage)
            ut.volume_to_surface_meshes(os.path.join(output_folder, "Tibia_cartilage_lateral.stl"), direction="Z")
            ut.volume_to_surface_meshes(os.path.join(output_folder, "Tibia_cartilage_medial.stl"), direction="Z")
            ut.volume_to_surface_meshes(os.path.join(output_folder, "Patella_cartilage.stl"), direction="Y")

            # Launch the morphing process
            print(f'Start of the morphing process for {patient_no}...')
            os.chdir('Morphing_scripts')
            process = subprocess.Popen([sys.executable, 'Mesh_processing.py'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            # Send the different inputs : mesh path, default
            process.stdin.write(os.path.join(model_dir, patient_no) + "\n")
            process.stdin.flush()
            process.stdin.write("\n")
            process.stdin.flush()
            process.stdin.close()
            # Capture errors in case of need
            output, error = process.communicate()
            print(error)
            os.chdir(root)

            # Clean the temporary files and sort the morphed results
            morphed_dir = os.path.join(model_dir, file.split(".")[0] + "_morphed")
            os.makedirs(morphed_dir, exist_ok=True)
            for stl in os.listdir(os.path.join(model_dir, patient_no, 'corresp')):
                shutil.move(os.path.join(model_dir, patient_no, 'corresp', stl), os.path.join(morphed_dir, stl))
            for file in os.listdir(os.path.join(model_dir, patient_no, 'remeshed')):
                if file.endswith("ACS.txt"):
                    shutil.move(os.path.join(model_dir, patient_no, 'remeshed', file), os.path.join(morphed_dir, file))
            shutil.rmtree(os.path.join(model_dir, patient_no))
            print(f'Knee No° {patient_no} Morphing process completed.')


if __name__ == "__main__":
    entry()
    while(input("Would you like to run another auto-morphing process ? (y/n): ")) == "y":
        entry()