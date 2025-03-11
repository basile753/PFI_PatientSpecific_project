import os
import subprocess
import nibabel as nib
import numpy as np
import h5py
import shutil
import pyvista as pv
from matplotlib.pyplot import legend


def create_No_empty_folders(path, start_No, end_No):
    """
    This function creates indexed folders in a specific directory (path) from a start No° and an end No°
    """
    for i in range(start_No, end_No+1):
        if not os.path.exists(path + str(i)):
            os.makedirs(path + str(i))

def execute_3dslicer(script: str, slicer_path: str):
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
    output_folder = os.path.join(file_path.split("_PRED")[0])
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
    execute_3dslicer("temp_export_stl_to_niftii.py", slicer_path=slicer_path)
    os.remove("temp_export_stl_to_niftii.py")
    for file in os.listdir(output_folder):
        i = int(file.split("_")[-1].split(".")[0])
        os.rename(os.path.join(output_folder, file), os.path.join(output_folder, segment_names[i-1] + ".stl"))
    print(f'files exported at {output_folder}.')

def volume_to_shell_meshes(prefix: str, path: str, thickness=0.1):
    """
    This function transform vol
    :param prefix: A string that the model's parts to transform must include in their name.
    Ex: "cartilage" for the cartilages parts
    :param path: The path to the model to transform
    """
    for mesh in os.listdir(path):
        if prefix in mesh:
            volume_mesh = pv.read(os.path.join(path, mesh))
            # Compute normals
            normals = volume_mesh.point_normals
            # Create an inner surface
            inner_shell = volume_mesh.copy()
            inner_shell.points -= normals * (thickness)  # Move inward

            # Combine both shells into one mesh
            shell_mesh = volume_mesh + inner_shell
            shell_mesh.save(os.path.join(path, mesh))
    return

def volume_to_surface_meshes(path: str, plot: bool = True, direction = "Z"):
    """
    Thus function create an open surface mesh from a shell/volume mesh using a ray tracing technique and saving only the upper surface (use for cartilages).
    WARNING: This function works only for relatively flat surfaces (such as Tibia and patella cartilages/meniscus, but won't work for the femur cartilage
    :param path: The path to the mesh to transform
    :param plot: Set to True to plot the result of the transformation.
    :param direction: ["Y", "-Y", "Z", "-Z"] Set the direction of the ray tracing technique
    (must be the surface normal direction, usually Z for upper meniscus and Tbia_cartilages, -Z for lower meniscus, Y for Patella_cartilage).
    """
    print(f"\tTransforming {path} into a surface mesh...")

    # Load the mesh
    m = pv.read(path)
    # Create a grid of rays to cast in every direction
    x_vals = np.linspace(m.bounds[0], m.bounds[1], int(np.sqrt(10000)))
    y_vals = np.linspace(m.bounds[2], m.bounds[3], int(np.sqrt(10000)))
    z_vals = np.linspace(m.bounds[4], m.bounds[5], int(np.sqrt(10000)))

    if direction in ["Y", "-Y"]:
        second_vals = z_vals
        direction_index = 1
    elif direction in ["Z", "-Z"]:
        second_vals = y_vals
        direction_index = 2
    # Collect the intersection points
    inf_intersection_points = []
    sup_intersection_points = []

    for x in x_vals:
        for s in second_vals:
            # Define the rays start point and direction
            if direction_index == 1:
                ray_start = np.array([x, m.center[direction_index] + 100, s])
                ray_end = ray_start + np.array([0, -1000, 0])  # A long enough distance
            if direction_index == 2:
                ray_start = np.array([x, s, m.center[direction_index] + 100])
                ray_end = ray_start + np.array([0, 0, -1000])  # A long enough distance

            # Cast the ray and check for intersection
            intersect = m.ray_trace(ray_start, ray_end, first_point = False)
            if len(intersect[0]) > 1:  # Check if there are 2 intersection points
                # Take the first intersection point
                sup_intersection_points.append(intersect[0][0])
                inf_intersection_points.append(intersect[0][1])

    # Convert the intersection points to meshes
    m_sup = pv.PolyData(sup_intersection_points).delaunay_2d(alpha=5)
    m_inf = pv.PolyData(inf_intersection_points).delaunay_2d(alpha=5)
    if direction[0] == "-":
        surface = m_inf
    else:
        surface = m_sup
    # Compute normals
    surface = surface.compute_normals()
    # Generate glyphs (arrows) to represent the normals
    surface_normals = surface.glyph(orient='Normals', scale=1, factor=2)

    if plot:  # Plot the results and check the normals direction
        plotter = pv.Plotter()
        plotter.add_mesh(m, color="gray", opacity=0.6, label="Original Mesh")
        plotter.add_mesh(surface, color="blue", opacity=0.8, label="Surface mesh")
        plotter.add_mesh(surface_normals, color='red')  # Normal vectors
        plotter.add_legend()
        plotter.add_text("Check the normals direction and flip if necessary")
        plotter.show()
        if input("Should the normals be flipped ? (y/n): ").lower() in ["y", "yes"]:
            surface.flip_normals()
    # Save the split meshes
    surface.save(path)
    print("\tTransformation finished.\n")

def check_side(mesh_dir: str):
    """
    Thus function check if the knee is a left or right knee, from the relative tibia/fibula position along X.
    :param mesh_dir: The path to the meshes directory.
    :return: strings "R" or "L" and also prints the side.
    """
    print(f"\t~~~~~ Checking if {mesh_dir} is a left or right knee. ~~~~~")
    flag = 0
    for mesh in os.listdir(mesh_dir):
        if mesh.endswith(".ply") or mesh.endswith(".stl"):
            if ("Fibula" in mesh):
                print("Fibula found")
                fibula_mesh = pv.read(os.path.join(mesh_dir, mesh))
                flag += 1
            elif ("Tibia" in mesh) and ("cartilage" not in mesh):
                print("Tibia found")
                tibia_mesh = pv.read(os.path.join(mesh_dir, mesh))
                flag += 1
    if flag != 2:
        raise FileNotFoundError(f"\nOne of these files are missing: 'tibia' and 'fibula' meshes within: {mesh_dir}\n. "
                            f"verify they are well named without error.")
    flag = 0
    if fibula_mesh.center[0] < tibia_mesh.center[0]:
        side = "L"
        flag += 1
    elif fibula_mesh.center[0] > tibia_mesh.center[0]:
        side = "R"
        flag += 1
    if flag != 1:
        raise ValueError(f"The tibia or fibula mesh file seems corrupted or not properly loaded. Please check them.")
        return
    print(f"\tKnee side: {side}")
    return side
