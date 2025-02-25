# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:19:09 2024

@author: Annagh Macie adapted & tunned by Antoine Basile
Updates the geometry of the scaled reference model to the person-specific geometry, updates the wrapping surfaces, and the muscle and ligament
attachment sites to the corresponding node/location on the person-specific geometry.
"""

import os
import numpy as np
import pyvista as pv
import matplotlib as plt
import sys
sys.path.append(r'Morphing_scripts\knee-model-tools')
import jam_updateModel
import ssm
import utils_bis as utb

def menisc_split(dir: str, plot: bool = True):
    """
    Thus function splits and save the meniscus in 2 inferior/superior open surfaces.
    :param dir: The directory to the Meniscus (MUST BE NAMED "C_Menisc_medial.ply", "C_Menisc_lateral.ply")
    :param plot: Set to True to plot the result of the split.
    """
    print(f"\tSplitting meniscus using ray tracing...")
    meniscus_list = ["C_Menisc_medial.ply", "C_Menisc_lateral.ply"]

    for menisc in meniscus_list:
        # Load the mesh
        m = pv.read(os.path.join(dir, menisc))
        # Create a grid of rays to cast in every direction
        x_vals = np.linspace(m.bounds[0], m.bounds[1], int(np.sqrt(10000)))
        y_vals = np.linspace(m.bounds[2], m.bounds[3], int(np.sqrt(10000)))
        # Collect the intersection points
        inf_intersection_points = []
        sup_intersection_points = []

        for x in x_vals:
            for y in y_vals:
                # Define the rays start point and direction
                ray_down_start = np.array([x, y, m.center[2]+100])
                ray_down_end = ray_down_start + np.array([0, 0, -1000])  # A long enough distance
                ray_up_start = np.array([x, y, m.center[2] - 100])
                ray_up_end = ray_up_start + np.array([0, 0, 1000])

                # Cast the ray and check for intersection
                sup_intersect = m.ray_trace(ray_down_start, ray_down_end, first_point=True)
                inf_intersect = m.ray_trace(ray_up_start, ray_up_end, first_point=True)
                if len(sup_intersect[0]) > 0:  # Check if there is an intersection
                    # Take the first intersection point
                    sup_intersection_points.append(sup_intersect[0])
                if len(inf_intersect[0]) > 0:  # Check if there is an intersection
                    inf_intersection_points.append(inf_intersect[0])

        # Convert the intersection points to meshes
        m_sup = pv.PolyData(sup_intersection_points).delaunay_2d(alpha=5)
        m_inf = pv.PolyData(inf_intersection_points).delaunay_2d(alpha=5)
        m_sup = m_sup.compute_normals()
        m_inf = m_inf.compute_normals()
        # Generate glyphs (arrows) to represent the normals
        m_inf_normals = m_inf.glyph(orient='Normals', scale=1, factor=2)
        m_sup_normals = m_sup.glyph(orient='Normals', scale=1, factor=2)


        if plot:  # Plot the results
            plotter = pv.Plotter()
            plotter.add_mesh(m, color="gray", style="wireframe", opacity=0.2, label="Original Mesh")
            plotter.add_mesh(m_inf, color="red", opacity=0.7, label="Inferior surface")
            plotter.add_mesh(m_sup, color="blue", opacity=0.7, label="Superior surface")
            plotter.add_mesh(m_inf_normals, color='orange')  # Normal vectors
            plotter.add_mesh(m_sup_normals, color='purple')  # Normal vectors
            plotter.add_text(f"Check the normals directions for {menisc}.")
            plotter.add_legend()
            plotter.show()
            check = input("Should the normals be flipped ? ('1' for Superior only, '2' for Inferior only, '3' for both): ")
            if check == "1":
                m_sup.flip_normals()
            if check == "2":
                m_inf.flip_normals()
            if check == "3":
                m_sup.flip_normals()
                m_inf.flip_normals()
        # Save the split meshes
        m_inf.save(os.path.join(dir, menisc.replace(".ply", "_inf.ply")))
        m_sup.save(os.path.join(dir, menisc.replace(".ply", "_sup.ply")))

    print("\tSplit finished.\n")

def tib_cartilages_merge(dir: str):
    """
    This function merge both lateral/medial parts of the tibia cartilage into a single file named "C_Tibia_cartilages_merged".
    :param dir: Directory where the cartilages mesh are stored (MUST BE NAMED "C_Tibia_cartilage_lateral" and "C_Tibia_cartilage_medial").
    """
    m_lat = pv.read(os.path.join(dir, "C_Tibia_cartilage_lateral.ply"))
    m_med = pv.read(os.path.join(dir, "C_Tibia_cartilage_medial.ply"))
    m_merged = m_lat + m_med
    m_merged.save(os.path.join(dir, "C_Tibia_cartilages_merged.ply"))
    return

def generate_tib_shift(path: str):
    """Load femur and tibia transformation matrices."""
    T_femur = np.loadtxt(os.path.join(path, 'Femur_ACS.txt'))
    T_tibia = np.loadtxt(os.path.join(path, 'Tibia_ACS.txt'))
    invTTib = np.linalg.inv(T_tibia)
    ft_diff = np.dot(invTTib, T_femur)
    return ft_diff[2, 3]  # Tib_Shift


def load_and_adjust_meshes(dir, tib_shift, patient_no):
    """Load and adjust the geometry meshes."""
    meshes = []
    order = ["Femur", "Tibia", "Patella",  #Bones
             "Femur_cartilage", "Tibia_cartilages_merged", "Patella_cartilage", #Cartilages
             "Menisc_lateral", "Menisc_medial", "Menisc_lateral_inf", "Menisc_medial_inf", "Menisc_lateral_sup", "Menisc_medial_sup"] #Meniscus
    for i in range(len(order)):
        flag = False
        for file in os.listdir(dir):
            if file.startswith("C_"):
                if file.split(".ply")[0].split("C_")[1] == order[i]:
                    flag = True
                    m = pv.PolyData(os.path.join(dir, file))
                    if ("Tibia" in file) or ("Menisc" in file):
                        m.points[:, 2] -= tib_shift
                    m.scale([1 / 1000, 1 / 1000, 1 / 1000], inplace=True) #Scale the meshes to correspond to meter unit
                    meshes.append(m)
        if flag == False:
            raise FileNotFoundError(f"The file {order[i]} was not found at: {os.path.join(dir, order[i])}.")
        else:
            geom_files = {
                'bone': meshes[:3],
                'cartilage': meshes[3:6],
                'meniscus': meshes[6:]
            }
    geom_files_remeshed = smith2019_remesh(geom_files)
    return mri_to_osim_transpose(dir, geom_files_remeshed, patient_no)

def mri_to_osim_transpose(dir, geom_files, patient_no):
    body_list = ['femur','tibia','patella']
    model_name = patient_no + "_GenericModelInjection"
    r_transposition = np.array([[0,1,0],[0,0,1],[1,0,0]])
    if not os.path.exists(os.path.join(dir,'Geometry')):
        os.mkdir(os.path.join(dir,'Geometry'))
    for surf in ['bone', 'cartilage', 'meniscus']:
        if surf == 'meniscus':
            i = 0
            for split in ['', '-inferior', '-superior']:
                for side in ['lateral', 'medial']:
                    geom_files[surf][i].points = np.transpose(np.matmul(r_transposition,geom_files[surf][i].points.transpose()))
                    geom_files[surf][i].save(os.path.join(dir,'Geometry',model_name+'-'+ side + '-' + surf + split + '.stl'))
                    print(model_name+'-'+ side + '-' + surf + split + '.stl')
                    i += 1
        else:
            for i in range(3):
                geom_files[surf][i].points = np.transpose(np.matmul(r_transposition,geom_files[surf][i].points.transpose()))
                geom_files[surf][i].save(os.path.join(dir,'Geometry',model_name+'-'+body_list[i]+'-'+surf+'.stl'))
                print(model_name+'-'+body_list[i]+'-'+surf+'.stl')
    return geom_files

def smith2019_remesh(geom_files, smith_dir = "./Morphing_scripts/insertion/Geometry", side="R"):
    """
    This function remesh the segmented cartilages and meniscus to have the same amount of point than in the smith2019 model.
    """
    #smith_file_names = [f"smith2019-{side}-femur-bone", f"smith2019-{side}-tibia-bone", f"smith2019-{side}-patella-bone", # Bones
    #                    f"smith2019-{side}-femur-cartilage", f"smith2019-{side}-tibia-cartilage", f"smith2019-{side}-patella-cartilage", #Cartilages
    #                    f"smith2019-{side}-lateral-meniscus", f"smith2019-{side}-medial-meniscus", f"smith2019-{side}-lateral-meniscus-inferior", f"smith2019-{side}-medial-meniscus-inferior", f"smith2019-{side}-lateral-meniscus-superior", f"smith2019-{side}-medial-meniscus-superior"] #Meniscus
    smith_file_names_to_remesh = [f"smith2019-{side}-femur-cartilage", f"smith2019-{side}-tibia-cartilage", f"smith2019-{side}-patella-cartilage", #Cartilages
                                  f"smith2019-{side}-lateral-meniscus-inferior", f"smith2019-{side}-medial-meniscus-inferior", f"smith2019-{side}-lateral-meniscus-superior", f"smith2019-{side}-medial-meniscus-superior"]  # Meniscus
    for i in range(3): # Remesh all the cartilages
        geom_files['cartilage'][i] = utb.ggremesh(geom_files['cartilage'][i], opts={'nb_pts': pv.PolyData(os.path.join(smith_dir, smith_file_names_to_remesh[i]) + ".stl").n_points,})
    for i in range(4): # Remesh the inferior/superior meniscus only
        geom_files['meniscus'][i+2] = utb.ggremesh(geom_files['meniscus'][i+2], opts={'nb_pts': pv.PolyData(os.path.join(smith_dir, smith_file_names_to_remesh[i+3]) + ".stl").n_points,})
    return geom_files

def update_model(geom_files, dir_files, output_dir, patient_no):
    """Update the model with new geometry and parameters."""
    ref_model_file = 'PFI_smith2019.osim'
    ref_model_other_files = ['lenhart-PFI_markers.xml', 'smith2019_reserve_actuators.xml']
    ref_geometry_dir = os.path.join(dir_files, 'Geometry')

    fitpts_file = os.path.join(dir_files, 'fitpts.json')
    ligament_info_file = os.path.join(dir_files, 'ligaments3.json')
    muscle_info_file = os.path.join(dir_files, 'muscles.json')

    model_name = patient_no + "_GenericModelInjection"

    jam_updateModel.copy_model_files(output_dir, dir_files, ref_model_file, ref_model_other_files, model_name)
    jam_updateModel.update_geometry(geom_files, model_name, ref_geometry_dir, ligament_info_file,
                                    muscle_info_file, fitpts_file, output_dir, show_plot=True)


def entry():
    """This is the entry function of the MSK insertion model script. Please use this script to insert a full knee model
    within a generic MSK .osim model for further simulation processes."""
    dir_knee_model = input(r"Enter the path to the knee models' folder to insert (default: ..\Data\RMIs\to_predict\results)"
                           "\nWARNING: The folders' names must have the 'morphed' suffix."
                           "\n\t--> ")
    if dir_knee_model == '':
        dir_knee_model = r'..\Data\RMIs\to_predict\results'
    dir_insertion_files = './Morphing_scripts/insertion'
    for folder in os.listdir(dir_knee_model):
        if ("morphed" in folder) and ("43" in folder):
            print(f"\nProcessing the insertion of {folder}...")
            output_dir = os.path.join(dir_knee_model, folder)
            patient_no = folder.split('_')[0]
            dir = os.path.join(dir_knee_model, folder)
            menisc_split(dir, plot=True)
            tib_cartilages_merge(dir)
            tib_shift = generate_tib_shift(dir)
            geom_files = load_and_adjust_meshes(dir, tib_shift, patient_no)
            update_model(geom_files, dir_insertion_files, output_dir, patient_no)


if __name__ == "__main__":
    entry()
