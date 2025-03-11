import os
import shutil
import sys
import opensim as osim
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
sys.path.append(r'Morphing_scripts\knee-model-tools')
sys.path.append(r'Morphing_scripts\insertion')
sys.path.append(r'Morphing_scripts\insertion\osimProcessing')
sys.path.append(r'Morphing_scripts\insertion\smith2019_scaling_scripts')
import utils_bis as utb
import utils as ut
from meshEditting import loadGeom, coordFrameFunc as frames, misc as miscTools, morphingTools
import scaleCOMAKmodel_popups as scm
from osimProcessing import osimTools
from osimProcessing import comakTools


def scale_smith2019_from_patient_static_trial(patient_dir, model_name, side = 'R', modeltype = ''):
    """
    Thus function scale the PFI_smith2019.osim model according to the patient's static trial (must be a .trc file).
    """
    # The processed can be improved to automatically find the .trc file, or make sure that the .trc file is kept in the whole process.
    pluginDir = r"Morphing_scripts\insertion\jam-plugin\build\Release\osimJAMPlugin"
    osim.common.LoadOpenSimLibrary(str(pluginDir))
    mass = float(input(f"\tThe smith2019 osim model needs to be scaled according to patient's static trial.\n"
          f"\tPLEASE make sure the static trial 'static.trc' file is present at: {patient_dir}.\n"
          f"\tThen enter the mass of the patient in kg here: "))
    if not os.path.exists(os.path.join(patient_dir, "static.trc")):
        raise FileNotFoundError(f"The file 'static.trc' does not exist at {patient_dir}")
    trc_dir = os.path.join(patient_dir, "static.trc")
    if side == 'L':
        xml_dir = os.path.join("Morphing_scripts", "insertion", "Geometry", "scale_smith_left.xml")
        gen_model_dir = os.path.join("Morphing_scripts", "insertion", "Geometry", "PFI_smith2019_L.osim")
    else:
        xml_dir = os.path.join("Morphing_scripts", "insertion", "Geometry", "scale_smith_right.xml")
        gen_model_dir = os.path.join("Morphing_scripts", "insertion", "Geometry", "PFI_smith2019_R.osim")
    scaleModelDir = scm.scaleOsimModel(patient_dir, model_name, trc_dir, mass, side, modeltype, xml_dir, gen_model_dir)
    scm.comakTools.scaleCOMAKcontactGeoms(scaleModelDir)
    print("\n\tsmith2019 successfully scaled using the patient's static MoCap.")
    return

def menisc_split(dir: str, plot: bool = True):
    """
    Thus function splits and save the meniscus in 2 inferior/superior open surfaces.
    :param dir: The directory to the Meniscus (MUST BE NAMED "C_Menisc_medial.ply", "C_Menisc_lateral.ply")
    :param plot: Set to True to plot the result of the split.
    """
    print(f"\n\tSplitting meniscus using ray tracing... You will have to check the normals direction that must be toward the outside of the mesh..")
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


def load_and_adjust_meshes(dir, tib_shift, patient_no, side):
    """Load and adjust the geometry meshes."""
    meshes = []
    order = ["Femur", "Tibia", "Patella",  #Bones
             "Femur_cartilage", "Tibia_cartilages_merged", "Patella_cartilage", #Cartilages
             "Menisc_lateral", "Menisc_medial", "Menisc_lateral_inf", "Menisc_medial_inf", "Menisc_lateral_sup", "Menisc_medial_sup", #Meniscus
             "Fibula"] # Fibula in last position because I'm too lazy to re-write the code with the right meshes order.
    for i in range(len(order)):
        flag = False
        for file in os.listdir(dir):
            if file.startswith("C_"):
                if file.split(".ply")[0].split("C_")[1] == order[i]:
                    flag = True
                    m = pv.PolyData(os.path.join(dir, file))
                    if ("Tibia" in file) or ("Menisc" in file) or ("Fibula" in file):
                        m.points[:, 2] -= tib_shift
                    m.scale([1 / 1000, 1 / 1000, 1 / 1000], inplace=True) #Scale the meshes to correspond to meter unit
                    meshes.append(m)
        if flag == False:
            raise FileNotFoundError(f"The file {order[i]} was not found at: {os.path.join(dir, order[i])}.")
    # Sort the parts
    geom_files = {
        'bone': meshes[:3],
        'cartilage': meshes[3:6],
        'meniscus': meshes[6:],
        'fibula': meshes[-1]}
    # Check the side orientation (right or left knee)
    geom_files_remeshed = smith2019_remesh(geom_files, side=side)
    mri_to_osim_transpose(dir, geom_files_remeshed, patient_no) # The transformed files are saved though this function
    return

def mri_to_osim_transpose(dir, geom_files, patient_no):
    """
    Thus function transpose the meshes from the MRI axes orientation to the Opensim orientation.
    The following transformation is applied: X --> Z, Y --> X, Z --> Y.
    :param dir:
    :param geom_files:
    :param patient_no:
    :return:
    """
    body_list = ['femur','tibia','patella']
    r_transposition = np.array([[0,1,0],[0,0,1],[1,0,0]]) # X --> Z, Y --> X, Z --> Y.
    for surf in ['bone', 'cartilage', 'meniscus']:
        if surf == 'meniscus':
            i = 0
            for split in ['', '-inferior', '-superior']:
                for side in ['lateral', 'medial']:
                    geom_files[surf][i].points = np.transpose(np.matmul(r_transposition,geom_files[surf][i].points.transpose()))
                    geom_files[surf][i].save(os.path.join(dir,'Geometry',patient_no+'-'+ side + '-' + surf + split + '.stl'))
                    i += 1
        else:
            for i in range(3):
                geom_files[surf][i].points = np.transpose(np.matmul(r_transposition,geom_files[surf][i].points.transpose()))
                geom_files[surf][i].save(os.path.join(dir,'Geometry',patient_no+'-'+body_list[i]+'-'+surf+'.stl'))
    # Addition of the Fibula transposition
    geom_files['fibula'].points = np.transpose(np.matmul(r_transposition, geom_files['fibula'].points.transpose()))
    geom_files['fibula'].save(os.path.join(dir, 'Geometry', patient_no + '-' + 'fibula' + '-' + 'bone' + '.stl'))
    return geom_files

def smith2019_remesh(geom_files, smith_dir = "./Morphing_scripts/insertion/Geometry", side="R"):
    """
    This function remesh the segmented cartilages and meniscus to have the same amount of point than in the smith2019 model.
    """
    smith_file_names_to_remesh = [f"smith2019-{side}-femur-cartilage", f"smith2019-{side}-tibia-cartilage", f"smith2019-{side}-patella-cartilage", #Cartilages
                                  f"smith2019-{side}-lateral-meniscus-inferior", f"smith2019-{side}-medial-meniscus-inferior", f"smith2019-{side}-lateral-meniscus-superior", f"smith2019-{side}-medial-meniscus-superior"]  # Meniscus
    for i in range(3): # Remesh all the cartilages
        geom_files['cartilage'][i] = utb.ggremesh(geom_files['cartilage'][i], opts={'nb_pts': pv.PolyData(os.path.join(smith_dir, smith_file_names_to_remesh[i]) + ".stl").n_points,})
    for i in range(4): # Remesh the inferior/superior meniscus only
        geom_files['meniscus'][i+2] = utb.ggremesh(geom_files['meniscus'][i+2], opts={'nb_pts': pv.PolyData(os.path.join(smith_dir, smith_file_names_to_remesh[i+3]) + ".stl").n_points,})
    return geom_files

def modify_scaled_smith2019_osim_file(dir_osim, model_name, side):
    """Update mesh file names in an OpenSim 4.3 .osim model file."""
    parser = ET.XMLParser(target=ET.TreeBuilder())
    tree = ET.parse(dir_osim, parser)
    root = tree.getroot()[0]  # First child of the root contains model data
    root.attrib['name'] = model_name  # Update model name
    side = side.lower()
    body_mesh_updates = {
        f"femur_distal_{side}": {"femur_bone": "femur-bone", "femur_cartilage": "femur-cartilage"},
        f"tibia_proximal_{side}": {"tibia_bone": "tibia-bone", "tibia_cartilage": "tibia-cartilage", "fibula_bone": "fibula-bone"},
        f"patella_{side}": {"patella_bone": "patella-bone", "patella_cartilage": "patella-cartilage"},
        f"meniscus_lateral_{side}": {f"meniscus_lateral_{side}": "lateral-meniscus"},
        f"meniscus_medial_{side}": {f"meniscus_medial_{side}": "medial-meniscus"},
    }
    # Update BodySet geometry
    body_set = root.find("BodySet")[0]
    for body_name, meshes in body_mesh_updates.items():
        for mesh_name, mesh_suffix in meshes.items():
            mesh_elem = body_set.findall(f"./Body[@name='{body_name}']/attached_geometry/Mesh[@name='{mesh_name}']/mesh_file")
            if mesh_elem:
                mesh_elem[0].text = f"{model_name}-{mesh_suffix}.stl"
            else:
                print(f"Warning: Mesh file element for {mesh_name} in {body_name} not found.")

    # Update ContactGeometrySet
    contact_mesh_updates = {
        "femur_cartilage": ["femur-cartilage", "femur-bone"],
        "tibia_cartilage": ["tibia-cartilage", "tibia-bone"],
        "patella_cartilage": ["patella-cartilage", "patella-bone"],
        "meniscus_med_sup": ["medial-meniscus-superior"],
        "meniscus_med_inf": ["medial-meniscus-inferior"],
        "meniscus_lat_sup": ["lateral-meniscus-superior"],
        "meniscus_lat_inf": ["lateral-meniscus-inferior"],
    }

    contact_set = root.find("ContactGeometrySet")[0]
    for contact_name, mesh_suffixes in contact_mesh_updates.items():
        for i, mesh_suffix in enumerate(mesh_suffixes):
            mesh_tag = "mesh_file" if i == 0 else "mesh_back_file"
            mesh_elem = contact_set.findall(f"./Smith2018ContactMesh[@name='{contact_name}']/{mesh_tag}")
            if mesh_elem:
                mesh_elem[0].text = f"{model_name}-{mesh_suffix}.stl"
            else:
                print(f"Warning: {mesh_tag} for {contact_name} not found.")

    # Write back to the .osim file
    tree.write(dir_osim.split(".osim")[0] + "_modified.osim", encoding="utf8", method="xml")
    print(f" Successfully modified osim file in {dir_osim}")

def copy_smith2019_files(smith2019_dir, dir):
    """
    This function copies all the smith2019 original files into the patient's Geometry folder.
    This step is a necessary requirement for the update_geometry() process.
    :param smith2019_dir: The smith2019 original directory path.
    :param dir: Patient's meshes path.
    """
    for file in os.listdir(smith2019_dir):
        if os.path.isfile(os.path.join(smith2019_dir, file)) and (not file.startswith(".")):
            shutil.copy(os.path.join(smith2019_dir, file), os.path.join(dir, file))

def update_geometry(patient_no, dir, scaledModel, side):
    """
    Thus function updates the generic model (smith2019) geometry with the muscle, ligements and wrapping surfaces
    according to the scaled .osim model (smith2019 scaled on patient).
    :param patient_no: The patient number.
    :param dir: The patient model directoey (where the patent's "Geometry" folder is located)
    :param scaledModel: The loaded scaled smith2019 model.
    :param side: R or L
    """
    model_name = f"smith2019_{patient_no}"
    ss = side.lower()
    Geom_dir = os.path.join(dir,'Geometry')
    fem = loadGeom.loadMesh(os.path.join(Geom_dir,patient_no+'-femur-bone.stl'))
    localFemurPoints = fem._points
    tib = loadGeom.loadMesh(os.path.join(Geom_dir,patient_no+'-tibia-bone.stl'))
    localTibiaPoints = tib._points
    pat = loadGeom.loadMesh(os.path.join(Geom_dir,patient_no+'-patella-bone.stl'))
    localPatellaPoints = pat._points
    ML = loadGeom.loadMesh(os.path.join(Geom_dir,patient_no+'-lateral-meniscus.stl'))
    localMLPoints = ML._points
    MM = loadGeom.loadMesh(os.path.join(Geom_dir,patient_no+'-medial-meniscus.stl'))
    localMMPoints = MM._points


    print('  ~  Remapping muscle, ligament points and wrapping surfaces  ~  ')

    bodyPathPoints = osimTools.getForcePathPointsOnBody(scaledModel , 'femur_distal_' + ss)
    forcePoints = np.array([bodyPathPoints[i]['location'] for i in bodyPathPoints.keys()])

    wrapPathPoints = osimTools.getWrapSurfacesOnBody(scaledModel, 'femur_distal_' + ss)
    wrapPoints = np.array([wrapPathPoints[i] for i in wrapPathPoints.keys()])

    srcMesh = loadGeom.loadMesh(os.path.join(Geom_dir,'Smith2019-R-femur-bone.stl')) if side == 'R' else loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-L-femur-bone.stl'))

    scaleT , hmfT = morphingTools.regScaleHMFGeomOutputTrans(srcMesh._points, localFemurPoints, False)
    morphedPoints = morphingTools.applyHMFoutputToPassivePointsWithProjection(scaleT , hmfT , srcMesh._points , forcePoints, localFemurPoints)
    scaledModel = osimTools.updPathPathsFromHMF(scaledModel, bodyPathPoints, morphedPoints)

    morphedPoints = morphingTools.applyHMFoutputToPassivePoints(scaleT , hmfT , srcMesh._points , wrapPoints)
    scaledModel = osimTools.updWrapPointsFromHMF(scaledModel, wrapPathPoints, morphedPoints, 'femur_distal_' + ss)

    bodyPathPoints = osimTools.getForcePathPointsOnBody(scaledModel , 'tibia_proximal_' + ss)
    forcePoints = np.array([bodyPathPoints[i]['location'] for i in bodyPathPoints.keys()])

    wrapPathPoints = osimTools.getWrapSurfacesOnBody(scaledModel, 'tibia_proximal_' + ss)
    wrapPoints = np.array([wrapPathPoints[i] for i in wrapPathPoints.keys()])

    srcMesh = loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-R-tibia-bone.stl')) if side == 'R' else loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-L-tibia-bone.stl'))

    scaleT , hmfT = morphingTools.regScaleHMFGeomOutputTrans(srcMesh._points, localTibiaPoints, False)
    morphedPoints = morphingTools.applyHMFoutputToPassivePointsWithProjection(scaleT , hmfT , srcMesh._points , forcePoints, localTibiaPoints)
    scaledModel = osimTools.updPathPathsFromHMF(scaledModel, bodyPathPoints, morphedPoints)

    morphedPoints = morphingTools.applyHMFoutputToPassivePoints(scaleT , hmfT , srcMesh._points , wrapPoints)
    scaledModel = osimTools.updWrapPointsFromHMF(scaledModel, wrapPathPoints, morphedPoints, 'tibia_proximal_' + ss)


    tmplig_list = ['LCL1', 'LCL2', 'LCL3', 'LCL4', 'PFL1', 'PFL2', 'PFL3', 'PFL4', 'PFL5']
    for selLig in tmplig_list:
        forceObj = scaledModel.getForceSet().get(selLig)
        geomPath = osim.GeometryPath.safeDownCast(forceObj.getPropertyByName('GeometryPath').getValueAsObject())
        ppSet = osim.PathPointSet.safeDownCast(geomPath.getPathPointSet())
        pp = osim.PathPoint.safeDownCast(ppSet.get((selLig +'-P2')))
        Location = pp.get_location()

        forceObj2 = scaledModel.getForceSet().get(selLig)
        geomPath2 = osim.GeometryPath.safeDownCast(forceObj2.getPropertyByName('GeometryPath').getValueAsObject())
        ppSet2 = osim.PathPointSet.safeDownCast(geomPath2.getPathPointSet())
        pp_personalisedModel = osim.PathPoint.safeDownCast(ppSet2.get(selLig + '-P2'))
        pp_personalisedModel.setLocation(Location)

        scaledModel.upd_ForceSet()

    bodyPathPoints = osimTools.getForcePathPointsOnBody(scaledModel , 'patella_' + ss)
    forcePoints = np.array([bodyPathPoints[i]['location'] for i in bodyPathPoints.keys()])

    wrapPathPoints = osimTools.getWrapSurfacesOnBody(scaledModel, 'patella_' + ss)
    wrapPoints = np.array([wrapPathPoints[i] for i in wrapPathPoints.keys()])

    srcMesh = loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-R-patella-bone.stl')) if side == 'R' else loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-L-patella-bone.stl'))

    scaleT , hmfT = morphingTools.regScaleHMFGeomOutputTrans(srcMesh._points, localPatellaPoints, False)
    morphedPoints = morphingTools.applyHMFoutputToPassivePointsWithProjection(scaleT , hmfT , srcMesh._points , forcePoints, localPatellaPoints)
    scaledModel = osimTools.updPathPathsFromHMF(scaledModel, bodyPathPoints, morphedPoints)

    morphedPoints = morphingTools.applyHMFoutputToPassivePoints(scaleT , hmfT , srcMesh._points , wrapPoints)
    scaledModel = osimTools.updWrapPointsFromHMF(scaledModel, wrapPathPoints, morphedPoints, 'patella_' + ss)

    scaledModel.upd_BodySet()

    bodyPathPoints = osimTools.getForcePathPointsOnBody(scaledModel , 'meniscus_lateral_' + ss)
    forcePoints = np.array([bodyPathPoints[i]['location'] for i in bodyPathPoints.keys()])

    srcMesh = loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-R-lateral-meniscus.stl')) if side == 'R' else loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-L-lateral-meniscus.stl'))

    scaleT , hmfT = morphingTools.regScaleHMFGeomOutputTrans(srcMesh._points, localMLPoints, False)
    morphedPoints = morphingTools.applyHMFoutputToPassivePointsWithProjection(scaleT , hmfT , srcMesh._points , forcePoints, localMLPoints)
    scaledModel = osimTools.updPathPathsFromHMF(scaledModel, bodyPathPoints, morphedPoints)

    scaledModel.upd_BodySet()

    bodyPathPoints = osimTools.getForcePathPointsOnBody(scaledModel , 'meniscus_medial_' + ss)
    forcePoints = np.array([bodyPathPoints[i]['location'] for i in bodyPathPoints.keys()])


    srcMesh = loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-R-medial-meniscus.stl')) if side == 'R' else loadGeom.loadMesh(os.path.join(Geom_dir,'smith2019-L-medial-meniscus.stl'))

    scaleT , hmfT = morphingTools.regScaleHMFGeomOutputTrans(srcMesh._points, localMMPoints, False)
    morphedPoints = morphingTools.applyHMFoutputToPassivePointsWithProjection(scaleT , hmfT , srcMesh._points , forcePoints, localMMPoints)
    scaledModel = osimTools.updPathPathsFromHMF(scaledModel, bodyPathPoints, morphedPoints)


    scaledModel.upd_BodySet()
    # Capsule_r
    scaledModel.getBodySet().get('femur_distal_' + ss).getWrapObjectSet().get('Capsule_' + ss).set_quadrant('all')

    # Med_lig
    scaledModel.getBodySet().get('tibia_proximal_' + ss).getWrapObjectSet().get('Med_Lig_' + ss).set_quadrant('-Z')

    # Med_ligP
    scaledModel.getBodySet().get('tibia_proximal_' + ss).getWrapObjectSet().get('Med_LigP_' + ss).set_quadrant('all')

    # PatTen
    scaledModel.getBodySet().get('patella_' + ss).getWrapObjectSet().get('PatTen_' + ss).set_quadrant('all')
    scaledModel.upd_BodySet()

    #Save the result
    print(f'  ~  Saving model as {model_name}_personalized.osim..  ~  ')
    scaledModel.printToXML(os.path.join(dir,model_name+'_personalized.osim'))

def entry():
    """This is the entry function of the MSK insertion model script. Please use this script to insert a full knee model
    within a generic MSK .osim model for further simulation processes."""
    # -----Inputs: the morphed segmentations' directory---------------------
    dir_knee_model = input(r"Enter the path to the knee models' folder to insert (default: ..\Data\RMIs\to_predict\results)"
                           "\nWARNING: The folders' names must have the 'morphed' suffix."
                           "\n\t--> ")
    if dir_knee_model == '':
        dir_knee_model = r'..\Data\RMIs\to_predict\results'
    dir_insertion_files = './Morphing_scripts/insertion'

    # -----Pass through every morphed segmentations in the segmentations' directory---------------------
    for folder in os.listdir(dir_knee_model):
        if ("morphed" in folder):
            print(f"\nProcessing the insertion of {folder} within the smith2019 model...")

            # -----Specific parameters---------------------
            output_dir = os.path.join(dir_knee_model, folder)
            patient_no = folder.split('_')[0]
            dir = os.path.join(dir_knee_model, folder)
            side = ut.check_side(output_dir)
            if not os.path.exists(os.path.join(dir, 'Geometry')):  # Necessary folder within the patient's directory for temporary files.
                os.mkdir(os.path.join(dir, 'Geometry'))
            dir_smith2019 = os.path.join(dir_insertion_files, "Geometry")

            # -----Scale the smith2019 model using patient's static mocap trial---------------------
            copy_smith2019_files(dir_smith2019, os.path.join(dir, "Geometry"))
            scale_smith2019_from_patient_static_trial(dir, "smith2019_" + patient_no, side=side)

            # -----Meniscus transversally split and tibia cartilages merged---------------------
            menisc_split(dir, plot=True)
            tib_cartilages_merge(dir)

            # -----Load the meshes, apply several transformations (Scaling, reposition, remesh, axis transposition)-----
            tib_shift = generate_tib_shift(dir)
            load_and_adjust_meshes(dir, tib_shift, patient_no, side)
            dir_osim = os.path.join(dir, "smith2019_" + patient_no + "_scaled_" + side + ".osim")
            scaledModel = "smith2019_" + patient_no + "_scaled_" + side + "_modified.osim"

            # -----Change the scaled smith2019 to fit the patient's knee part's names, then loads it--------------
            modify_scaled_smith2019_osim_file(dir_osim, patient_no, side=side)
            loadedModel = comakTools.loadCOMAKModel(os.path.join(dir, scaledModel))

            # -----Update the smith2019 scaled model with the patient's knee geometry, save it and vizualise it.---------------------
            update_geometry(patient_no, output_dir, loadedModel, side)

            # -----Cleaning the segmentation's folder---------------------
            for file in os.listdir(dir):
                if ((not file.startswith('C_')) and (not file.endswith('personalized.osim')) and
                        (not file.endswith('ACS.txt')) and (not file.startswith("static")) and (not file == "Geometry")):
                    if os.path.isfile(os.path.join(dir, file)):
                        os.remove(os.path.join(dir, file))
                    else:
                        shutil.rmtree(os.path.join(dir, file))
            for file in os.listdir(os.path.join(dir, 'Geometry')):
                if os.path.isfile(os.path.join(dir, file)) and (not file.startswith(patient_no)):
                    os.remove(os.path.join(dir, "Geometry", file))
            print("Insertion process finished and files cleaned.")



if __name__ == "__main__":
    entry()