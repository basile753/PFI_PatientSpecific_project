#%% -*- coding: utf-8 -*-
"""
This script processes patient-specific meshes by:
1. Remeshing the meshes to match the reference model's points.
2. Aligning meshes to the Anatomical Coordinate System (ACS).
3. Using Coherent Point Drift (CPD) for node correspondence with the reference model.

@authors: Annagh Macie ft Antoine Basile
"""

import os
import pyvista as pv
import numpy as np
import sys

# Add necessary paths for external libraries
sys.path.append(r'.\knee-model-tools')
import utils_bis as utb
import ssm
import buildACS
import gbcpd

# Directories
DIR_MAIN = os.getcwd()
MESH_DIR = input('Enter the path to the meshes to process (default: ./Data/meshes): ')
if MESH_DIR == '':
    MESH_DIR = './Data/meshes'
if not os.path.exists(MESH_DIR):
    raise FileNotFoundError(f'{MESH_DIR} is not a valid directory')
DIR_REF = input('Enter the path to the reference models (default: ./Data/reference_meshes): ')
if DIR_REF == '':
    DIR_REF = './Data/reference_meshes'
if not os.path.exists(DIR_REF):
    raise FileNotFoundError(f'{DIR_REF} is not a valid directory')
DIR_CORRESP = os.path.join(MESH_DIR, 'corresp') # path where the morphed result will be stored
os.makedirs(DIR_CORRESP, exist_ok=True)
DIR_BCPD = DIR_MAIN + r'\bcpd-master\win'  # file path where GBCPD exe is found

# File and Body Definitions
BODY_NAMES = [
    'Femur', 'Tibia', 'Patella', 'Femur_cartilage', 'Tibia_cartilage_lateral', 'Tibia_cartilage_medial',
    'Patella_cartilage', 'Menisc_lateral', 'Menisc_medial', 'Fibula'
]

# --- HELPER FUNCTIONS ---

def load_ref_points(dir_ref, body_names):
    """Loads the number of points for reference meshes."""
    n_pts_ref = []
    for body in body_names:
        mesh = pv.PolyData(os.path.join(dir_ref, f'{body}.stl'))
        n_pts_ref.append(mesh.n_points)
        print(f'The reference mesh for {body} has {mesh.n_points} points.')
    return np.array(n_pts_ref)


def remesh_body(mesh_dir, body_names, n_pts_ref):
    """Remeshes each body to match the number of points of the reference model."""
    bodies = []
    os.makedirs(os.path.join(mesh_dir, 'remeshed'), exist_ok=True)
    for i, name in enumerate(body_names):
        fname = f'{name}.stl'
        mesh = pv.PolyData(os.path.join(mesh_dir, fname))
        n_target = int(n_pts_ref[i])
        remeshed_mesh = utb.ggremesh(mesh, opts={'nb_pts': n_target}, ggremesh_prog=r'geogram\win64\bin\vorpalite.exe')
        remeshed_mesh.save(os.path.join(mesh_dir, 'remeshed', fname))
        bodies.append(ssm.meshSet([remeshed_mesh]))
        print(f'The mesh {name} has been remeshed to {n_target} points.')
    return bodies

def create_femur_acs(bodies, mesh_dir):
    """
    Create the Anatomical Coordinate System (ACS) for the femur.

    Parameters:
        bodies (list): List of remeshed body meshes.
        mesh_dir (str): Directory for saving ACS files.
    """

    for m in bodies[0].meshes:
        height = np.ptp(m.points[:, 2])
        width = np.ptp(m.points[:, 0])

        if height / width < 1.4:
            d_add = 1.4 * width - height
            Itop = (m.point_normals[:, 2] > 0.8) & (m.points[:, 2] > m.points[:, 2].max() - 10)
            m.points[Itop, 2] += d_add
            m = utb.ggremesh(m, opts={'nb_pts': 10000})
            T = buildACS.buildfACS(m, plotACS=False)
        else:
            T = buildACS.buildfACS(m, plotACS=False)

        np.savetxt(os.path.join(mesh_dir, 'remeshed', 'Femur_ACS.txt'), T)


def create_tibia_acs(bodies, mesh_dir):
    """
    Create the Anatomical Coordinate System (ACS) for the tibia.

    Parameters:
        bodies (list): List of remeshed body meshes.
        mesh_dir (str): Directory for saving ACS files.
    """

    for m in bodies[1].meshes:
        height = np.ptp(m.points[:, 2])
        width = np.ptp(m.points[:, 0])

        if height / width < 1.1:
            d_add = 1.1 * width - height
            Ibot = (m.point_normals[:, 2] < -0.8) & (m.points[:, 2] < m.points[:, 2].min() + 10)
            m.points[Ibot, 2] -= d_add
            m = utb.ggremesh(m, opts={'nb_pts': 10000})
            I_ant = np.argmin(m.points[:, 1])
            T = buildACS.buildtACS(m, m.points[I_ant, :], plotACS=False)
        else:
            I_ant = np.argmin(m.points[:, 1])
            T = buildACS.buildtACS(m, m.points[I_ant, :], plotACS=False)

        np.savetxt(os.path.join(mesh_dir, 'remeshed', 'Tibia_ACS.txt'), T)

def create_patella_acs(bodies, mesh_dir):
    """See other ACS docstrings"""
    for m in bodies[2].meshes:
        T = buildACS.buildpACS(m, 'R', plotACS=False)
        np.savetxt(os.path.join(mesh_dir, 'remeshed', 'Patella_ACS.txt'), T)

def align_meshes_to_ACS(body_names, mesh_dir):
    """Aligns the meshes to their Anatomical Coordinate Systems (ACS)."""
    aligned_meshes = []
    for body in body_names:
        fname = body + '.stl'
        m = pv.PolyData(os.path.join(mesh_dir, 'remeshed', fname))
        # Every part that is attached to a body (cartilages) have the same ACS alignment
        tibia_related = ['Tibia', 'Menisc', 'Fibula']
        if 'Patella' in body:
            part = 'Patella'
        elif 'Femur' in body:
            part = 'Femur'
        else:
            for elem in tibia_related:
                if elem in body:
                    part = 'Tibia'
        T = np.loadtxt(os.path.join(mesh_dir, 'remeshed', part + '_ACS.txt'))
        m = m.transform(np.linalg.inv(T))
        aligned_meshes.append(m)

    return aligned_meshes


def perform_cpd(meshes_aligned, ref_files, dir_ref, dir_corresp, bodynames, dir_bcpd):
    """Performs Coherent Point Drift (CPD) on aligned meshes."""
    gbcpd_results = []
    refs = [pv.PolyData(os.path.join(dir_ref, file)) for file in ref_files]
    # Check if the knee is a left knee (from the relative position Fibula/Tibia)
    if meshes_aligned[-1].center[0]<meshes_aligned[1].center[0]:
        refs = reflect_meshes(refs) # Reflect the reference model to correspond to a left knee.
    for i, mesh in enumerate(meshes_aligned):
        ref = refs[i]
        result = gbcpd.run_gbcpd([mesh], ref, dir_corresp, bodyname=bodynames[i], labels='C', dir_bcpd=dir_bcpd, outsuffix="")
        gbcpd_results.append(result)

    return gbcpd_results

def plot_cpd_results(results, body_names):
    """Plots the CPD results part by part."""
    for i, result in enumerate(results):
        utb.plotpatch(result, opts={'opacity': [1], 'color': ['grey'], 'title': [f'Morphed {body_names[i]}']})

def reflect_meshes(bodies):
    """Reflect every mesh from the list of meshes (bodies) along the X axis and also flip the normals back
    (as they are flipped during the reflection process)."""
    reflected_bodies  = [mesh.reflect((1, 0, 0)) for mesh in bodies]  # reflects along the x axis
    for mesh in reflected_bodies:
        mesh.flip_normals()
    return reflected_bodies

# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    #%% Load reference points
    n_pts_ref = load_ref_points(DIR_REF, BODY_NAMES)

    #%% Remesh patient-specific meshes
    bodies = remesh_body(MESH_DIR, BODY_NAMES, n_pts_ref)

    #%% Create the ACS for every body
    create_femur_acs(bodies, MESH_DIR)
    create_tibia_acs(bodies, MESH_DIR)
    create_patella_acs(bodies, MESH_DIR)

    #%% Align meshes to ACS
    aligned_meshes = align_meshes_to_ACS(BODY_NAMES, MESH_DIR)

    #%% Plot aligned meshes
    #utb.plotpatch(aligned_meshes, opts={'opacity': [0.5] * 10,
    #                           'color': ['blue', 'red', 'green', 'yellow', 'pink', 'purple', 'black', 'orange', 'white',
    #                                     'grey']})

    #%% Perform CPD for node correspondence
    ref_files = [name + ".stl" for name in BODY_NAMES]
    cpd_results = perform_cpd(aligned_meshes, ref_files, DIR_REF, DIR_CORRESP, BODY_NAMES, DIR_BCPD)

    #%% Plot the morphing results
    #plot_cpd_results(cpd_results, BODY_NAMES)
