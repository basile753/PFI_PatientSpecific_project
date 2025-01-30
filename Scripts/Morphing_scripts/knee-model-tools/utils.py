# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:57:15 2023

@author: aclouthi
"""

import os
import pyvista as pv
import numpy as np
# from stl import mesh
# import vtk
import math
from scipy.optimize import leastsq
import re
from matplotlib import pyplot as plt
import subprocess


def mass_properties(mesh):
    '''
    Compute the basic properties for a mesh object based on the Divergence Theorem.
    
    Adapted from mass_properties.m. Note not all outputs from the original Matlab
    function are computed here.

    Parameters
    ----------
    mesh : pyvista.PolyData
        pyvista mesh object.

    Returns
    -------
    centroid : numpy.ndarray
        The centroid of the object.
    eigenvalues : numpy.ndarray
        The eigenvalues about the centre of mass.
    CoM_eigenvectors : numpy.ndarray
        The eigenvectors, which are unit vectors describing the inertial axes.

    '''
    
    # pts = mesh.points
    # cns = mesh.faces.reshape((-1,4))
    # cns = cns[:,1:]
    
    # -- Centroid -- #
    centroid = np.zeros(3)
    func_sum = np.zeros(3)
    for i in range(mesh.n_faces): #range(cns.shape[0]):
        p = mesh.get_cell(i).points #pts[cns[i,:]]
        
        # edge vectors
        ijk = np.zeros((3,3),dtype=float)
        ijk[0,:] = p[1,:] - p[0,:]
        ijk[1,:] = p[2,:] - p[0,:]
        ijk[2,:] = p[2,:] - p[1,:]
        
        u = np.cross(ijk[0,:],ijk[1,:])
        area = np.linalg.norm(u)/2
        if np.linalg.norm(u) != 0:
            u = u / np.linalg.norm(u)
        else:
            u = np.zeros(3)
            
        avg = p.mean(axis=0)
        # volume of current triangle
        t_vol = area * np.multiply(u,avg)
        
        func_sum = func_sum + np.multiply(t_vol,avg)
    
    func_sum = func_sum / 2
    centroid = func_sum / mesh.volume
    
    # -- Inertia -- #
    func_sum_inertia = np.zeros(3,dtype=float)
    func_sum_prod = np.zeros(3,dtype=float)
    for i in range(mesh.n_faces): #range(cns.shape[0]):
        p = mesh.get_cell(i).points - centroid#pts[cns[i,:],:] - centroid #mesh.center_of_mass() #np.array([3.17618046518640,14.0927729059696,18.1750888691410])
        # edge vectors
        ijk = np.zeros((3,3),dtype=float)
        ijk[0,:] = p[1,:] - p[0,:]
        ijk[1,:] = p[2,:] - p[0,:]
        ijk[2,:] = p[2,:] - p[1,:]
        
        
        u = np.cross(ijk[0,:],ijk[1,:])
        area = np.linalg.norm(u)/2
        if np.linalg.norm(u) != 0:
            u = u / np.linalg.norm(u)
        else:
            u = np.zeros(3)
    
        # volume elements
        avg = p.mean(axis=0)
        
        # inertia
        func_sum_inertia = func_sum_inertia + area * np.multiply(u,avg**3)
        # product of inertia
        func_sum_prod  = func_sum_prod + area * np.multiply(np.multiply(u[[1,0,2]],avg[[1,0,2]]**2),avg[[0,2,1]])
        
    func_sum_inertia = func_sum_inertia/3
    func_sum_prod = -0.5 * func_sum_prod
    
    I_CoM = np.array([[func_sum_inertia[1:].sum(), func_sum_prod[0], func_sum_prod[1]],
                      [func_sum_prod[0], func_sum_inertia[[0,2]].sum(), func_sum_prod[2]],
                      [func_sum_prod[1], func_sum_prod[2], func_sum_inertia[:2].sum()]])
    eigenvalues,CoM_eigenvectors = np.linalg.eig(I_CoM)
    idx_sort = np.argsort(eigenvalues)
    CoM_eigenvectors = CoM_eigenvectors[:,idx_sort]
    eigenvalues = eigenvalues[idx_sort]
    # CoM_eigenvectors_sorted = np.zeros((3,3),dtype=float)
    # for i in range(3):
    #     CoM_eigenvectors_sorted[:,i] = CoM_eigenvectors[:,idx_sort[i]]
       # CoM_eigenvectors_sorted[:,i] = np.sign(eigenvalues[idx_sort[i]])*CoM_eigenvectors[:,idx_sort[i]]
    
    
    # check if determinant = -1 (left-handed cs)
    if np.abs(np.linalg.det(CoM_eigenvectors)+1) < 1e-9:
        # CoM_eigenvectors[:,1] = -CoM_eigenvectors[:,1]
        CoM_eigenvectors = -CoM_eigenvectors
    
    return centroid,eigenvalues,CoM_eigenvectors
        
       
def sliceProperties(mesh,mesh_inertia,T_inertia,slice_thickness):
    '''
    Determine the properties of each slice.
    
    Adapted from sliceProperties.m, written by Daniel Miranda and Evan Leventhal
    at Brown University. 

    Parameters
    ----------
    mesh : pyvista.PolyData
        pyvista mesh object.
    mesh_inertia : pyvista.PolyData
        the mesh registered to its inertial axes and centroid.
    T_inertia : np.ndarray
        4x4 pose matrix of inertial axes with origin at centroid
    slice_thickness : float
        The thickness between slices.

    Returns
    -------
    output : dict
        Properties of each slice
        output['area'] : slice cross-sectional area
        output['centroid'] : centroid of each slice
        output['index'] : slice index
        output['min_ML_pt'] : minimum medial-lateral point
        output['max_ML_pt'] : maximum medial-lateral point
        output['ML_vector'] : medial-lateral vector

    '''
    
    # lets figure out which way along X of the inertial axes points me towards the femur condyles. 
    # The centroid (0,0,0) now should be closer to the condyles since its larger and has more mass.
    if (max(mesh_inertia.points[:,0]) < abs(min(mesh_inertia.points[:,0]))):
        # if the max value is smaller, then we are pointed the wrong way, flip X & Y to keep us straight
        T_inertia[:3,:2] = -T_inertia[:3,:2]
        mesh_inertia = mesh.transform(np.linalg.inv(T_inertia),inplace=False)
    
    max_x = np.amax(mesh_inertia.points[:,0])
    min_x = np.amin(mesh_inertia.points[:,0])
    
    r_y=np.empty((math.ceil(abs(min_x-max_x)/slice_thickness), 1))
    r_z=np.empty((math.ceil(abs(min_x-max_x)/slice_thickness), 1))
    area=np.empty((math.ceil(abs(min_x-max_x)/slice_thickness), 1))
    centroid_slice=np.empty((math.ceil(abs(min_x-max_x)/slice_thickness), 3))
    min_y_pt=np.empty((math.ceil(abs(min_x-max_x)/slice_thickness), 3))
    max_y_pt=np.empty((math.ceil(abs(min_x-max_x)/slice_thickness), 3))
    
    for i in range(math.ceil(abs(min_x-max_x)/slice_thickness)):      
        # Find slice points
        poly_pts_index = np.where(np.logical_and((mesh_inertia.points[:,0] >= (min_x + (i)*slice_thickness)), 
                                                 (mesh_inertia.points[:,0] < (min_x + (i+1)*slice_thickness))))
        
        # Find length and width of bounding box
        r_y[i,0] = np.amax(mesh_inertia.points[poly_pts_index,1])-np.amin(mesh_inertia.points[np.asarray(poly_pts_index),1])
        r_z[i,0] = np.amax(mesh_inertia.points[poly_pts_index,2])-np.amin(mesh_inertia.points[np.asarray(poly_pts_index),2])
        
        area[i,0] = r_y[i,0]*r_z[i,0] # calculate area of bounding box
        centroid_slice[i,0]=np.mean(mesh_inertia.points[poly_pts_index,0]) # calculate x coordinate centroid
        centroid_slice[i,1]=np.amin(mesh_inertia.points[poly_pts_index,1]) + \
                            (np.amax(mesh_inertia.points[poly_pts_index,1])- \
                             np.amin(mesh_inertia.points[poly_pts_index,1]))/2 # calculate y coordinate centroid
        centroid_slice[i,2]=np.amin(mesh_inertia.points[poly_pts_index,2]) + \
                            (np.amax(mesh_inertia.points[poly_pts_index,2])- \
                             np.amin(mesh_inertia.points[poly_pts_index,2]))/2 # calculate z coordinate centroid
    
        min_y_pt[i,0]=centroid_slice[i,0]
        min_y_pt[i,1] = np.amin(mesh_inertia.points[poly_pts_index,1])
        min_y_pt[i,2]=centroid_slice[i,2]
    
        max_y_pt[i,0]=centroid_slice[i,0]
        max_y_pt[i,1]=np.amax(mesh_inertia.points[poly_pts_index,1])      
        max_y_pt[i,2]=centroid_slice[i,2]

    
    min_y_pt_TF = np.transpose(np.matmul(T_inertia,np.concatenate((min_y_pt.transpose(),np.ones((1,min_y_pt.shape[0]))),axis=0)))
    max_y_pt_TF = np.transpose(np.matmul(T_inertia,np.concatenate((max_y_pt.transpose(),np.ones((1,max_y_pt.shape[0]))),axis=0)))
    
    max_ry_index=np.argmax(r_y)  
    
    centroid_slice = np.transpose(np.matmul(T_inertia,
                                  np.concatenate((centroid_slice.transpose(),np.ones((1,centroid_slice.shape[0]))),axis=0)))
    
    index = np.indices((1,area.shape[0]))
    index = index[1,0,:]
    
    
    output = {'area': area,'centroid': centroid_slice[:,:3], 'index': index, 
              'min_ML_pt': min_y_pt_TF[max_ry_index,:3], 'max_ML_pt': max_y_pt_TF[max_ry_index,:3],
              'ML_vector': max_y_pt_TF[max_ry_index]-min_y_pt_TF[max_ry_index]}
    output['ML_vector'] = output['ML_vector'] / np.linalg.norm(output['ML_vector'])
    return output
    
    # min_y_pt_TF=transformShell(min_y_pt,rt_i,1,1);
    # max_y_pt_TF=transformShell(max_y_pt,rt_i,1,1);
    # [max_r_y max_r_y_index]=max(r_y);
        
        # % move centroid pts back to CT space
        # output.area=area;
        # output.centroid=transformShell(centroid,rt_i,1,1);
        # output.index=1:length(area);
        # output.min_ML_pt=min_y_pt_TF(max_r_y_index,:);
        # output.max_ML_pt=max_y_pt_TF(max_r_y_index,:);
        # output.ML_vector=unit(output.max_ML_pt-output.min_ML_pt); % insure ML_vector is a unit vector 


# IGNORE THIS, NOT FULLY WORKING
# need to include checks for normal direction and make sure created triangles are within the bounds of the mesh
# def fill_hole(mesh):
#     edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
#     # plotpatch(mesh,points_list=edges.points)
    
#     # get the order of the points on the edge
#     # edge_pts_sorted = np.zeros(edges.points.shape)
#     # remain = np.ones(edges.points.shape[0],dtype=bool) # indices of unsorted edge points
#     # edge_pts_sorted[0:2,:] = edges.cell_points(0)
#     # remain[0] = False
#     cell_pt_ids = np.zeros((edges.n_cells,2),dtype=int)
#     for i in range(cell_pt_ids.shape[0]):
#         cell_pt_ids[i,:] = edges.cell_point_ids(i)
#     cell_pt_ids_sorted = np.zeros(cell_pt_ids.shape,dtype=int)
#     cell_pt_ids_sorted[0,:] = cell_pt_ids[0,:]
    
#     cell_pt_ids[0,0] = -1
#     for i in range(1,cell_pt_ids.shape[0]-1):
#         j = np.where(cell_pt_ids[:,0]==cell_pt_ids_sorted[i-1,1])
#         cell_pt_ids_sorted[i,:] = cell_pt_ids[j,:]
#         # edge_pts_sorted[i+1,:] = edges.cell_points(cell_pt_ids_sorted[i,1])[1,:]
        
#         cell_pt_ids[j,0] = -1
#     j = np.where(cell_pt_ids[:,0]==cell_pt_ids_sorted[-2,1])
#     cell_pt_ids_sorted[-1,:] = cell_pt_ids[j,:]
#     # edge_pts_sorted[i+1,:] = edges.cell_points(3)[1,:]
    
#     new_faces = np.zeros((edges.n_cells-2,4),dtype=int)
#     new_faces[:,0] = 3
#     new_faces[:,1] = mesh.find_closest_point(edges.points[cell_pt_ids_sorted[0,0],:])
#     for i in range(new_faces.shape[0]):
#         new_faces[i,2] = mesh.find_closest_point(edges.points[cell_pt_ids_sorted[i+1,0],:])
#         new_faces[i,3] = mesh.find_closest_point(edges.points[cell_pt_ids_sorted[i+1,1],:])
#         # new_faces[i,2] = mesh.find_closest_point(edge_pts_sorted[i+1,:])
#         # new_faces[i,3] = mesh.find_closest_point(edge_pts_sorted[i+2,:])
    
#     mesh.faces = np.concatenate((mesh.faces,new_faces.reshape(-1)))
#     # test = mesh.copy()
#     # test.faces = np.concatenate((test.faces,new_faces.reshape(-1)))
#     # plotpatch(test)
    
#     return mesh
    

    
# function for Procrustes alignment, taken from MATLAB
def procrustes(X,Y,scale=True):
    '''
    Procrustes Analysis
    * adapted from Matlab procrustes.m
    
    Determines a linear transformation (translation,
    reflection, orthogonal rotation, and scaling) of the points in the
    matrix Y to best conform them to the points in the matrix X.  The
    "goodness-of-fit" criterion is the sum of squared errors.  PROCRUSTES
    returns the minimized value of this dissimilarity measure in D.  D is
    standardized by a measure of the scale of X, given by

       sum(sum((X - repmat(mean(X,1), size(X,1), 1)).^2, 1))

    i.e., the sum of squared elements of a centered version of X.  However,
    if X comprises repetitions of the same point, the sum of squared errors
    is not standardized.

    X and Y are assumed to have the same number of points (rows), and
    PROCRUSTES matches the i'th point in Y to the i'th point in X.  Points
    in Y can have smaller dimension (number of columns) than those in X.
    In this case, PROCRUSTES adds columns of zeros to Y as necessary.

    Z = b * Y * T + c.
 
   References:
     [1] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
     [2] Gower, J.C. and Dijskterhuis, G.B., Procrustes Problems, Oxford
         Statistical Science Series, Vol 30. Oxford University Press, 2004.
     [3] Bulfinch, T., The Age of Fable; or, Stories of Gods and Heroes,
         Sanborn, Carter, and Bazin, Boston, 1855.

    Parameters
    ----------
    X : numpy.ndarray
        n x 3 matrix of point coordinates for target/reference mesh.
    Y : numpy.ndarray
        n x 3 matrix of point coordinates for the mesh to be transformed.
    scale : bool, optional
        If True, compute a procrustes solution that includes a scale component. 
        The default is True.

    Returns
    -------
    Z : numpy.ndarray
        The transformed mesh, now aligned with X.
    T : numpy.ndarray
        The orthogonal rotation and reflection component of the transformation 
        that maps Y to Z.
    b : float
        The scale component of the transformation that maps Y to Z.
    c : numpu.ndarray
        The translation component of the transformation that maps Y to Z.
    d : float
        the standardized distance.

    '''
    
    
    X0 = X - np.tile(X.mean(axis=0),(X.shape[0],1))
    Y0 = Y - np.tile(Y.mean(axis=0),(Y.shape[0],1))
    
    ssqX = np.square(X0).sum(axis=0)
    ssqY = np.square(Y0).sum(axis=0)
    constX = (ssqX <= np.square(np.abs(np.spacing(1)*X.shape[0]*X.mean(axis=0)))).any()
    constY = (ssqY <= np.square(np.abs(np.spacing(1)*X.shape[0]*Y.mean(axis=0)))).any()
    ssqX = ssqX.sum()
    ssqY = ssqY.sum()
    
    if (not constX) and (not constY):
        # The "centred" Frobenius norm
        normX = np.sqrt(ssqX) # == sqrt(trace(X0*X0'))
        normY = np.sqrt(ssqY)
    
        # Scale to equal (unit) norm
        X0 = X0 / normX
        Y0 = Y0 / normY
        
        # Make sure they're in the same dimension space
        if Y.shape[1] < X.shape[1]:
            Y0 = np.concatenate((Y0,np.zeros(Y.shape[0],X.shape[1]-Y.shape[1])))
            
        # The optimum rotation matrix of Y
        A = np.matmul(X0.transpose(),Y0)
        [L,D,M] = np.linalg.svd(A)
        T = np.matmul(M.transpose(),L.transpose())
        
        # can include code to force reflection or no here
        
        # The minimized unstandardized distance D(X0,b*Y0*T) is
        # ||X0||^2 + b^2*||Y0||^2 - 2*b*trace(T*X0'*Y0)
        traceTA = D.sum()
        
        if scale == True:
            b = traceTA * normX /normY # the optimum scaling of Y
            d = 1 - traceTA**2 # the standardized distance between X and b*Y*T+c
            Z = normX * traceTA * np.matmul(Y0,T) + np.tile(X.mean(axis=0),(X.shape[0],1))
        else:
            b = 1
            d = 1 + ssqY/ssqX - 2*traceTA*normY/normX # The standardized distance between X and Y*T+c.
            Z = normY * np.matmul(Y0,T) + np.tile(X.mean(axis=0),(X.shape[0],1))
        
        c = X.mean(axis=0) - b * np.matmul(Y.mean(axis=0),T)
    
    # The degenerate cases: X all the same, and Y all the same.
    elif constX:
        d = 0
        Z = np.tile(X.mean(axis=0),(X.shape[0],1))
        T = np.eye(Y.shape[1],X.shape[1])
        b = 0
        c = Z
    else: # constX and constY
        d = 1
        Z = np.tile(X.mean(axis=0),(X.shape[0],1))
        T = np.eye(Y.shape[1],X.shape[1])
        b = 0
        c = Z
        
    return Z, T, b, c, d


def read_iv(file_path):
    '''
    Import an open inventor .iv mesh file and return a pyvista PolyData object.

    Parameters
    ----------
    file_path : string
        Path to .iv file.

    Returns
    -------
    mesh : pyvista.PolyData
        mesh object.

    '''
    
    
    pts_list = []
    cns_list = []

    re.IGNORECASE = True
    with open(file_path,'r') as f:
        txt = f.read()
        
        # vertices
        m1 = re.search('point\s(.*)\[',txt)
        m2 = re.search('\]\s\}.*\n\s Ind',txt)
        ptstxt = txt[m1.span()[1]+1:m2.span()[0]]
        # tokens = re.findall(r'[-+]?(?:\d*\.*\d+)\s[-+]?(?:\d*\.*\d+)\s[-+]?(?:\d*\.*\d+),',ptstxt)
        tokens = re.findall(r'-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?\s-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?\s-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?,',ptstxt)
        for i in range(len(tokens)):
            pts_list.append(re.findall("-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?",tokens[i]))
            
        # faces
        m1 = re.search('coordIndex\s(.*)\[',txt)
        txt2 = txt[m1.span()[1]+1:]
        m2 = re.search('\]\s\}',txt2)
        cnstxt = txt2[:m2.span()[0]]
        tokens = re.findall(r'[-+]?(?:\d+),\s[-+]?(?:\d+),\s[-+]?(?:\d+)',cnstxt)
        for i in range(len(tokens)):
            cns_list.append(re.findall(r'[-+]?(?:\d+)',tokens[i]))
            
    pts = np.array(pts_list,dtype=np.float32)
    cns = np.array(cns_list,dtype=np.int64)
    cns = np.concatenate((3*np.ones((cns.shape[0],1),dtype=int),cns),axis=1)
    cns = cns.reshape(-1)
    
    mesh = pv.PolyData(pts,cns)

    return mesh


def read_asc(file_path):
    '''
    Import a .asc mesh file and return a pyvista PolyData object.

    Parameters
    ----------
    file_path : string
        Path to .asc file.

    Returns
    -------
    mesh : pyvista.PolyData
        mesh object.

    '''

    with open(file_path,'r') as f:
        # f=open(file_path,'r') 
        txt = f.readline()
        txt = f.readline()
        n_vertices, n_faces = re.findall(r'\d+',txt)
        n_vertices = int(n_vertices)
        n_faces = int(n_faces)
        
        txt = f.readline() # first line is something else apparently?
        
        pts = np.zeros((n_vertices,3),dtype=np.float32)
        for i in range(n_vertices):
            txt = f.readline()
            data = re.findall(r'-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?\s',txt)
            pts[i,:] = np.array(data[:3],dtype=np.float32)
        
        cns = np.zeros((n_faces,4),dtype=np.int64)
        for i in range(n_faces):
            txt = f.readline()
            data = re.findall(r'\d+',txt)
            cns[i,:] = np.array(data,dtype=np.int64)
            
    cns = cns.reshape(-1)
    
    mesh = pv.PolyData(pts,cns)

    return mesh


def unit(v):
    '''
    Create unit vector

    Parameters
    ----------
    v : numpy.ndarray
        Vector.

    Returns
    -------
    v_unit : numpy.ndarray
        Unit vector of v.

    '''
    v_unit = v / np.linalg.norm(v)
    return v_unit

def angle_diff(v1,v2):
    '''
    Determine angle between two vectors

    Parameters
    ----------
    v1 : numpy.ndarray
        vector 1.
    v2 : numpy.ndarray
        vector 2.

    Returns
    -------
    ang : float
        Angle in degrees.

    '''
    
    ang = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180/np.pi
    
    return ang



def rotmat(angle,axis,deg='rad'):
    '''
    Create a 4x4 transformation matrix for a rotation about one major axis

    Parameters
    ----------
    angle : float
        angle to rotate.
    axis : string
        'x','y', or 'z' - axis to rotate about.
    deg : string, optional
        'deg' if angle is in degrees, and 'rad' if in radians. The default is 'rad'.

    Returns
    -------
    None.

    '''

    if deg == 'deg':
        angle = angle*np.pi/180
    T = np.eye(4,dtype=float)
    
    if axis.lower() == 'x':
        T[1,1] = np.cos(angle)
        T[1,2] =-np.sin(angle)
        T[2,1] = np.sin(angle)
        T[2,2] = np.cos(angle)
    elif axis.lower() == 'y':
        T[0,0] = np.cos(angle)
        T[0,2] = np.sin(angle)
        T[2,0] =-np.sin(angle)
        T[2,2] = np.cos(angle)
    elif axis.lower() == 'z':
        T[0,0] = np.cos(angle)
        T[0,1] =-np.sin(angle)
        T[1,0] = np.sin(angle)
        T[1,1] = np.cos(angle)

    return T

def ea2r(angle_x,angle_y,angle_z,sequence='XYZ',deg='rad'):
    '''
    

    Parameters
    ----------
    angle_x : float
        angle of rotation about y axis.
    angle_y : float
        angle of rotation about y axis.
    angle_z : float
        angle of rotation about z axis.
    sequence : string, optional
        Euler sequence to use. The default is 'XYZ'.
    deg : string, optional
        'deg' if angle is in degrees, and 'rad' if in radians. The default is 'rad'.

    Returns
    -------
    R : numpy.array
        3x3 rotation matrix

    '''
    Rx = rotmat(angle_x,'x',deg)
    Ry = rotmat(angle_y,'y',deg)
    Rz = rotmat(angle_z,'z',deg)
    
    if sequence == 'XYZ':
        R = np.matmul(Rx,np.matmul(Ry,Rz))
    elif sequence == 'ZXY': # used by SIMM
        R = np.matmul(Rz,np.matmul(Rx,Ry))
    elif sequence == 'ZYX':
        R = np.matmul(Rz,np.matmul(Ry,Rx))
    # add more as needed
    
    R = R[:3,:3] # just want rotation matrix
    
    return R
 
def r2ea(R,sequence='XYZ',deg='rad'):
    '''
    Decompose a 3x3 rotation matrix into the Euler angles for a given sequence

    Parameters
    ----------
    R : numpy.array
        3x3 rotation matrix.
    sequence : string, optional
        Euler sequence to use. The default is 'XYZ'.
    deg : string, optional
        'deg' if angle is in degrees, and 'rad' if in radians. The default is 'rad'.

    Returns
    -------
    angles : np.array
        x, y, and z Euler angles

    '''
    angles = np.zeros(3,dtype=float)
    if sequence == 'XYZ':
         # R = Rx*Ry*Rz
         # R = [cos(y)cos(z) -cos(y)sin(z)     sin(y);
         #         *               *       -sin(x)cos(y);
         #         *               *        cos(x)cos(y)];
        angles[0] = np.arctan2(-R[1,2],R[2,2]) #x
        angles[1] = np.arcsin(R[0,2]) #y
        angles[2] = np.arctan2(-R[0,1],R[0,0]) # z
    elif sequence == 'ZYX':
         # R = Rz*Ry*Rx
         # R = [cos(y)cos(z)      *           *        ;
         #      cos(y)sin(z)      *           *        ;
         #        -sin(y)    sin(x)cos(y) cos(x)cos(y)];
        angles[0] = np.arctan2(R[2,1],R[2,2])
        angles[1] = np.arcsin(-R[0,2])
        angles[2] = np.arctan2(R[1,0],R[0,0])
    elif sequence == 'XZY':
         # R = Rx*Rz*Ry
         # R = [cos(y)cos(z)   -sin(z)    sin(y)cos(z) ;
         #           *       cos(x)cos(z)      *       ;
         #           *       sin(x)cos(z)      *      ];
        angles[0] = np.arctan2(R[2,1],R[1,1])
        angles[1] = np.arctan2(R[0,2],R[0,0])
        angles[2] = np.arcsin(-R[0,1])
    elif sequence == 'ZXY':
         # R = Rz*Rx*Ry
         # R = [      *       -cos(x)sin(z)      *       ;
         #            *        cos(x)cos(z)      *       ;
         #      -cos(x)sin(y)      sin(x)   cos(x)cos(y)];
        angles[0] = np.arcsin(R[2,1])
        angles[1] = np.arctan2(-R[2,0],R[2,2]) 
        angles[2] = np.arctan2(-R[0,1],R[1,1])
    elif sequence == 'YXZ':
         # R = Ry*Rx*Rz
         # R = [     *            *        cos(x)sin(y) ;
         #      cos(x)sin(z) cos(x)cos(z)    -sin(x)    ; 
         #           *            *        cos(x)cos(y)];
        angles[0] = np.arcsin(-R[1,2])
        angles[1] = np.arctan2(R[0,2],R[2,2])
        angles[2] = np.arctan2(R[1,0],R[1,1])
    elif sequence == 'YXY':
         # R = Ry*Rx*Ry
         # R = [      *        sin(x)*sin(y)          *       ; 
         #      sin(x)*sin(y2)      cos(x)       -cos(y2)*sin(x);
         #            *         sin(x)cos(y)          *      ];
        angles[0] = np.arccos(R[1,1])
        angles[1] = np.arctan2(R[0,1],R[2,1])
        angles[2] = np.arctan2(R[1,0],-R[1,2])
 
    if deg == 'deg':
        angles = angles*180/np.pi
 
    return angles

   
def circ_fit(points,showPlot=False):
    '''
    Fit a circle to a set of points.
    From https://www.mathworks.com/matlabcentral/fileexchange/5557-circle-fit

    Parameters
    ----------
    points : numpy.array
        A n x 2 array of point coordinates.

    Returns
    -------
    radius : float
        Circle radius.
    centre : np.array
        Circle centre.

    '''
    points1 = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)
    pointssq = -np.sum(points[:,0:2]**2,axis=1)
    a = np.matmul(np.linalg.pinv(points1),pointssq.T)
    # a = np.matmul(np.linalg.inv(np.matmul(points1.T,points1)),np.matmul(points1.T,pointssq.T))
    centre = -0.5 * a[0:2]
    radius = np.sqrt((a[0]**2 + a[1]**2)/4-a[2])
    
    if showPlot == True:
        th = np.linspace(0,2*np.pi,30)
        x = radius * np.cos(th) + centre[0]
        y = radius * np.sin(th) + centre[1]
        plt.plot(points[:,0],points[:,1],'.')
        plt.plot(x,y,'r')
        plt.axis('equal')
    

    return radius,centre
                      
# # This doesn't work well
# def lscylinder(pts,params0):
#     # fit ls cylinder (https://stackoverflow.com/questions/43784618/fit-a-cylinder-to-scattered-3d-xyz-point-data)
#     # pts = points to fit
#     # params0 = inital guess for cylinder parameters: x0[0], x0[1], x rotation, y rotation, radius
#     #            x0 = coords of centre (x and y coordinates)
#     #            x rotation = rotation angle (radians) about x-axis
#     #            y rotation = rotation angle (radians) about y-axis
#     #            radius = radius
    
    
    
#     fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) 
#                                   - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + \
#                                   (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2
#     errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 
#     params,success = leastsq(errfunc,params0,args=(pts[:,0],pts[:,1],pts[:,2]),maxfev=1000)
    
#     return params

def gr(x,y):
    '''
    For Givens plane rotation
    
    Adapted from /Matlab_tools/KneeACS/Tools/gr.m
    Created by I M Smith 08 Mar 2002

    Parameters
    ----------
    x : float
        DESCRIPTION.
    y : float
        DESCRIPTION.

    Returns
    -------
    U : numpy.array
        2x2 rotation matrix [c s; -s c], with U * [x y]' = [z 0]'
    c : float
        cosine of the rotation angle
    s : float
        sine of the rotation angle

    '''
    if y == 0:
        c = 1
        s = 0
    elif np.abs(y) > np.abs(x):
        t = x/y
        s = 1/np.sqrt(1+t*t)
        c = t*s
    else:
        t = y/x
        c = 1/np.sqrt(1+t*t)
        s=t*c
    U = np.array([[c,s],[-s,c]])
    
    return U, c, s

def rot3z(a):
    '''
    Form rotation matrix U to rotate the vector a to a point along
    the positive z-axis. 
    
    Adapted from /Matlab_Tools/KneeACS/Tools/rot3z.m
    Created by I M Smith 2 May 2002

    Parameters
    ----------
    a : numpy.array
        3x1 array.

    Returns
    -------
    U : numpy.array
        3x3 array. Rotation matrix with U * a = [0 0 z]', z > 0. 

    '''

    # form first Givens rotation
    W, c1, s1 = gr(a[1], a[2])
    z = c1*a[1] + s1*a[2]
    V = np.array([[1,0,0],[0,s1,-c1],[0,c1,s1]])

    # form second Givens rotation
    W, c2, s2 = gr(a[0],z);
    
    # check positivity
    if c2 * a[0] + s2 * z < 0:
        c2 = -c2
        s2 = -s2
     
    W = np.array([[s2,0,-c2],[0,1,0],[c2,0,s2]])
    U = np.matmul(W,V)
      
    return U

def fgrrot3(theta,R0=np.eye(3)):
    '''
    Form rotation matrix R = R3*R2*R1*R0 and its derivatives using right-
    handed rotation matrices.
             R1 = [ 1  0   0 ]  R2 = [ c2 0  s2 ] and R3 = [ c3 -s3 0 ]
                  [ 0 c1 -s1 ],      [ 0  1   0 ]          [ s3  c3 0 ].
                  [ 0 s1  c2 ]       [-s2 0  c2 ]          [  0   0 1 ]
                  
    Adapted from Matlab_Tools/KneeACS/Tools/fgrrot3.m
    by I M Smith 27 May 2002

    Parameters
    ----------
    theta : numpy.array
        Array of plane rotation angles (t1,t2,t3).
    R0 : numpy.array
        3x3 rotation matrix, optional with default = I.

    Returns
    -------
    R : numpy.array
        3x3 rotation matrix.
    DR1 : numpy.array
        Derivative of R wrt t1.
    DR2 : numpy.array
        Derivative of R wrt t2.
    DR3 : numpy.array
        Derivative of R wrt t3.

    '''
    ct = np.cos(theta)
    st = np.sin(theta)
    R1 = np.array([[1,0,0],[0,ct[0],-st[0]],[0,st[0],ct[0]]])
    R2 = np.array([[ct[1],0,st[1]],[0,1,0],[-st[1],0,ct[1]]])
    R3 = np.array([[ct[2],-st[2],0],[st[2],ct[2],0],[0,0,1]])
    R = np.matmul(R3,np.matmul(R2,R1))
    # evaluate derivative matrices
    # drrot3  function
    dR1 = np.array([[0,0,0],[0,-R1[2,1],-R1[1,1]],[0,R1[1,1],-R1[2,1]]])
    dR2 = np.array([[-R2[0,2],0,R2[0,0]],[0,0,0],[-R2[0,0],0,-R2[0,2]]])
    dR3 = np.array([[-R3[1,0],-R3[0,0],0],[R3[0,0],-R3[2,0],0],[0,0,0]])
    DR1 = np.matmul(R3,np.matmul(R2,np.matmul(dR1,R0)))
    DR2 = np.matmul(R3,np.matmul(dR2,np.matmul(R1,R0)))
    DR3 = np.matmul(dR3,np.matmul(R2,np.matmul(R1,R0)))
    
    return R, DR1, DR2, DR3

def fgcylinder(a,X,w):
    '''
    Function and gradient calculation for least-squares cylinder fit.
    
    Adapted from Matlab_Tools/KneeACS/Tools/fgcylinder.m
    by I M Smith 27 May 2002

    Parameters
    ----------
    a : numpy.array
        Parameters [x0 y0 alpha beta s].
    X : numpy.array
        Array [x y z] where x = vector of x-coordinates, 
        y = vector of y-coordinates and z = vector of z-coordinates. .
    w : numpy.array
        Weights.

    Returns
    -------
    f : numpy.array
        Signed distances of points to cylinder:
         f(i) = sqrt(xh(i)^2 + yh(i)^2) - s, where 
         [xh yh zh]' = Ry(beta) * Rx(alpha) * ([x y z]' - [x0 y0 0]').
         Dimension: m x 1.
    J : numpy.array
        Jacobian matrix df(i)/da(j). Dimension: m x 5.

    '''
    m = X.shape[0]
    # if no weights are specified, use unit weights
    # if w == None:
    #     w = np.ones((m,1))
    
    x0 = a[0]
    y0 = a[1]
    alpha = a[2]
    beta = a[3]
    s = a[4]
    
    R, DR1, DR2, _ = fgrrot3(np.array([alpha,beta,0]))
    
    Xt = np.matmul(X - np.array([x0,y0,0]),R.T)
    rt = np.linalg.norm(Xt[:,:2],axis=1)
    Nt = np.zeros((m,3))
    Nt[:,0] = np.divide(Xt[:,0],rt)
    Nt[:,1] = np.divide(Xt[:,1],rt)
    f = np.divide(Xt[:,0]**2,rt) + np.divide(Xt[:,1]**2,rt)
    f = f - s
    f = np.multiply(w,f)
    
    # form the Jacobian matrix
    J = np.zeros((m,5))
    A1 = np.matmul(R,np.array([-1,0,0]).T)
    J[:,0] = A1[0] * Nt[:,0] + A1[1] * Nt[:,1]
    A2 = np.matmul(R,np.array([0,-1,0]).T)
    J[:,1] = A2[0] * Nt[:,0] + A2[1] * Nt[:,1]
    A3 = np.matmul(X-np.array([x0,y0,0]),DR1.T)
    J[:,2] = np.multiply(A3[:,0],Nt[:,0]) + np.multiply(A3[:,1],Nt[:,1])
    A4 = np.matmul(X-np.array([x0,y0,0]),DR2.T)
    J[:,3] = np.multiply(A4[:,0],Nt[:,0]) + np.multiply(A4[:,1],Nt[:,1])
    J[:,4] = -1 * np.ones(m)
    
    return f,J
    
def nlss11(ai,tol,p1,p2):
    '''
    Nonlinear least squares solver. Minimize f'*f.
    
    Adapted from /Matlab_Tools/KneeACS/Tools/nlss11.m
    by AB Forbes, CMSC, NPL

    Parameters
    ----------
    ai : numpy.array
        Optimisation parameters, intial estimates.
    tol : numpy.array
        Convergence tolerances [tolr tols]', where 
          tolr = relative tolerance, and 
          tols = scale for function values. 
          Dimension: 2 x 1. 
    p1 : numpy.array
        DESCRIPTION.
    p2 : numpy.array
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    a : numpy.array
        Solution estimates of the optimisation parameters.
          Dimension: n x 1.
    f : numpy.array
        Functions evaluated at a.
          Dimension: m x 1.
          Constraint: m >= n..
    R : numpy.array
        Triangular factor of the Jacobian matrix evaluated at a.
          Dimension: n x n.
    GNlog : list
        Log of the Gauss-Newton iterations. 
          Rows 1 to niter contain 
          [iter, norm(f_iter), |step_iter|, |gradient_iter|]. 
          Row (niter + 1) contains 
          [conv, norm(d), 0, 0]. 
          Dimension: (niter + 1) x 4. 

    '''
    a0 = ai
    n = len(a0)
    
    if n == 0:
        raise ValueError('Empty vector of parameter estimates')

    mxiter = int(100 + np.ceil(np.sqrt(n)))
    conv = 0
    niter = 0
    eta = 0.01
    GNlog = []
    
    # G-N iterations
    while (niter < mxiter) and (conv ==0):
        f0,J = fgcylinder(a0,p1,p2)
        if niter == 0:
            # scale by norm of columns of J
            mJ,nJ = J.shape
            scale = np.linalg.norm(J,axis=0)
        
        m = len(f0)
        # check on m, n
        if (niter==0) and (m<n):
            raise ValueError('Number of observation less than number of parameters')
        
        # Calculate update step and gradient
        F0 = np.linalg.norm(f0)
        _,Rqr = np.linalg.qr(np.concatenate((J,np.expand_dims(f0,axis=1)),axis=1))
        Ra = np.triu(Rqr)
        R = Ra[:nJ,:nJ]
        q = Ra[:nJ,nJ]
        p = np.matmul(np.linalg.inv(-R),q.T)
        g = 2 * np.matmul(R.T,q.T)
        G0 = np.matmul(g,p.T)
        a1 = a0 + p
        niter = niter+1
        
        # Check on convergence
        f1,J1 = fgcylinder(a1,p1,p2)
        F1 = np.linalg.norm(f1)
        # Gauss-Newton convergence conditions
        # from Matlab_Tools/KneeACS/Tools/gncc2.m
        #gncc2(F0, F1, p, g, scale, tol(1)=tolr, tol(2)=scalef);
        conv = 0
        sp = np.max(np.abs(p * scale))
        sg = np.max(np.abs(g / scale))
        c = np.full(5,np.nan)       
        c[0] = sp / (tol[1] * (tol[0]**0.7))
        c[1] = np.abs(F0-F1) / (tol[0]*tol[1])
        c[2] = sg / ((tol[0]**0.7)*tol[1])
        c[3] = F1 / (tol[1] * np.finfo(float).eps**0.7)
        c[4] = sg / ((np.finfo(float).eps**0.7)*tol[1])
        if (c[0] < 1) and (c[1] < 1) and (c[2] < 1):
            conv = 1
        elif (c[3] < 1) or (c[4] < 1):
            conv = 1

        if conv != 1:
            # otherwise check on reduction of sum of squares
            # evaluate f at a1
            rho = (F1-F0) * (F1+F0)/G0
            if rho < eta:
                tmin = np.max(np.array([0.001,1/(2*(1-rho))]))
                a0 = a0 + tmin * p
            else:
                a0 = a0 + p
        GNlog.append([niter,F0,sp,sg])
    
    a = a0 + p
    f = f1
    GNlog.append([conv,F1,0,0])
    
    return a, f, R, GNlog
        

# X = pts
# w = None #np.ones(X.shape[0])
# x0 = X.mean(axis=0)
# a0 = np.array([1,0,0])
# r0 = ((np.max(X[:,1])-np.min(X[:,1])) + (np.max(X[:,2])-np.min(X[:,2])))/2
# x0n, an, rn, stats = lscylinder(X,x0,a0,r0)

def lscylinder(X,x0,a0,r0,tolp=0.1,tolg=0.1,w=None):
    '''
    Least-squares cylinder using Gauss-Newton
    
    Adapted from /Matlab_Tools/KneeACS/Tools/lscylinder.m
    by I M Smith 27 May 2002

    Parameters
    ----------
    X : numpy.array
        Array [x y z] where x = vector of x-coordinates, 
        y = vector of y-coordinates and z = vector of z-coordinates.
        Dimension: m x 3. 
    x0 : numpy.array
        Estimate of the point on the axis. 
        Dimension: 3 x 1. 
    a0 : numpy.array
        Estimate of the axis direction. 
        Dimension: 3 x 1.
    r0 : float
        Estimate of the cylinder radius. 
        Dimension: 1 x 1.
    tolp : float, optional
        Tolerance for test on step length. The default is 0.1.
    tolg : float, optional
        Tolerance for test on gradient. The default is 0.1.
    w : numpy.array, optional
        Weights. The default is None. If None, it will be an array of ones.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    x0n : numpy.array
        Estimate of the point on the axis. Dimension: 3x1
    an : numpy.array
        Estimate of the axis direction. Dimension: 3x1
    rn : float
        Estimate of the cylinder radius.
    stats : dict
        Dictionary of dditonal statistics and results.
        stats = {'sigmah':sigmah,'conv':conv,'Vx0n':Vx0n,'Van':Van,'urn':urn,'GNlog':GNlog,
                 'a':a,'R0':R0,'R':R}
        sigmah   Estimate of the standard deviation of the weighted residual errors. 
                Dimension: 1 x 1. 
 
        conv     If conv = 1 the algorithm has converged, if conv = 0 the algorithm
                has not converged and x0n, rn, d, and sigmah are current estimates. 
                Dimension: 1 x 1. 
 
        Vx0n     Covariance matrix of point on the axis. Dimension: 3 x 3. 

        Van      Covariance matrix of axis direction. Dimension: 3 x 3. 

        urn      Uncertainty in cylinder radius. Dimension: 1 x 1. 
 
        GNlog    Log of the Gauss-Newton iterations. 
                Rows 1 to niter contain [iter, norm(f_iter), |step_iter|, |gradient_iter|]. 
                Row (niter + 1) contains [conv, norm(d), 0, 0]. 
                Dimension: (niter + 1) x 4. 
 
        a        Optimisation parameters at the solution. Dimension: 5 x 1. 
 
        R0       Fixed rotation matrix. Dimension: 3 x 3. 
 
        R        Upper-triangular factor of the Jacobian matrix at the solution. 
                Dimension: 5 x 5.     

    '''
    m = X.shape[0]
    if m < 5:
        raise ValueError('At least 5 data points required')
    
    if w is None:
        w = np.ones(m)

    # find the centroid of the data
    xb = X.mean(axis=0)
    
    # transform the data to close to standard position via a rotation 
    # followed by a translation 
    R0 = rot3z(a0) # U * a0 = [0 0 1]' 
    x1 = np.matmul(R0,x0)
    xb1 = np.matmul(R0,xb) 
    # find xp, the point on axis nearest the centroid of the rotated data 
    t = x1 + (xb1[2] - x1[2]) * np.array([0,0,1]) 
    X2 = np.matmul(X,R0.T) - t
    x2 = x1 - t
    xb2 = xb1 - t
     
    ai = np.array([0,0,0,0,r0]) 
    tol = np.array([tolp,tolg])     
    
    # Gauss-Newton algorithm to find estimates of roto-translation parameters
    # that transform the data so that the best-fit circle is one in the standard
    # position
    a, d, R, GNlog = nlss11(ai,tol,X2,w)
    
    # inverse transformation to find axis and point on axis corresponding to 
    # original data
    rn = a[4]
    R3, DR1, DR2, DR3 = fgrrot3(np.array([a[2],a[3],0]))
    an = np.matmul(R0.T,np.matmul(R3.T,np.array([0,0,1]).T))
    p = np.matmul(R3,(xb2-np.array([a[0],a[1],0])).T)
    pz = np.array([0,0,p[2]])
    x0n = np.matmul(R0.T,(t + np.array([a[0],a[1],0]) + np.matmul(R3.T,pz.T)).T)
    
    nGN = len(GNlog)
    conv = GNlog[nGN-1][0]
    if conv == 0:
        print(' *** Gauss-Newton algorithm has not converged ***')
    
    # Calculate statistics
    dof = m - 5
    sigmah = np.linalg.norm(d)/np.sqrt(dof)
    ez = np.array([0,0,1])
    G = np.zeros((7,5))
    # derivatives of x0n
    dp1 = np.matmul(R3,np.array([-1,0,0]).T)
    dp2 = np.matmul(R3,np.array([0,-1,0]).T)
    dp3 = np.matmul(DR1,(xb2 - np.array([a[0],a[1],0])).T)
    dp4 = np.matmul(DR2,(xb2 - np.array([a[0],a[1],0])).T)
    G[0:3,0] = np.matmul(R0.T,np.array([1,0,0]) + np.matmul(R3.T,np.array([0,0,np.matmul(dp1.T,ez)]).T))
    G[0:3,1] = np.matmul(R0.T,np.array([0,1,0]) + np.matmul(R3.T,np.array([0,0,np.matmul(dp2.T,ez)]).T))
    G[0:3,2] = np.matmul(R0.T,np.matmul(DR1.T,np.array([0,0,np.matmul(p.T,ez)]).T) + \
                    np.matmul(R3.T,np.array([0,0,np.matmul(dp3.T,ez)]).T))
    G[0:3,3] = np.matmul(R0.T,np.matmul(DR2.T,np.array([0,0,np.matmul(p.T,ez)]).T) + \
                    np.matmul(R3.T,np.array([0,0,np.matmul(dp4.T,ez)]).T))
    # derivatives of an
    G[3:6,2] = np.matmul(R0.T,np.matmul(DR1.T,np.array([0,0,1]).T))
    G[3:6,3] = np.matmul(R0.T,np.matmul(DR2.T,np.array([0,0,1]).T))
    # derivatives of rn
    G[6,4] = 1
    Gt = np.matmul(np.linalg.inv(R.T),sigmah*G.T)
    Va = np.matmul(Gt.T,Gt)
    Vx0n = Va[0:3,0:3] # covariance matrix for x0n
    Van = Va[3:6,3:6] # covariance matrix for an
    urn = np.sqrt(Va[6,6]) # uncertainty in rn
    
    stats = {'d':d,'sigmah':sigmah,'conv':conv,'Vx0n':Vx0n,'Van':Van,'urn':urn,'GNlog':GNlog,
             'a':a,'R0':R0,'R':R}
    
    return x0n,an,rn,stats
     

def surfature(X,Y,Z):
    '''
    Adapted from https://www.mathworks.com/matlabcentral/fileexchange/11168-surface-curvature

     SURFATURE -  COMPUTE GAUSSIAN AND MEAN CURVATURES OF A SURFACE
       [K,H] = SURFATURE(X,Y,Z), WHERE X,Y,Z ARE 2D ARRAYS OF POINTS ON THE
       SURFACE.  K AND H ARE THE GAUSSIAN AND MEAN CURVATURES, RESPECTIVELY.
       SURFATURE RETURNS 2 ADDITIONAL ARGUEMENTS,
       [K,H,Pmax,Pmin] = SURFATURE(...), WHERE Pmax AND Pmin ARE THE MINIMUM
       AND MAXIMUM CURVATURES AT EACH POINT, RESPECTIVELY.
    '''
   
    # First Derivatives
    Xv,Xu = np.gradient(X)
    Yv,Yu = np.gradient(Y)
    Zv,Zu = np.gradient(Z)
    
    # Second Derivatives
    Xuv,Xuu = np.gradient(Xu)
    Yuv,Yuu = np.gradient(Yu)
    Zuv,Zuu = np.gradient(Zu)   
    Xvv,Xuv = np.gradient(Xv)
    Yvv,Yuv = np.gradient(Yv)
    Zvv,Zuv = np.gradient(Zv)
    
    # Reshape 2D arrays into vectors
    Xu = np.vstack((Xu.reshape(-1,order='F'),Yu.reshape(-1,order='F'),Zu.reshape(-1,order='F'))).T
    Xv = np.vstack((Xv.reshape(-1,order='F'),Yv.reshape(-1,order='F'),Zv.reshape(-1,order='F'))).T
    Xuu = np.vstack((Xuu.reshape(-1,order='F'),Yuu.reshape(-1,order='F'),Zuu.reshape(-1,order='F'))).T
    Xuv = np.vstack((Xuv.reshape(-1,order='F'),Yuv.reshape(-1,order='F'),Zuv.reshape(-1,order='F'))).T
    Xvv = np.vstack((Xvv.reshape(-1,order='F'),Yvv.reshape(-1,order='F'),Zvv.reshape(-1,order='F'))).T

    # First fundamental coefficients of the surface
    E = np.sum(Xu*Xu,axis=1) # row-wise dot product
    F = np.sum(Xu*Xv,axis=1)
    G = np.sum(Xv*Xv,axis=1)
    
    m = np.cross(Xu,Xv)
    p = np.sqrt(np.sum(m*m,axis=1))
    n = m / np.vstack((p,p,p)).T
    
    # Second fundamental Coeffecients of the surface
    L = np.sum(Xuu*n,axis=1)
    M = np.sum(Xuv*n,axis=1)
    N = np.sum(Xvv*n,axis=1)
    
    # Gaussian curvature
    K = (L*N - M**2) / (E*G - F**2)
    K = K.reshape(Z.shape,order='F') # order goes down column first
    
    # Mean curvature
    H = (E*N + G*L - 2*F*M) / (2*(E*G - F**2))
    H = H.reshape(Z.shape,order='F')
    
    # Principal curvatures
    Pmin = -(H + np.sqrt(H**2-K))
    Pmax = -(H - np.sqrt(H**2-K))
    
    # Principal curvature directions
    Pmin_vec = Pmin.reshape(-1,order='F') # switch back to vector
    umin = np.ones((E.shape[0],2))
    umin[:,1] = (E*G-F**2)*(-Pmin_vec-(G*L-F*M)/(E*G-F**2))/(G*M-F*N)
    umin = umin/np.tile(np.linalg.norm(umin,axis=1),(2,1)).T
    umin1 = umin[:,0].reshape(Z.shape,order='F')
    umin2 = umin[:,1].reshape(Z.shape,order='F')
    
    Pmax_vec = Pmax.reshape(-1,order='F')
    umax = np.ones((E.shape[0],2))
    umax[:,1] = (E*G-F**2)*(-Pmax_vec-(G*L-F*M)/(E*G-F**2))/(G*M-F*N)
    umax = umax/np.tile(np.linalg.norm(umax,axis=1),(2,1)).T
    umax1 = umax[:,0].reshape(Z.shape,order='F')
    umax2 = umax[:,1].reshape(Z.shape,order='F')
    
    # i = 1
    # I = np.array([[E[i],F[i]],[F[i],G[i]]])
    # II = np.array([[L[i],M[i]],[M[i],N[i]]])
    # W = np.matmul(np.linalg.inv(I),II)
    # ki,ui = np.linalg.eig(W)
    
    # umax = np.ones(2)
    # umax[1] = (E[i]*G[i]-F[i]**2)*(-Pmin[i,0]-(G[i]*L[i]-F[i]*M[i])/(E[i]*G[i]-F[i]**2))/(G[i]*M[i]-F[i]*N[i])
    # # umax[1] = (-Pmin[i,0]-W[0,0])/W[0,1]
    # umax = umax/np.linalg.norm(umax)
    
     
    return K, H, Pmax, Pmin, umax1, umax2, umin1, umin2

def ggremesh(mesh,opts=None,ggremesh_prog=None):
    '''
    This function uses the external library Geogram to remesh the input
    pyvista mesh. In particular the code "vorpalite" is used. An additional 
    option structure may be provided where users can set particular parameters 
    for Geogram. 
    
    ***NOTE***: The vorpalite executable function is required. 
    A compiled version can be obtained from the GIBBON toolbox:
    https://github.com/gibbonCode/GIBBON/tree/master/lib_ext/geogram
    
    Geogram website:
    http://alice.loria.fr/index.php/software/4-library/75-geogram.html 
    
    Geogram license: 
    http://alice.loria.fr/software/geogram/doc/html/geogram_license.html
   
    LÃ©vy B., Bonneel N. (2013) Variational Anisotropic Surface Meshing with
    Voronoi Parallel Linear Enumeration. In: Jiao X., Weill JC. (eds)
    Proceedings of the 21st International Meshing Roundtable. Springer,
    Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-33573-0_21 
    
    See also: 
    http://alice.loria.fr/publications/papers/2012/Vorpaline_IMR/vorpaline.pdf
    https://www.ljll.math.upmc.fr/hecht/ftp/ff++days/2013/BrunoLevy.pdf
    

    This wrapper function was adapted from the GIBBON Matlab toolbox.
    https://www.gibboncode.org/

    Parameters
    ----------
    mesh : pyvista.PolyData
        The input mesh to be remeshed.
        
    opts : dict, optional
        A dictionary of options for the ggremesh execuatable. If opts=None, 
        the default values will be used.The default is None.
        The options and their default values are
            opts['nb_pts']=mesh.n_points #resample with same number of points
            opts['anisotropy']=0 #Use anisotropy (~=0) to capture geometry or favour isotropic triangles (=0)
            opts['pre']['max_hole_area']=100 #Max hole area for pre-processing step
            opts['pre']['max_hole_edges']=0 #Max number of hole edges for pre-processing step
            opts['post']['max_hole_area']=100 #Max hole area for post-processing step
            opts['post']['max_hole_edges']=0 #Max number of hole edges for post-processing step
            
    ggremesh_prog : string, optional
        The file path to the vorpalite executable file. If none, a default path is
        used (user should change this to the location on their computer). 
        The default is None.

    Returns
    -------
    mesh_out : pyvista.PolyData
        Resulting remeshed mesh.

    '''
    if ggremesh_prog == None:
        ggremesh_prog = r'D:\Antoine\TN10_uOttawa\codes\Seg_SSM\Seg_SSM\geogram\win64\bin\vorpalite.exe'
        # Note select the vorpalite file from lin64/bin or mac64/bin for those os

    # create temporary input/output files
    inputFileName = 'temp.ply'
    outputFileName = 'temp_out.ply'
    
    mesh.save(inputFileName)
    
    if opts == None:
        opts = {}
    
    # set default options for missing options
    if 'nb_pts' not in opts:
        opts['nb_pts']=mesh.n_points #resample with same number of points
    if 'anisotropy' not in opts:
        opts['anisotropy']=0 #Use anisotropy (~=0) to capture geometry or favour isotropic triangles (=0)
    if 'pre' not in opts:
        opts['pre']={}
    if 'max_hole_area' not in opts['pre']:
        opts['pre']['max_hole_area']=100 #Max hole area for pre-processing step
    if 'max_hole_edges' not in opts['pre']:
        opts['pre']['max_hole_edges']=0 #Max number of hole edges for pre-processing step
    if 'post' not in opts:
        opts['post']={}
    if 'max_hole_area' not in opts['post']:
        opts['post']['max_hole_area']=100 #Max hole area for post-processing step
    if 'max_hole_edges' not in opts['post']:
        opts['post']['max_hole_edges']=0 #Max number of hole edges for post-processing step
    
    cmd_str = '\"' + ggremesh_prog + '\" \"' + inputFileName + '\" \"' + outputFileName + '\"'
    
    for key in opts:
        if type(opts[key]) == dict:
            for subkey in opts[key]:
                cmd_str = cmd_str + ' ' + key + ':' + subkey + '=%.16g' % opts[key][subkey]
        else:
            cmd_str = cmd_str + ' ' + key + '=%.16g' % opts[key]

    subprocess.run(cmd_str,shell=True)
    
    mesh_out = pv.PolyData(outputFileName)
    
    os.remove(inputFileName)
    os.remove(outputFileName)
    
    return mesh_out


# ---- Plotting functions ---- #

def show_cs(T,pl):
    '''
    Plot a coordinate system.

    Parameters
    ----------
    T : np.ndarray
        4x4 pose matrix.
    pl : pyvista.Plotter
        Plot object to add CS to.

    Returns
    -------
    pl : pyvista.Plotter
        Plot object with CS.

    '''
    cs = pv.Arrow(start=T[:3,3],direction=T[:3,0],scale=70,shaft_radius=0.02,tip_radius=0.06)
    pl.add_mesh(cs,color='r')
    cs = pv.Arrow(start=T[:3,3],direction=T[:3,1],scale=70,shaft_radius=0.02,tip_radius=0.06)
    pl.add_mesh(cs,color='g')
    cs = pv.Arrow(start=T[:3,3],direction=T[:3,2],scale=70,shaft_radius=0.02,tip_radius=0.06)
    pl.add_mesh(cs,color='b')
    
    return pl

def plotpatch(mesh_list,cs_list=[],points_list=[],opts=None):
    '''
    Plot meshes, coordinate systems, and/or points.

    * Note, sometimes you will get an error if you don't specify opts. Just add opts={}.

    Parameters
    ----------
    mesh_list : list
        List of pyvista.PolyData meshes.
    cs_list : list, optional
        List of 4x4 np.ndarrays for coordinate systems. The default is [].
    points_list : list, optional
        List of np.ndarrays of 3D point coordinates. The default is [].
    opts : dict, optional
        Options for plot. The default is {}.

    Returns
    -------
    None.

    '''
    if opts == None:
        opts = {}
    
    if type(mesh_list) != list:
        mesh_list = [mesh_list]
    if type(cs_list) != list:
        cs_list = [cs_list]
    if type(points_list) != list:
        points_list = [points_list]
    if 'color' not in opts:
        opts['color'] = [np.array([.7,.7,.7])]*len(mesh_list)
    if 'style' not in opts:
        opts['style'] = ['surface']*len(mesh_list)
    if 'show_edges' not in opts:
        opts['show_edges'] = [True]*len(mesh_list)
    if 'edge_color' not in opts:
        opts['edge_color'] = ['k']*len(mesh_list)
    if 'opacity' not in opts:
        opts['opacity'] = [1.0]*len(mesh_list)
    if 'point_color' not in opts:
        opts['point_color'] = ['r']*len(points_list)
    if 'point_size' not in opts:
        opts['point_size'] = [5.0]*len(points_list)
    for k in opts.keys():
        if type(opts[k]) != list:
            opts[k] = [opts[k]]
        
    
    pv.set_plot_theme('document')
    pl = pv.Plotter(notebook=False)
    pl.disable_anti_aliasing()
    for i in range(len(mesh_list)):
        pl.add_mesh(mesh_list[i],color=opts['color'][i],style=opts['style'][i],show_edges=opts['show_edges'][i],
                    edge_color=opts['edge_color'][i],opacity=opts['opacity'][i])
    for i in range(len(cs_list)):
        show_cs(cs_list[i],pl)
    for i in range(len(points_list)):
        pl.add_points(points_list[i],color=opts['point_color'][i],point_size=opts['point_size'][i])
    if 'legend_entries' in opts:
        pl.add_legend(opts['legend_entries'],bcolor='w',border=True)
    if 'title' in opts:
        pl.add_title(opts['title'][0])
    pl.show_grid()
    pl.show_axes()
    pl.show()
    pl.close()