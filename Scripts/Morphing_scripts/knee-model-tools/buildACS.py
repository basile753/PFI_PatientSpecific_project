# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:58:39 2023

These functions build anatomical coordinate systems from 3D models of the distal femur,
proximal tibia, and patella. 

Miranda DL, Rainbow MJ, Leventhal EL, Crisco JJ, Fleming BC. 
Automatic determination of anatomical coordinate systems for 
three-dimensional bone models of the isolated human knee. 
J Biomech. 2010 May 28;43(8):1623–6. 

Rainbow, M. J. et al. Automatic determination of an anatomical
coordinate system for a three-dimensional model of the human patella. 
J Biomech (2013). doi:10.1016/j.jbiomech.2013.05.024


@author: Adapted from Matlab by aclouthi
"""

# import os
import pyvista as pv
import numpy as np
# import math
# import fitting 
# import re
import utils_bis as utb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import qr

# filepath = r'C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Queens\Shape\UW\HealthyBoneAndCartilageModels\ACLC01_R_Femur.stl' # file name of 3-D femur model (e.g. femur0123.iv)

# slice_thickness = 0.625;    # slice thickness value
# filepath = r'C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Queens\Shape\UW\HealthyBoneAndCartilageModels\ACLC01_R_Femur.stl'



def buildfACS(mesh,slice_thickness=0.625,plotACS=False):
    '''
    This function builds an anatomical coordinate system from a 3D model of the distal
    femur using its diaphysis and condyles. 
    Based on: 
      Miranda DL, Rainbow MJ, Leventhal EL, Crisco JJ, Fleming BC. 
      Automatic determination of anatomical coordinate systems for 
      three-dimensional bone models of the isolated human knee. 
      J Biomech. 2010 May 28;43(8):1623–6. 
    
    Parameters
    ----------
    mesh : string or pyvista PolyData mesh 
        Either the filepath to a mesh file or a pyvista PolyData mesh. This is the mesh to create the ACS for.
    slice_thickness : float, optional
        Slice thickness to use to slice mesh for cross-sectional properties. The default is 0.625.
    plotACS : bool, optional
        Set to True to produce a plot showing the mesh and ACS. The default is False.

    Returns
    -------
    fACS : numpy array
        4x4 pose matrix of the femur anatomical coordinate system that transforms from local to global.
        x = medial-lateral
        y = anterior-posterior
        z = superior-inferior

    '''
    
    if type(mesh) == str:
        # Load points and connections of 3-D femur model 
        if '.iv' in mesh:
            mesh = utb.read_iv(mesh)
        else:
            mesh=pv.PolyData(mesh)
    
    # Determine inertial properties
    centroid,evals,inertial_axes = utb.mass_properties(mesh)
    T_inertia = np.zeros((4,4),dtype=float)
    T_inertia[:3,:3] = inertial_axes
    T_inertia[:3,3] = centroid
    T_inertia[3,3] = 1
    
    # register points to inertial axes
    mesh_inertia = mesh.transform(np.linalg.inv(T_inertia),inplace=False)
    
    # lets figure out which way along X of the inertial axes points me towards the femur condyles. 
    # The centroid (0,0,0) now should be closer to the condyles since its larger and has more mass.
    if (max(mesh_inertia.points[:,0]) < abs(min(mesh_inertia.points[:,0]))):
        # if the max value is smaller, then we are pointed the wrong way, flip X & Y to keep us straight
        T_inertia[:3,:2] = -T_inertia[:3,:2]
        mesh_inertia = mesh.transform(np.linalg.inv(T_inertia),inplace=False)
    # plotpatch(mesh,cs_list=[T_inertia],opts={'opacity':0.7})
    
    #  Determine axial slice properties of the 3-D femur model
    slice_props = utb.sliceProperties(mesh,mesh_inertia,T_inertia,slice_thickness)
    
    # --- Isolate femoral diaphysis --- #
    # determine point where shaft beings
    area_max_index = np.argmax(slice_props['area'])
    r = (np.amax(slice_props['area']) - np.amin(slice_props['area']))/2 # half range of the max-min cross-sectional area
    d = np.abs(slice_props['area'] - r)
    condyle_end_index = np.argmin(d[area_max_index:]) + area_max_index + 1
    shaft_start_index = int(np.round(1.3*condyle_end_index))
    
    min_distance_index = np.argmin(np.abs(slice_props['index'] - slice_props['index'][shaft_start_index]))
    bottom_crop_pt = slice_props['centroid'][min_distance_index,:]
    bottom_crop_pt[:2] = centroid[:2]
    
    mesh_diaphysis = mesh.clip(normal=-1*T_inertia[:3,0],origin=bottom_crop_pt) 
    mesh_diaphysis.fill_holes(100,inplace=True)
    mesh_diaphysis = mesh_diaphysis.compute_normals() # centroid is wrong if you dont' do this
    # edges = mesh_diaphysis.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False)
    # close hole so that inertia computes properly
    # edge_faces = []
    # for i in range(edges.n_points):
    #     edge_faces.append(mesh_diaphysis.find_containing_cell(edges.points[i,:]))
    # edge_faces = np.unique(edge_faces)
    # newfaces = np.zeros(edge_faces.shape[0],dtype=int)
    # for i in range(edge_faces.shape[0]):
        
    centroid_diaphysis,evals_diaphysis,inertial_axes_diaphysis = utb.mass_properties(mesh_diaphysis)
    
    
    # In the original code, the first eigenvector (ie smallest eval) was selected as long axis
    # With the hole fill method, that doesn't seem to work.
    # Instead, I moved some code from fCondyles here that compares to the vector from mesh com to diaphysis com
    
    # make sure diaphysis vector is pointing towards the distal femur
    correct_direction = centroid - centroid_diaphysis
    dp = utb.unit(np.matmul(inertial_axes_diaphysis.T,correct_direction)) # dot product of each column with correct_direction
    idx = np.argmax(np.abs(dp))
    diaphysis_vector = inertial_axes_diaphysis[:,idx]
    if np.arccos(dp[idx]) * 180/np.pi > 90:
        diaphysis_vector = -diaphysis_vector
    
    # v = pv.Arrow(start=centroid_diaphysis,direction=diaphysis_vector,scale=70,shaft_radius=0.02,tip_radius=0.06)
    # utb.plotpatch([mesh_diaphysis,v],opts={'opacity': [0.8,1],'color':['grey','red']})
    
    
    # -- Isolate femoral condyles -- #
    pt_multiplication_factor = 500
    
    # determine where the vector through diaphysis intersects bottom of condyles
    distal_pt,_ = mesh.ray_trace(centroid_diaphysis,centroid_diaphysis+diaphysis_vector*pt_multiplication_factor)
    if len(distal_pt.shape) > 1:
        # if multiple intersections were found, take the last one
        distal_pt = distal_pt[-1,:]
    
    # create rotation matrix using the y- and z-axes from the full inertial coordinate system, and the x-axis as the shaft vector
    # set x-axis as the shaft vector pointing distally. z-axis is determined 
    # first because it is required to point posterior and the y-axis can 
    # point in any direction
    R_inertia_with_diaphysis = np.eye(3)
    R_inertia_with_diaphysis[:,0] = diaphysis_vector
    R_inertia_with_diaphysis[:,2] = utb.unit(np.cross(diaphysis_vector,T_inertia[:3,1]))
    R_inertia_with_diaphysis[:,1] = utb.unit(np.cross(R_inertia_with_diaphysis[:,2],diaphysis_vector))
        
    # make sure that the z-axis determined above is pointing posterior by
    # comparing its direction to the vector going from the shaft centroid to the full model centroid
    if utb.angle_diff(correct_direction,R_inertia_with_diaphysis[:,2]) > 90:
        R_inertia_with_diaphysis[:,1:3] = -R_inertia_with_diaphysis[:,1:3] 
    
    # determine most proximal point where condyles should be cropped (on surface) the point on the surface in the direction of the z-axis
    proximal_pt,_ = mesh.ray_trace(slice_props['centroid'][condyle_end_index,:],
                                 slice_props['centroid'][condyle_end_index,:]+R_inertia_with_diaphysis[:,2]*pt_multiplication_factor)
    if len(proximal_pt.shape) > 1:
        # if multiple intersections were found, take the last one
        proximal_pt = proximal_pt[-1,:]
    
    # create a crop normal vector, normal to the plane with the vector connecting
    # the proximal point to the distal point, and the R_inertia_with_diaphysis y-axis 
    condyle_crop_u = utb.unit(np.cross(utb.unit(distal_pt-proximal_pt),R_inertia_with_diaphysis[:,1]))
    R_crop_inertia_y = utb.unit(np.cross(condyle_crop_u,distal_pt-proximal_pt)) # used later to check cylinder direction
    
    # 1st condyles crop based on inertial RT at proximal point
    mesh_condyles = mesh.clip(normal=-condyle_crop_u,origin=proximal_pt) 
    # mesh_condyles.fill_holes(100,inplace=True)
    
    # Cylinder fit to 1st condyles crop
    # bounding box dimensions
    dim = np.amax(mesh_condyles.points,axis=0) - np.amin(mesh_condyles.points,axis=0)
    dim_idx = np.argsort(dim)
    a0_p1_idx = np.argmax(mesh_condyles.points[:,dim_idx[2]])
    a0_p2_idx = np.argmin(mesh_condyles.points[:,dim_idx[2]])
    
    x0 = mesh_condyles.points.mean(axis=0)
    a0 = utb.unit(mesh_condyles.points[a0_p2_idx,:]-mesh_condyles.points[a0_p1_idx,:])
    r0 = dim[dim_idx[:2]].mean()/2
    
    # an,xn,rn,error = fitting.fit(mesh_condyles.points,guess_angles=[(np.cos(a0[1]),np.cos(a0[0]))])
    xn, an, rn, stats = utb.lscylinder(mesh_condyles.points,x0,a0,r0)
    
    # cylinder = pv.Cylinder(center=xn,direction=an,radius=rn,height=dim[dim_idx[2]]+20)
    # utb.plotpatch([mesh_condyles,cylinder],points_list=[mesh_condyles.points[a0_p1_idx,:],mesh_condyles.points[a0_p2_idx]],
    #           opts={'color': ['grey','cyan'],'opacity' : [.5,.5],'show_edges':[True,False]})
    
    
    # Repeat finding the crop planes based on the axis through the cylinder fit of the original condyle cropping
    # make sure both inertia medial lateral axis and cylinder medial lateral axis are pointing in the same direction  
    if utb.angle_diff(an,R_crop_inertia_y) > 90:
        an = -an
    
    # create a new crop rotation matrix with the x-axis being the vector
    # connecting the proximal point to the distal point. again, the z-axis is 
    # determined first because it is required to point posterior and the y-axis
    # can point in any direction. instead of using the inertial axis to 
    # determine the z-axis, the vector through the cylinder fit is used.
    cylinder_crop_u = utb.unit(np.cross(distal_pt-proximal_pt,an))
    
    # 2nd condyles crop based on cylinder fit or original condyles crop
    mesh_condyles_cylinder = mesh.clip(normal=-cylinder_crop_u,origin=proximal_pt) 
    
    # Fit cylinder to condyles
    dim = np.amax(mesh_condyles_cylinder.points,axis=0) - np.amin(mesh_condyles.points,axis=0)
    dim_idx = np.argsort(dim)
    a0_p1_idx = np.argmax(mesh_condyles_cylinder.points[:,dim_idx[2]])
    a0_p2_idx = np.argmin(mesh_condyles_cylinder.points[:,dim_idx[2]])
    
    x0 = mesh_condyles_cylinder.points.mean(axis=0)
    a0 = utb.unit(mesh_condyles_cylinder.points[a0_p2_idx,:]-mesh_condyles_cylinder.points[a0_p1_idx,:])
    r0 = (np.ptp(mesh_condyles_cylinder.points[:,dim_idx[1]])/2 + np.ptp(mesh_condyles_cylinder.points[:,dim_idx[0]])/2)/2
    
    # an,xn,rn,error = fitting.fit(mesh_condyles_cylinder.points,guess_angles=[(np.cos(a0[1]),np.cos(a0[0]))])
    xn, an, rn, stats = utb.lscylinder(mesh_condyles_cylinder.points,x0,a0,r0)
    
    # cylinder = pv.Cylinder(center=xn,direction=an,radius=rn,height=dim[dim_idx[2]]+20)
    # utb.plotpatch([mesh,cylinder],points_list=[mesh_condyles.points[a0_p1_idx,:],mesh_condyles.points[a0_p2_idx]],
    #           opts={'color': ['grey','cyan'],'opacity' : [.5,.5],'show_edges':[True,False]})
    
    # --- Create femoral ACS --- #
    
    # Create rotation matrix from cylinder fit vector and diaphysis vector
    # make sure that the medial lateral axis is pointing in a direction that
    # will allow long axis to be pointing proximal and anterior posterior axis
    # pointing posterior.  if this is not the case negate the cylinder fit axis
    
    if utb.angle_diff(correct_direction,np.cross(diaphysis_vector,an)) > 90:
        an = -an
    
    fACS = np.eye(4)
    fACS[:3,0] = utb.unit(an)
    fACS[:3,1] = utb.unit(np.cross(-diaphysis_vector,an))
    fACS[:3,2] = utb.unit(np.cross(fACS[:3,0],fACS[:3,1]))
    fACS[:3,3] = xn
    
    if plotACS == True:
        utb.plotpatch([mesh],cs_list=[fACS],opts={'opacity':[.7]})
        
    return fACS

# filepath = r'C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Queens\Shape\UW\HealthyBoneAndCartilageModels\ACLC01_R_Tibia.stl'
# slice_thickness = 0.625
# anterior_pt = np.array([10.44,26.87,-52.37])
# plotACS = True

def buildtACS(mesh,anterior_pt,slice_thickness=0.625,plotACS=False):
    '''
    This function builds an anatomical coordinate system from a 3D model of the proximal
    femur using its plateau.
    
    Based on: 
      Miranda DL, Rainbow MJ, Leventhal EL, Crisco JJ, Fleming BC. 
      Automatic determination of anatomical coordinate systems for 
      three-dimensional bone models of the isolated human knee. 
      J Biomech. 2010 May 28;43(8):1623–6. 

    Parameters
    ----------
    mesh : string or pyvista.PolyData
        Either the filepath to a mesh file or a pyvista PolyData mesh. This is the mesh to create the ACS for.
    anterior_pt : numpy.array
        1x3 vector for the coordinates of any point on the anterior half of the tibia.
    slice_thickness : float, optional
        Slice thickness to use to slice mesh for cross-sectional properties. The default is 0.625.
    plotACS : bool, optional
        Set to True to produce a plot showing the mesh and ACS. The default is False.

    Returns
    -------
    tACS : numpy array
        4x4 pose matrix of the femur anatomical coordinate system that transforms from local to global.
        x = medial-lateral
        y = anterior-posterior
        z = superior-inferior

    '''

    # if filepath
    if type(mesh) == str:
        # Load points and connections of 3-D femur model 
        if '.iv' in mesh:
            mesh = utb.read_iv(mesh)
        else:
            mesh=pv.PolyData(mesh)   
    
    # Determine inertia properties and create transformation matrix using inertial axes and centroid
    centroid,evals,inertial_axes = utb.mass_properties(mesh)
    T_inertia = np.eye(4,dtype=float)
    T_inertia[:3,:3] = inertial_axes
    T_inertia[:3,3] = centroid
    # utb.plotpatch(mesh,cs_list=[T_inertia])
    
    # Register points to inertial axes
    mesh_inertia = mesh.transform(np.linalg.inv(T_inertia),inplace=False)
    
    #  Determine axial slice properties of the 3-D tibia model
    slice_props = utb.sliceProperties(mesh,mesh_inertia,T_inertia.copy(),slice_thickness)
    
    # --- Isolate tibial plateau --- #
    widest_slice_index = np.argmax(slice_props['area'])
    widest_pt = slice_props['centroid'][widest_slice_index,:]
    
    # lets figure out which way along X of the inertial axes points me towards the tibial plateau. 
    # The centroid (0,0,0) now should be closer to the plateau since its larger and has more mass.
    if (max(mesh_inertia.points[:,0]) > abs(min(mesh_inertia.points[:,0]))):
        # if the max value is greater, then we are pointed the wrong way, flip X & Y to keep us straight
        T_inertia[:3,:2] = -T_inertia[:3,:2]
        mesh_inertia = mesh.transform(np.linalg.inv(T_inertia),inplace=False)
    
    # we now want to change the coordinate system, so that z points in the
    # positive z direction. To do so, make z the new x, and x the negated z, we
    # are basically rotating around the y axis by 90°
    T_positive_z = np.eye(4)
    T_positive_z[:3,0] = -T_inertia[:3,2]
    T_positive_z[:3,1] = T_inertia[:3,1]
    T_positive_z[:3,2] = T_inertia[:3,0]
    # utb.plotpatch(mesh,cs_list=[T_positive_z])
    
    # Crop tibial plateau
    mesh_plateau_initial = mesh.clip(normal=-T_positive_z[:3,2],origin=widest_pt) 
    mesh_plateau_initial_filled = mesh_plateau_initial.fill_holes(100,inplace=False)
    # utb.plotpatch(mesh_plateau_initial)
    
    # flip normals for filled hole
    mesh_plateau_initial_filled.flip_normals()
    # mesh_plateau_initial_filled.plot_normals(mag=2,opacity=.5)
    
    mesh_plateau_initial.faces = np.concatenate((mesh_plateau_initial.faces,
                                                 mesh_plateau_initial_filled.faces[mesh_plateau_initial.faces.shape[0]:]))
    
    centroid_plateau,evals_plateau,inertial_axes_plateau = utb.mass_properties(mesh_plateau_initial)
    
    # create transformation matrix from the inertial axes and center of mass
    # and then orient it in the positive z direction in order to make second crop upwards
    # lets figure out which way along Z of the inertial axes points me towards
    # the tibial platau. the full centroid should be below the tibial plateau centroid
    correct_direction = utb.unit(centroid_plateau - centroid)
    if utb.angle_diff(correct_direction,inertial_axes_plateau[:,2]) > 90:
        inertial_axes_plateau[:3,1:3] = -inertial_axes_plateau[:3,1:3]
    
    # crop tibial plateau again using the inertial axes of the tibial plateau
    # create a 4x4 transformation matrix from rotation matrix and bottom crop
    # pt to be used for cropping
    mesh_plateau = mesh.clip(normal=-inertial_axes_plateau[:,2],origin=widest_pt)
    # mesh_plateau = utb.fill_hole(mesh_plateau)
    mesh_plateau_filled = mesh_plateau.fill_holes(100,inplace=False)
    # utb.plotpatch(mesh_plateau)
    
    # flip normals for filled hole
    mesh_plateau_filled.flip_normals()
    # mesh_plateau_filled.plot_normals(mag=2,opacity=.5)
    
    mesh_plateau.faces = np.concatenate((mesh_plateau.faces,
                                                 mesh_plateau_filled.faces[mesh_plateau.faces.shape[0]:]))
    
    centroid_plateau,evals_plateau,inertial_axes_plateau = utb.mass_properties(mesh_plateau)
    
    # --- Calculate tibia ACS --- #
    # Assign diaphysis vector as the largest inertial axis of the cropped plateau
    diaphysis_vector = inertial_axes_plateau[:,2]
    
    # check to make sure long axis is pointing proximal by comparing its
    # direction to the direction from the center of mass of the full tibia to
    # the center of mass of the cropped tibial plateau
    correct_direction = utb.unit(centroid_plateau - centroid)
    if utb.angle_diff(correct_direction,diaphysis_vector) > 90:
        diaphysis_vector=-diaphysis_vector
    
    # make sure anterior posterior axis is pointing forward
    # check to make sure anterior posterior axis is poinging anterior by
    # comparing its direction to the direction from the center of mass of the
    # tibial plateau crop to the specified anterior point
    anterior_direction = inertial_axes_plateau[:,1]
    if utb.angle_diff(utb.unit(anterior_pt-centroid_plateau),anterior_direction) > 90:
        anterior_direction = -anterior_direction
    
    # creat rotation matrix from the diaphysis vector and remaining inertial axes
    tACS = np.eye(4,dtype=float)
    tACS[:3,2] = diaphysis_vector
    tACS[:3,1] = anterior_direction
    tACS[:3,0] = utb.unit(np.cross(anterior_direction,diaphysis_vector))
    tACS[:3,3] = centroid_plateau
    
    if plotACS == True:
        utb.plotpatch([mesh],cs_list=[tACS],opts={'opacity':[.7]})
    
    return tACS

# filepath = r'C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Queens\Shape\UW\HealthyBoneAndCartilageModels\ACLC01_R_Patella.stl'
# side = 'R'
    
def buildpACS(mesh,side,plotACS=False): 
    '''
    This function builds an anatomical coordinate system from a 3D model of the 
    patella using its surface topography.
    
    Based on: 
      Rainbow, M. J. et al. Automatic determination of an anatomical
      coordinate system for a three-dimensional model of the human patella. 
      J Biomech (2013). doi:10.1016/j.jbiomech.2013.05.024

    Parameters
    ----------
    mesh : string or pyvista.PolyData
        Either the filepath to a mesh file or a pyvista PolyData mesh. This is the mesh to create the ACS for.
    side : string
        Either 'R' or 'L'. Right or left patella.
    plotACS : bool, optional
        Set to True to produce a plot showing the mesh and ACS. The default is False.

    Returns
    -------
    pACS : numpy array
        4x4 pose matrix of the femur anatomical coordinate system that transforms from local to global.
        x = medial-lateral
        y = anterior-posterior
        z = superior-inferior

    '''

    if type(mesh) == str:
         # Load points and connections of 3-D femur model 
        if '.iv' in mesh:
            mesh = utb.read_iv(mesh)
        else:
            mesh=pv.PolyData(mesh)
    
    
    centroid,_,CoM_eigenvectors = utb.mass_properties(mesh)
    
    # set eig3 to z-axis (A/P axis)
    ACS_L_P = np.eye(4,dtype=float)
    ACS_L_P[:3,:3] = CoM_eigenvectors
    ACS_L_P[:3,3] = centroid
    # utb.plotpatch(mesh,cs_list=ACS_L_P)
    
    
    # --- Check patella AP --- #
    # eigenvectors may be flipped 180 degree from intended orienation (from
    # posterior to anterior)  Use patella shape to check axis orienation
    
    # This function fits a 4th order polynomial to the front and back of the
    # patella and choses the side with the best R^2 as the front.  Need to
    # implement a better way to do this.
    surface_res = 0.5
    fit_order = 4
    
    mesh_local = mesh.transform(np.linalg.inv(ACS_L_P),inplace=False)
    
    # z < 0 fit
    I = np.where(mesh_local.points[:,2] < 0)[0]
    poly = PolynomialFeatures(degree=fit_order)
    X = poly.fit_transform(mesh_local.points[I,0:2])
    mdl_neg = LinearRegression()
    mdl_neg.fit(X,mesh_local.points[I,2])
    r2_neg = mdl_neg.score(poly.transform(mesh_local.points[I,0:2]),mesh_local.points[I,2])
    # p_pred = np.zeros((len(I),3))
    # p_pred[:,0:2] = mesh_local.points[I,0:2].copy()
    # p_pred[:,2] = mdl_neg.predict(poly.transform(mesh_local.points[I,0:2]))
    # utb.plotpatch(mesh_local,points_list=p_pred,opts={'opacity':.3})
    
    # z > 0 fit
    I = np.where(mesh_local.points[:,2] > 0)[0]
    poly = PolynomialFeatures(degree=fit_order)
    X = poly.fit_transform(mesh_local.points[I,0:2])
    mdl_pos = LinearRegression()
    mdl_pos.fit(X,mesh_local.points[I,2])
    r2_pos = mdl_pos.score(poly.transform(mesh_local.points[I,0:2]),mesh_local.points[I,2])
    # p_pred = np.zeros((len(I),3))
    # p_pred[:,0:2] = mesh_local.points[I,0:2].copy()
    # p_pred[:,2] = mdl_pos.predict(poly.transform(mesh_local.points[I,0:2]))
    # utb.plotpatch(mesh_local,points_list=p_pred,opts={'opacity':.3})
    
    # check direction of 3rd inertial axis
    if r2_pos < r2_neg:
        ACS_L_P[:3,2] = -ACS_L_P[:3,2]
        ACS_L_P[:3,0] = -ACS_L_P[:3,0]
    
    # correct z-axis so it is oriented from posteior to anterior
    # --- patellaIGuess --- #
    # use curvature to determine the lateral aspect of the patella
    articular_fit_order = 6
    surface_res = 0.5
    
    mesh_local = mesh.transform(np.linalg.inv(ACS_L_P),inplace=False)
    
    percentO = 0.1 
    # fit polynomial to posterior surface
    I = np.where(mesh_local.points[:,2] < np.amin(mesh_local.points[:,2])*percentO)[0]
    xy = mesh_local.points[I,0:2]
    z = mesh_local.points[I,2]
    stdind = np.sqrt(np.diag(np.cov(xy.T)))
    xy = np.matmul(xy,np.diag(1/stdind))
    poly = PolynomialFeatures(degree=articular_fit_order)
    # X = poly.fit_transform(mesh_local.points[I,0:2])
    X = poly.fit_transform(xy)
    
    # based on polyfitn matlab function, using LinearRegression doesn't seem to work properly
    scalefact = np.ones(X.shape[1])
    for i in range(X.shape[1]):
        for j in range(2):
            scalefact[i] = scalefact[i]/(stdind[j]**poly.powers_[i,j])
    Q,R,E = qr(X,mode='economic',pivoting=True)
    mdl_coeffs = np.zeros(X.shape[1])
    mdl_coeffs[E] = np.matmul(np.linalg.inv(R),np.matmul(Q.T,z))
    mdl_coeffs = mdl_coeffs * scalefact
    
    # mdl = LinearRegression()
    # mdl.fit(X,mesh_local.points[I,2])
    
    Xgrid,Ygrid = np.meshgrid(np.arange(np.amin(mesh_local.points[I,0]),np.amax(mesh_local.points[I,0]),surface_res),
                              np.arange(np.amin(mesh_local.points[I,1]),np.amax(mesh_local.points[I,1]),surface_res))
    Zgrid = np.zeros(Xgrid.shape)
    for i in range(Xgrid.shape[1]):
        # Zgrid[:,i] = mdl.predict(poly.transform(np.hstack((Xgrid[:,i:i+1],Ygrid[:,i:i+1]))))
        Zgrid[:,i] = np.matmul(poly.transform(np.hstack((Xgrid[:,i:i+1],Ygrid[:,i:i+1]))),mdl_coeffs)

    ## p_pred = np.zeros((len(I),3))
    ## p_pred[:,0:2] = mesh_local.points[I,0:2].copy()
    ## p_pred[:,2] = mdl.predict(poly.transform(mesh_local.points[I,0:2]))
    ## utb.plotpatch(mesh_local,points_list=p_pred,opts={'opacity':.3})   
    
    K, H, Pmax, Pmin, *u = utb.surfature(Xgrid,Ygrid,Zgrid)
    # Krms_all = np.sqrt((Pmin**2+Pmax**2)/2)
    
    I = np.unravel_index(np.argmax(Pmin),Pmin.shape)
    
    latDimpleL = np.array([Xgrid[I],Ygrid[I],Zgrid[I]])
    # grid = pv.StructuredGrid(Xgrid, Ygrid, Zgrid)
    # pl = pv.Plotter(notebook=False)
    # pl.add_mesh(mesh_local,opacity=.5)
    # pl.add_mesh(grid,scalars=Pmin.T.reshape(-1),show_edges=True)
    # pl.add_points(latDimpleL)
    # pl.show()
    ## utb.plotpatch(mesh_local,points_list=latDimpleL,opts={'opacity':.5})
    ## utb.plotpatch([mesh_local,grid],points_list=latDimpleL,opts={'opacity':[.5,.5]})
    
    # check that this is not a false min
    d = np.amin(np.linalg.norm(mesh_local.points - latDimpleL,axis=1))
    d_xy = np.amin(np.linalg.norm(mesh_local.points[:,:2] - latDimpleL[:2],axis=1))
    tempPmin = Pmin.copy()
    while (d > 2.0) or (d_xy > 0.1):
        tempPmin[I] = -1
        I = np.unravel_index(np.argmax(tempPmin),Pmin.shape)
        latDimpleL = np.array([Xgrid[I],Ygrid[I],Zgrid[I]])
        d = np.amin(np.linalg.norm(mesh_local.points - latDimpleL,axis=1))
        d_xy = np.amin(np.linalg.norm(mesh_local.points[:,:2] - latDimpleL[:2],axis=1))
    
    latDimpleG = np.matmul(ACS_L_P,np.concatenate((latDimpleL,np.array([1]))).T)
    latDimpleG = latDimpleG[:3]
    # utb.plotpatch(mesh,points_list=latDimpleG,opts={'opacity': .5})
    
    vertical = utb.unit(np.cross(latDimpleG-centroid,ACS_L_P[:3,2]))
    lateral = utb.unit(np.cross(ACS_L_P[:3,2],vertical))
    
    ACS_steer = ACS_L_P.copy()
    ACS_steer[:3,0] = vertical
    ACS_steer[:3,1] = lateral
    
    # objective function 
    def evalRidgeDistances(coordRot,ACS_steer,pts):
        samples = 200
        zpercent = 0.25
        zrot = utb.rotmat(coordRot,'z')
        
        ACSrot = np.matmul(ACS_steer,zrot)
        steerPtsL = np.matmul(np.linalg.inv(ACSrot),np.concatenate((pts.T,np.ones((1,pts.shape[0]))),axis=0)).T
        moveaxis_inc = np.linspace(0.9*np.amin(steerPtsL[:,0]),0.8*np.amax(steerPtsL[:,0]),samples)
        
        xdev = np.zeros(samples,dtype=float)
        for i in range(samples):
            Ipoi = np.where((steerPtsL[:,0]>moveaxis_inc[i]) & (steerPtsL[:,0]<moveaxis_inc[i]+2) & \
                            (steerPtsL[:,2]<zpercent*np.amin(steerPtsL[:,2])))[0]
            tempPts = steerPtsL[Ipoi,:3]
            II = np.argmin(tempPts[:,2])
            xdev[i] = tempPts[II,1]
        
        return np.std(xdev)
    
    
    
    bounds = Bounds(lb=-89.5*np.pi/180,ub=89.5*np.pi/180)
    res = minimize(evalRidgeDistances,x0=5.0*np.pi/180,args=(ACS_steer,mesh.points),bounds=bounds,method='Nelder-Mead',tol=1e-8)
    
    zrot = utb.rotmat(res.x[0],'z')
    
    patT = np.matmul(ACS_steer,zrot)
    patT = patT[:,[1,2,0,3]]
    
    if side == 'L':
        patT[:,0] = -patT[:,0]
        patT[:,2] = -patT[:,2]
    
    # refine 
    def evalRidgeDistances2(coordRot,ACS_steer,pts):
        samples = 100
        zpercent = 0.2
        zrot = utb.rotmat(coordRot,'z')
        
        ACSrot = np.matmul(ACS_steer,zrot)
        steerPtsL = np.matmul(np.linalg.inv(ACSrot),np.concatenate((pts.T,np.ones((1,pts.shape[0]))),axis=0)).T
        moveaxis_inc = np.linspace(0.55*np.amin(steerPtsL[:,0]),0.88*np.amax(steerPtsL[:,0]),samples)
        
        xdev = np.zeros(samples,dtype=float)
        for i in range(samples):
            Ipoi = np.where((steerPtsL[:,0]>moveaxis_inc[i]) & (steerPtsL[:,0]<moveaxis_inc[i]+2) & \
                            (steerPtsL[:,2]<zpercent*np.amin(steerPtsL[:,2])))[0]
            tempPts = steerPtsL[Ipoi,:3]
            II = np.argmin(tempPts[:,2])
            xdev[i] = tempPts[II,1]
        
        return np.std(xdev)
    
    ACS_steer2 = patT[:,[2,0,1,3]]
    bounds = Bounds(lb=-15*np.pi/180,ub=15*np.pi/180)
    res2 = minimize(evalRidgeDistances2,x0=5.0*np.pi/180,args=(ACS_steer2,mesh.points),bounds=bounds,method='Nelder-Mead',tol=1e-8)
    
    zrot = utb.rotmat(res2.x[0],'z')
    
    pACS = np.matmul(ACS_steer2,zrot)
    pACS = pACS[:,[1,2,0,3]]
    
    if plotACS == True:
        utb.plotpatch([mesh],cs_list=[pACS],opts={'opacity':[.7]})
    
    return pACS