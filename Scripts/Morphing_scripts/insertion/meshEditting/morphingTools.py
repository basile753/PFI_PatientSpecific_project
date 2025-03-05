"""
Functions or fitting -
At the moment only host-mesh fitting for models 
maybe PC ?
"""  

import numpy as np
import sys, os
import itertools
import copy
import csv
import vtk

import scipy 
from scipy.spatial import cKDTree

from gias2.fieldwork.field.tools import fitting_tools
from gias2.fieldwork.field import geometric_field
from gias2.fieldwork.field import geometric_field_fitter as GFF
from gias2.common import transform3D
from gias2.registration import alignment_fitting as af

def regScaleHMFGeomWithPassive(source_points_fitting_selected, target_points,source_points_passive_selected , verify):
    

    """
    This function takes a source and target points and a set of passive points

    Source is the data you will fit to the target points - the passive points are 
    related to the source data but are not explicitly used for the fitting.

    Example

    Source: Bone segmentations
    Target: Generic Bone models
    Passive points: Set of ligament points on the Source Bone
    
    """
    if verify is True:
        from mayavi import mlab
          
    #=============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 10.0 # host mesh padding around slave points
    host_elem_type = 'quad444' # quadrilateral cubic host elements
    host_elems = [1,1,1] # a single element host mesh
    maxit = 35
    sobd = [4,4,4]
    sobw = 1e-10
    xtol = 1e-12
    data_coord=[0.0,0.0,0.0]

    #=============================================================#
    # rigidly register source points to target point

  
  # TYPICALLY USED THIS ONE
    reg1_T, source_points_fitting_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.deg2rad((0,0,0,0,0,0)),
                                                    outputErrors=1
                                                    )
    # add isotropic scaling to rigid registration
    reg2_T, source_points_fitting_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.hstack([reg1_T, 1.0]),
                                                    outputErrors=1
                                                    )
    ## apply same transforms to the passive slave points
    source_points_passive_reg2 = transform3D.transformRigidScale3DAboutP(
                                    source_points_passive_selected,
                                    reg2_T,
                                    source_points_fitting_selected.mean(0)
                                    ) 
        
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0))
        mlab.points3d(source_points_fitting_reg2[:,0],source_points_fitting_reg2[:,1],source_points_fitting_reg2[:,2], color = (0,0,1))
        mlab.show()
    
    source_points_all = np.vstack([
                        source_points_fitting_reg2,
                        source_points_passive_reg2,
                        ])
    #=============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points

    # define some slave obj funcs
    target_tree = cKDTree(target_points)

    # distance between each source fitting point and its closest target point
    # this it is the fastest
    # should not be used if source has more geometry than target
    def slave_func_sptp(x):
        d = target_tree.query(x)[0]
        return d

    # distance between each target point and its closest source fitting point
    # should not use if source has less geometry than target
    def slave_func_tpsp(x):
        sourcetree = cKDTree(x)
        d = sourcetree.query(target_points)[0]
        return d

    # combination of the two funcs above
    # this gives the most accurate result
    # should not use if source and target cover different amount of
    # geometry
    def slave_func_2way(x):
        sourcetree = cKDTree(x)
        d_tpsp = sourcetree.query(target_points)[0]
        d_sptp = target_tree.query(x)[0]
        return np.hstack([d_tpsp, d_sptp])

    slave_func = slave_func_tpsp


    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
                    source_points_all.T,
                    host_mesh_pad,
                    host_elem_type,
                    host_elems,
                    )

    # calculate the emdedding (xi) coordinates of passive
    # source points.
    source_points_passive_xi = host_mesh.find_closest_material_points(
                                source_points_passive_reg2,
                                initGD=[50,50,50],
                                verbose=True,
                                )[0]
    # make passive source point evaluator function
    eval_source_points_passive = geometric_field.makeGeometricFieldEvaluatorSparse(
                                    host_mesh, [1,1],
                                    matPoints=source_points_passive_xi,
                                    )

    # host mesh fit
    host_x_opt, source_points_fitting_hmf,\
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
                            host_mesh,
                            source_points_fitting_reg2,
                            slave_func,
                            max_it=maxit,
                            sob_d=sobd,
                            sob_w=sobw,
                            verbose=True,
                            xtol=xtol
                            )
                            
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0))
        mlab.points3d(source_points_fitting_hmf[:,0],source_points_fitting_hmf[:,1],source_points_fitting_hmf[:,2], color = (0,0,1))
        mlab.show()            
                            
    # evaluate the new positions of the passive source points
    source_points_passive_hmf = eval_source_points_passive(host_x_opt).T

    return source_points_passive_hmf, source_points_passive_reg2
    
def regScaleHMFGeomWithPassiveWithProjection(source_points_fitting_selected, target_points,source_points_passive_selected , verify):
    

    """
    This function takes a source and target points and a set of passive points

    Source is the data you will fit to the target points - the passive points are 
    related to the source data but are not explicitly used for the fitting.

    Example

    Source: Bone segmentations
    Target: Generic Bone models
    Passive points: Set of ligament points on the Source Bone
    
    """
    if verify is True:
        from mayavi import mlab
          
    #=============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 10.0 # host mesh padding around slave points
    host_elem_type = 'quad444' # quadrilateral cubic host elements
    host_elems = [1,1,1] # a single element host mesh
    maxit = 35
    sobd = [4,4,4]
    sobw = 1e-10
    xtol = 1e-12
    data_coord=[0.0,0.0,0.0]

    #=============================================================#
    # rigidly register source points to target point

  
  # TYPICALLY USED THIS ONE
    reg1_T, source_points_fitting_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.deg2rad((0,0,0,0,0,0)),
                                                    outputErrors=1
                                                    )
    # add isotropic scaling to rigid registration
    reg2_T, source_points_fitting_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.hstack([reg1_T, 1.0]),
                                                    outputErrors=1
                                                    )
    ## apply same transforms to the passive slave points
    source_points_passive_reg2 = transform3D.transformRigidScale3DAboutP(
                                    source_points_passive_selected,
                                    reg2_T,
                                    source_points_fitting_selected.mean(0)
                                    ) 
        
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0))
        mlab.points3d(source_points_fitting_reg2[:,0],source_points_fitting_reg2[:,1],source_points_fitting_reg2[:,2], color = (0,0,1))
        mlab.show()
    
    source_points_all = np.vstack([
                        source_points_fitting_reg2,
                        source_points_passive_reg2,
                        ])
    #=============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points

    # define some slave obj funcs
    target_tree = cKDTree(target_points)

    # distance between each source fitting point and its closest target point
    # this it is the fastest
    # should not be used if source has more geometry than target
    def slave_func_sptp(x):
        d = target_tree.query(x)[0]
        return d

    # distance between each target point and its closest source fitting point
    # should not use if source has less geometry than target
    def slave_func_tpsp(x):
        sourcetree = cKDTree(x)
        d = sourcetree.query(target_points)[0]
        return d

    # combination of the two funcs above
    # this gives the most accurate result
    # should not use if source and target cover different amount of
    # geometry
    def slave_func_2way(x):
        sourcetree = cKDTree(x)
        d_tpsp = sourcetree.query(target_points)[0]
        d_sptp = target_tree.query(x)[0]
        return np.hstack([d_tpsp, d_sptp])

    slave_func = slave_func_tpsp


    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
                    source_points_all.T,
                    host_mesh_pad,
                    host_elem_type,
                    host_elems,
                    )

    # calculate the emdedding (xi) coordinates of passive
    # source points.
    source_points_passive_xi = host_mesh.find_closest_material_points(
                                source_points_passive_reg2,
                                initGD=[50,50,50],
                                verbose=True,
                                )[0]
    # make passive source point evaluator function
    eval_source_points_passive = geometric_field.makeGeometricFieldEvaluatorSparse(
                                    host_mesh, [1,1],
                                    matPoints=source_points_passive_xi,
                                    )

    # host mesh fit
    host_x_opt, source_points_fitting_hmf,\
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
                            host_mesh,
                            source_points_fitting_reg2,
                            slave_func,
                            max_it=maxit,
                            sob_d=sobd,
                            sob_w=sobw,
                            verbose=True,
                            xtol=xtol
                            )
                            
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0))
        mlab.points3d(source_points_fitting_hmf[:,0],source_points_fitting_hmf[:,1],source_points_fitting_hmf[:,2], color = (0,0,1))
        mlab.show()            
                            
    # evaluate the new positions of the passive source points
    source_points_passive_hmf = eval_source_points_passive(host_x_opt).T

    proPoints = source_points_passive_hmf
    cntr=0
    
    for x in source_points_passive_hmf:
        proPoints[cntr,:] = projectPoint(x,target_points)
        cntr = cntr +1
        
        
    proPoints = np.array(proPoints)

    return source_points_passive_hmf, proPoints
    
def projectPoint(point,boneVert):
    
    boneX=boneVert[:,0]
    boneY=boneVert[:,1]
    boneZ=boneVert[:,2]
      
    
    # define number of vert in Tendon
    #iTen = 0
    iBone = range(0,np.size(boneX))

    # create empty array
    dist=np.zeros((1,(len(iBone))))
    
    tempTenPt=point
    tempTenPt=np.array(tempTenPt)[np.newaxis]
    for j in iBone:
        tempBonePt=boneVert[j,:]
        tempBonePt=np.array(tempBonePt)[np.newaxis]
        dist[0,j]=abs(np.linalg.norm(tempTenPt - tempBonePt))

    # dist is a n x m array which contains the distance between each point on the 
    # tendon surface and each point on the pelvis surface
    # rows = Tendon points
    # columns = pelvis points
            
    # calculate the minimum distance for each point on the tendon
    #minTDistArray=dist.min(axis=1)
   # minTInd= np.where(minTDistArray == minTDistArray.min())
    ind = np.where(dist == dist.min())[1]
    newPoint = boneVert[ind]

    return newPoint

def regScaleHMFGeomOutputTrans(source_points_fitting_selected, target_points, verify):
    

    """
    This function takes a source and target points and a set of passive points

    Source is the data you will fit to the target points - the passive points are 
    related to the source data but are not explicitly used for the fitting.

    Example

    Source: Bone segmentations
    Target: Generic Bone models
    Passive points: Set of ligament points on the Source Bone
    
    """
    if verify is True:
        from mayavi import mlab
          
    #=============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 10.0 # host mesh padding around slave points
    host_elem_type = 'quad444' # quadrilateral cubic host elements
    host_elems = [1,1,1] # a single element host mesh
    maxit = 35
    sobd = [4,4,4]
    sobw = 1e-10
    xtol = 1e-12
    data_coord=[0.0,0.0,0.0]

    #=============================================================#
    # rigidly register source points to target point

  
  # TYPICALLY USED THIS ONE
    reg1_T, source_points_fitting_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.deg2rad((0,0,0,0,0,0)),
                                                    outputErrors=1
                                                    )
    # add isotropic scaling to rigid registration
    reg2_T, source_points_fitting_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.hstack([reg1_T, 1.0]),
                                                    outputErrors=1
                                                    )
    ## apply same transforms to the passive slave points
    #source_points_passive_reg2 = transform3D.transformRigidScale3DAboutP(
    #                                source_points_passive_selected,
    #                                reg2_T,
    #                                source_points_fitting_selected.mean(0)
    #                                ) 
        
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0))
        mlab.points3d(source_points_fitting_reg2[:,0],source_points_fitting_reg2[:,1],source_points_fitting_reg2[:,2], color = (0,0,1))
        mlab.show()
    
    #source_points_all = np.vstack([
    #                    source_points_fitting_reg2,
    #                    source_points_passive_reg2,
    #                    ])
                        
    source_points_all  = source_points_fitting_reg2                  
    #=============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points

    # define some slave obj funcs
    target_tree = cKDTree(target_points)

    # distance between each source fitting point and its closest target point
    # this it is the fastest
    # should not be used if source has more geometry than target
    def slave_func_sptp(x):
        d = target_tree.query(x)[0]
        return d

    # distance between each target point and its closest source fitting point
    # should not use if source has less geometry than target
    def slave_func_tpsp(x):
        sourcetree = cKDTree(x)
        d = sourcetree.query(target_points)[0]
        return d

    # combination of the two funcs above
    # this gives the most accurate result
    # should not use if source and target cover different amount of
    # geometry
    def slave_func_2way(x):
        sourcetree = cKDTree(x)
        d_tpsp = sourcetree.query(target_points)[0]
        d_sptp = target_tree.query(x)[0]
        return np.hstack([d_tpsp, d_sptp])

    slave_func = slave_func_tpsp


    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
                    source_points_all.T,
                    host_mesh_pad,
                    host_elem_type,
                    host_elems,
                    )

    # calculate the emdedding (xi) coordinates of passive
    # source points.
    #source_points_passive_xi = host_mesh.find_closest_material_points(
    #                            source_points_passive_reg2,
    #                            initGD=[50,50,50],
    #                            verbose=True,
    #                            )[0]
    # make passive source point evaluator function
    #eval_source_points_passive = geometric_field.makeGeometricFieldEvaluatorSparse(
    #                                host_mesh, [1,1],
    #                                matPoints=source_points_passive_xi,
    #                                )

    # host mesh fit
    host_x_opt, source_points_fitting_hmf,\
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
                            host_mesh,
                            source_points_fitting_reg2,
                            slave_func,
                            max_it=maxit,
                            sob_d=sobd,
                            sob_w=sobw,
                            verbose=True,
                            xtol=xtol
                            )
                            
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0))
        mlab.points3d(source_points_fitting_hmf[:,0],source_points_fitting_hmf[:,1],source_points_fitting_hmf[:,2], color = (0,0,1))
        mlab.show()            
                            
    # evaluate the new positions of the passive source points
   # source_points_passive_hmf = eval_source_points_passive(host_x_opt).T

    return  reg2_T , host_x_opt

def applyHMFoutputToPassivePointsWithProjection(reg2_T , host_x_opt , source_points , passive_points, target_points ):
    
          
    #=============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 10.0 # host mesh padding around slave points
    host_elem_type = 'quad444' # quadrilateral cubic host elements
    host_elems = [1,1,1] # a single element host mesh
    maxit = 35
    sobd = [4,4,4]
    sobw = 1e-10
    xtol = 1e-12
    data_coord=[0.0,0.0,0.0]

    #=============================================================#
    # rigidly register source points to target point

    ## apply same transforms to the passive slave points
    source_points_passive_reg2 = transform3D.transformRigidScale3DAboutP(
                                    passive_points,
                                    reg2_T,
                                    source_points.mean(0)
                                    ) 
    
    source_points_fitting_reg2 = transform3D.transformRigidScale3DAboutP(
                                    source_points,
                                    reg2_T,
                                    source_points.mean(0)
                                    ) 
    
    
    source_points_all = np.vstack([
                        source_points_fitting_reg2,
                        source_points_passive_reg2,
                        ])
    #=============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points
    
    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
                    source_points_all.T,
                    host_mesh_pad,
                    host_elem_type,
                    host_elems,
                    )

    # calculate the emdedding (xi) coordinates of passive
    # source points.
    source_points_passive_xi = host_mesh.find_closest_material_points(
                                source_points_passive_reg2,
                                initGD=[50,50,50],
                                verbose=True,
                                )[0]
    # make passive source point evaluator function
    eval_source_points_passive = geometric_field.makeGeometricFieldEvaluatorSparse(
                                    host_mesh, [1,1],
                                    matPoints=source_points_passive_xi,
                                    )
                            
    # evaluate the new positions of the passive source points
    source_points_passive_hmf = eval_source_points_passive(host_x_opt).T

    proPoints = source_points_passive_hmf
    cntr=0
    
    for x in source_points_passive_hmf:
        #proPoints[cntr,:] = projectPoint(x,target_points)
        proPoints[cntr,:] = projectPoint(x*1000,target_points*1000)/1000
        cntr = cntr +1
        
        
    proPoints = np.array(proPoints)

    return  proPoints
    
def applyHMFoutputToPassivePoints(reg2_T , host_x_opt , source_points , passive_points ):
    
          
    #=============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 10.0 # host mesh padding around slave points
    host_elem_type = 'quad444' # quadrilateral cubic host elements
    host_elems = [1,1,1] # a single element host mesh
    maxit = 35
    sobd = [4,4,4]
    sobw = 1e-10
    xtol = 1e-12
    data_coord=[0.0,0.0,0.0]

    #=============================================================#
    # rigidly register source points to target point

    ## apply same transforms to the passive slave points
    source_points_passive_reg2 = transform3D.transformRigidScale3DAboutP(
                                    passive_points,
                                    reg2_T,
                                    source_points.mean(0)
                                    ) 
    
    source_points_fitting_reg2 = transform3D.transformRigidScale3DAboutP(
                                    source_points,
                                    reg2_T,
                                    source_points.mean(0)
                                    ) 
    
    
    source_points_all = np.vstack([
                        source_points_fitting_reg2,
                        source_points_passive_reg2,
                        ])
    #=============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points
    
    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
                    source_points_all.T,
                    host_mesh_pad,
                    host_elem_type,
                    host_elems,
                    )

    # calculate the emdedding (xi) coordinates of passive
    # source points.
    source_points_passive_xi = host_mesh.find_closest_material_points(
                                source_points_passive_reg2,
                                initGD=[50,50,50],
                                verbose=True,
                                )[0]
    # make passive source point evaluator function
    eval_source_points_passive = geometric_field.makeGeometricFieldEvaluatorSparse(
                                    host_mesh, [1,1],
                                    matPoints=source_points_passive_xi,
                                    )
                            
    # evaluate the new positions of the passive source points
    source_points_passive_hmf = eval_source_points_passive(host_x_opt).T

    return  source_points_passive_hmf
 
def regScaleHMFGeom(source_points_fitting_selected, target_points, verify ):
    

    """
    This function takes a source and target points 

    Source is the data you will fit to the target points - 
    Example
    
    """
    if verify is True:
        from mayavi import mlab
        
    #=============================================================================#
    # fititng parameters for host mesh fitting
    host_mesh_pad = 10.0 # host mesh padding around slave points
    host_elem_type = 'quad444' # quadrilateral cubic host elements
    host_elems = [1,1,1] # a single element host mesh
    maxit = 35
    sobd = [4,4,4]
    sobw = 1e-10
    xtol = 1e-12
    data_coord=[0.0,0.0,0.0]

    #=============================================================#
    # rigidly register source points to target points
    reg1_T, source_points_fitting_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.deg2rad((0,0,0,0,0,0)),
                                                    outputErrors=1
                                                    )

    # add isotropic scaling to rigid registration using initial registration as starting point
    reg2_T, source_points_fitting_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
                                                    source_points_fitting_selected,
                                                    target_points,
                                                    xtol=1e-9,
                                                    sample=1000,
                                                    t0=np.hstack([reg1_T, 1.0]),
                                                    outputErrors=1
                                                    )
            
    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0), scale_factor = 0.00025)
        mlab.points3d(source_points_fitting_reg2[:,0],source_points_fitting_reg2[:,1],source_points_fitting_reg2[:,2], color = (0,0,1), scale_factor = 0.00025)
        mlab.show()

            
    #=============================================================#
    # host mesh fit source fitting points to target points and
    # apply HMF transform to passive source points

    # define some slave obj funcs
    target_tree = cKDTree(target_points)

    # distance between each source fitting point and its closest target point
    # this it is the fastest
    # should not be used if source has more geometry than target
    def slave_func_sptp(x):
        d = target_tree.query(x)[0]
        return d

    # distance between each target point and its closest source fitting point
    # should not use if source has less geometry than target
    def slave_func_tpsp(x):
        sourcetree = cKDTree(x)
        d = sourcetree.query(target_points)[0]
        return d

    # combination of the two funcs above
    # this gives the most accurate result
    # should not use if source and target cover different amount of
    # geometry
    def slave_func_2way(x):
        sourcetree = cKDTree(x)
        d_tpsp = sourcetree.query(target_points)[0]
        d_sptp = target_tree.query(x)[0]
        return np.hstack([d_tpsp, d_sptp])

    slave_func = slave_func_2way

    # make host mesh
    host_mesh = GFF.makeHostMeshMulti(
                    source_points_fitting_reg2.T,
                    host_mesh_pad,
                    host_elem_type,
                    host_elems,
                    )


    # host mesh fit
    host_x_opt, source_points_fitting_hmf,\
    slave_xi, rmse_hmf = fitting_tools.hostMeshFitPoints(
                            host_mesh,
                            source_points_fitting_reg2,
                            slave_func,
                            max_it=maxit,
                            sob_d=sobd,
                            sob_w=sobw,
                            verbose=True,
                            xtol=xtol
                            )


    if verify is True:
        mlab.points3d(target_points[:,0],target_points[:,1],target_points[:,2], color = (1,0,0), scale_factor = 0.00025 )
        mlab.points3d(source_points_fitting_hmf[:,0],source_points_fitting_hmf[:,1],source_points_fitting_hmf[:,2], color = (0,0,1), scale_factor = 0.00025)
        mlab.show()   

    return source_points_fitting_hmf