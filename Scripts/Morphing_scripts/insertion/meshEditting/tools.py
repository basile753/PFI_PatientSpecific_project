'''
Written by; Bryce A Killen
contact: bryce.killen@kuleuven.be
'''

import numpy as np
from meshEditting import loadGeom
import meshlabxml as mlx
from meshEditting import misc
import os
import os.path as path
import vtk 

from gias2.mesh import vtktools
from gias2.common import transform3D

import meshlabxml as mlx
import keyboard
# set up a function as we are repeatededly doign it 
def scaleMesh(inputMeshDir , scaleFactors):

    # Load the mesh
    meshReader= vtktools.Reader()
    meshReader.setFilename(inputMeshDir)
    meshReader.read()
    
    # load poitns and faces
    faces = meshReader._triangles
    points = meshReader._points
  
    # scale the points
    scaledPoints = (transform3D.transformScale3D(points,[scaleFactors[0],scaleFactors[1],scaleFactors[2]]))


    return scaledPoints, faces

def mirrorAndFlipFaces(fileIn):
    ## Define meshalb info
    MESHLABSERVER_PATH = 'C:\\Program Files\\VCG\\MeshLab' 
    os.environ['PATH'] += os.pathsep + MESHLABSERVER_PATH 
    ml_version='2016.12'
    # load the local STL
    _,_,_, vert, faces = loadGeom.loadSTL(fileIn)

    # create an ampty list
    svert  = []
    # "mirror" all the Z vertices
    for v in vert:
       mv = v
       mv[2] = v[2] *-1
       svert.append(mv)

    # Now re_save the mirrored z vals
    # create vtkPolyData for "mirrored"
    polydata = misc.polygons2Polydata(svert , faces )
    # define the output directory
    fileOut = fileIn[:-4] + '_mirror.stl'
    # save the "Z mirrored STL"
    loadGeom.saveSTL(polydata,fileOut)

    # create a "filter script file"
    vertFlipMirrorFaces = mlx.FilterScript(file_in = fileOut, file_out = fileOut, ml_version='2016.12') 
    # flip the normals of the faces to face outwards
    mlx.normals.flip(vertFlipMirrorFaces, force_flip=True)
    # run and save 
    vertFlipMirrorFaces.run_script()
    
    return

def flipFaces(fileIn):
    ## Define meshalb info
    MESHLABSERVER_PATH = 'C:\\Program Files\\VCG\\MeshLab' 
    os.environ['PATH'] += os.pathsep + MESHLABSERVER_PATH 
    ml_version='2016.12'
    
    # create a "filter script file"
    vertFlipMirrorFaces = mlx.FilterScript(file_in = fileIn, file_out = fileIn, ml_version='2016.12') 
    # flip the normals of the faces to face outwards
    mlx.normals.flip(vertFlipMirrorFaces, force_flip=True)
    # run and save 
    vertFlipMirrorFaces.run_script()
    
    return
 
def projectPointToSurf(point,surfVert):
    
    surfX=surfVert[:,0]
    surfY=surfVert[:,1]
    surfZ=surfVert[:,2]
      
    
    # define number of vert in Tendon
    #iTen = 0
    iSurf = range(0,np.size(surfX))

    # create empty array
    dist=np.zeros((1,(len(iSurf))))
    
    tempPt=point
    tempPt=np.array(tempPt)[np.newaxis]
    for j in iSurf:
        tempSurfPt=surfVert[j,:]
        tempSurfPt=np.array(tempSurfPt)[np.newaxis]
        dist[0,j]=np.linalg.norm(tempPt - tempSurfPt)   

    # dist is a n x m array which contains the distance between each point on the 
 
    ind = np.where(dist == dist.min())[1]
    newPoint = surfVert[ind]

    return newPoint 


def uniformMeshResamplingMLX(inputMeshDir):
    MESHLABSERVER_PATH = 'C:\\Program Files\\VCG\\MeshLab' 
    os.environ['PATH'] += os.pathsep + MESHLABSERVER_PATH 

    uniMeshRes = mlx.FilterScript(file_in = inputMeshDir, file_out= inputMeshDir, ml_version = '2016.12')
    mlx.remesh.uniform_resampling(uniMeshRes, multisample=True)
    uniMeshRes.run_script()
    return 
    
def laplacianSmoothMLX(inputMeshDir , its):
    smth = mlx.FilterScript(file_in = inputMeshDir , file_out= inputMeshDir, ml_version = '2016.12')
    mlx.smooth.laplacian( smth , iterations = its) 
    smth.run_script()
    return
    
def simplifyMLX(inputMeshDir, nFaces):
    simp = mlx.FilterScript(file_in = inputMeshDir , file_out= inputMeshDir, ml_version = '2016.12')
    mlx.remesh.simplify(simp, texture = False, faces = nFaces)
    simp.run_script()
    return

    
####### VTK IMPLEMENTATION
def reduceTriPoints(pts , tris, tar):

    # calcualte decimation percentage
    src = np.shape(pts)[0]
    per = 1 - (tar/src)
    
    # craete polyData object from points
    pd = misc.polygons2Polydata(pts, tris)

    # convert poly data to triangular mesh
    tri = vtk.vtkTriangleFilter()
    tri.SetInputDataObject(pd)
    tri.Update()

    # Preserve the topology of the mesh. This may limit the total reduction prossible,
    # which has been specified to 90%
    deci = vtk.vtkDecimatePro()
    deci.SetInputConnection(tri.GetOutputPort())
    deci.SetTargetReduction(per)
    deci.SetPreserveTopology(0)
    deci.Update()

    reducedPoly = deci.GetOutput()

    return reducedPoly
    
def triSmoother(pts, tris , its):

    # craete polyData object from points
    pd = misc.polygons2Polydata(pts, tris)

    # Take the points which are reducded and smooth them !
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(pd)
    smoother.SetNumberOfIterations(its)
    smoother.Update()  
    
    # output the polyData
    smoothPD = smoother.GetOutput()
    return smoothPD
    
   