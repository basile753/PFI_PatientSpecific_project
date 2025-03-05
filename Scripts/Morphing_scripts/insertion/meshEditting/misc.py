"""
misc funcs
"""  
import numpy as np
import os
import math  
from gias2.mesh import vtktools
import vtkmodules as vtk
from vtk.util.numpy_support import vtk_to_numpy
from numpy import array


def uniqueRows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))    

def listOnlyFolders(baseDir):

    fullList = os.listdir(baseDir)
    folderList = list()    
    
    for item in fullList:
        if os.path.isdir(os.path.join(baseDir, item)):
            folderList.append(item)

    
    return folderList
    
def listOnlyFiles(baseDir):

    fullList = os.listdir(baseDir)
    fileList = list()    
    
    for item in fullList:
        if os.path.isdir(os.path.join(baseDir, item)) == 0:
            fileList.append(item)

    
    return fileList

# COPIED FUNCTION FROM GAIT2392GEOMCUSTOMIZER STEP
# all credit Ju Zhang

def polygons2Polydata(vertices, faces):
    """
    Uses create a vtkPolyData instance from a set of vertices and
    faces.

    Inputs:
    vertices: (nx3) array of vertex coordinates
    faces: list of lists of vertex indices for each face
    clean: run vtkCleanPolyData
    normals: run vtkPolyDataNormals

    Returns:
    P: vtkPolyData instance
    """
    # define points
    points = vtk.vtkPoints()
    for x, y, z in vertices:
        points.InsertNextPoint(x, y, z)

    # create polygons
    polygons = vtk.vtkCellArray()
    for f in faces:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(f))
        for fi, gfi in enumerate(f):
            polygon.GetPointIds().SetId(fi, gfi)
        polygons.InsertNextCell(polygon)

    # create polydata
    P = vtk.vtkPolyData()
    P.SetPoints(points)
    P.SetPolys(polygons)

    return P  

def getPointsAndTriFromPolyData(pd):

    # first get the points
    ptsArr = pd.GetPoints().GetData()
    pts = vtk_to_numpy(ptsArr)
    
    #now get the triangles
    triArr = pd.GetPolys().GetData()
    triO = vtk_to_numpy(triArr)
    # reshape for triangular meshes
    X = array(triO).reshape((-1,4))
    tri= X[:,1:].copy()
    
    return pts , tri 