"""
Reads STL - function from GIAS library

Concatenates readSTL result from Reader for use in "fitting" functions

"""

import numpy as np
import vtk
import scipy as sp
from os import path

# load gias 2
from gias2.fieldwork.field import geometric_field
from gias2.musculoskeletal.bonemodels import bonemodels
from gias2.musculoskeletal.bonemodels import lowerlimbatlas
from gias2.mesh import vtktools
import os

def loadMesh(filename):
    mesh = vtktools.Reader()
    mesh.setFilename(filename)
    mesh.read()
    
    return mesh
    

def loadSTL(fileName):

    r = vtktools.Reader()
    r.readSTL(fileName)

    xCoords=r._points[:,0]
    yCoords=r._points[:,1]
    zCoords=r._points[:,2]  
    
    faces = r._triangles
    
    #return xCoords , yCoords,  zCoords , r # PREVIOUS VERSION
    return xCoords , yCoords,  zCoords , r._points, faces
    
def saveSTL(polydata,outputDir):
    w=vtk.vtkSTLWriter()
    w.SetInputData(polydata)
    w.SetFileName(outputDir)
    w.SetFileTypeToASCII()
    w.Write()    