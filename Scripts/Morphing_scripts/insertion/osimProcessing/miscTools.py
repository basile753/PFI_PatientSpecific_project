"""
Reads STL - function from GIAS library

Concatenates readSTL result from Reader for use in "fitting" functions

"""
import os
import numpy as np
import vtk
from scipy.interpolate import interp1d
from gias2.mesh import vtktools


def loadSTL(fileName):
    """
    Load an STL file and extract points and triangles.

    Parameters:
    fileName (str): Path to the STL file.

    Returns:
    tuple: A tuple containing the points (numpy array) and triangles (numpy array).
    """
    r = vtktools.Reader()
    r.readSTL(fileName)

    xCoords = r._points[:, 0]
    yCoords = r._points[:, 1]
    zCoords = r._points[:, 2]

    return r._points, r._triangles


def saveSTL(polydata, outputDir):
    """
    Save a VTK polydata object to an STL file.

    Parameters:
    polydata (vtk.vtkPolyData): The polydata object to save.
    outputDir (str): Output file path.
    """
    w = vtk.vtkSTLWriter()
    w.SetInputData(polydata)
    w.SetFileName(outputDir)
    w.SetFileTypeToASCII()
    w.Write()


def polygons2Polydata(vertices, faces):
    """
    Create a vtkPolyData instance from vertices and faces.

    Parameters:
    vertices (ndarray): (nx3) array of vertex coordinates.
    faces (list): List of lists of vertex indices for each face.

    Returns:
    vtk.vtkPolyData: A vtkPolyData instance.
    """
    # Define points
    points = vtk.vtkPoints()
    for x, y, z in vertices:
        points.InsertNextPoint(x, y, z)

    # Create polygons
    polygons = vtk.vtkCellArray()
    for f in faces:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(f))
        for fi, gfi in enumerate(f):
            polygon.GetPointIds().SetId(fi, gfi)
        polygons.InsertNextCell(polygon)

    # Create polydata
    P = vtk.vtkPolyData()
    P.SetPoints(points)
    P.SetPolys(polygons)

    return P


def interpolate101(origX, origY, nPoints):
    """
    Interpolate data points to resample at 101 evenly spaced points.

    Parameters:
    origX (ndarray): Original X values.
    origY (ndarray): Original Y values.
    nPoints (int): Number of points to interpolate.

    Returns:
    tuple: Interpolated X and Y values.
    """
    ifunc = interp1d(origX, origY)
    newX = np.linspace(origX[0], origX[-1], nPoints)  # New X values
    newY = ifunc(newX)  # Resample points

    return newX, newY


def stringToArray(inputString):
    """
    Convert a space-separated string of numbers into a 1x3 numpy array.

    Parameters:
    inputString (str): Space-separated numbers.

    Returns:
    ndarray: A 1x3 numpy array.
    """
    intArray = np.array(inputString.split(), dtype=float)
    outputArray = np.zeros((1, 3))
    outputArray[0, :] = intArray[:3]

    return outputArray


def calcEuclDist(A, B):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
    A (array-like): Coordinates of the first point.
    B (array-like): Coordinates of the second point.

    Returns:
    float: Euclidean distance between A and B.
    """
    A = np.array(A)
    B = np.array(B)

    return np.linalg.norm(B - A)
