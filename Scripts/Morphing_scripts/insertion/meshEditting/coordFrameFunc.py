"""
functiosn for creating coordiante freams and transformation matrices
"""

# import general modules
import os
import numpy as np
from numpy import array
from numpy import linalg 
from numpy import math 

import copy
from os import path
import vtk

from xml.etree import ElementTree as ET

def createCoordFrameYZ(yPos,yNeg,zPos,zNeg):
    
    axisY=np.subtract(yPos,yNeg)
    axisY=axisY / linalg.norm(axisY)
    
    tempAxisZ=np.subtract(zPos,zNeg)
    tempAxisZ=tempAxisZ / linalg.norm(tempAxisZ)
    print("axisY shape:", axisY.shape)
    print("tempAxisZ shape:", tempAxisZ.shape)
    axisX= np.cross(axisY, tempAxisZ)
    axisX= axisX / linalg.norm(axisX)
    
    axisZ= np.cross(axisX,axisY)
    axisZ= axisZ / linalg.norm(axisZ)
    
    # convert all to transposed column vector of type array
    # make (1,3) array 1 row 3 columns
    axisX=np.array(axisX)[np.newaxis]
    axisY=np.array(axisY)[np.newaxis]
    axisZ=np.array(axisZ)[np.newaxis]
    # transpose (3,1) array 3 row 1 column
    axisX=axisX.transpose()
    axisY=axisY.transpose()
    axisZ=axisZ.transpose()
    
    # concatenate into matrix
    rot=np.concatenate((axisX,axisY,axisZ),1)

    return rot

def createCoordFrameYX(yPos,yNeg,xPos,xNeg):

    axisY=np.subtract(yPos,yNeg)
    axisY=axisY / linalg.norm(axisY)
    
    tempAxisX=np.subtract(xPos,xNeg)
    tempAxisX=tempAxisX / linalg.norm(tempAxisX)
    
    axisZ= np.cross( tempAxisX , axisY)
    axisZ= axisZ / linalg.norm(axisZ)
    
    axisX= np.cross(axisY,axisZ)
    axisX= axisX / linalg.norm(axisX)
    
    # convert all to transposed column vector of type array
    # make (1,3) array 1 row 3 columns
    #axisX=np.array(axisX)[np.newaxis]
    #axisY=np.array(axisY)[np.newaxis]
    #axisZ=np.array(axisZ)[np.newaxis]
    # transpose (3,1) array 3 row 1 column
    axisX=axisX.transpose()
    axisY=axisY.transpose()
    axisZ=axisZ.transpose()
    
    # concatenate into matrix
    rot=np.concatenate((axisX,axisY,axisZ),1)

    return rot    
    
def createCoordFrameXZ(xPos,xNeg,zPos,zNeg):

    tempAxisX=np.subtract(xPos,xNeg)
    tempAxisX=tempAxisX / linalg.norm(tempAxisX)

    axisZ= np.subtract(zPos, zNeg)
    axisZ= axisZ / linalg.norm(axisZ)
    
    axisY= np.cross(axisZ, tempAxisX)
    axisY=axisY / linalg.norm(axisY)
    
    axisX= np.cross(axisY, axisZ)
    axisX=axisX / linalg.norm(axisX)
    
    # convert all to transposed column vector of type array
    # make (1,3) array 1 row 3 columns
    #axisX=np.array(axisX)[np.newaxis]
    #axisY=np.array(axisY)[np.newaxis]
    #axisZ=np.array(axisZ)[np.newaxis]
    # transpose (3,1) array 3 row 1 column
    axisX=axisX.transpose()
    axisY=axisY.transpose()
    axisZ=axisZ.transpose()
    
    # concatenate into matrix
    rot=np.concatenate((axisX,axisY,axisZ),1)
    
    return rot 

def createCoordFramePat(patZ,yPos,yNeg):

    axisY=np.subtract(yPos,yNeg)
    axisY=axisY/linalg.norm(axisY)
    
    axisX=np.cross(patZ,yNeg)
    axisX= axisX / linalg.norm(axisX)
    
    axisZ= np.cross(axisX,axisY)
    axisZ= axisZ / linalg.norm(axisZ)
        
    # convert all to transposed column vector of type array
    # make (1,3) array 1 row 3 columns
    #axisX=np.array(axisX)[np.newaxis]
    #axisY=np.array(axisY)[np.newaxis]
    #axisZ=np.array(axisZ)[np.newaxis]
    # transpose (3,1) array 3 row 1 column
    axisX=axisX.transpose()
    axisY=axisY.transpose()
    axisZ=axisZ.transpose()
    
    # concatenate into matrix
    rot=np.concatenate((axisX,axisY,axisZ),1)
    
    return rot 
    
def convertCoordXYZ(inputArray,order,pol):
# convert coordinate array from input to opensim XYZ 
# outputArray  
    outputArray=np.zeros((1,3))
# conver the input tuple to an array
    if str(np.shape(inputArray)) == '(,3L)' or str(np.shape(inputArray)) == '(3L,)':
        inputArray=np.array(inputArray)[np.newaxis]
    elif str(np.shape(inputArray)) == '(3L,1L)' or str(np.shape(inputArray)) == '(1L,3L)': 
        print ('shape correct')
    
    
##############################################################        
##########  USE THE FIRST COLULMN OF INPUT ARRAY #############
##############################################################   
     
    if order[0] =='x':
        if pol[0] =='+':
            outputArray[0,0]=inputArray[0,0]
        elif pol[0] == '-':
            outputArray[0,0]=inputArray[0,0] *-1
            
    elif order[0] =='y':
        if pol[0] =='+':
            outputArray[0,1]=inputArray[0,0]
        elif pol[0] == '-':
            outputArray[0,1]=inputArray[0,0] *-1
            
    elif order[0] == 'z':
        if pol[0] =='+':
            outputArray[0,2]=inputArray[0,0]
        elif pol[0] == '-':
            outputArray[0,2]=inputArray[0,0] *-1
            
##############################################################        
##########  USE THE SECOND COLULMN OF INPUT ARRAY #############
############################################################## 
    if order[1] =='x':
        if pol[1] =='+':
            outputArray[0,0]=inputArray[0,1]
        elif pol[1] == '-':
            outputArray[0,0]=inputArray[0,1] *-1
            
    elif order[1] =='y':
        if pol[1] =='+':
            outputArray[0,1]=inputArray[0,1]
        elif pol[1] == '-':
            outputArray[0,1]=inputArray[0,1] *-1
            
    elif order[1] == 'z':
        if pol[1] =='+':
            outputArray[0,2]=inputArray[0,1]
        elif pol[1] == '-':
            outputArray[0,2]=inputArray[0,1] *-1

##############################################################        
##########  USE THE THIRD COLULMN OF INPUT ARRAY #############
############################################################## 
    if order[2] =='x':
        if pol[2] =='+':
            outputArray[0,0]=inputArray[0,2]
        elif pol[2] == '-':
            outputArray[0,0]=inputArray[0,2] *-1
            
    elif order[2] =='y':
        if pol[2] =='+':
            outputArray[0,1]=inputArray[0,2]
        elif pol[2] == '-':
            outputArray[0,1]=inputArray[0,2] *-1
            
    elif order[2] == 'z':
        if pol[2] =='+':
            outputArray[0,2]=inputArray[0,2]
        elif pol[2] == '-':
            outputArray[0,2]=inputArray[0,2] *-1


    return outputArray
    
def createTransformFromRot(Rot,Origin):
    # create padding
    pad=0,0,0,1
    pad=np.array(pad)[np.newaxis]
    
    # create transform
    Transform=np.concatenate((Rot,Origin),1)
    Transform=np.concatenate((Transform,pad),0)

    return Transform
   
def transformCoord(Transform,Coord):
    inputCoord=np.zeros((4,1))
    inputCoord[0,0]=Coord[0,0]
    inputCoord[1,0]=Coord[0,1]
    inputCoord[2,0]=Coord[0,2]
    inputCoord[3,0]= 1
    
    outputCoord= linalg.solve(Transform,inputCoord)
    
    coordOutput=np.zeros((3,1))
    coordOutput[0,0]=outputCoord[0,0]
    coordOutput[1,0]=outputCoord[1,0]
    coordOutput[2,0]=outputCoord[2,0]

    return coordOutput   
    
def transformPts_loop(pts, transform):
    newPts = list()
    
    for p in pts:
        p = np.reshape(p, (1,3))
        newPts.append(transformCoord(transform, p))

    return np.array((newPts))    
    
def convertVertXYZ(vertices,order,pol):
    emp=np.zeros(np.shape(vertices))
   
    k=0
    for x in vertices:
        emp[k,:]=convertCoordXYZ(x,order,pol)
        k=k+1

    return emp
    
def calcRelativeRot(parent,child):

    newChild=np.transpose(child)
    newRot=np.dot(parent,newChild)
    
    return newRot
  
def rotationMatrixToEulerAngles(R) :
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
 
    #assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])    
    
def transformPts(vertices,T):

    vertices =  np.hstack((vertices, np.ones((np.shape(vertices)[0],1)))) 
    try:
        newVertices=linalg.solve(T,vertices)
        # remove padding
        outVert=newVertices[0:3,:]
    except:
        newVertices=linalg.solve(T,np.transpose(vertices))
        # remove padding
        outVert=np.transpose(newVertices[0:3,:])
    
    return outVert
     
def defineAxisOrderFromLndmrks(lndmks):

    # if conversion dict is listed, the name of lndmks in the dictioanry need
    # to be editted to show xpos xneg etc. expected format where keys are 
    # xPos, yPos etc etc, where the entry is the name of the landmakr in the existing
    # dictioanry 
    
    #for c in convDict():
    #    lndmks[c] = lndmks[convDict[c]]


    # define axis order
    xp = np.argmax(abs(lndmks['xPos'] - lndmks['xNeg']))
    yp = np.argmax(abs(lndmks['yPos'] - lndmks['yNeg']))
    zp = np.argmax(abs(lndmks['zPos'] - lndmks['zNeg']))

    ## X
    if xp == 0:
        oneO ='x'
        if (lndmks['xPos'] - lndmks['xNeg'])[xp] > 0:
            oneP = '+'
        else:
            oneP = '-'
    elif xp == 1:
        twoO = 'x'
        if (lndmks['xPos'] - lndmks['xNeg'])[xp] > 0:
            twoP = '+'
        else:
            twoP = '-'
    elif xp == 2:
        threeO = 'x'        
        if (lndmks['xPos'] - lndmks['xNeg'])[xp] > 0:
            threeP = '+'
        else:
            threeP = '-'

    ## Y
    if yp == 0:
        oneO ='y'
        if (lndmks['yPos'] - lndmks['yNeg'])[yp] > 0:
            oneP = '+'
        else:
            oneP = '-'
    elif yp == 1:
        twoO = 'y'
        if (lndmks['yPos'] - lndmks['yNeg'])[yp] > 0:
            twoP = '+'
        else:
            twoP = '-'
    elif yp == 2:
        threeO = 'y' 
        if (lndmks['yPos'] - lndmks['yNeg'])[yp] > 0:
            threeP = '+'
        else:
            threeP = '-'
    ## Z
    if zp == 0:
        oneO ='z'
        if (lndmks['zPos'] - lndmks['zNeg'])[zp] > 0:
            oneP = '+'
        else:
            oneP = '-'
    elif zp == 1:
        twoO = 'z'
        if (lndmks['zPos'] - lndmks['zNeg'])[zp] > 0:
            twoP = '+'
        else:
            twoP = '-'
    elif zp == 2:
        threeO = 'z' 
        if (lndmks['zPos'] - lndmks['zNeg'])[zp] > 0:
            threeP = '+'
        else:
            threeP = '-'


    order = oneO + twoO + threeO
    pol = oneP + twoP + threeP
    
    return order , pol
    
def arbitraryLocalTransform(pts):
    # calculate the avaerage in each 3 axes ( x,y,z)
    arbmean = np.mean(pts, axis = 0)
    # transform
    localPts = pts - arbmean

    return localPts
    
def arbitraryLocalTransform_outputTrans(pts):
    # calculate the avaerage in each 3 axes ( x,y,z)
    arbmean = np.mean(pts, axis = 0)
    # transform
    localPts = pts - arbmean

    return localPts , arbmean