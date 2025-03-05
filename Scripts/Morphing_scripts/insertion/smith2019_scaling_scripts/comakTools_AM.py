# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:30:17 2024

@author: qwerty
"""
import os
import shutil
# import opensim as osim
import numpy as np
import pyvista as pv
import pandas as pd
import xml.etree.ElementTree as ET
import json
from osimProcessing import miscTools
from gias2.common import transform3D
# import fitting
import opensim as osim

def loadCOMAKModel(moddir):
    pluginDir = r"Morphing_scripts\insertion\jam-plugin\build\Release\osimJAMPlugin"
    osim.common.LoadOpenSimLibrary(str(pluginDir))
    model = osim.Model(moddir)
    return model

def loadReferenceBiomechanics():
    import scipy.io as sio

    mdir = os.path.join(r'C:\Users\qwerty\Documents\Annagh\Python\OSim_Killen\forInstall\customModules\osimProcessing\data\referenceBiomechanics.mat')
    matfiledict = dict()
    
    matfile = sio.loadmat(mdir , squeeze_me = True , struct_as_record = False)['referenceBiomechanics']
       
    for dt in matfile._fieldnames: # loop kineamitcs/moments 
        matfiledict[dt] = dict()
        for vd in matfile.__dict__[dt]._fieldnames: # loop dofs
            matfiledict[dt][vd] = matfile.__dict__[dt].__dict__[vd]
    
    
    return matfiledict

def loadMLSplit():
    mldir = os.path.join(r'C:\Users\qwerty\Documents\Annagh\Python\OSim_Killen\forInstall\customModules\osimProcessing\data\mlsplit.npy')
    mlsplit = np.load(mldir, allow_pickle=True).item()
    return mlsplit
    
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


## Scale the contact geeomteries

def geomScaler(uspath, scales):

    # first load the geom
    vert,faces = miscTools.loadSTL(uspath)

    # scale the vertices
    scaledVert = (transform3D.transformScale3D(vert,[scales[0],scales[1],scales[2]]))

    # create polydata
    spd = miscTools.polygons2Polydata(scaledVert,faces)

    # output the scaled geoms
    spath = uspath[:-4] + '_scaled.stl'
    miscTools.saveSTL(spd , spath )

    return  
   
def scaleCOMAKcontactGeoms(modelDir):

    #modellevels = modelDir.split('/')[0:-1]
    #modelFolder = str()
    #for m in modellevels:
    #    modelFolder = modelFolder + m + '/'


    modelFolder = modelDir.replace( modelDir.split('\\')[-1],'')
    # Load the model using the comak tool so all the properties are there
    osimModel = loadCOMAKModel(modelDir)

    # define if it is left or right - can be found usin l/r distal femur
    # get the body set
    bodySet= osimModel.getBodySet()
    # loop through the body set

    if bodySet.hasComponent('femur_distal_l'):
        side = 'left'
        femBody = bodySet.get('femur_distal_l')
        tibBody = bodySet.get('tibia_proximal_l')
        patBody = bodySet.get('patella_l')
    elif bodySet.hasComponent('femur_distal_r'):
        side = 'right'
        femBody = bodySet.get('femur_distal_r')
        tibBody = bodySet.get('tibia_proximal_r')
        patBody = bodySet.get('patella_r')
    else:
        IOError('Model supplied model does not have the require contact geometry names - femur_distal')

    # Femur Section

    # get the scale factors
    # change the code from standard because they rewrote the API

    #_#_#_#_#_ Femur 
    femScalesOS = femBody.get_attached_geometry(0).get_scale_factors()
    femScales=np.ones(3)
    for x in range(0,3):
        femScales[x]=femScalesOS.get(x)

    # Now get the path to the contact geom 

    femContact = osimModel.getContactGeometrySet().get('femur_cartilage')
    femContactMesh = osim.PropertyString_getAs(femContact.getPropertyByName('mesh_file')).getValue()
    femBoneMesh = osim.PropertyString_getAs(femContact.getPropertyByName('mesh_back_file')).getValue()
    # scale and save
    #print(os.path.join(modelFolder, 'Geometry', femContactMesh))
    geomScaler(os.path.join(modelFolder, 'Geometry', femContactMesh) , femScales)
    geomScaler(os.path.join(modelFolder, 'Geometry', femBoneMesh) , femScales)
    # set the contact geometry paths to the scaled versions
    osim.PropertyString_getAs(femContact.getPropertyByName('mesh_file')).setValue(femContactMesh[:-4] + '_scaled.stl')
    osim.PropertyString_getAs(femContact.getPropertyByName('mesh_back_file')).setValue(femBoneMesh[:-4] + '_scaled.stl')


    #_#_#_#_#_ Tibia
    tibScalesOS = tibBody.get_attached_geometry(0).get_scale_factors()
    tibScales=np.ones(3)
    for x in range(0,3):
        tibScales[x]=tibScalesOS.get(x)

    tibContact = osimModel.getContactGeometrySet().get('tibia_cartilage')
    tibContactMesh = osim.PropertyString_getAs(tibContact.getPropertyByName('mesh_file')).getValue()
    tibBoneMesh = osim.PropertyString_getAs(tibContact.getPropertyByName('mesh_back_file')).getValue()
    # scale and save
    geomScaler(os.path.join(modelFolder,  'Geometry', tibContactMesh) , tibScales)
    geomScaler(os.path.join(modelFolder,  'Geometry', tibBoneMesh) , tibScales)
    # set the contact geometry paths to the scaled versions
    osim.PropertyString_getAs(tibContact.getPropertyByName('mesh_file')).setValue(tibContactMesh[:-4] + '_scaled.stl')
    osim.PropertyString_getAs(tibContact.getPropertyByName('mesh_back_file')).setValue(tibBoneMesh[:-4] + '_scaled.stl')


    #_#_#_#_#_ Patella
    try:
        patScalesOS = patBody.get_attached_geometry(0).get_scale_factors()
        patScales=np.ones(3)
        for x in range(0,3):
            patScales[x]=patScalesOS.get(x)
        patContact = osimModel.getContactGeometrySet().get('patella_cartilage')
        patContactMesh = osim.PropertyString_getAs(patContact.getPropertyByName('mesh_file')).getValue()
        patBoneMesh = osim.PropertyString_getAs(patContact.getPropertyByName('mesh_back_file')).getValue()
        # scale and save
        geomScaler(os.path.join(modelFolder,  'Geometry', patContactMesh) , patScales)
        geomScaler(os.path.join(modelFolder,  'Geometry', patBoneMesh) , patScales)
        # set the contact geometry paths to the scaled versions
        osim.PropertyString_getAs(patContact.getPropertyByName('mesh_file')).setValue(patContactMesh[:-4] + '_scaled.stl')
        osim.PropertyString_getAs(patContact.getPropertyByName('mesh_back_file')).setValue(patBoneMesh[:-4] + '_scaled.stl')
    except:
        print(' No Patella Contact Geom -- skipping ')

    # print the updated model 
    osimModel.printToXML(modelDir)
    return
      
        
## Create the settings xmls to run the COMAK w/f
def copyGenericJointSplines(trialInfo):

    if trialInfo['side'].lower() == 'right':
        genpath = os.path.join(r'C:\Users\qwerty\Documents\Annagh\Python\OSim_Killen\genericModels\secondary_coordinate_constraint_functions_r.xml')
    else:
        genpath = os.path.join(r'C:\Users\qwerty\Documents\Annagh\Python\OSim_Killen\genericModels\secondary_coordinate_constraint_functions_l.xml')
    
    iksetpath = os.path.join(trialInfo['baseoutputdir'], 'comak-inverse-kinematics','secondary_coordinate_constraint_functions.xml')
    
    shutil.copy(genpath, iksetpath)
    return