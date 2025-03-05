# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:23:39 2024

@author: qwerty
"""

import os
import shutil
from osimProcessing import xmlEditor
from osimProcessing import miscTools
import numpy as np
import opensim as osim
from vtk import vtkXMLPolyDataReader as reader 
from vtk import vtkXMLPolyDataWriter as writer 
from vtk.util import numpy_support
from gias2.common import transform3D
import scipy.signal as sig
from pathlib import Path
SELF_DIR = Path(__file__).parent


def loadCOMAKModel(moddir):
    pluginDir = r"Morphing_scripts\insertion\jam-plugin\build\Release\osimJAMPlugin"
    osim.common.LoadOpenSimLibrary(str(pluginDir))
    model = osim.Model(moddir)
    return model

def loadReferenceBiomechanics():
    import scipy.io as sio
    mdir = os.path.join(SELF_DIR, 'data', 'referenceBiomechanics.mat')
    matfiledict = {}
    matfile = sio.loadmat(mdir, squeeze_me=True, struct_as_record=False)['referenceBiomechanics']
    for dt in matfile._fieldnames:
        matfiledict[dt] = {vd: getattr(matfile.__dict__[dt], vd) for vd in matfile.__dict__[dt]._fieldnames}
    return matfiledict

def loadMLSplit():
    mldir = os.path.join(SELF_DIR, 'data', 'mlsplit.npy')
    mlsplit = np.load(mldir, allow_pickle=True).item()
    return mlsplit

def listOnlyFolders(baseDir):
    return [item for item in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, item))]

def listOnlyFiles(baseDir):
    return [item for item in os.listdir(baseDir) if not os.path.isdir(os.path.join(baseDir, item))]

def geomScaler(uspath, scales):
    vert, faces = miscTools.loadSTL(uspath)
    scaledVert = transform3D.transformScale3D(vert, scales)
    spd = miscTools.polygons2Polydata(scaledVert, faces)
    spath = uspath[:-4] + '_scaled.stl'
    miscTools.saveSTL(spd, spath)
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
    #### The position of the patella needs to be updated to reflect the scaled position, this is done
    # use a frame femur_distal_side added to geenric model
    # first get the joint 
    pfJoint = osimModel.getJointSet().get('pf_' + side[0])
    pfFrame = pfJoint.get_frames(0)
    # define the new transaltion values
    pft = osim.Vec3() 
    # new pt = old pt * sf - oldpt
    # (0.053 0.005 0.004)
    if side == 'right': 
        pft.set(0 , 0.053 * patScales[0] - 0.053)
        pft.set(1 , 0.005 * patScales[1] - 0.005)
        pft.set(2 , 0.004 * patScales[2] - 0.004)
    if side == 'left':
        pft.set(0 , 0.053 * patScales[0] - 0.053)
        pft.set(1 , 0.005 * patScales[1] - 0.005)
        pft.set(2 , (-0.004 * patScales[2] - -0.004) - 0.008)
        
        
    # now set in frame 
    pfFrame.set_translation(pft)

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