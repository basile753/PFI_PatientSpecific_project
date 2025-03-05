import os
import sys
import pdb
from osimProcessing import xmlEditor
from osimProcessing import comakTools
from osimProcessing import miscTools
import opensim as osim
import shutil
import numpy as np
from pathlib import Path
SELF_DIR = Path(__file__).parent
pluginDir = r"Morphing_scripts\insertion\jam-plugin\build\Release\osimJAMPlugin"
osim.common.LoadOpenSimLibrary(str(pluginDir))

    
def getTendonForce(currModel, currState, refDOF, MTUoi, ranges):
    """
    Returns the tendon force for a specific muscle across a range of motion using the OpenSim API.
    """
    nc = currModel.getNumCoordinates()

    # Reset all coordinates to zero
    for c in range(nc):
        currModel.getCoordinateSet().get(c).setValue(currState, 0)

    # Reference DOF and muscle
    dofOI = currModel.getCoordinateSet().get(refDOF)
    MTUOI = currModel.getMuscles().get(MTUoi)

    # Generate range of motion in radians
    if len(ranges) < 2:
        minVal, maxVal = dofOI.getRangeMin(), dofOI.getRangeMax()
        radRange = np.arange(minVal, maxVal + np.deg2rad(1), np.deg2rad(1))
    else:
        radRange = np.deg2rad(ranges)

    # Calculate tendon force
    tendonForce = []
    for rad in radRange:
        dofOI.setValue(currState, rad)
        currState = currModel.updWorkingState()
        tendonForce.append(MTUOI.getTendonForce(currState))

    return tendonForce

def getPassiveFiberForce(currModel, currState, refDOF, MTUoi, ranges):
    ##############################################################################   
    #  returns the moment arm for a specifc muscle using OPensim API
    #  Only set up for one muscle across a range of motion and once DOF
    #  Initally written for Knee MA

    #  INPUT
    #  modleDIr= char with model directory
    #  currentModel= opensim object of model
    #  currentState= opensim object of state
    #  referenceCoordinate= the coordainte you want MA taken to
    #  refernceDOI= the DOF you are moving the ROM through
    #  muscleOIName= name of the mtu you want to analyse
    #  rangeOfMotionDeg= array with each angel you want tested in degrees

    # OUTPUT
    # fibrelength , tendonlength and MTU lengths
    
    ##############################################################################
    
    nc = currModel.getNumCoordinates()
    # reset all coords to zeros
    for c in range(nc):
        currModel.getCoordinateSet().get(c).setValue(currState, 0)
    
    # Call input data 
    # DOF
    
    dofOI = currModel.getCoordinateSet().get(refDOF)
    MTUOI = currModel.getMuscles().get(MTUoi)
    
    if len(ranges) < 2:
        minVal, maxVal = dofOI.getRangeMin(), dofOI.getRangeMax()
        radRange = np.arange(minVal, maxVal + np.deg2rad(1), np.deg2rad(1))
    else:
        radRange = np.deg2rad(ranges)

    # Calculate tendon force
    passiveFiberForce = []
    for rad in radRange:
        dofOI.setValue(currState, rad)
        currState = currModel.updWorkingState()
        passiveFiberForce.append(MTUOI.getPassiveFiberForce(currState))
  
    
    return passiveFiberForce


# Force path points

def getForcePathPointsOnBody(model, bodyName):
    # init empty dictionary
    bodyPathPoints = {}
    pc = 0
    
    # Load the model
    osimModel = comakTools.loadCOMAKModel(model) if isinstance(model, str) else model
    
    # Get the force set size
    print(f"Force set size: {osimModel.getForceSet().getSize()}")  # Debugging line
    
    # Iterate through the force set
    for fi in range(osimModel.getForceSet().getSize()):
        force = osimModel.getForceSet().get(fi)
        
        # Get path points for the current force
        tempPathPoints = getForcePathPoints(osimModel, force.getName())
        print(f"Force: {force.getName()}, Path Points: {tempPathPoints}")  # Debugging line
        
        # Iterate through path points
        for pp, ppData in tempPathPoints.items():
            if ppData['bodyName'] == bodyName:
             bodyPathPoints[pc] = {
                 'name': force.getName(),
                 'ind': pp,
                 'location': ppData['location']
             }
             pc += 1

    return bodyPathPoints


def getLigamentPathPointsOnBody(model, bodyName):
    """
    Retrieves ligament path points attached to a specific body.
    """
    bodyPathPoints = {}
    pc = 0
    osimModel = comakTools.loadCOMAKModel(model) if isinstance(model, str) else model

    for fi in range(osimModel.getForceSet().getSize()):
        force = osimModel.getForceSet().get(fi)
        if force.getConcreteClassName() == 'Blankevoort1991Ligament':
            tempPathPoints = getForcePathPoints(osimModel, force.getName())
            for pp, ppData in tempPathPoints.items():
                if ppData['bodyName'] == bodyName:
                    bodyPathPoints[pc] = {
                        'name': force.getName(),
                        'ind': pp,
                        'location': ppData['location']
                    }
                    pc += 1
    return bodyPathPoints
        
    
def getLigamentPathPointsWholeModel(model):
    # init emptyDict
    bodyPathPoints = {}
    pc = 0
    osimModel = comakTools.loadCOMAKModel(model) if isinstance(model, str) else model

    for fi in range(osimModel.getForceSet().getSize()):
        force = osimModel.getForceSet().get(fi)
        if force.getConcreteClassName() == 'Blankevoort1991Ligament':
            tempPathPoints = getForcePathPoints(osimModel, force.getName())
            for pp, ppData in tempPathPoints.items():
                bodyPathPoints[pc] = {
                        'name': force.getName(),
                        'ind': pp,
                        'location': ppData['location']
                    }
                pc += 1
    return bodyPathPoints
   

def getForcePathPoints(model, forceName):
    """
    Extracts path points for a specified force in the model.

    Args:
        model: The OpenSim model object.
        forceName: The name of the force.

    Returns:
        A dictionary where keys are the path point indices and values are dictionaries with
        'bodyName' (name of the body/frame) and 'location' (numpy array of 3D coordinates).
    """
    # Initialize the dictionary to hold the path points info
    
    # Retrieve the force object
    ppInfo = {}

    # Retrieve the force object
    try:
        forceObj = model.getForceSet().get(forceName)
        print(f"Force '{forceName}' is of type '{type(forceObj).__name__}' (raw type: Force).")
    except Exception as e:
        print(f"Error retrieving force: {e}")
        

    # Downcast to specific types
    geomPath = None

    # Check if the force is a Millard2012EquilibriumMuscle
    if isinstance(forceObj, osim.Force):
        if osim.Millard2012EquilibriumMuscle.safeDownCast(forceObj):
            geomPath = osim.Millard2012EquilibriumMuscle.safeDownCast(forceObj).getGeometryPath()
            print(f"Force '{forceName}' is of type 'Millard2012EquilibriumMuscle'.")
        elif osim.Blankevoort1991Ligament.safeDownCast(forceObj):
            geomPath = osim.Blankevoort1991Ligament.safeDownCast(forceObj).get_GeometryPath()
            print(f"Force '{forceName}' is of type 'Blankevoort1991Ligament'.")
        elif osim.PathActuator.safeDownCast(forceObj):
            geomPath = osim.PathActuator.safeDownCast(forceObj).getGeometryPath()
            print(f"Force '{forceName}' is of type 'PathActuator'.")
        else:
            print(f"Force '{forceName}' is not a recognized path-based force. Skipping.")
            return ppInfo
    else:
        print(f"Force '{forceName}' is not derived from Force. Skipping.")
        return ppInfo

    # Access the PathPointSet from the GeometryPath
    try:
        ppSet = geomPath.getPathPointSet()
    except Exception as e:
        print(f"Error accessing PathPointSet for force {forceName}: {e}")
        return ppInfo

    # Iterate through each path point
    for pi in range(ppSet.getSize()):
        # Initialize dictionary entry for this path point
        ppInfo[pi] = {'bodyName': None, 'location': None}

        # Retrieve the path point object
        tpp = ppSet.get(pi)

        # Cast to concrete PathPoint
        tppConcrete = osim.AbstractPathPoint.safeDownCast(tpp)
        print (tppConcrete)
        if tppConcrete is None:
            print(f"Error: PathPoint {pi} is not of the correct subclass. Skipping.")
            continue
        
        # Get the body name from the parent frame
        try:
            #bodyName = tppConcrete.getBodyName()
            bodyName = tppConcrete.getPropertyByIndex(1).toString()
            bodyName = bodyName.split('/')[-1]
            ppInfo[pi]['bodyName'] = bodyName
        except Exception as e:
            print(f"Error retrieving bodyName for path point {pi}: {e}")
            continue

        # Get the location of the path point
        try:
            locationVec = tppConcrete.getPropertyByIndex(2).toString()  # Get the Vec3 location
            numbers_str = locationVec.strip('()').split()
            location = np.array([float(num) for num in numbers_str])
            ppInfo[pi]['location'] = location
        except Exception as e:
            print(f"Error retrieving location for path point {pi}: {e}")
            continue
   
    return ppInfo

def updPathPathsFromHMF(model, bodyPathPointsDict, newPoints):

    for pi in range(0, np.size(bodyPathPointsDict.keys())):
        forceName = bodyPathPointsDict[pi]['name']
        ppInd = bodyPathPointsDict[pi]['ind']
        newLocation = newPoints[pi]
        setForcePathPoint(model,forceName , ppInd , newLocation)
  
    return model    

def setForcePathPoint(model, forceName, ppInd, newLocation):
    """
    Updates the location of a specified path point for a given force in the OpenSim model.

    Args:
        model: The OpenSim model object.
        forceName: The name of the force.
        ppInd: Index of the path point to update.
        newLocation: New location as a list or numpy array [x, y, z].

    Returns:
        None
    """
    try:
        forceObj = model.getForceSet().get(forceName)
        if forceObj is None:
            print(f"Error: Force '{forceName}' not found in model.")
            return
    except Exception as e:
        print(f"Error retrieving force '{forceName}': {e}")
        return

    geomPath = None
    if isinstance(forceObj, osim.Force):
        if osim.Millard2012EquilibriumMuscle.safeDownCast(forceObj):
            geomPath = osim.Millard2012EquilibriumMuscle.safeDownCast(forceObj).getGeometryPath()
        elif osim.Blankevoort1991Ligament.safeDownCast(forceObj):
            geomPath = osim.Blankevoort1991Ligament.safeDownCast(forceObj).get_GeometryPath()
        elif osim.PathActuator.safeDownCast(forceObj):
            geomPath = osim.PathActuator.safeDownCast(forceObj).getGeometryPath()
        else:
            print(f"Force '{forceName}' is not a recognized path-based force. Skipping.")
            return
    else:
        print(f"Force '{forceName}' is not derived from Force. Skipping.")
        return

    try:
        ppSet = geomPath.getPathPointSet()
        if ppInd >= ppSet.getSize():
            print(f"Error: Path point index {ppInd} is out of range.")
            return
    except Exception as e:
        print(f"Error accessing PathPointSet: {e}")
        return

    pp = ppSet.get(ppInd)
    ppConcrete = osim.AbstractPathPoint.safeDownCast(pp)
    if ppConcrete is None:
        print(f"Error: PathPoint {ppInd} is not of the correct subclass. Skipping.")
        return

    try:
        if not isinstance(newLocation, osim.Vec3):
            newLocationVec = osim.Vec3(*newLocation)
        else:
            newLocationVec = newLocation
        
        ppConcrete.setLocation(newLocationVec)
        model.updForceSet()
        print(f"Successfully updated PathPoint {ppInd} for force '{forceName}'.")
    except Exception as e:
        print(f"Error setting new location for PathPoint {ppInd}: {e}")


    # Update the force set in the model
    model.updForceSet()
    
    return model  

def getForceCurrentLength(model, state, forceName):

    try:
        forceObj = model.getForceSet().get(forceName)
    except:
        model = comakTools.loadCOMAKModel(model)
        forceObj = model.getForceSet().get(forceName)
        
    if isinstance(forceObj, osim.Force):
        if osim.Millard2012EquilibriumMuscle.safeDownCast(forceObj):
            geomPath = osim.Millard2012EquilibriumMuscle.safeDownCast(forceObj).getGeometryPath()
            print(f"Force '{forceName}' is of type 'Millard2012EquilibriumMuscle'.")
        elif osim.Blankevoort1991Ligament.safeDownCast(forceObj):
            geomPath = osim.Blankevoort1991Ligament.safeDownCast(forceObj).get_GeometryPath()
            print(f"Force '{forceName}' is of type 'Blankevoort1991Ligament'.")
        elif osim.PathActuator.safeDownCast(forceObj):
            geomPath = osim.PathActuator.safeDownCast(forceObj).getGeometryPath()
            print(f"Force '{forceName}' is of type 'PathActuator'.")
        else:
            print(f"Force '{forceName}' is not a recognized path-based force. Skipping.")
            return 
    else:
        print(f"Force '{forceName}' is not derived from Force. Skipping.")
        return 
    
    len = geomPath.getLength(state)

    return len

# Muscels

# single body
def getMusclesOnBody(model , bodyName):
    # init emptyDict
    muscList = list()
    # empty counter
    if type(model) is str:
        # Load the model as 
        osimModel = comakTools.loadCOMAKModel(model)
    else:
        osimModel = model 
        
    # get the forceSet
    #forceSet = osimModel.getForceSet()
    forceSet = osimModel.getMuscles()
    # now loop through each force
    for fi in range(0, forceSet.getSize()):
        # for each force return the pathpoints
        tempPathPoints = getForcePathPoints(osimModel, forceSet.get(fi).getName())
        # now loop through each of the pathpoints for that force
        for pp in tempPathPoints.keys():
            if tempPathPoints[pp]['bodyName'] == bodyName:
                if muscList.count(forceSet.get(fi).getName()) is 0:
                    muscList.append(forceSet.get(fi).getName())

    return muscList
#  multiple bodies
def getMusclesOnBodies(model, bodyList):

    fullMuscList = list()
    
    for bod in bodyList:
        tmuscLst= getMusclesOnBody(model , bod)
        
        for t in tmuscLst:
            if fullMuscList.count(t) is 0:
                fullMuscList.append(t)    

    return fullMuscList

# calibrate normalise length
def calMuscStaticPropMTULength(refModel , ssModel, muscName):
# Scale the tendon slack length and optimal fiber length based on a scale
#factor calculate using the static length of the MTU

    # for the targetmodel i.e., scaled/generic model 
    # get current length
    refLen = getForceCurrentLength(refModel, muscName)
    # for the personaslied
    # get teh current length
    ssLen = getForceCurrentLength(ssModel, muscName) 
    # calc ratio
    #lenR = refLen/tarLen
    lenR = ssLen/refLen
    # reset the optimal fibre length
    ssModel.getMuscles().get(muscName).setOptimalFiberLength(refModel.getMuscles().get(muscName).getOptimalFiberLength()*lenR)
    # reset the tendon slack length
    ssModel.getMuscles().get(muscName).setTendonSlackLength(refModel.getMuscles().get(muscName).getTendonSlackLength()*lenR)

    ssModel.updMuscles()
      
    return ssModel

def calMuscStaticPropNormValEquilMusc(refModel , ssModel, muscName):
    
    # initialise the models
    refModel.equilibrateMuscles(refModel.initSystem())
    refState = refModel.updWorkingState()
    
    ssModel.equilibrateMuscles(ssModel.initSystem())
    ssState = ssModel.updWorkingState()

# First lets do optimal fibre or fiber length     
    # get the reference normalised length 
    refNFL = refModel.getMuscles().get(muscName).getNormalizedFiberLength(refState)
    # ze mathz for this calc
    #OFL  = FiberLength/NormalizedFiberLEngth where FiberLenght is from SS model and normalized if from test model
    # reset the optimal fibre length
    ssModel.getMuscles().get(muscName).setOptimalFiberLength(ssModel.getMuscles().get(muscName).getFiberLength(ssState)/refNFL)
    
    ssModel.updMuscles()
    
# Second lets change tendon slack length
    # get the reference train in the tendon
    refTS = refModel.getMuscles().get(muscName).getTendonStrain(refState) + 1 # add the 1 to change to a %
    # reset the tendon slack length
    ssModel.getMuscles().get(muscName).setTendonSlackLength(ssModel.getMuscles().get(muscName).getTendonLength(ssState)/refTS )

    ssModel.updMuscles()
      
    return ssModel

def calMuscStaticPropNormVal(refModel , ssModel, muscName):
    
    # initialise the models
    #refModel.equilibrateMuscles(refModel.initSystem())
    #refState = refModel.updWorkingState()
    
    #ssModel.equilibrateMuscles(ssModel.initSystem())
    #ssState = ssModel.updWorkingState()

# First lets do optimal fibre or fiber length     
    # get the reference normalised length 
    refNFL = refModel.getMuscles().get(muscName).getNormalizedFiberLength(refModel.initSystem())
    # ze mathz for this calc
    #OFL  = FiberLength/NormalizedFiberLEngth where FiberLenght is from SS model and normalized if from test model
    # reset the optimal fibre length
    ssModel.getMuscles().get(muscName).setOptimalFiberLength( ssModel.getMuscles().get(muscName).getFiberLength(ssModel.initSystem())/refNFL)
    ssModel.updMuscles()
    
# Second lets change tendon slack length
    # get the reference train in the tendon
    refTS = refModel.getMuscles().get(muscName).getTendonStrain(refModel.initSystem()) + 1 # add the 1 to change to a %
    # reset the tendon slack length
    ssModel.getMuscles().get(muscName).setTendonSlackLength(ssModel.getMuscles().get(muscName).getTendonLength(ssModel.initSystem())/refTS )
    ssModel.updMuscles()
      
    return ssModel

# Wrapping surfaces

def getWrapSurfacesOnBody(model, bodyName):

    wrapDict = dict()
    
    for i in range(0, model.getBodySet().get(bodyName).getWrapObjectSet().getSize()):
    
        t = list()
        vecT = model.getBodySet().get(bodyName).getWrapObjectSet().get(i).get_translation()
        
        for x in range(0,3):
            t.append(vecT.get(x))
            
        t = np.array(t)    
        wrapDict[model.getBodySet().get(bodyName).getWrapObjectSet().get(i).getName()] = t

    return wrapDict

def updWrapPointsFromHMF(model, wrapDict , newPoints, body):
    
    # counter for newPoints array
    c = 0
    # loop through the points in wrap dict
    for i in wrapDict.keys():
        # get the wrapping surface
        ws = model.getBodySet().get(body).getWrapObjectSet().get(i)
        
        # new vec for trans 
        nTrans = osim.Vec3()
        tempPoint = newPoints[c]
        for ii in range(0,3):
            nTrans.set(ii, tempPoint[ii])
            
        ws.set_translation(nTrans) 
        # counter it
        c=c+1
    return model
    
# Ligament properties

def updLigamentSlackLength(modelDir, ligName, l0):
    
    # load the model as an xml/dict
    m = xmlEditor.readGenericXML(modelDir)
    # return all the ligs info
    allLigs = m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament']

    # loop through all the ligamets
    for i in range(0,np.size(allLigs)):
        # get the name
        tn=(allLigs[i]['@name'])
        # check name 
        if tn == ligName:
            # if yes - reset it
            m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament'][i]['slack_length'] = l0
            # After updating the model we need to save it again - 
            xmlEditor.saveDictAsXML(m , modelDir)
            return
    return

def getLigamentSlackLength(modelDir, ligName):
    
    # load the model as an xml/dict
    m = xmlEditor.readGenericXML(modelDir)
    # return all the ligs info
    allLigs = m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament']

    # loop through all the ligamets
    for i in range(0,np.size(allLigs)):
        # get the name
        tn=(allLigs[i]['@name'])
        # check name 
        if tn == ligName:
            # if yes - reset it
            ligSL = m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament'][i]['slack_length']
            return np.float(ligSL)

    return

def updLigamentStiffness(modelDir, ligName, k):

    # load the model as an xml/dict
    m = xmlEditor.readGenericXML(modelDir)
    # return all the ligs info
    allLigs = m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament']

    # loop through all the ligamets
    for i in range(0,np.size(allLigs)):
        # get the name
        tn=(allLigs[i]['@name'])
        # check name 
        if tn == ligName:
            # if yes - reset it
            m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament'][i]['linear_stiffness'] = k
            # After updating the model we need to save it again - 
            xmlEditor.saveDictAsXML(m , modelDir)
            return
    return

def getLigamentStiffness(modelDir, ligName):
    
    # load the model as an xml/dict
    m = xmlEditor.readGenericXML(modelDir)
    # return all the ligs info
    allLigs = m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament']

    # loop through all the ligamets
    for i in range(0,np.size(allLigs)):
        # get the name
        tn=(allLigs[i]['@name'])
        # check name 
        if tn == ligName:
            # if yes - reset it
            ligk = m['OpenSimDocument']['Model']['ForceSet']['objects']['Blankevoort1991Ligament'][i]['linear_stiffness']
            return np.float(ligk)

    return
 

# Scale models

def scaleOActiveModel(outputDirectory, modelName, trcDir, mass, side , modeltype ):
    # side - either left or right
    # modeltype - eitehr medial , lateral, or full
    
    ## other osim doesnt lie kit
    if os.path.isdir(os.path.join(outputDirectory, 'Geometry')) is False:
        shutil.copytree(os.path.join(SELF_DIR ,'setupFiles' , 'Geometry'), os.path.join(outputDirectory, 'Geometry')  )
      

    # Written specifically to scale the OpenSim model used for the OActive 
    # dataset - and therefore have to define the left and the right side as the contact 
    # is only defined for one side

    ## define the path to subject - set as empty to use full paths
    pts = ''
    
    ## use the var side to determine to use the left or right scale and osim models 
    if side.lower() == 'left':
        if modeltype.lower() == 'full':
            genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleLeft.xml')
            genModelDir = os.path.join(SELF_DIR ,'setupFiles', 'OActiveLeft.osim')
        elif modeltype.lower() == 'medial':
            genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleLeft.xml')
            genModelDir = os.path.join(SELF_DIR ,'setupFiles', 'OActiveLeft_medial.osim')     
        elif modeltype.lower() == 'lateral':
            genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleLeft.xml')
            genModelDir = os.path.join(SELF_DIR ,'setupFiles', 'OActiveLeft_lateral.osim')      
        
        
    elif side.lower() == 'right':
        if modeltype.lower() == 'full':
            genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleRight.xml')
            genModelDir = os.path.join(SELF_DIR ,'setupFiles' , 'OActiveRight.osim')
        elif modeltype.lower() == 'medial':
            genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleRight.xml')
            genModelDir = os.path.join(SELF_DIR ,'setupFiles' , 'OActiveRight_medial.osim')
        elif modeltype.lower() == 'lateral':
            genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleRight.xml')
            genModelDir = os.path.join(SELF_DIR ,'setupFiles' , 'OActiveRight_lateral.osim')

    ## create the scale tool
    scaleTool = osim.ScaleTool(genSetupDir)
    scaleTool.setSubjectMass(mass)
    
    ## set the time range
    tr = osim.ArrayDouble()
    tr.setSize(2)
    # load the trc to get the times
    trcf = osim.Storage(trcDir)
    tr.set(0, trcf.getFirstTime())
    tr.set(1, trcf.getLastTime())
    
    scaleTool.getModelScaler().setTimeRange(tr)
    scaleTool.getMarkerPlacer().setTimeRange(tr)
     
    ## add the model to the model maker
    scaleTool.getGenericModelMaker().setModelFileName(genModelDir)
    
    ## add the static TRC to the model scaler and makrer aplcer 
    scaleTool.getMarkerPlacer().setMarkerFileName(trcDir)
    scaleTool.getModelScaler().setMarkerFileName(trcDir)
    
    ## set the output model directory in both marker placer and model scaler
    if modeltype == 'full':
        outModelDir = os.path.join(outputDirectory, modelName + '_scaled_' + side + '.osim' )
    else:
        outModelDir = os.path.join(outputDirectory, modelName + '_scaled_' + side + '_' + modeltype + '.osim' )
        
    scaleTool.getModelScaler().setOutputModelFileName(outModelDir)
    scaleTool.getMarkerPlacer().setOutputModelFileName(outModelDir)
    
    ## set the ouput scale file
    outputScale = os.path.join(outputDirectory, modelName + '_scaleFactors_' + side + '.txt')
    scaleTool.getModelScaler().setOutputScaleFileName(outputScale)
    
    ## set the output motion file
    outputMotion =  os.path.join(outputDirectory, modelName + '_staticPose_' + side + '.mot')
    scaleTool.getMarkerPlacer().setOutputMotionFileName(outputMotion)

    ## prtin the scal setup
    scaleTool.printToXML(os.path.join(outputDirectory, modelName + '_scaleSetup_' + side + '.xml'))

    ## execute both the model scaler and marker placer    
    scaleTool.getModelScaler().processModel(osim.Model(genModelDir), pts,mass)
    scaleTool.getMarkerPlacer().processModel(osim.Model(outModelDir), pts)


    return outModelDir
   
def scaleOsimModel(outputDirectory, modelName, trcDir, mass, side, modeltype, genSetupDir, genModelDir  ):

# A copy of the above OACtive model scaler written under a different name, to make this
# function work for your own data, you need to edit the genSetupDir , and genModelDir
# directories to reflect your own generic scaleSetup and model f iles

    ## define the path to subject - set as empty to use full paths
    pts = ''
    
    ## use the var side to determine to use the left or right scale and osim models 
    #if side.lower() == 'left':
    #    genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleLeft.xml')
    #    genModelDir = os.path.join(SELF_DIR ,'setupFiles', 'OActiveLeft.osim')
    #elif side.lower() == 'right':
    #    genSetupDir = os.path.join(SELF_DIR ,'setupFiles' , 'oactiveScaleRight.xml')
    #    genModelDir = os.path.join(SELF_DIR ,'setupFiles' , 'OActiveRight.osim')

    ## create the scale tool
    scaleTool = osim.ScaleTool(genSetupDir)
    scaleTool.setSubjectMass(mass)
    
    ## set the time range
    tr = osim.ArrayDouble()
    tr.setSize(2)
    # load the trc to get the times
    trcf = osim.Storage(trcDir)
    tr.set(0, trcf.getFirstTime())
    tr.set(1, trcf.getLastTime())
    
    scaleTool.getModelScaler().setTimeRange(tr)
    scaleTool.getMarkerPlacer().setTimeRange(tr)
     
    ## add the model to the model maker
    scaleTool.getGenericModelMaker().setModelFileName(genModelDir)
    
    ## add the static TRC to the model scaler and makrer aplcer 
    scaleTool.getMarkerPlacer().setMarkerFileName(trcDir)
    scaleTool.getModelScaler().setMarkerFileName(trcDir)
    
    ## set the output model directory in both marker placer and model scaler
    outModelDir = os.path.join(outputDirectory, modelName + '_scaled_' + side + '.osim' )
    scaleTool.getModelScaler().setOutputModelFileName(outModelDir)
    scaleTool.getMarkerPlacer().setOutputModelFileName(outModelDir)
    
    ## set the ouput scale file
    outputScale = os.path.join(outputDirectory, modelName + '_scaleFactors_' + side + '.txt')
    scaleTool.getModelScaler().setOutputScaleFileName(outputScale)
    
    ## set the output motion file
    outputMotion =  os.path.join(outputDirectory, modelName + '_staticPose_' + side + '.mot')
    scaleTool.getMarkerPlacer().setOutputMotionFileName(outputMotion)

    ## prtin the scal setup
    scaleTool.printToXML(os.path.join(outputDirectory, modelName + '_scaleSetup_' + side + '.xml'))

    ## execute both the model scaler and marker placer    
    scaleTool.getModelScaler().processModel(osim.Model(genModelDir), pts,mass)
    scaleTool.getMarkerPlacer().processModel(osim.Model(outModelDir), pts)

    return outModelDir
   
def scaleOsimModelManualSf(outputDirectory, modelName, trcDir, mass, side, modeltype, genSetupDir, genModelDir, sfDict  ):

# A copy of the above OACtive model scaler written under a different name, to make this
# function work for your own data, you need to edit the genSetupDir , and genModelDir
# directories to reflect your own generic scaleSetup and model f iles

    ## define the path to subject - set as empty to use full paths
    pts = ''
    
    ## create the scale tool
    scaleTool = osim.ScaleTool(genSetupDir)
    scaleTool.setSubjectMass(mass)
    
    ## set the time range
    tr = osim.ArrayDouble()
    tr.setSize(2)
    # load the trc to get the times
    trcf = osim.Storage(trcDir)
    tr.set(0, trcf.getFirstTime())
    tr.set(1, trcf.getLastTime())
    
    scaleTool.getModelScaler().setTimeRange(tr)
    scaleTool.getMarkerPlacer().setTimeRange(tr)
     
    ## setup the manual scale factors 
    # get the scale
    scaleSet = scaleTool.getModelScaler().getScaleSet()
    # loop throgh the sfDict
    
    for sfname in sfDict:
        #create a scale 
        tss = osim.Scale()
        tss.setApply(True)
        tss.setName(sfname)
        tss.setSegmentName(sfname)
        tss.setScaleFactors(osim.Vec3(sfDict[sfname],sfDict[sfname],sfDict[sfname]))
        scaleSet.cloneAndAppend(tss)
    
    ## add the model to the model maker
    scaleTool.getGenericModelMaker().setModelFileName(genModelDir)
    
    ## add the static TRC to the model scaler and makrer aplcer 
    scaleTool.getMarkerPlacer().setMarkerFileName(trcDir)
    scaleTool.getModelScaler().setMarkerFileName(trcDir)
    
    ## set the output model directory in both marker placer and model scaler
    outModelDir = os.path.join(outputDirectory, modelName + '_scaled_' + side + '.osim' )
    scaleTool.getModelScaler().setOutputModelFileName(outModelDir)
    scaleTool.getMarkerPlacer().setOutputModelFileName(outModelDir)
    
    ## set the ouput scale file
    outputScale = os.path.join(outputDirectory, modelName + '_scaleFactors_' + side + '.txt')
    scaleTool.getModelScaler().setOutputScaleFileName(outputScale)
    
    ## set the output motion file
    outputMotion =  os.path.join(outputDirectory, modelName + '_staticPose_' + side + '.mot')
    scaleTool.getMarkerPlacer().setOutputMotionFileName(outputMotion)

    ## prtin the scal setup
    scaleTool.printToXML(os.path.join(outputDirectory, modelName + '_scaleSetup_' + side + '.xml'))

    ## execute both the model scaler and marker placer    
    scaleTool.getModelScaler().processModel(osim.Model(genModelDir), pts,mass)
    scaleTool.getMarkerPlacer().processModel(osim.Model(outModelDir), pts)

    return outModelDir
   

def getMarkerPosition(model , markerName):
    # get marker location
    locVec = model.getMarkerSet().get(markerName).get_location()
    # empty vec
    loc=np.ones(3)
    for x in range(0,3):
        loc[x]=locVec.get(x)
        
    return loc 

def getBodyScaleFactors(modelDir , body):
    
    # first load the model
    osimModel = comakTools.loadCOMAKModel(modelDir)
    # get the body set
    bodySet = osimModel.getBodySet()
    # ge the body
    body = bodySet.get(body)

    scaleOS =  body.get_attached_geometry(0).get_scale_factors()
    bodyScales=np.ones(3)
    for x in range(0,3):
        bodyScales[x]=scaleOS.get(x)

    return bodyScales
 

 
def getOsimArrAsList(arr):
    
    # create empty list
    listObj  = list()

    # loop through column lables and append to list
    for i in range(0,arr.getSize()):
        listObj.append(arr.get(i))

    return listObj

def loadStorageFile(storageDir):

    # load in the storage file

    storageFile = osim.Storage(storageDir)
     
    
    # now we want to convert it from an opensim file into abs
    # dictionary so we can use it
    
    # get the column labels
    colLabelsOsim = storageFile.getColumnLabels()
    # return them as a python list
    #colLabels = getOsimArrAsList(colLabelsOsim)
    
    # get the time column
    timeOsim = osim.ArrayDouble()
    storageFile.getTimeColumnWithStartTime(timeOsim, storageFile.getFirstTime())
    time = getOsimArrAsList(timeOsim)
    
    # create a dictionary
    storageFileDict = dict()
    #storageFileDict['time'] = time
    
    # now loop through each coumn and add to dict
    for i in range(0 , colLabelsOsim.getSize()):
        # create empty array
        tempArr = osim.ArrayDouble()
        # get the values for a specific var
        storageFile.getDataColumn(colLabelsOsim.get(i), tempArr)
        # convert values to a list instead of osim array
        arr = getOsimArrAsList(tempArr)
        # add to dictioanry
        storageFileDict[colLabelsOsim.get(i)] = arr

    storageFileDict['time'] = list(np.round(time,2))
    
    
    return storageFileDict
  
def getValsFromStorage(storageFile , storageKey):

    vals = storageFile[storageKey]
    # normalise
    normVals = miscTools.interpolate101(range(0,np.size(vals)) , vals , 101)

    return normVals

def getValsFromStorageTR(storageFile , storageKey, timeRange):

    try: # if you give indexes this will work
        i0 = timeRange[0]
        i1 = timeRange[1]
        vals = storageFile[storageKey][i0:i1]
    except: # if you give time ranges this will work 
        i0 = storageFile['time'].index(timeRange[0])
        i1 = storageFile['time'].index(timeRange[1])
        vals = storageFile[storageKey][i0:i1]
    
    # normalise
    normVals = miscTools.interpolate101(range(0,np.size(vals)) , vals , 101)

    return normVals

def getValsFromStorageRaw(storageFile , storageKey, timeRange):

    # get the values from the dictioanry
    if timeRange is False:
        vals = storageFile[storageKey]
    else:
        i0 = storageFile['time'].index(timeRange(0))
        i1 = storageFile['time'].index(timeRange(1))
        vals = storageFile[storageKey][i0:i1]

    return vals

def getValsFromStorageRawTR(storageFile , storageKey, timeRange):

    # get the values from the dictioanry
    if timeRange is False:
        vals = storageFile[storageKey]
    else:
        i0 = storageFile['time'].index(timeRange(0))
        i1 = storageFile['time'].index(timeRange(1))
        vals = storageFile[storageKey][i0:i1]

    return vals