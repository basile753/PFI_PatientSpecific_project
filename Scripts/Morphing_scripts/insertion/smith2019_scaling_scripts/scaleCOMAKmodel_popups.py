
from osimProcessing import osimTools
import comakTools_AM as comakTools
import os
import scipy.io
import opensim as osim
def loadCOMAKModel(moddir):
    pluginDir = r"Morphing_scripts\insertion\jam-plugin\build\Release\osimJAMPlugin"
    osim.common.LoadOpenSimLibrary(str(pluginDir))
    model = osim.Model(moddir)
    return model
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

#r = tk.Tk()
#r.withdraw()


#outputDirectory = str(tkfd.askdirectory(initialdir = '../Data/', title = "Select the output directory for your model"))
modelName = "ankleFootModel"
#trcDir = str(tkfd.askopenfilename( initialdir = outputDirectory, title = "Select the input static .trc file", filetypes=[("TRC File", ".trc")]))
mass = 76.8
side = 'R'
modeltype =''
#genSetupDir = str(tkfd.askopenfilename(initialdir = outputDirectory, title = "Select the generic scaling setiup xml ", filetypes=[("XML File", ".xml")]))
#genModelDir = str(tkfd.askopenfilename(initialdir = outputDirectory, title = "Select the generic model to be scaled ", filetypes=[("OSIM model", ".osim")]))

#scaleModelDir = scaleOsimModel(outputDirectory, modelName, trcDir, mass, side, modeltype, genSetupDir, genModelDir)
#comakTools.scaleCOMAKcontactGeoms(scaleModelDir)