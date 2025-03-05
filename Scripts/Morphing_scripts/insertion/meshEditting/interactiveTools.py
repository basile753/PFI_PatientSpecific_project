'''
Written by; Bryce A Killen
contact: bryce.killen@kuleuven.be
'''
import numpy as np
from mayavi import mlab
from gias2.mesh import vtktools
import keyboard
from meshEditting import coordFrameFunc as frames 

def pickPointsToDefineAxes(segDir , verify = False):

    # function triggerd when key is pressed 
    def logpts(p1,p2):
        appptlog(ptlog, cp)

    # function to append point log
    def appptlog(ptlog, pt):
        #global ptlog
        ptlog.append(pt)
        print('Selected pt number ' + (str(len(ptlog))))
  
    # function for when mouse is clicked
    def picker_callback(picker_obj):
        global ptlog
        global cp
        picked = picker_obj.actors
        if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
            cp = picker_obj.point_id
            ptPlot.mlab_source.set(x= seg._points[cp][0] , y=seg._points[cp][1], z=seg._points[cp][2])
            

    # make it global so it can be passed through to the 
    global ptlog
    global cp
    ptlog=list()
    cp=0
    
    
    # laod the geometry
    seg = vtktools.Reader()
    seg.setFilename(segDir)
    seg.read()
    

    # Create figure
    fig = mlab.figure(1 , size = (1000,1000))
    # when a key is pressed it will select the chosen point - then add it to the list
    fig.scene.interactor.add_observer('KeyPressEvent', logpts)
    #from mayavi.api import Engine
    #engine = Engine()
    #engine.start()
    #if len(engine.scenes) == 0:
    #    engine.new_scene()
    #scene = engine.scenes[0]

    # graph the mesh
    #col = tuple(np.random.rand(3))
    col = tuple((0.420129941640336, 0.7875126970512139, 0.24999310026295096))
    # add a local transformation to ensure it segmentation is in a good position
    seg._points = frames.arbitraryLocalTransform(seg._points)
    
    if verify is True:
        mesh = mlab.triangular_mesh(seg._points[:,0] , seg._points[:,1] , seg._points[:,2] , seg._triangles, color = col)
    else:
        mesh = mlab.triangular_mesh(seg._points[:,0] , seg._points[:,1] , seg._points[:,2] , seg._triangles)
        
    ptPlot = mlab.points3d(0,0,0, color = (0,0,0) , mode = 'axes' , scale_factor = 25)
    # add title
    mlab.title('Choose the following points in order \n xPositive, xNegative, yPositive, yNegative, zPositive, zNegative \n where possible use "X Y Z views" on the toolbar \n if youre happy with the point, press Enter' ,  size = 250)
    # add function on mouse click from figure
    fig.on_mouse_pick(picker_callback)
    mlab.show()

    # use ptLog to convert index to names
    ii = range(0, len(ptlog))
    f = ('xPos',  'xNeg', 'yPos' ,'yNeg' ,'zPos' ,'zNeg')
    lndmks = dict()
    for i in ii:
        lndmks[f[i]] = seg._points[ptlog[i],:]

    # This will re-graph the points and geom
    if verify is True:
        mesh = mlab.triangular_mesh(seg._points[:,0] , seg._points[:,1] , seg._points[:,2] , seg._triangles, color = col, opacity = 0.5)

        for p in lndmks.keys():
            mlab.text3d(lndmks[p][0], lndmks[p][1],lndmks[p][2], p)

        mlab.show()
    # return the points 
    return lndmks