# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:32:41 2023

@author: aclouthi
"""
import os
import sys
import shutil
import numpy as np
import pyvista as pv
import pandas as pd
import xml.etree.ElementTree as ET
import json
import utils_bis as utb
opensim_path = input(r"Enter your OpenSim/bin folder path (default: D:\Programmes\OpenSim 4.2\bin): ")
if opensim_path == "":
    opensim_path = r"D:\Programmes\OpenSim 4.2\bin"
os.environ['PATH'] = opensim_path
sys.path.append(r'opensim')
import opensim as osim




def copy_model_files(dir_model,dir_ref_model,ref_model_file,ref_model_other_files,model_name):
    '''
    This function copies over the files associated with the model to a new folder
    so they can be used for a new model. The .osim file fill be copied and renamed
    and any files included in the list ref_model_other_files will also be copied.

    Parameters
    ----------
    dir_model : string
        Directory where the new model will be saved.
    dir_ref_model : string
        Directory for the reference model.
    ref_model_file : string
        Full path to the reference model.
    ref_model_other_files : list
        List of full paths to other files to copy.
    model_name : string
        Name for the output model. It will be saved as model_name.osim.

    Returns
    -------
    None.

    '''
    # --- Copy generic model to new folder --- #
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    shutil.copy2(os.path.join(dir_ref_model,ref_model_file),os.path.join(dir_model,model_name+'.osim'))
    for filename in ref_model_other_files:
        shutil.copy2(os.path.join(dir_ref_model,filename),dir_model)
    # shutil.copytree(os.path.join(dir_ref_model,'Geometry'),os.path.join(dir_model,'Geometry'),
    #                 ignore=shutil.ignore_patterns('*.vtp'))


def update_geometry(geom_files,model_name,ref_geometry_dir,ligament_info_file,muscle_info_file,
                    fitpts_file,dir_model, show_plot=False):
    '''
    The function updates the .stl file paths, ligament attachments, and wrapping surfaces
    at the knee for new knee mesh files. 

    Parameters
    ----------
    geom_files : dict
        A dictionary with keys 'bone' and 'cartilage'. Each key contains a list of 
        pyvista.PolyData meshes for the knee, tibia, and patella (in that order).
        geom_files = {'bone': [pyvista.PolyData,pyvista.PolyData,pyvista.PolyData],
                      'cartilage': [pyvista.PolyData,pyvista.PolyData,pyvista.PolyData]}
        The keys may also contain file paths to .stl or .iv files for these meshes.
    model_name : string
        The name of the model (i.e., the model file will be model_name.osim).
    ligament_info_file : string
        Full path to a .json file that contains the node IDs in the mesh and which 
        body they attach to for each ligament in the model.
    muscle_info_file : string
        Full path to a .json file that contans the node IDs in the mesh and which body 
        they attach to for each muscle that will be updated based on the new geometry.
    fitpts_file : string
        Full path to a .json file that contains the node IDs to use to create
        the wrapping surfaces that will be updated.
    dir_model : string
        Path of the directory where the model will be saved.
    show_plot : bool, optional
        Show a plot of the bone and cartilage meshes with the wrapping surfaces 
        displayed. The default is False.

    Returns
    -------
    None.

    '''
    
    
    # --- Import geometry to use --- #
    if type(geom_files['bone'][0]) == str:
        meshes = {'bone': [None]*3,'cartilage': [None]*3}
        for i in range(3):
            for surf in ['bone','cartilage']:
                if '.iv' in geom_files[surf][i]:
                   meshes[surf][i] = utb.read_iv(geom_files[surf][i])
                else:
                   meshes[surf][i] = pv.PolyData(geom_files[surf][i])
    else:
        meshes = geom_files            
        
    # --- Write geometry files to model folder --- #
    body_list = ['femur','tibia','patella']
    R_MRI_to_osim = np.array([[0,1,0],[0,0,1],[1,0,0]])
    if not os.path.exists(os.path.join(dir_model,'Geometry')):
        os.mkdir(os.path.join(dir_model,'Geometry'))
    for i in range(3):
        for surf in ['bone','cartilage']:
            meshes[surf][i].points = np.transpose(np.matmul(R_MRI_to_osim,meshes[surf][i].points.transpose()))
            meshes[surf][i].save(os.path.join(dir_model,'Geometry',model_name+'-'+body_list[i]+'-'+surf+'.stl'))
    body_list = ['lateral','medial']
    for i in range(2):
        for surf in ['meniscus']:
                meshes[surf][i].points = np.transpose(np.matmul(R_MRI_to_osim,meshes[surf][i].points.transpose()))
                meshes[surf][i].save(os.path.join(dir_model,'Geometry',model_name+'-'+body_list[i]+'-'+surf+'.stl'))
    for i in range(2):
        for surf in ['meniscus']:
                meshes[surf][i+2].points = np.transpose(np.matmul(R_MRI_to_osim,meshes[surf][i+2].points.transpose()))
                meshes[surf][i+2].save(os.path.join(dir_model,'Geometry',model_name+'-'+body_list[i]+'-'+surf+'-inferior.stl'))        
    for i in range(2):
        for surf in ['meniscus']:
                meshes[surf][i+4].points = np.transpose(np.matmul(R_MRI_to_osim,meshes[surf][i+4].points.transpose()))
                meshes[surf][i+4].save(os.path.join(dir_model,'Geometry',model_name+'-'+body_list[i]+'-'+surf+'-superior.stl'))
    # # check alignment between this femur and Smith2019 model
    reffem = pv.PolyData(os.path.join(ref_geometry_dir,'Smith2019-R-femur-bone.stl'))
    #utb.plotpatch([meshes['bone'][0],reffem],opts={'color': ['grey','red'],'opacity':[.7,.7],
    #                                               'show_edges': [False]*2,'legend_entries': [['current','grey'],['Smith','r']]})

    # -------------------------------------------------- #
    # ----- Update ligament and muscle attachments ----- #
    # -------------------------------------------------- #
    
    with open(ligament_info_file, 'r') as f:
        ligament_info = json.load(f)
    ligament_info = pd.DataFrame(ligament_info)
    with open(muscle_info_file, 'r') as f:
        muscle_info_all = json.load(f)
    muscle_info_all = pd.DataFrame(muscle_info_all)
    muscle_info = muscle_info_all[muscle_info_all['node'].map(lambda d: len(d)) > 0] # muscles to update
    
    tibiaML = meshes['bone'][1].points[:,2].max() - meshes['bone'][1].points[:,2].min()
    tibiaAP = meshes['bone'][1].points[:,0].max() - meshes['bone'][1].points[:,0].min()
    
    # --- Generate new wrap surfaces --- #
    with open(fitpts_file, 'r') as f:
        fitpts = json.load(f)
    
    wrapSurface = pd.DataFrame(data=[[None]*8]*6,columns=['body','name','type','xyz_body_rotation','translation', 
                    'radius','length','dimensions'])
    
    i_me = np.argmin(meshes['bone'][0].points[:,2])
    i_le = np.argmax(meshes['bone'][0].points[:,2])
    ME = meshes['bone'][0].points[i_me,:]
    LE = meshes['bone'][0].points[i_le,:]
    
    # KnExt_at_fem_r
    I = fitpts[13]['num']
    pts = meshes['bone'][0].points[I,:]
    h = 2 * np.abs(ME[2] - LE[2])
    x0 = pts.mean(axis=0)
    r = pts[:,0].max() - x0[0]
    wrapSurface.loc[0] = ['femur_r','KnExt_at_fem_r','WrapCylinder',np.array([0,0,0]),x0,r,h,None]
    
    # KnExt_vasint_at_fem_r
    I = fitpts[14]['num']
    pts = meshes['bone'][0].points[I,:]
    x0 = pts.mean(axis=0)
    r = pts[:,0].max() - x0[0]
    wrapSurface.loc[1] = ['femur_r','KnExt_vasint_at_fem_r','WrapCylinder',np.array([0,0,0]),x0,r,h,None]
    
    # cylinder1 = pv.Cylinder(center=wrapSurface.translation[0],direction=[0,0,1],
    #                         radius=wrapSurface.radius[0],height=wrapSurface.length[0])
    # cylinder2 = pv.Cylinder(center=wrapSurface.translation[1],direction=[0,0,1],
    #                         radius=wrapSurface.radius[1],height=wrapSurface.length[1])
    # utb.plotpatch([meshes['bone'][0],meshes['cartilage'][0],cylinder1,cylinder2],points_list=[meshes['bone'][0].points[fitpts[14]['num']]],
    #               opts={'color': ['grey','white','c','m'],'show_edges': [False]*4,
    #                     'opacity': [0.5]*4,'legend_entries': [['KnExt_at_fem_r','c'],['KnExt_vasint_at_fem_r','m']]})
    
    # Gastroc_at_Condyles_r
    I = fitpts[16]['num']
    pts = meshes['bone'][0].points[I,:]
    x0 = (ME+LE)/2
    x0[0:2] = pts[6,0:2]
    r = np.array([0.0,0.0,0.0])
    r[0] = np.abs(pts[3,0] - pts[2,0])/2
    r[1] = np.abs(pts[5,1] - pts[4,1])/2
    r[2] = 1.875*np.abs(ME[2]-LE[2])
    a0 = pts[1,:]-pts[0,:]
    a0 = a0 / np.linalg.norm(a0)
    thx = np.arctan2(a0[1],a0[2]) # rotation about x
    a01 = np.matmul(np.array([[1,0,0],[0,np.cos(thx),-np.sin(thx)],[0,np.sin(thx),np.cos(thx)]]),a0.transpose())
    thy = np.arctan2(a01[0],a01[2])
    an = np.array([thx,thy,0])
    wrapSurface.loc[2] = ['femur_r','Gastroc_at_Condyles_r','WrapEllipsoid',an,x0,None,None,r]
    
    
    # ellipse = pv.ParametricEllipsoid(r[0],r[1],r[2])
    # ellipse = ellipse.rotate_x(an[0]*180/np.pi,inplace=False)
    # ellipse = ellipse.rotate_y(an[1]*180/np.pi,inplace=False)
    # utb.plotpatch([meshes['bone'][0],ellipse], points_list=[meshes['bone'][0].points[fitpts[16]['num']]],
    #               opts={'color': ['grey','c'],'opacity': [.5,.5],
    #                     'show_edges': [False]*2,'legend_entries': [['Gastroc_at_Condyles_r','c']]})
    
    # Capsule_r
    I = fitpts[21]['num']
    pts = meshes['bone'][0].points[I,:]
    x0 = (ME+LE)/2
    a0 = (LE - ME)/np.linalg.norm(LE-ME)
    r0 = (np.linalg.norm(pts[:,0:2] - x0[0:2],axis=1)).mean()
    h = 3*np.abs(LE[2]-ME[2])
    p0 = [x0[0],x0[1],0,0,r0] # x0[0], x0[1], x rotation, y rotation, radius
    
    # an,xn,rn,error = fitting.fit(pts,guess_angles=[(np.cos(a0[1]),np.cos(a0[0]))])
    xn, an, rn, stats = utb.lscylinder(pts,x0,a0,r0)
    
    wrapSurface.loc[3] = ['femur_distal_r','Capsule_r','WrapCylinder',an,xn,rn,h,None]
    
    # cylinder = pv.Cylinder(center=wrapSurface.translation[3],direction=wrapSurface.xyz_body_rotation[3],
    #                         radius=wrapSurface.radius[3],height=wrapSurface.length[3])
    # utb.plotpatch([meshes['bone'][0],cylinder],points_list=[pts],
    #               opts={'color': ['grey','c'],'show_edges': [False]*2,'opacity': [0.5]*2,'legend_entries': [['Capsule_r','c']]})
    
    # Med_lig_r
    I = fitpts[18]['num']
    pts = meshes['bone'][1].points[I,:]
    r = np.zeros(3,dtype=float)
    r[0] = np.abs(pts[3,0]-pts[2,0]) * 0.7
    r[1] = np.abs(pts[5,1]-pts[4,1]) * 1.2
    r[2] = np.abs(pts[1,2]-pts[0,2]) * 0.75
    x0n = np.zeros(3,dtype=float)
    x0n[0] = (pts[3,0]+pts[2,0])/2
    x0n[1] = (pts[5,1]+pts[4,1])/2 - 0.1*r[1]
    x0n[2] = (pts[1,2]+pts[0,2])/2 + 0.4*r[2]
    a0 = (pts[2,:]-pts[3,:])/np.linalg.norm((pts[2,:]-pts[3,:]))
    an = np.zeros(3,dtype=float)
    an[0] = 5 * np.pi/180
    an[1] = -np.arcsin(a0[2])
    an[2] = np.arcsin(a0[1])
    wrapSurface.loc[4] = ['tibia_proximal_r','Med_Lig_r','WrapEllipsoid',an,x0n,None,None,r]
    
    # ellipse = pv.ParametricEllipsoid(r[0],r[1],r[2])
    # ellipse = ellipse.rotate_x(an[0]*180/np.pi,inplace=False)
    # ellipse = ellipse.rotate_y(an[1]*180/np.pi,inplace=False)
    # ellipse = ellipse.rotate_z(an[2]*180/np.pi,inplace=False)
    # ellipse = ellipse.translate(x0n)
    # utb.plotpatch([meshes['bone'][1],ellipse], points_list=[pts],
    #               opts={'color': ['grey','c'],'opacity': [.5,.5],'show_edges': [False]*2,'legend_entries': [['Med_lig_r','c']]})
    
    # Med_LigP_r
    I = fitpts[19]['num']
    pts = meshes['bone'][1].points[I,:]
    r = np.zeros(3,dtype=float)
    r[0] = np.linalg.norm(pts[3,:]-pts[2,:]) * 0.8
    r[2] = np.linalg.norm(pts[3,:]-pts[1,:]) * 1.4
    r[1] = r[2]
    a0 = pts[2,:]-pts[3,:]
    x0n = pts[3,:] + a0*0.4
    a0 = a0/np.linalg.norm(a0)
    an = np.zeros(3,dtype=float)
    an[0] = 0
    an[1] = -np.arcsin(a0[2])
    an[2] = np.arcsin(a0[1])
    wrapSurface.loc[5] = ['tibia_proximal_r','Med_LigP_r','WrapEllipsoid',an,x0n,None,None,r]
    
    
    # ellipse2 = pv.ParametricEllipsoid(r[0],r[1],r[2])
    # ellipse2 = ellipse2.rotate_x(an[0]*180/np.pi,inplace=False)
    # ellipse2 = ellipse2.rotate_y(an[1]*180/np.pi,inplace=False)
    # ellipse2 = ellipse2.rotate_z(an[2]*180/np.pi,inplace=False)
    # ellipse2 = ellipse2.translate(x0n)
    # utb.plotpatch([meshes['bone'][1],ellipse,ellipse2], points_list=[meshes['bone'][1].points[fitpts[18]['num'],:],pts],
    #               opts={'color': ['grey','c','m'],'opacity': [.5]*3,'show_edges': [False]*3,'point_color': ['b','r'],
    #                     'legend_entries': [['Med_LigP_r','c'],['Med_LigP_r','m']]})
    
    
    # PatTen_r
    I = fitpts[20]['num']
    pts = meshes['bone'][2].points[I,:]
    an = np.zeros(3,dtype=float)
    x0n = np.zeros(3,dtype=float)
    x0n[0] = pts[2:4,0].mean()
    x0n[1] = pts[4:6,1].mean()
    x0n[2] = pts[0:2,2].mean()
    r = np.zeros(3,dtype=float)
    r[0] = np.abs(pts[3,0]-pts[2,0])/2
    r[1] = np.abs(pts[4,1]-pts[5,1])/2
    r[2] = np.abs(pts[1,2]-pts[0,2])/2
    wrapSurface.loc[6] = ['patella_r','PatTen_r','WrapEllipsoid',an,x0n,None,None,r]
    
    # ellipse = pv.ParametricEllipsoid(r[0],r[1],r[2])
    # ellipse = ellipse.translate(x0n)
    # utb.plotpatch([meshes['bone'][2],ellipse], points_list=[pts],
    #               opts={'color': ['grey','c'],'opacity': [.5,.5],'show_edges': [False]*2,'legend_entries': [['PatTen_r','c']]})
    
    
    if show_plot==True:
        pv.set_plot_theme('document')
        pl = pv.Plotter(notebook=False)
        pl.disable_anti_aliasing()
        actor = []
        for i in range(2):
            actor = pl.add_mesh(meshes['bone'][i],color=np.array([.7,.7,.7]),opacity=.5)
        m = meshes['bone'][2].copy()
        m.points[:,0] = m.points[:,0] + 0.055
        actor = pl.add_mesh(m,color=np.array([.7,.7,.7]),opacity=.5)
        for i in range(wrapSurface.shape[0]):
            if wrapSurface.type[i] == 'WrapCylinder':
                if wrapSurface.xyz_body_rotation[i].sum() == 0:
                    direction = np.array([0,0,1])
                else: 
                    direction = wrapSurface.xyz_body_rotation[i]
                cylinder = pv.Cylinder(center=wrapSurface.translation[i],direction=direction,
                                        radius=wrapSurface.radius[i],height=wrapSurface.length[i])
                actor = pl.add_mesh(cylinder,color='cyan',opacity=.5)
            if wrapSurface.type[i] == 'WrapEllipsoid':
                ellipse = pv.ParametricEllipsoid(wrapSurface.dimensions[i][0],wrapSurface.dimensions[i][1],wrapSurface.dimensions[i][2])
                ellipse = ellipse.rotate_x(wrapSurface.xyz_body_rotation[i][0]*180/np.pi,inplace=False)
                ellipse = ellipse.rotate_y(wrapSurface.xyz_body_rotation[i][1]*180/np.pi,inplace=False)
                ellipse = ellipse.rotate_z(wrapSurface.xyz_body_rotation[i][2]*180/np.pi,inplace=False)
                if wrapSurface.body[i] == 'patella_r':
                    ellipse = ellipse.translate(wrapSurface.translation[i]+np.array([0.055,0,0]),inplace=False)
                else:
                    ellipse = ellipse.translate(wrapSurface.translation[i],inplace=False)
                actor = pl.add_mesh(ellipse,color='c',opacity=.5)
        pl.show_grid()
        pl.show_axes()
        pl.show()
        pl.close()
    
    
    # -- Update .osim file -- #
    
    parser = ET.XMLParser(target=ET.TreeBuilder())
    tree = ET.parse(os.path.join(dir_model,model_name+'.osim'),parser)
    root = tree.getroot()[0]
    
    root.attrib['name'] = model_name
    
    # update body set geometry (used for visualization only)
    BodySet = root.find('BodySet')[0]
    BodySet.findall("./Body[@name='femur_distal_r']/attached_geometry/Mesh[@name='femur_bone']/mesh_file")[0].text = \
        model_name+'-femur-bone.stl'
    BodySet.findall("./Body[@name='femur_distal_r']/attached_geometry/Mesh[@name='femur_cartilage']/mesh_file")[0].text = \
        model_name+'-femur-cartilage.stl'
    BodySet.findall("./Body[@name='tibia_proximal_r']/attached_geometry/Mesh[@name='tibia_bone']/mesh_file")[0].text = \
        model_name+'-tibia-bone.stl'
    BodySet.findall("./Body[@name='tibia_proximal_r']/attached_geometry/Mesh[@name='tibia_cartilage']/mesh_file")[0].text = \
        model_name+'-tibia-cartilage.stl'
    BodySet.findall("./Body[@name='patella_r']/attached_geometry/Mesh[@name='patella_bone']/mesh_file")[0].text = \
        model_name+'-patella-bone.stl'
    BodySet.findall("./Body[@name='patella_r']/attached_geometry/Mesh[@name='patella_cartilage']/mesh_file")[0].text = \
        model_name+'-patella-cartilage.stl'
    BodySet.findall("./Body[@name='meniscus_lateral_r']/attached_geometry/Mesh[@name='meniscus_lateral_r']/mesh_file")[0].text = \
         model_name+'-lateral-meniscus.stl'
    BodySet.findall("./Body[@name='meniscus_medial_r']/attached_geometry/Mesh[@name='meniscus_medial_r']/mesh_file")[0].text = \
         model_name+'-medial-meniscus.stl'        
    # update contact geometry (used in simulation)
    ContactSet = root.find('ContactGeometrySet')[0]
    ContactSet.findall("./Smith2018ContactMesh[@name='femur_cartilage']/mesh_file")[0].text = \
        model_name+'-femur-cartilage.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='femur_cartilage']/mesh_back_file")[0].text = \
        model_name+'-femur-bone.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='tibia_cartilage']/mesh_file")[0].text = \
        model_name+'-tibia-cartilage.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='tibia_cartilage']/mesh_back_file")[0].text = \
        model_name+'-tibia-bone.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='patella_cartilage']/mesh_file")[0].text = \
        model_name+'-patella-cartilage.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='patella_cartilage']/mesh_back_file")[0].text = \
        model_name+'-patella-bone.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='meniscus_med_sup']/mesh_file")[0].text = \
        model_name+'-medial-meniscus-superior.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='meniscus_med_inf']/mesh_file")[0].text = \
        model_name+'-medial-meniscus-inferior.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='meniscus_lat_sup']/mesh_file")[0].text = \
        model_name+'-lateral-meniscus-superior.stl'
    ContactSet.findall("./Smith2018ContactMesh[@name='meniscus_lat_inf']/mesh_file")[0].text = \
        model_name+'-lateral-meniscus-inferior.stl'        
    # update wrapping surfaces
    # get offset for femur and tibia and add to wrap surface translations
    JointSet = root.find('JointSet')[0]
    femur_r_offset = np.fromstring(
        JointSet.findall("./WeldJoint[@name='femur_femur_distal_r']/frames/PhysicalOffsetFrame/translation")[0].text,
        dtype=float,sep=' ')
    tibia_r_offset = np.fromstring(
        JointSet.findall("./WeldJoint[@name='tibia_tibia_proximal_r']/frames/PhysicalOffsetFrame/translation")[0].text,
        dtype=float,sep=' ')
    for i in range(wrapSurface.shape[0]):
        if wrapSurface.body[i] == 'femur_r':
            wrapSurface.translation[i] = wrapSurface.translation[i] + femur_r_offset
        if wrapSurface.body[i] == 'tibia_r':
            wrapSurface.translation[i] = wrapSurface.translation[i] + tibia_r_offset
        BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/xyz_body_rotation" % \
                        (wrapSurface.body[i],wrapSurface.type[i],wrapSurface.name[i]))[0].text = \
                        ' '.join(map(str,wrapSurface.xyz_body_rotation[i]))
        BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/translation" % \
                        (wrapSurface.body[i],wrapSurface.type[i],wrapSurface.name[i]))[0].text = \
                        ' '.join(map(str,wrapSurface.translation[i]))
        if wrapSurface.type[i] == 'WrapCylinder':
            BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/radius" % \
                            (wrapSurface.body[i],wrapSurface.type[i],wrapSurface.name[i]))[0].text = \
                            str(wrapSurface.radius[i])
            BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/length" % \
                            (wrapSurface.body[i],wrapSurface.type[i],wrapSurface.name[i]))[0].text = \
                            str(wrapSurface.length[i])
        if wrapSurface.type[i] == 'WrapEllipsoid':
            BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/dimensions" % \
                            (wrapSurface.body[i],wrapSurface.type[i],wrapSurface.name[i]))[0].text = \
                            ' '.join(map(str,wrapSurface.dimensions[i]))
    
    # update muscle attachments
    ForceSet = root.find('ForceSet')[0]
    for i in muscle_info.index:
        for j in range(len(muscle_info.node[i])):
            if muscle_info.node[i][j] is not None:
                if muscle_info.segment[i][j] == 'femur_r':
                    p = meshes['bone'][0].points[muscle_info.node[i][j],:]
                    p = p + femur_r_offset
                elif muscle_info.segment[i][j] == 'tibia_r':
                    p = meshes['bone'][1].points[muscle_info.node[i][j],:]
                    p = p + tibia_r_offset
                elif muscle_info.segment[i][j] == 'patella_r':
                    p = meshes['bone'][2].points[muscle_info.node[i][j],:]
                    
                ForceSet.findall("./Millard2012EquilibriumMuscle[@name='%s']/GeometryPath/" %  (muscle_info.name[i]) + \
                                 "PathPointSet/objects/PathPoint[@name='%s-P%d']/location" % \
                                 (muscle_info.name[i],j+1))[0].text = ' '.join(map(str,p))
    
    # update ligament attachments
    for i in ligament_info.index:
        for j in range(len(ligament_info.node[i])):
            if ligament_info.node[i][j] is not None:
                if ligament_info.segment[i][j] == 'femur_r':
                    p = meshes['bone'][0].points[ligament_info.node[i][j],:]
                    p = p + femur_r_offset
                elif ligament_info.segment[i][j] == 'tibia_r':
                    p = meshes['bone'][1].points[ligament_info.node[i][j],:]
                    p = p - tibia_r_offset
                elif ligament_info.segment[i][j] == 'patella_r':
                    p = meshes['bone'][2].points[ligament_info.node[i][j],:]
                elif ligament_info.segment[i][j] == 'femur_distal_r':
                    p = meshes['bone'][0].points[ligament_info.node[i][j],:]
                elif ligament_info.segment[i][j] == 'tibia_proximal_r':
                    p = meshes['bone'][1].points[ligament_info.node[i][j],:]
                    # points that are attached to fibula are shifted out, since SSM has no fibula
                    if ligament_info['shift'][i]:
                        p = p + np.multiply(ligament_info['shift'][i][j],np.array([tibiaAP,0,tibiaML]))
                elif ligament_info.segment[i][j] == 'meniscus_medial_r':
                    p = meshes['meniscus'][1].points[ligament_info.node[i][j],:]
                elif ligament_info.segment[i][j] == 'meniscus_lateral_r':
                    p = meshes['meniscus'][0].points[ligament_info.node[i][j],:]
                print(i)
                ForceSet.findall("./Blankevoort1991Ligament[@name='%s']/GeometryPath/" %  (ligament_info.name[i]) + \
                                 "PathPointSet/objects/PathPoint[@name='%s-P%d']/location" % \
                                 (ligament_info.name[i],j+1))[0].text = ' '.join(map(str,p))
            
      
    tree.write(os.path.join(dir_model,model_name+'.osim'),encoding='utf8',method='xml')




def update_ligament_slack_lengths(model_name,ligament_info_file,dir_model,settle_sto_file,
                                  model_name_out=None,sec_coords=None):
    '''
    This function updates the slack lengths in a model based on pre-defined 
    reference strains and a reference pose resulting from a settle simulation.

    Parameters
    ----------
    model_name : string
        Name of model. i.e., the model file is model_name.osim
    ligament_info_file : string
        Full path to a .json file that contains the reference strains for each ligament in the model.
    dir_model : string
        Path for the folder containing the model.
    settle_sto_file : string
        Full path to the .sto file of a settle simulation that will be used as the reference
        pose for calculating ligament strains. The last row of the .sto file will be used.
    model_name_out : string, optional
        Name to use for the output model with updated ligament slack lengths. The default is None.
        If None, it will be model_name + '_slack_len_updated'
    sec_coords : list, optional
        List of secondary coordinates in the model. The default is None. If None,
        the secondary coordinates for a right knee in the lenhart2015 model will be used.

    Returns
    -------
    None.

    '''
    
    # import results of settle simulation to use as reference pose
    settle_sto = osim.TimeSeriesTable(settle_sto_file)
    sto_data = settle_sto.getMatrix().to_numpy()
    sto_cols = settle_sto.getColumnLabels()
    
    # import model and initialize 
    if '.osim' in model_name:
        model_name = model_name.replace('.osim','')
    model = osim.Model(os.path.join(dir_model,model_name+'.osim'))
    state = model.initSystem()
    
    # Update the state with the coordinates from the last row of the settle sim
    # Also update default values for secondary coordinates
    if sec_coords == None:
        sec_coords = ['knee_add_r','knee_rot_r','knee_tx_r','knee_ty_r','knee_tz_r',
                      'pf_flex_r','pf_rot_r','pf_tilt_r','pf_tx_r','pf_ty_r','pf_tz_r']
    coords = model.getCoordinateSet()
    for i in range(coords.getSize()):   
        coord = coords.get(i)
        name = coord.getName()
        if name in sec_coords:
            i_sto = [idx for idx, s in enumerate(sto_cols) if name+'/value' in s]
            if len(i_sto) > 0:
                coord.setValue(state,sto_data[-1,i_sto][0])
                coord.setDefaultValue(sto_data[-1,i_sto][0])
            else:
                print('Can\'t find %s in .sto file' % name)
            
    # import ligament info file
    with open(ligament_info_file, 'r') as f:
        ligament_info = json.load(f)
    ligament_info = pd.DataFrame(ligament_info)

    # update ligament slack lengths based on the reference strain and the reference pose
    forceset = model.getForceSet()
    for i in range(forceset.getSize()):
        force = forceset.get(i)
        if force.getConcreteClassName() == 'Blankevoort1991Ligament':
            
            lig = osim.Blankevoort1991Ligament.safeDownCast(force)
            name = lig.getName()
            # length = lig.getLength(state)
            lig.setSlackLengthFromReferenceStrain(ligament_info[ligament_info.name == name].ReferenceStrain.values[0],state)
    
    if model_name_out == None:
        model_name_out = model_name + '_slack_len_updated'
    
    model.setName(model_name_out)
    
    model.printToXML(os.path.join(dir_model,model_name_out+'.osim'))
    
