# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:26:48 2024

@author: qwerty
"""

import os
os.add_dll_directory(r'C:\opensim-core\bin')
import pyvista as pv
import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'C:\Users\qwerty\Documents\Annagh\Python')
import utils_bis as utb
import ssm
import buildACS

#Segmentation file from which to export segments
mesh_dir=r'C:\Users\qwerty\Documents\Annagh\PFI\15\meshes'
degree = '5deg'

#%% Get num pts in lenhart model
dir_smith = r'C:\Users\qwerty\Documents\Annagh\Python\jam-resources-python\models\knee_healthy\smith2019\Geometry'

n_pts_smith = np.zeros(12)
i = 0
for surf in ['bone','cartilage']:
    for b in ['femur','tibia','patella']:
        m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-'+b+'-'+surf+'.stl'))
        n_pts_smith[i] = m.n_points
        i = i+1
m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-lateral-meniscus.stl'))        
n_pts_smith[6] = m.n_points
m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-medial-meniscus.stl'))        
n_pts_smith[7] = m.n_points
m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-lateral-meniscus-inferior.stl'))        
n_pts_smith[8] = m.n_points
m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-lateral-meniscus-superior.stl'))        
n_pts_smith[9] = m.n_points
m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-medial-meniscus-inferior.stl'))        
n_pts_smith[10] = m.n_points
m = pv.PolyData(os.path.join(dir_smith,'smith2019-R-medial-meniscus-superior.stl'))        
n_pts_smith[11] = m.n_points
#%% Remesh
# Remeshing the meshes to have fewer triangles, because they are super tiny in these meshes
# Use the number of nodes in lenhart model as the target



body_names = ['Femur','Tibia','Patella','Femur Cartilage','Tibia Cartilage','Patella Cartilage', 'Lat Meniscus', 'Med Meniscus', 'Lat Meniscus Inf', 'Lat Meniscus Sup', 'Med Meniscus Inf','Med Meniscus Sup']


bodies = []
for b in range(0,len(body_names)):
    meshlist = []
    fname = body_names[b]+'.stl'
    m = pv.PolyData(os.path.join(mesh_dir,fname))
    if b < 3:
        n_target = n_pts_smith[b]
    elif b == 6:
        n_target = n_pts_smith[b]
    elif b == 7:
        n_target = n_pts_smith[b]
    else:
            # this is the full mesh rather than just one surface, if just using surface remove *2.1
            #n_target = int(n_pts_lenhart[b] * 2.1) 
        n_target = int(n_pts_smith[b])
    m = utb.ggremesh(m,opts={'nb_pts':n_target}) # remesh
    #     if info.loc[info.MTR_ID==pID].Knee.iloc[0] == 'L': # flip if left knee (for shape model only)
    #         m = m.reflect((1,0,0),inplace=False)
    #         m.flip_normals()
    if b == 9:
        m.flip_normals()
    if b == 11:
        m.flip_normals()
    meshlist.append(m)
    m.save(os.path.join(mesh_dir,degree,'remeshed',fname)) # save remeshed geometry
    meshes = ssm.meshSet(meshlist)
    bodies.append(meshes)

#%% Save and switch ACS to new folder
Trot = utb.rotmat(5,'y',deg='deg')
TF = np.loadtxt(os.path.join(mesh_dir,'original','remeshed','Femur_ACS.txt'))
TP = np.loadtxt(os.path.join(mesh_dir,'original','remeshed','Patella_ACS.txt'))
TT = np.loadtxt(os.path.join(mesh_dir,'original','remeshed','Tibia_ACS.txt'))
utb.plotpatch([],cs_list=[TT, np.matmul(TT, Trot)])
TT = np.matmul(TT,Trot)
np.savetxt(os.path.join(mesh_dir,degree,'remeshed','Femur_ACS.txt'),TF)
np.savetxt(os.path.join(mesh_dir,degree,'remeshed','Patella_ACS.txt'),TP)
np.savetxt(os.path.join(mesh_dir,degree,'remeshed','Tibia_ACS.txt'),TT)

#%% Load meshes from saved files
bodies = []
hw_ratio = []
# meshlist = []
# ACS_list = []
# hw_list = []
for b in range(3): #len(body_names)):
    meshlist = []
    ACS_list = []
    hw_list = []
    
    fname = body_names[b]+'.stl'
    m = pv.PolyData(os.path.join(mesh_dir,degree,'remeshed',fname))
    T = np.loadtxt(os.path.join(mesh_dir,degree,'remeshed',body_names[b]+'_ACS.txt'))
    m = m.transform(np.linalg.inv(T))
    ACS_list.append(T)
    meshlist.append(m)
    hw_list.append(np.ptp(m.points[:,2])/np.ptp(m.points[:,0]))
    hw_ratio.append(hw_list)
    #meshes = ssm.meshSet(meshlist,ACSs=ACS_list)
    meshes = m
    bodies.append(meshes)
for b in range(3,6): #len(body_names)):
    meshlist = []
    ACS_list = []
    hw_list = []
    
    fname = body_names[b]+'.stl'
    m = pv.PolyData(os.path.join(mesh_dir,degree,'remeshed',fname))
    T = np.loadtxt(os.path.join(mesh_dir,degree,'remeshed',body_names[b-3]+'_ACS.txt'))
    m = m.transform(np.linalg.inv(T))
    ACS_list.append(T)
    meshlist.append(m)
    hw_list.append(np.ptp(m.points[:,2])/np.ptp(m.points[:,0]))
    hw_ratio.append(hw_list)
    #meshes = ssm.meshSet(meshlist,ACSs=ACS_list)
    meshes = m
    m.save(os.path.join(mesh_dir,degree,'remeshed',fname))
        #utb.plotpatch(bodies[b].meshes,opts={'opacity': [.5]*len(bodies[0].meshes)})
for b in range(6,12): #len(body_names)):
    meshlist = []
    ACS_list = []
    hw_list = []
    
    fname = body_names[b]+'.stl'
    m = pv.PolyData(os.path.join(mesh_dir,degree,'remeshed',fname))
    T = np.loadtxt(os.path.join(mesh_dir,degree,'remeshed','Tibia_ACS.txt'))
    m = m.transform(np.linalg.inv(T))
    ACS_list.append(T)
    meshlist.append(m)
    hw_list.append(np.ptp(m.points[:,2])/np.ptp(m.points[:,0]))
    hw_ratio.append(hw_list)
    meshes = m
    bodies.append(meshes)

    #meshes = ssm.meshSet(meshlist,ACSs=ACS_list)
    meshes = m
    m.save(os.path.join(mesh_dir,degree,'remeshed',fname))
        #utb.plotpatch(bodies[b].meshes,opts={'opacity': [.5]*len(bodies[0].meshes)})
meshes_aligned = []
meshes_aligned.append(bodies[0])
meshes_aligned.append(bodies[1])
meshes_aligned.append(bodies[2])
meshes_aligned.append(bodies[3])
meshes_aligned.append(bodies[4])

# i = 0
# for m in range(len(bodies[i].meshes)):
#         #utb.plotpatch(m,cs_list=ACS_list[i])
#     #m = m.transform(np.linalg.inv(ACS_list[i]))
#     meshes_aligned.append(m)
#     i=i+1
utb.plotpatch(meshes_aligned,opts={'opacity': [0.5,0.5,0.5,0.5,0.5], 'color': ['blue','red','green','yellow','pink']})
#%%
meshes = []
m = pv.PolyData(os.path.join(mesh_dir,'original','remeshed/Tibia.stl')) # femur bone
meshes.append(m)
m = pv.PolyData(os.path.join(mesh_dir,degree,'remeshed/Tibia.stl')) # femur bone
meshes.append(m)
T_tibia = np.loadtxt(os.path.join(mesh_dir,'original','remeshed','Tibia_ACS.txt')) 
T_tibiarot = np.loadtxt(os.path.join(mesh_dir,degree,'remeshed','Tibia_ACS.txt'))
cs_list=[]
cs_list.append(T_tibia)
cs_list.append(T_tibiarot)
utb.plotpatch(meshes,cs_list,opts={'opacity':[.7,.7]})
#%% First CPD-Femur
import gbcpd
dir_corresp = r'C:\Users\qwerty\Documents\Annagh\PFI\15\meshes\5deg\corresp'
hw_ratio = np.zeros(len(meshes_aligned))

i = 0

hw_ratio[i] = np.ptp(meshes_aligned[i].points[:,2])/np.ptp(meshes_aligned[i].points[:,0])
# Get initial reference mesh
#target_hw_ratio = np.median(hw_ratio)
#Iref = np.argmin(np.sqrt((hw_ratio-target_hw_ratio)**2))        


dir_ACLC = r'C:/Users/qwerty/Documents/Annagh/Python/JAM-data'
ACLC = pv.PolyData(os.path.join(dir_ACLC,'ACLC_mean_Femur.ply'))
ref = ACLC
target_hw_ratio = np.ptp(ref.points[:,2])/np.ptp(ref.points[:,0])
# target_h = np.ptp(ref.points[:,0]) * target_hw_ratio
# origin = np.array([0,0,target_h+ref.points[:,2].min()])
#ref = ref.clip(normal='z',origin=origin)
utb.plotpatch(ref)
# Or look at them to see which are nice
#for i in range(len(meshes_aligned)):
# if hw_ratio[i] > target_hw_ratio:
#     utb.plotpatch(meshes_aligned[i])
#     m = meshes_aligned[i].copy()
                     #opts={'title':'%d : '%i + IDs})
        



# Crop all - use manually extruded meshes
meshes_hw = []
hw_ratio_new = np.zeros(len(meshes_aligned))
#for i in range(len(meshes_aligned)):

    # else:
    #     m = meshes_extrude_aligned[i].copy()
# target_h = np.ptp(m.points[:,0]) * target_hw_ratio
# origin = np.array([0,0,target_h+m.points[:,2].min()])
# m = m.clip(normal='z',origin=origin)
# m = m.fill_holes(100)
# m = utb.ggremesh(m)
# meshes_hw.append(m)
# hw_ratio_new[i] = np.ptp(m.points[:,2])/np.ptp(m.points[:,0])
# utb.plotpatch(m)

# # Crop/extend meshes to same h/w ratio
# meshes_hw = []
# hw_ratio_new = np.zeros(len(meshes_aligned))
# for i in range(len(meshes_aligned)):
if hw_ratio[i] > target_hw_ratio:
        target_h = np.ptp(meshes_aligned[i].points[:,0]) * target_hw_ratio
        # clip_z = target_h+meshes_aligned[i].points[:,2].min()
        origin = np.array([0,0,target_h+meshes_aligned[i].points[:,2].min()])
        m = meshes_aligned[i].clip(normal='z',origin=origin)
        m = m.fill_holes(100)
else:
        m = meshes_aligned[i].copy()
        d_add = target_hw_ratio*np.ptp(m.points[:,0]) - np.ptp(m.points[:,2])
        Itop = (m.point_normals[:,2]>.8) & (m.points[:,2] > m.points[:,2].max()-10)
        m.points[Itop,2] = m.points[Itop,2].max() + d_add
m = utb.ggremesh(m)
meshes_hw.append(m)
hw_ratio_new[i] = np.ptp(m.points[:,2])/np.ptp(m.points[:,0])
utb.plotpatch(m)
    

meshes_corresp_f = gbcpd.run_gbcpd(meshes_hw,ref,dir_corresp,bodyname='Femur',labels='C')
gbcpd_results_F = gbcpd.check_gbcpd_results(meshes_hw,meshes_corresp_f,labels=None,show_plot=True,rmse_method='closest_point')

##CPD Tibia
i = 1
meshes_hw = []
hw_ratio[i] = np.ptp(meshes_aligned[i].points[:,2])/np.ptp(meshes_aligned[i].points[:,0])
       


dir_ACLC = r'C:/Users/qwerty/Documents/Annagh/Python/JAM-data'
ACLC = pv.PolyData(os.path.join(dir_ACLC,'ACLC_mean_Tibia.ply'))
ref = ACLC
target_hw_ratio = np.ptp(ref.points[:,2])/np.ptp(ref.points[:,0])
# target_h = np.ptp(ref.points[:,0]) * target_hw_ratio
# origin = np.array([0,0,target_h+ref.points[:,2].min()])
#ref = ref.clip(normal='z',origin=origin)
utb.plotpatch(ref)

if hw_ratio[i] > target_hw_ratio:
        target_h = np.ptp(meshes_aligned[i].points[:,0]) * target_hw_ratio
        # clip_z = target_h+meshes_aligned[i].points[:,2].min()
        origin = np.array([0,0,target_h+meshes_aligned[i].points[:,2].min()])
        m = meshes_aligned[i].clip(normal='z',origin=origin)
        m = m.fill_holes(100)
else:
        m = meshes_aligned[i].copy()
        d_add = target_hw_ratio*np.ptp(m.points[:,0]) - np.ptp(m.points[:,2])
        Itop = (m.point_normals[:,2]<.8) & (m.points[:,2] < m.points[:,2].min()+10)
        m.points[Itop,2] = m.points[Itop,2].min() - d_add
m = utb.ggremesh(m)
meshes_hw.append(m)
hw_ratio_new[i] = np.ptp(m.points[:,2])/np.ptp(m.points[:,0])
utb.plotpatch(m)
    

meshes_corresp_t = gbcpd.run_gbcpd(meshes_hw,ref,dir_corresp,bodyname='Tibia',labels='C')
gbcpd_results_T = gbcpd.check_gbcpd_results(meshes_hw,meshes_corresp_t,labels=None,show_plot=True,rmse_method='closest_point')

##CPD Patella
i = 2
meshes_hw = []
hw_ratio[i] = np.ptp(meshes_aligned[i].points[:,2])/np.ptp(meshes_aligned[i].points[:,0])
       


dir_ACLC = r'C:/Users/qwerty/Documents/Annagh/Python/JAM-data'
ACLC = pv.PolyData(os.path.join(dir_ACLC,'ACLC_mean_Patella.ply'))
ref = ACLC
target_hw_ratio = np.ptp(ref.points[:,2])/np.ptp(ref.points[:,0])
utb.plotpatch(ref)
m = meshes_aligned[i].copy()
m = utb.ggremesh(m)
meshes_hw.append(m)

utb.plotpatch(m)
    

meshes_corresp_p = gbcpd.run_gbcpd(meshes_hw,ref,dir_corresp,bodyname='Patella',labels='C')
gbcpd_results_P = gbcpd.check_gbcpd_results(meshes_hw,meshes_corresp_p,labels=None,show_plot=True,rmse_method='closest_point')

#CPD Lat Meniscus
i = 3
meshes_hw = []
hw_ratio[i] = np.ptp(meshes_aligned[i].points[:,2])/np.ptp(meshes_aligned[i].points[:,0])
dir_ACLC = r'C:/Users/qwerty/Documents/Annagh/Python/JAM-data'
ACLC = pv.PolyData(os.path.join(dir_ACLC,'lateral_meniscus.ply'))
ref = ACLC
target_hw_ratio = np.ptp(ref.points[:,2])/np.ptp(ref.points[:,0])
utb.plotpatch(ref)
m = meshes_aligned[i].copy()
m = utb.ggremesh(m)
meshes_hw.append(m)

utb.plotpatch(m)
    

meshes_corresp_lm = gbcpd.run_gbcpd(meshes_hw,ref,dir_corresp,bodyname='Lateral_Meniscus',labels='C')
gbcpd_results_lm = gbcpd.check_gbcpd_results(meshes_hw,meshes_corresp_p,labels=None,show_plot=True,rmse_method='closest_point')

#CPD Med Meniscus
i = 4
meshes_hw = []
hw_ratio[i] = np.ptp(meshes_aligned[i].points[:,2])/np.ptp(meshes_aligned[i].points[:,0])
dir_ACLC = r'C:/Users/qwerty/Documents/Annagh/Python/JAM-data'
ACLC = pv.PolyData(os.path.join(dir_ACLC,'medial_meniscus.ply'))
ref = ACLC
target_hw_ratio = np.ptp(ref.points[:,2])/np.ptp(ref.points[:,0])
utb.plotpatch(ref)
m = meshes_aligned[i].copy()
m = utb.ggremesh(m)
meshes_hw.append(m)

utb.plotpatch(m)
    

meshes_corresp_mm = gbcpd.run_gbcpd(meshes_hw,ref,dir_corresp,bodyname='Medial_Meniscus',labels='C')
gbcpd_results_mm = gbcpd.check_gbcpd_results(meshes_hw,meshes_corresp_p,labels=None,show_plot=True,rmse_method='closest_point')
