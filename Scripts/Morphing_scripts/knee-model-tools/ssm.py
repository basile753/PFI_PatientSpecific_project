# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:36:49 2023

@author: aclouthi
"""

import scipy.io
import os
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import sys
sys.path.append(r'D:\Antoine\TN10_uOttawa\codes\Seg_SSM\Seg_SSM\knee-model-tools')
import utils_bis as utb


class meshSet:
    def __init__(self,meshes=None,name=None,Tprocrustes=None,ACSs=None):
        self.meshes = meshes
        self.name = name
        self.Tprocrustes = Tprocrustes
        self.ACSs = ACSs
        self.n_meshes = len(meshes)
        self.n_points = meshes[0].points.shape[0]
        self.n_faces = meshes[0].n_faces

    def procrustes_align(self,refmesh=None,scale=False):
        if refmesh is None:
            refmesh = self.meshes[0].copy()
        
        T_list = []
        for i in range(len(self.meshes)):
            Z,R,b,c,_ = utb.procrustes(refmesh.points,self.meshes[i].points,scale=scale)
            T = np.eye(4)
            T[:3,:3] = R.T * b
            T[:3,3] = c
            T_list.append(T)
            
            self.meshes[i].points = Z.copy()
        self.Tprocrustes = T_list
    
    def get_mean(self):
        X = np.zeros((self.n_meshes,self.n_points*3),dtype=float)
        for i in range(self.n_meshes):
            X[i,:] = np.reshape(self.meshes[i].points,(1,-1))
        points = np.reshape(X.mean(axis=0),(-1,3))
        self.mean = pv.PolyData(points,self.meshes[0].faces)
            

def splitKnee(mesh,refmeshes,translations=None):   
    npts = np.zeros(len(refmeshes),dtype=int)
    for i in range(len(refmeshes)):
        npts[i] = refmeshes[i].n_points
    
    meshes = []
    pts = mesh.points.copy()
    pts = np.reshape(pts,(-1,3))
    for j in range(len(refmeshes)):
        p = pts[npts[:j].sum():npts[:j+1].sum(),:]
        if translations is not None:
            p = p + translations[j,:]
        meshes.append(pv.PolyData(p,refmeshes[j].faces))
    
    return meshes

def prep_SSM_data(bodies):
    # bodies - a list of meshSets for each body included (e.g., femur, femur cartilage, tibia)
    
    # get number of vertices and faces in each body
    npts = np.zeros(len(bodies),dtype=int)
    nfaces = np.zeros(len(bodies),dtype=int)
    for i in range(len(bodies)):
        npts[i] = bodies[i].n_points
        nfaces[i] = bodies[i].n_faces
        
    # Compile X matrix
    X = np.zeros((bodies[0].n_meshes,npts.sum()*3),dtype=float)
    for i in range(X.shape[0]):
        for j in range(len(npts)):
            X[i,npts[:j].sum()*3:npts[:j+1].sum()*3] = np.reshape(bodies[j].meshes[i].points,(1,-1))
    
    # faces for all bodies combined
    faces = bodies[0].meshes[0].faces.copy()
    for i in range(1,len(nfaces)):
        cns = np.reshape(bodies[i].meshes[0].faces.copy(),(-1,4))
        cns[:,1:] = cns[:,1:] + npts[:i].sum()         
        faces = np.append(faces,np.reshape(cns,(-1)))
        
    return X, npts, nfaces, faces

def SSM_PCA(X,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    U = pca.components_.transpose() # modes of variation
    Z = np.matmul(X - pca.mean_,U) # scores
    
    return U, Z, pca

def SSM_PLS(X,y,n_components):
    pls = PLSRegression(n_components=n_components,scale=False) # if scale is not False, the reconstruction doesn't work!
    pls.fit(X,y)
    
    return pls

def mesh_from_SSM(Xbar,shape_vector,score,faces,refmeshes,save_dir=None,model_name=None,
                  body_names=['Femur','Tibia','Patella','Femur_Cartilage','Tibia_Cartilage','Patella_Cartilage']):
    pts = Xbar + score * shape_vector
    pts = np.reshape(pts,(-1,3))
    mesh = pv.PolyData(pts,faces)
    meshes = splitKnee(mesh,refmeshes)
    if save_dir is not None:
        for j in range(0,len(meshes)):
            meshes[j].save(os.path.join(save_dir,model_name+'_'+body_names[j]+'.ply'))
    
    return meshes

def animateSSM_pv(Xbar,shape_vector,inc,faces,refmeshes,translations=None,score_label='SD',title=''):
    # Xbar is the mean of the points from all bodies, reshaped to a row vector
    # shape_vector is the loading vector used to define the shape change/
    #    for PCA, shape_vector = np.std(Z[:,pc]) * U[:,pc].transpose()
    #    for PLS, shape_vector = np.matmul(pls.y_loadings_[0],pls.x_loadings_.T)/(np.linalg.norm(pls.y_loadings_)**2)
    # score0 is the score to start at. 
    #   for PCA, the score is the number of standard deviations of the PC scores
    #   for PLS, the score is the difference in the measurement from the mean (y_i-y.mean())
    # inc is the increment to increase/decrease the score by
    # faces is a list containing the faces for each mesh in the knee
    # refmeshes is a list of the pyvista meshes for the reference meshes
    # translations is the translation to apply to each body for the animation
    # score_label is what to label the score as (PC for PCA or the measurement name for PLS)
    # title is the title of the plot
    def increase_sd(meshes,shape_vector,text,inc,score_label):
        global score
        npts = np.zeros(len(meshes),dtype=int)
        for i in range(len(meshes)):
            npts[i] = meshes[i].n_points
        for i in range(0,len(meshes)):
            pts = meshes[i].points.copy()
            pts = np.reshape(pts,-1)
            pts = pts + inc * shape_vector[npts[:i].sum()*3:npts[:i+1].sum()*3]
            meshes[i].points = np.reshape(pts,(-1,3))
            pl.update_coordinates(meshes[i].points,mesh=meshes[i])
        score = score + inc
        text.SetText(0,score_label+' = %.1f' % score)

        
        
    def decrease_sd(meshes,shape_vector,text,inc,score_label):
        global score
        npts = np.zeros(len(meshes),dtype=int)
        for i in range(len(meshes)):
            npts[i] = meshes[i].n_points
        for i in range(0,len(meshes)):
            pts = meshes[i].points.copy()
            pts = np.reshape(pts,-1)
            pts = pts - inc * shape_vector[npts[:i].sum()*3:npts[:i+1].sum()*3]
            meshes[i].points = np.reshape(pts,(-1,3))
            pl.update_coordinates(meshes[i].points,mesh=meshes[i])
        score = score - inc
        text.SetText(0,score_label+' = %.1f' % score)
    
    global score
    score = 0
    pts = Xbar + score*shape_vector
    pts = np.reshape(pts,(-1,3))
    mesh = pv.PolyData(pts,faces)
    meshes = splitKnee(mesh,refmeshes,translations)
    pl = pv.Plotter(notebook=False)
    pl.add_text('a=increase, d=decrease',position='lower_right')
    pl.add_title(title)
    text_actor = pl.add_text(score_label+' = %.1f' % score, position="lower_left")
    for i in range(3):
        pl.add_mesh(meshes[i],color='gray')
    for i in range(3,6):
        pl.add_mesh(meshes[i],color='cyan')
    pl.add_key_event('a',lambda: increase_sd(meshes,shape_vector,text_actor,inc,score_label))
    pl.add_key_event('d',lambda: decrease_sd(meshes,shape_vector,text_actor,inc,score_label))
    pl.show()

def animateSSM_paraview(Xbar,shape_vector,scores,faces,refmeshes,translations,label,ssmfld):
    # Xbar is the mean of the points from all bodies, reshaped to a row vector
    # shape_vector is the loading vector used to define the shape change/
    #    for PCA, shape_vector = np.std(Z[:,pc]) * U[:,pc].transpose()
    #    for PLS, shape_vector = np.matmul(pls.y_loadings_[0],pls.x_loadings_.T)/(np.linalg.norm(pls.y_loadings_)**2)
    # scores is the range of scores to create the animation from 
    #   for PCA, the score is the number of standard deviations of the PC scores
    #   for PLS, the score is the difference in the measurement from the mean (y_i-y.mean())
    # faces is a list containing the faces for each mesh in the knee
    # refmeshes is a list of the pyvista meshes for the reference meshes
    # translations is the translation to apply to each body for the animation
    # label is the text that will appear at the start of the filenames
    # ssmfld is the folder to write the files to
    i = 1
    for s in scores:
        pts = Xbar + s * shape_vector
        pts = np.reshape(pts,(-1,3))
        mesh = pv.PolyData(pts,faces)
        meshes = splitKnee(mesh,refmeshes,translations)
        for j in range(0,3):
            meshes[j].save(os.path.join(ssmfld,label+'_B'+str(j+1)+'_'+str(i)+'.ply'))
        for j in range(3,6):
            meshes[j].save(os.path.join(ssmfld,label+'_B'+str(j+1)+'_'+str(i)+'.ply'),
                           texture=np.tile(np.array([0,1,1],dtype=np.uint8),(meshes[j].n_points,1)))
        i = i+1