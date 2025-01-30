# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:43:54 2023

@author: aclouthi
"""


import os
import numpy as np
import pyvista as pv
import subprocess
import time
import sys
sys.path.append(r'\knee-model-tools')
import utils as ut



def run_gbcpd(meshlist,refmesh,dir_ouput,bodyname='mesh',labels=None,outsuffix='_corresp',
              dir_bcpd = r'C:\Users\qwerty\Documents\Annagh\Python\bcpd-master\win',
              omg=0.0,bet=0.7,lmd=100,gma=1,K=100,J=300,c=1e-6,n=500,nrm='x',
              dwn='B,5000,0.02',tau=0.5,kern_type='geodesic'):
    
    '''
    This runs GBCPD which can be downloaded from https://github.com/ohirose/bcpd.
    
    It runs Geodesic-Based Bayesian Coherent Point Drift to deform a reference mesh
    to match each mesh in a list of meshes so that there is node correspondence among the
    set of meshes. The meshes with corresponding nodes are written as .ply files.
    
    Citation: O. Hirose, "Geodesic-Based Bayesian Coherent Point Drift," IEEE TPAMI, Oct 2022.
    
    The default parameters are taken from the FACE01 GBCPD example provided in the 
    source code Matlab examples. 
    
    Parameters
    ----------
    meshlist : list of pyvista.PolyData
        List of meshes to obtain correspondence for.
    refmesh : pyvista.PolyData
        The source/reference mesh. This will be deformed to match each of the meshes in meshlist.
    dir_ouput : string
        Path to a directory where the resulting meshes will be written.
    bodyname : string, optional
        Name of the body of interest (e.g., 'Femur','Tibia'). The default is 'mesh'.
    labels : list of strings, optional
        labels for each mesh in the mesh list. E.g., participant IDs. These will be
        used to name the output meshes. If None, it will be mesh001, mesh002, etc. 
        The default is None.
    outsuffix : string, optional
        Suffix to append to file name for saved corresponding meshes. 
        The default is '_corresp'.
    dir_bcpd : string, optional
        Directory containing bcpd.exe. 
    omg : float, optional
        Omega. Outlier probability in (0,1). The larger the more robust agains outliers,
        but the less sensitive to the data points. Range [0.0,0.3]. The default is 0.0.
    bet : float, optional
        Beta. Positive. It controls the range where deformation vectors are smoothed.
        The larger, the smoother. Range [0.1,2.5]. The default is 0.7.
    lmd : float, optional
        Lambda. Positive. It controls the expected length of deformation vectors. 
        Smaller is longer. Range [1,5000]. The default is 100.
    gma : float, optional
        Gamma. Positive. It defines the randomness of the point matching at the 
        beginning of the optimization. How much the initial alignment is considered.
        The smaller, the more considered. Range [0.1,3.0]. The default is 1.
    K : int, optional
        Used in Nystrom method. #Nystrom samples for computing the coherence matrix G_YY.
        The smaller, the faster. Range [70,300]. The default is 100.
    J : int, optional
        Used in Nystrom method. #Nystrom samples for computing mjatching probabilities, P. 
        The smaller, the faster. Range [300,600]. The default is 300.
    c : float, optional
        Convergence tolerance. The default is 1e-6.
    n : int, optional
        The maximum number of VB loops. The default is 500.
    nrm : char, optional
        Chooses a normalization option by specifying the argument of the option, e.g., -ux.
            e: Each of X and Y is normalized separately (default).
            x: X and Y are normalized using the location and the scale of X.
            y: X and Y are normalized using the location and the scale of Y.
            n : Normalization is skipped (not recommended).
            The default is 'x'.
    dwn : string, optional
        Downsampling. Changes the number of points. E.g., -D'B,10000,0.08'.
            1st argument: One of the symbols: [X,Y,B,x,y,b]; x: target; y: source; b: both, upper: voxel, lower: ball.
            2nd argument: The number of points to be extracted by the downsampling.
            3rd argument: The voxel size or ball radius required for downsampling. 
            The default is 'B,5000,0.02'. Not currently used.
    tau : float, optional
        Tau. The rate controlling the balance between geodesic and Gaussian kernels. 
        Range [0.0,1.0]. The default is 0.5.
    kern_type : string, optional
        Kernel type. Use '1','2', or '3', for the standard kernels. Use 'geodesic' for
        the geodesic kernel. '1' if the default standard kernel. The default is 'geodesic'.

    Returns
    -------
    meshes_corresp : list of pyvista.PolyData
        List of resulting meshes.

    '''
    t0 = time.time()

    if labels == None:
        labels = []
        for i in range(meshlist):
            labels.append('mesh%03d' % i)

    np.savetxt(os.path.join(dir_ouput,bodyname+'_ref_p.txt'),refmesh.points,delimiter='\t')
    np.savetxt(os.path.join(dir_ouput,bodyname+'_ref_f.txt'),np.reshape(refmesh.faces,(-1,4))[:,1:],delimiter='\t')
    
    if kern_type == 'geodesic':
        kern = 'geodesic,%s,%s' % (tau,os.path.join(dir_ouput,bodyname+'_ref_f.txt'))
    else:
        kern = kern_type # should be '1','2',or '3' if not geodesic
    
    meshes_corresp = []
    for i in range(len(meshlist)):
        np.savetxt(os.path.join(dir_ouput,bodyname+'%02d_p.txt' % i),meshlist[i].points,delimiter='\t')

        cmd_str = '\"' + os.path.join(dir_bcpd,'bcpd.exe') + '\" -x \"' + \
                os.path.join(dir_ouput,bodyname+'%02d_p.txt' % i) + '\" ' + \
                '-y \"' + os.path.join(dir_ouput,bodyname+'_ref_p.txt') + '\" ' 
        cmd_str = cmd_str + '-w%s -b%s -l%s -g%s ' % (omg,bet,lmd,gma)
        cmd_str = cmd_str + '-J%s -K%s -p -u%s ' % (J,K,nrm)
        cmd_str = cmd_str + '-c%s -n%s -h -r1 -ux -sy ' % (c,n)
        if kern != None:
            cmd_str = cmd_str + '-G' + kern + ' '
        cmd_str = cmd_str + '-o \"' + os.path.join(dir_ouput,bodyname+'%02d_' % i) +'\"\n'
        
        subprocess.run(cmd_str,shell=True)

        T = np.loadtxt(os.path.join(dir_ouput,bodyname+'%02d_y.txt' % i))
        deformedZ = pv.PolyData(T,refmesh.faces)
        deformedZ.save(os.path.join(dir_ouput,labels[i]+'_'+bodyname+outsuffix+'.ply'))
        meshes_corresp.append(deformedZ)
        file_names = ['%02d_comptime.txt', '%02d_info.txt', '%02d_p.txt', '%02d_y.txt']
        for file in file_names:
            file_path = os.path.join(dir_ouput,bodyname+file % i)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print('Done mesh %d/%d' % (i+1,len(meshlist)))
        
    os.remove(os.path.join(dir_ouput,bodyname+'_ref_p.txt'))
    os.remove(os.path.join(dir_ouput,bodyname+'_ref_f.txt'))

    t1 = time.time()
    print('Correspondence for %d meshes in %.2f min' % (len(meshlist),(t1-t0)/60))
    return meshes_corresp
       


def check_gbcpd_results(meshlist,meshes_corresp,labels=None,show_plot=True,rmse_method='closest_point'):
    '''
    Check the results of GBCPD by calculating the error between the correspondance and
    originial meshes and through a plot.

    Parameters
    ----------
    meshlist : list of pyvista.PolyData
        List of originial meshes.
    meshes_corresp : list of pyvista.PolyData
        list of corresponding meshes resulting from GBCPD.
    labels : list of strings, optional
        List of labels for each participant/item. The default is None.
    show_plot : bool, optional
        Show plot or not. The default is True.
    rmse_method : string, optional
        'closest_point' to calculate the error between points in the corresponding mesh 
        and each closest point on the surface of the original mesh. 
        'raytrace' to calculate the error by finding the distance from the points 
        on the correspondance mesh and the surface of the original mesh in the direction of
        the point normals. This is much slower.
        The default is 'closest_point'.

    Returns
    -------
    rmse : numpy.ndarray
        The RMSE for each mesh in meshlist.

    '''
    
    if labels == None:
        labels = []
        for i in range(len(meshlist)):
            labels.append('%03d' % i)
            
    n_row = int(np.ceil(np.sqrt(len(meshes_corresp))))
    n_col = int(np.ceil(len(meshes_corresp) / n_row))
    
    rmse = np.zeros(len(meshlist))
    if rmse_method == 'closest_point':
        for i in range(len(meshes_corresp)):
            _,pt = meshlist[i].find_closest_cell(meshes_corresp[i].points,return_closest_point=True)
            rmse[i] = np.sqrt(np.mean(np.linalg.norm(meshes_corresp[i].points - pt,axis=1)**2))
    elif rmse_method == 'raytrace':
        for i in range(len(meshes_corresp)):
            normals = meshes_corresp[i].compute_normals(cell_normals=False,point_normals=True)
    
            d = np.zeros(normals.n_points)
            for j in range(normals.n_points):
                ip,_ = meshlist[i].ray_trace(normals.points[j,:]-normals['Normals'][j,:]*100,
                                          normals.points[j,:]+normals['Normals'][j,:]*100,first_point=False)
    
                d[j] = np.amin(np.linalg.norm(ip - normals.points[j,:],axis=1))
            rmse[i] = np.sqrt(np.mean(d**2))
    
    if show_plot == True:
        pv.set_plot_theme('document')
        #pl = pv.Plotter(notebook=False)
        pl = pv.Plotter(notebook=False,shape=(n_row,n_col))
        pl.disable_anti_aliasing()
        for i in range(len(meshes_corresp)):
            pl.subplot(int(np.floor(i/n_col)),i%n_col)
            pl.add_mesh(meshlist[i],color='b',opacity=.5,label='Target')
            pl.add_mesh(meshes_corresp[i],color='r',opacity=.5,show_edges=True,edge_color='r',label='Deformed Ref')
            pl.add_text(labels[i]+' RMSE=%.2f' % rmse[i],font_size=16)
            if i == len(meshes_corresp)-1:
                pl.add_legend(bcolor=None,size=(.4,.4))
        pl.show()
        pl.close()

    return rmse