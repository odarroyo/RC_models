# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:41:05 2025

@author: HOME
"""
from openseespy.opensees import *
import matplotlib.pyplot as plt
import opsvis as opsv
import opseestools.analisis3D as an
import opseestools.utilidades as ut
import numpy as np
import pandas as pd
import joblib as jb
import time
import tempfile
import multiprocessing
from joblib import Parallel, delayed
import os as os

#%%

records= ["GM01.txt", "GM02.txt", "GM03.txt", "GM04.txt", "GM05.txt", "GM06.txt", "GM07.txt", "GM08.txt", "GM09.txt", "GM10.txt", "GM11.txt", "GM12.txt", "GM13.txt", "GM14.txt", "GM15.txt",
         "GM16.txt", "GM17.txt", "GM18.txt", "GM19.txt", "GM20.txt", "GM21.txt", "GM22.txt", "GM23.txt", "GM24.txt", "GM25.txt", "GM26.txt", "GM27.txt", "GM28.txt", "GM29.txt", "GM30.txt",
          "GM31.txt", "GM32.txt", "GM33.txt", "GM34.txt", "GM35.txt", "GM36.txt", "GM37.txt", "GM38.txt", "GM39.txt", "GM40.txt", "GM41.txt", "GM42.txt", "GM43.txt", "GM44.txt"]

Nsteps= [3000, 3000, 2000, 2000, 5590, 5590, 4535, 4535, 9995, 9995, 7810, 7810, 4100, 4100, 4100, 4100, 5440, 5440, 6000, 6000, 2200, 2200, 11190, 11190, 7995, 7995, 7990, 7990, 2680, 2300, 8000, 8000, 2230, 2230, 1800, 1800, 18000, 18000, 18000, 18000, 2800, 2800, 7270, 7270]
DTs= [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.01,0.01,0.01,0.01,0.005,0.005,0.05,0.05,0.02,0.02,0.0025,0.0025,0.005,0.005,0.005,0.005,0.02,0.02,0.005,0.005,0.01,0.01,0.02,0.02,0.005,0.005,0.005,0.005,0.01,0.01,0.005,0.005]
GMcode = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]

Nsteps2 = Nsteps[1::2]
DTs2 = DTs[1::2]
#%%

def rundyn(ind):
    wipe()
    model('basic','-ndm',3,'-ndf',6)


    #%% Definicion de nodos
    coordx = [0, 6, 12]
    coordy = [0, 6, 12, 18, 24]
    coordz = [0, 3, 6, 9, 12, 15]
    mp = max(coordx)*max(coordy)*0.6 # para meterle una masa de 600kg por m2
    masas = [mp]*5

    coords = ut.creategrid3D(coordx,coordy,coordz,1,masas)
    fixZ(0.0,1,1,1,1,1,1)

    #%% Definición de materiales (son según norma colombiana NSR-10, similares detallamiento especial del ACI)
    fc = 28
    fy = 420
    noconf, conf, acero = ut.col_materials(fc,fy)

    #%% Definición de elementos

    Bcol = 0.5 # base de la columna
    Hcol = 0.5 # altura de la columna
    Bvig = 0.35 # base de la viga
    Hvig = 0.45 # altura de la viga

    c = 0.05  # recubrimiento de las secciones

    As4 = 0.000127 # area barra #4
    As5 = 0.0002 # area barra #5
    As6 = 0.000286
    As7 = 0.000387 # area barra #7

    col30x30 = 101 # tag de la columna
    vig30x40 = 201 # tag de la viga

    ut.create_rect_RC_section(col30x30, Hcol, Bcol, c, conf, noconf, acero, 4, As6, 4, As6, 6, As6)
    ut.create_rect_RC_section(vig30x40, Hvig, Bvig, c, conf, noconf, acero, 4, As5, 5, As5)

    #%% Creando los elementos
    coltags = [col30x30,col30x30, col30x30, col30x30, col30x30] # one section tag per floor
    # cols, vigx, vigy = ut.create_elements3D(coordx, coordy, coordz, col30x30, vig30x40, vig30x40)
    cols, vigx, vigy, sectag_col = ut.create_elements3D(coordx, coordy, coordz, coltags, vig30x40, vig30x40)
    # opsv.plot_model(node_labels=0,gauss_points=False)
    # plt.show()

    #%% Creando la losa
    hslab = 0.15 # altura de la losa
    Eslab = 1000*4400*(28)**0.5 # módulo de elasticidad de la losa
    pois = 0.3 # relación de Poisson de la losa


    ut.create_slabs(coordx, coordy, coordz, hslab, Eslab*0.5, pois)
    # ut.create_slabs_NL(coordx, coordy, coordz, hslab, Eslab, pois) # para la losa no lineal
    # vfo.plot_model(show_nodetags='no', show_eletags='no', show_nodes='yes')

    #%% Cargando las vigas
    floorx = -20  
    floory = -20
    roofx = -10 
    roofy = -10
    ut.load_beams3D(-20, -10, -20, -10, vigx, vigy, coordx, coordy)

    #%% calculando modos

    # eig = eigen('-fullGenLapack',(len(coordz)-1)*2)
    # modalProperties('-print', '-unorm', '-file', 'modal.txt')

    # vfo.plot_modeshape(scale=15, contour='X', modenumber=1)

    #%% Analizando el modelo

    an.gravedad()
    # plt.show()
    loadConst('-time',0.0)
    recs = records[2*ind:2*(ind+1)]
    tiempo,techo,techo2,techoT,node_disp,node_vel,node_acel,node_disp2,node_acel2,forces,driftX,driftY = an.dinamicoBD3(recs, DTs2[ind], Nsteps2[ind], DTs2[ind], 9.81*1, 0.025, 5, 1, [10000,1,2,3,4,5], [10000],eletype='frame')
    return tiempo,techo,techo2,node_acel,node_acel2,forces,driftX,driftY
    
#%%
# tiempo,techo,techo2,node_acel,node_acel2,forces,driftX,driftY = rundyn(0)
#%%

num_cores = multiprocessing.cpu_count() # esta linea identifica el número de nucleos totales del PC.
# En equipos con SMT identifica los núcleos físicos y lógicos. La recomendación si se va a seguir usando el PC es dejar dos núcleos físicos libres
stime = time.time()
# resultados devuelve de momento cuatro cosas. La primera es el indice del terremoto, la segunda es el factor escalar, la tercera el tiempo del registro y la cuarta es el desplazamiento de techo
resultados = Parallel(n_jobs=num_cores)(delayed(rundyn)(ind) for ind in range(len(DTs2))) # loop paralelo
etime = time.time()
ttotal = etime - stime
print('tiempo de ejecucion: ',ttotal,'segundos')
    
jb.dump(resultados, 'model_linear_slab_XX')

#%%

