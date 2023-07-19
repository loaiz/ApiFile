# -*- coding: utf-8 -*-
"""
@author: 

██╗░░░██╗███████╗███████╗███████╗██████╗░░██████╗░█████╗░███╗░░██╗  ██╗░░░░░░█████╗░░█████╗░██╗███████╗░█████╗░
╚██╗░██╔╝██╔════╝██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗████╗░██║  ██║░░░░░██╔══██╗██╔══██╗██║╚════██║██╔══██╗
░╚████╔╝░█████╗░░█████╗░░█████╗░░██████╔╝╚█████╗░██║░░██║██╔██╗██║  ██║░░░░░██║░░██║███████║██║░░███╔═╝███████║
░░╚██╔╝░░██╔══╝░░██╔══╝░░██╔══╝░░██╔══██╗░╚═══██╗██║░░██║██║╚████║  ██║░░░░░██║░░██║██╔══██║██║██╔══╝░░██╔══██║
░░░██║░░░███████╗██║░░░░░███████╗██║░░██║██████╔╝╚█████╔╝██║░╚███║  ███████╗╚█████╔╝██║░░██║██║███████╗██║░░██║
░░░╚═╝░░░╚══════╝╚═╝░░░░░╚══════╝╚═╝░░╚═╝╚═════╝░░╚════╝░╚═╝░░╚══╝  ╚══════╝░╚════╝░╚═╝░░╚═╝╚═╝╚══════╝╚═╝░░╚═╝
"""

from facenet_pytorch import InceptionResnetV1, MTCNN
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import h5py
import cv2
import os
        
def preprocesamiento_imagen(ruta_imagen):
    
    imagen_original=cv2.imread(ruta_imagen)
    imagen_grises=cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    imagen_grises_rgb=cv2.cvtColor(imagen_grises,cv2.COLOR_GRAY2RGB)
    imagen_grises_rgb = Image.fromarray(imagen_grises_rgb)
    
    return imagen_grises_rgb


def deteccion_rostro_haar(modelo,imagen_grises_rgb):

    img = modelo.detect(imagen_grises_rgb)
    (x0, y0, x1, y1) = (img[0][0][0], img[0][0][1], img[0][0][2], img[0][0][3])
    
    if img is not None:
        deteccion=True
        rostros=1
        rostros_detectados = imagen_grises_rgb.crop((x0, y0, x1, y1)).resize((160,160))
    else:
        deteccion=False
        rostros=0
        rostros_detectados = None
    
    return rostros_detectados, deteccion, rostros

def definicion_conjunto(df,path,modelo_deteccion):

    set_salida=[]
    set_no_rostro_detectado=[]
    set_etiquetas=[]
    
    for index, row in df.iterrows():
        img_gr=preprocesamiento_imagen(path)
        img,det,ros=deteccion_rostro_haar(modelo_deteccion,img_gr)
        
        if det==True:
            set_salida.append(img)
            set_etiquetas.append(row['IDENTIFICACION'])
        else:
            set_no_rostro_detectado.append(row['IDENTIFICACION'])
    
    return set_salida, set_etiquetas, set_no_rostro_detectado

def lectura_kernel(nombre_kernel):
    
    archivo_kernel=h5py.File(nombre_kernel,'r')
    
    return archivo_kernel


def realizar_estimacion_similitud(transformacion,modelo_reconocimiento,nombre_kernel,umbral_identificacion,rostro_identificar):
    
    similitud=[]
    kernel_modelo = lectura_kernel(nombre_kernel)
    lista_kernel = list(kernel_modelo.keys())
    tensor_identificar = modelo_reconocimiento((transformacion(rostro_identificar)/255).unsqueeze(0))
    
    for sujeto in lista_kernel:
        vector_sujeto = kernel_modelo[sujeto][:]
        modulo_sujeto = np.linalg.norm(vector_sujeto)
        vector_identificar = tensor_identificar.detach().numpy()
        modulo_identificar = np.linalg.norm(vector_identificar)
        coef_sim = np.dot(vector_sujeto,vector_identificar.T)/(modulo_sujeto*modulo_identificar)
        similitud.append(coef_sim)
        
    kernel_modelo.close()
    posicion = similitud.index(max(similitud))
    
    if max(similitud) >= umbral_identificacion:
        identificado = True
        sujeto = lista_kernel[posicion]
    else:
        identificado = False
        sujeto = 'Desconocido'

    return identificado, sujeto, max(similitud)


def entrenamiento_facenet(transformacion,modelo_reconocimiento,lista_rostros,lista_etiquetas):
    
    archivo_destino= r'C:\Users\Yeferson Loaiza\OneDrive\Documentos\Face api\Modelo\kernel_modelo.hdf5'
    archivo_hdf5=h5py.File(archivo_destino, 'a')
    
    for rostro, etiqueta in zip(lista_rostros,lista_etiquetas):
        vector_caracteristico=modelo_reconocimiento((transformacion(rostro)/255).unsqueeze(0))

        try:
            archivo_hdf5.create_dataset(etiqueta, data=vector_caracteristico.detach().numpy())
        except:
            archivo_hdf5.__delitem__(etiqueta)
            archivo_hdf5.create_dataset(etiqueta, data=vector_caracteristico.detach().numpy())

    archivo_hdf5.close()  
    return True
