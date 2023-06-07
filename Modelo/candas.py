# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:33:05 2022

@author: Yeferson Loaiza
"""
import pandas as pd
# from sagemaker import get_execution_role
# import boto3
import pickle
import cv2
import numpy as np
# from matplotlib import pyplot as plt
# import seaborn as sns
import tensorflow as tf
import pandas as pd
from skimage import io
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

#Pandas
def crear(Cedula):
    datos = {'Nombre':
    ['Chaloi','Mariso','Yolanda','Tina'],'Cedula':
    ['100','90','100','80']}

    df = pd.DataFrame(datos)

    df.to_csv('DatosExportados.csv',header=True, index=False)

    print(df)

    data = {'Product': ['AAA','BBB'],
            'Price': ['210','250']}

    df = pd.DataFrame(data)
    df['Price'] = pd.to_numeric(df['Price'])

    print (df)
    print (df.dtypes)
    return df

def editar(Nombre,Cedula):
        
        print(Nombre,Cedula)
        df = pd.read_csv('DatosExportados.csv')
        df = df.reset_index(drop=True)
        datos = {'Nombre':
                [Nombre],'Cedula':
                [Cedula]}
        
        datos = pd.DataFrame(datos)
        print(datos)
        print(df)
        # df1 = df.append(datos, ignore_index = True)
        df1 = pd.concat([df, datos]).reset_index(drop=True)
        df1.to_csv('DatosExportados.csv',header=True, index=False)

        print(df1)
        
def list(cedula):
        df = pd.read_csv('DatosExportados.csv')
        print(df)
        # Obras=df[df["Cedula"] != cedula].index
        # print(Obras)
        # Obra=df.drop(Obras)
        # time.sleep(1)
        # Obra.reset_index(inplace=True, drop=True)
        # # Facturacion[:,1]
        # Obra.iloc[0]
        # nombre = Obra.iloc[:, 2]
        # contador = nombre[0]
        
        df = df.loc[df['Cedula'] == int(cedula)]
        
        df = df.iloc[0]
        
        print(df)
        
        return df
        
def preprocesamiento_imagen(ruta_imagen):
    
    imagen_original=io.imread(ruta_imagen)
    
    imagen_grises=cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    imagen_grises_rgb=cv2.cvtColor(imagen_grises,cv2.COLOR_GRAY2RGB)
    return imagen_grises_rgb


def deteccion_rostro_haar(imagen_grises_rgb):
    
    deteccion_rostros=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #(path_modelo_rostro)
    rostros_detectados=deteccion_rostros.detectMultiScale(imagen_grises_rgb,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    area=[]
    #for (x, y, Dx, Dy) in rostros_detectados:
    #    cv2.rectangle(imagen_grises_rgb, (x, y), (x+Dx, y+Dy), (0, 255, 0), 3)
    
    if len(rostros_detectados)>=1:
        deteccion=True
        rostros=1
        for (x, y, Dx, Dy) in rostros_detectados:
            area.append(Dx*Dy)
        seleccion=area.index(max(area))
        rostros_detectados=rostros_detectados[seleccion]
        (x,y,Dx,Dy)=rostros_detectados
        cv2.rectangle(imagen_grises_rgb, (x, y), (x+Dx, y+Dy), (0, 255, 0), 3)
    else:
        deteccion=False
        rostros=0
    
    
    
    #if len(rostros_detectados)==0:
     #   deteccion=False
     #   rostros=0
    #else:
     #   deteccion=True
      #  rostros=len(rostros_detectados)
    # print(rostros)
    print(imagen_grises_rgb, deteccion, rostros, rostros_detectados)
    return imagen_grises_rgb, deteccion, rostros, rostros_detectados
