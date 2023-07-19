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


from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, FastAPI, HTTPException, status,Form
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
from sagemaker import get_execution_role
from datetime import datetime, timedelta
from passlib.context import CryptContext
from matplotlib import pyplot as plt
from fastapi import UploadFile, File
from jose import JWTError, jwt
from pydantic import BaseModel
from Modelo import candas
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import boto3
import h5py
import cv2
import os


app = FastAPI()

upload_dir = os.path.join(os.path.dirname(__file__), "uploads")

Model = os.path.join(os.path.dirname(__file__), "Modelo")

edit_dir = os.path.join(os.path.dirname(__file__),)

@app.post("/reconocimiento")
async def upload_image_gray(file: UploadFile = File(...)):
    
    contents = await file.read()
    
    with open(os.path.join(upload_dir, file.filename), "wb") as f:
        f.write(contents)
        
    archivo_real = upload_dir+'/'+ file.filename
    
    # Obtener la ruta completa del archivo subido
    file_path = os.path.join(upload_dir, file.filename)
    
    modelo_deteccion_rostros=MTCNN(margin=20, select_largest=True, post_process=False, keep_all=False)
    df_deteccion=pd.DataFrame([],columns=['IMAGEN','IDENTIFICACION','DETECCION','ROSTROS_DETECTADOS'])
    img_gr_fo = candas.preprocesamiento_imagen(archivo_real)
    
    img_fo,det_fo,ros_fo = candas.deteccion_rostro_haar(modelo_deteccion_rostros,img_gr_fo)
    
    df_deteccion = pd.concat([df_deteccion,pd.DataFrame(
    [{
        'IMAGEN':archivo_real,
        'IDENTIFICACION':file.filename.split('-')[0],
        'DETECCION': det_fo,
        'ROSTROS_DETECTADOS':ros_fo,
    }])],ignore_index=True)
    
    df_bloque=df_deteccion[df_deteccion['ROSTROS_DETECTADOS']==1]
    
    df_row_number=pd.concat([df_bloque,pd.DataFrame(df_bloque.groupby(['IDENTIFICACION'])['IDENTIFICACION'].rank(method='first'),columns=['IDENTIFICACION']).rename({'IDENTIFICACION':'REGISTROS'},axis=1)],axis=1)
    
    df_train=df_row_number[df_row_number['REGISTROS']==1]
    df_test=df_row_number[df_row_number['REGISTROS']>1]
     
    set_train, set_train_etiqueta, set_no_detectado_train=candas.definicion_conjunto(df_train,archivo_real,modelo_deteccion_rostros)
    
    set_test, set_test_etiqueta, set_no_detectado_test=candas.definicion_conjunto(df_test,archivo_real,modelo_deteccion_rostros)
    
    conjunto_train=set(set_train_etiqueta)
    conjunto_test=set(set_test_etiqueta)
    identificados_reales=conjunto_train & conjunto_test
    no_identificados_reales=list(conjunto_test-identificados_reales)
    df_solucion=pd.DataFrame([],columns=['IDENTIFICADO','ETIQUETA_ESTIMADA','SIMILITUD'])
    modelo_facenet=InceptionResnetV1(pretrained='vggface2').eval()
    
    transform = transforms.Compose([
    transforms.PILToTensor()
    ])
    
    identificado, sujeto, similitud = candas.realizar_estimacion_similitud(transform,modelo_facenet,r'C:\Users\Yeferson Loaiza\OneDrive\Documentos\Face api\Modelo\kernel_modelo.hdf5', 0.7,img_fo)
    
    df_solucion = pd.concat([df_solucion,pd.DataFrame(
        [{
            'IDENTIFICADO' : identificado,
            'ETIQUETA_ESTIMADA' : sujeto,
            'SIMILITUD' : similitud[0][0]
        }])],ignore_index=True)
    
    df_estimacion = df_test
    df_estimacion.reset_index(inplace=True)
    df_estimacion = df_estimacion.drop(['index'],axis=1)
    df_resultado = pd.concat([df_estimacion,df_solucion],axis=1)
    df_resultado['ETIQUETA_CORRECTA']=df_resultado.apply(lambda row: row['IDENTIFICACION']==row['ETIQUETA_ESTIMADA'],axis=1)
    p = df_resultado.loc[(df_resultado['ETIQUETA_CORRECTA']==True),['ETIQUETA_CORRECTA']].count()['ETIQUETA_CORRECTA']/df_resultado['ETIQUETA_CORRECTA'].count()
    print(p)

    return ros_fo,det_fo, p, sujeto




@app.post("/Entrenamiento")
async def upload_image_gray(file: UploadFile = File(...)):
    contents = await file.read()
    with open(os.path.join(upload_dir, file.filename), "wb") as f:
        f.write(contents)
    archivo_real = upload_dir+'/'+ file.filename
    
    print(archivo_real)
    # Obtener la ruta completa del archivo subido
    file_path = os.path.join(upload_dir, file.filename)
    
    modelo_deteccion_rostros=MTCNN(margin=20, select_largest=True, post_process=False, keep_all=False)
    df_deteccion=pd.DataFrame([],columns=['IMAGEN','IDENTIFICACION','DETECCION','ROSTROS_DETECTADOS'])
    img_gr_fo = candas.preprocesamiento_imagen(archivo_real)
    
    img_fo,det_fo,ros_fo = candas.deteccion_rostro_haar(modelo_deteccion_rostros,img_gr_fo)
    
    df_deteccion = pd.concat([df_deteccion,pd.DataFrame(
    [{
        'IMAGEN':archivo_real,
        'IDENTIFICACION':file.filename.split('-')[0],
        'DETECCION': det_fo,
        'ROSTROS_DETECTADOS':ros_fo,
    }])],ignore_index=True)
    
    df_bloque=df_deteccion[df_deteccion['ROSTROS_DETECTADOS']==1]
    
    df_row_number=pd.concat([df_bloque,pd.DataFrame(df_bloque.groupby(['IDENTIFICACION'])['IDENTIFICACION'].rank(method='first'),columns=['IDENTIFICACION']).rename({'IDENTIFICACION':'REGISTROS'},axis=1)],axis=1)
    
    df_train=df_row_number[df_row_number['REGISTROS']==1]
    df_test=df_row_number[df_row_number['REGISTROS']>1]
     
    set_train, set_train_etiqueta, set_no_detectado_train=candas.definicion_conjunto(df_train,archivo_real,modelo_deteccion_rostros)
    
    set_test, set_test_etiqueta, set_no_detectado_test=candas.definicion_conjunto(df_test,archivo_real,modelo_deteccion_rostros)
    
    conjunto_train=set(set_train_etiqueta)
    conjunto_test=set(set_test_etiqueta)
    
    identificados_reales=conjunto_train & conjunto_test
    no_identificados_reales=list(conjunto_test-identificados_reales)
    
    df_solucion=pd.DataFrame([],columns=['IDENTIFICADO','ETIQUETA_ESTIMADA','SIMILITUD'])
    modelo_facenet=InceptionResnetV1(pretrained='vggface2').eval()
    
    transform = transforms.Compose([
    transforms.PILToTensor()
    ])
    
    entrenamiento = candas.entrenamiento_facenet(transform,modelo_facenet,set_train,set_train_etiqueta)
    
    return entrenamiento
