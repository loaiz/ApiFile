from fastapi import Depends, FastAPI, HTTPException, status,Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import UploadFile, File
import os
# import asyncpg
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
from Modelo import candas
import pandas as pd
import json
# from flask import Flask, send_file
from PIL import Image
from matplotlib import pyplot as plt


SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

#modulo usuarios
db = {
    "tim": {
        "username": "tim",
        "full_name": "Tim Ruscica",
        "email": "tim@gmail.com",
        "hashed_password": "$2b$12$HxWHkvMuL7WrZad6lcCfluNFj1/Zp63lvP5aUrKlSTYtoFzPXHOtu",
        "disabled": False
    }
}



class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str or None = None


class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None
    hashed_password: str  # Added password field


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


app = FastAPI()

upload_dir = os.path.join(os.path.dirname(__file__), "uploads")

edit_dir = os.path.join(os.path.dirname(__file__),)


@app.post("/imagen")
async def upload_image_gray(file: UploadFile = File(...), cedula: str = Form(...), nombre: str = Form(...)):
    contents = await file.read()
    with open(os.path.join(upload_dir, file.filename), "wb") as f:
        f.write(contents)
    archivo_real = upload_dir+'/'+ file.filename
    
    print(archivo_real)
    # Obtener la ruta completa del archivo subido
    file_path = os.path.join(upload_dir, file.filename)
    
    # print(file_path)
    # candas.crear(cedula)
    
    foto = candas.preprocesamiento_imagen(archivo_real)
    candas.deteccion_rostro_haar(foto)
    plt.imshow(foto)
    plt.show()
    print(foto)
    # foto.save('imagen_generada.png')
    # Descargar el archivo en el computador del usuario
    # send_file('imagen_generada.png', as_attachment=True)
    return FileResponse(file_path, filename=file.filename)

@app.post("/editar")
async def upload_image(file: UploadFile = File(...), cedula: str = Form(...), nombre: str = Form(...)):
    contents = await file.read()
    with open(os.path.join(upload_dir, file.filename), "wb") as f:
        f.write(contents)
    
    print(edit_dir)
    # Obtener la ruta completa del archivo subido
    file_path = os.path.join(edit_dir, 'DatosExportados.csv')
    
    # candas.crear(cedula)
    
    candas.editar(nombre,cedula)
    
    # Descargar el archivo en el computador del usuario
    return FileResponse(file_path, filename='DatosExportados.csv')

@app.post("/listar")
async def upload_image(cedula: str = Form(...)):
    # candas.crear(cedula)
    # file_path = os.path.join(upload_dir, file.filename)
    
    result = candas.list(cedula)
    
    
@app.post("/Prueba")    
async def upload_image(Numero1: str = Form(...), Numero2: str = Form(...)):
    # candas.crear(cedula)
    # file_path = os.path.join(upload_dir, file.filename)
    
    result = float(Numero1) + float(Numero2)
    
    data = {
    'Numero1': Numero1,
    'Numero2': Numero2,
    'sUMA': result
    }
    
    print(data)
    
    
    # Descargar el archivo en el computador del usuario
    return json.dumps(data)
