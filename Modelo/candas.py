# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:33:05 2022

@author: Yeferson Loaiza
"""
import pandas as pd

#Pandas
def crear(Cedula):
    datos = {'Nombre':
    ['Chaloi','Mariso','Yolanda','Tina'],'Cedula':
    ['100','90','100','80']}

    df = pd.DataFrame(datos)

    df.reset_index().to_csv('DatosExportados.csv',header=True, index=False)

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
        return df1
        
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
        
        
