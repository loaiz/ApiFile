# Api Example file request
Este proyecto tiene como funcion subir una imagen en cualquier formato y convertir su rgb de colores a escalas de grises y detectar un rostro
## Requerimientos

- Python 3.7+

## Configurar proyecto

En la raiz del proyecto:

1. Crear virtual environment: `python -m venv .venv`
2. Cargar virtual environment: para linux y mac `source .venv/bin/activate`; para windows `venv/Scripts/activate`
3. Instalar dependencias: `pip install -r requirements.txt`
4. Se puede ejecutar de modo que auto-recargue cambios: `uvicorn main:app --reload`.
5. una vez el servidor este ejecutando podemos revisar su funcionamiento desde `http://127.0.0.1:8000/docs`
6. la clave encriptada es: `tim1234`
