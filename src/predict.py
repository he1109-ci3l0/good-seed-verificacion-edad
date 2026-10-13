import argparse
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / 'models' / 'modelo_2.keras'
METRICAS = RAIZ / 'resultados' / 'metricas.json'
TAMANO_IMAGEN = 224


def umbral_recomendado():
    if METRICAS.exists():
        return json.loads(METRICAS.read_text(encoding='utf-8'))['umbral']['elegido']
    return 30


def estimar_edad(ruta_foto, modelo):
    from tensorflow.keras.utils import img_to_array, load_img

    imagen = load_img(ruta_foto, color_mode='rgb', target_size=(TAMANO_IMAGEN, TAMANO_IMAGEN))
    lote = np.expand_dims(img_to_array(imagen) / 255.0, axis=0)
    return float(modelo.predict(lote, verbose=0)[0][0])


def decidir(edad_estimada, umbral):
    if edad_estimada < umbral:
        return 'Pedir identificación oficial'
    return 'No se activa la solicitud automática; la decisión de venta sigue siendo del cajero'


def main():
    parser = argparse.ArgumentParser(description='Estima la edad de una foto de rostro y aplica la política de umbral.')
    parser.add_argument('foto', help='Ruta de la foto del rostro (JPG o PNG)')
    parser.add_argument('--modelo', default=str(MODELO), help='Ruta del modelo .keras')
    parser.add_argument('--umbral', type=float, default=None, help='Umbral T en años (por defecto, el de metricas.json)')
    args = parser.parse_args()

    from tensorflow.keras.models import load_model

    umbral = args.umbral if args.umbral is not None else umbral_recomendado()
    modelo = load_model(args.modelo, compile=False)
    edad = estimar_edad(args.foto, modelo)
    print(f'Edad estimada: {edad:.1f} años')
    print(f'Umbral T: {umbral:g} años')
    print(f'Decisión: {decidir(edad, umbral)}')


if __name__ == '__main__':
    main()
