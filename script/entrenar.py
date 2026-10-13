import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

import numpy as np
import pandas as pd
import tensorflow as tf

from src import datos
from src.modelo import create_model, crear_modelo_congelado, modelo_constante, train_model


def guardar_predicciones(numero, test_data, predicciones):
    tabla = pd.DataFrame({
        'file_name': test_data.filenames[:len(predicciones)],
        'real_age': test_data.labels[:len(predicciones)],
        'edad_estimada': np.asarray(predicciones).ravel(),
    })
    tabla.to_csv(f'resultados/predicciones_modelo_{numero}.csv', index=False)
    mae = (tabla['edad_estimada'] - tabla['real_age']).abs().mean()
    print(f'MAE de validación del modelo {numero}: {mae:.4f} (n = {len(tabla):,})')


def main():
    parser = argparse.ArgumentParser(description='Entrena los modelos de estimación de edad de Good Seed.')
    parser.add_argument('--modelo', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--tamano', type=int, default=224)
    parser.add_argument('--lote', type=int, default=32)
    parser.add_argument('--epocas', type=int, default=None)
    parser.add_argument('--prueba', action='store_true')
    parser.add_argument('--datos', default='data')
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(datos.RANDOM_STATE)
    datos.TAMANO_IMAGEN = args.tamano
    datos.TAMANO_LOTE = args.lote
    os.makedirs('models', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)

    print(f'Modelo {args.modelo} | imagen {args.tamano}x{args.tamano} | lote {args.lote} | prueba: {args.prueba}')
    inicio = time.time()
    train_data = datos.load_train(args.datos)
    test_data = datos.load_test(args.datos)

    if args.modelo == 0:
        mediana, predicciones = modelo_constante(train_data, test_data)
        print(f'Mediana de edad del entrenamiento: {mediana}')
        guardar_predicciones(0, test_data, predicciones)
        return

    epocas_por_defecto = {1: 5, 2: 20}
    epocas = 1 if args.prueba else (args.epocas or epocas_por_defecto[args.modelo])
    pasos = 2 if args.prueba else None
    forma = (args.tamano, args.tamano, 3)
    model = crear_modelo_congelado(forma) if args.modelo == 1 else create_model(forma)

    model = train_model(model, train_data, test_data, epochs=epocas,
                        steps_per_epoch=pasos, validation_steps=pasos)
    print(f'Tiempo de entrenamiento: {(time.time() - inicio) / 60:.1f} min')

    predicciones = model.predict(test_data, steps=pasos, verbose=0)
    guardar_predicciones(args.modelo, test_data, predicciones)
    print(f'Tiempo total: {(time.time() - inicio) / 60:.1f} min')


if __name__ == '__main__':
    main()
