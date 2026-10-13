import os

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RANDOM_STATE = 12345
VALIDATION_SPLIT = 0.25
TAMANO_IMAGEN = 224
TAMANO_LOTE = 32


def leer_etiquetas(path):

    """
    Lee labels.csv y mezcla sus filas antes de dividir.
    """

    labels = pd.read_csv(os.path.join(path, 'labels.csv'))

    return labels.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load_train(path):

    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """

    train_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, validation_split=VALIDATION_SPLIT)
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=leer_etiquetas(path),
        directory=os.path.join(path, 'final_files'),
        x_col='file_name',
        y_col='real_age',
        target_size=(TAMANO_IMAGEN, TAMANO_IMAGEN),
        batch_size=TAMANO_LOTE,
        class_mode='raw',
        subset='training',
        seed=RANDOM_STATE,
    )

    return train_gen_flow


def load_test(path):

    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta
    """

    test_datagen = ImageDataGenerator(rescale=1/255, validation_split=VALIDATION_SPLIT)
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=leer_etiquetas(path),
        directory=os.path.join(path, 'final_files'),
        x_col='file_name',
        y_col='real_age',
        target_size=(TAMANO_IMAGEN, TAMANO_IMAGEN),
        batch_size=TAMANO_LOTE,
        class_mode='raw',
        subset='validation',
        shuffle=False,
        seed=RANDOM_STATE,
    )

    return test_gen_flow
