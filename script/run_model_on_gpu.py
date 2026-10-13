
import os

import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

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


def create_model(input_shape):

    """
    Define el modelo
    """

    backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential(name='modelo_2')
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Entrena el modelo dados los parámetros
    """

    callbacks = [
        EarlyStopping(monitor='val_mae', mode='min', patience=3, restore_best_weights=True),
        ModelCheckpoint(f'models/{model.name}.keras', monitor='val_mae', mode='min', save_best_only=True),
        CSVLogger(f'resultados/registro_{model.name}.csv'),
    ]
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )

    return model


