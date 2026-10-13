import numpy as np
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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


def crear_modelo_congelado(input_shape):

    """
    Define el modelo 1: ResNet50 congelada con una cabeza entrenable
    """

    backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    backbone.trainable = False
    model = Sequential(name='modelo_1')
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def modelo_constante(train_data, test_data):

    """
    Modelo 0: predice para todas las fotos la mediana de edad del entrenamiento
    """

    mediana = float(np.median(train_data.labels))

    return mediana, np.full(len(test_data.labels), mediana)


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
