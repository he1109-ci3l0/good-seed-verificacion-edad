import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RANDOM_STATE = 12345
RANGOS = [0, 12, 17, 24, 34, 49, 64, np.inf]
GRUPOS = ['0-12', '13-17', '18-24', '25-34', '35-49', '50-64', '65+']
UMBRALES = list(range(18, 31))
LIMITE_MENORES = 1.0
REPETICIONES_BOOTSTRAP = 1000


def leer_predicciones(numero):
    tabla = pd.read_csv(RAIZ / 'resultados' / f'predicciones_modelo_{numero}.csv')
    tabla['error'] = tabla['edad_estimada'] - tabla['real_age']
    tabla['grupo'] = pd.cut(tabla['real_age'], bins=RANGOS, labels=GRUPOS, include_lowest=True)
    return tabla


def mae(tabla):
    return float(tabla['error'].abs().mean())


def error_por_grupo(tabla):
    resumen = tabla.groupby('grupo', observed=False).agg(
        fotos=('error', 'size'),
        mae=('error', lambda e: e.abs().mean()),
        sesgo=('error', 'mean'),
    )
    return resumen.reset_index()


def tasas_umbral(tabla, umbral):
    menores = tabla[tabla['real_age'] < 18]
    adultos = tabla[tabla['real_age'] >= 18]
    menores_no_detectados = float((menores['edad_estimada'] >= umbral).mean() * 100)
    adultos_con_identificacion = float((adultos['edad_estimada'] < umbral).mean() * 100)
    return menores_no_detectados, adultos_con_identificacion


def curva_umbral(tabla):
    filas = []
    for umbral in UMBRALES:
        menores_no_detectados, adultos_con_identificacion = tasas_umbral(tabla, umbral)
        filas.append({
            'umbral': umbral,
            'menores_no_detectados_pct': menores_no_detectados,
            'adultos_con_identificacion_pct': adultos_con_identificacion,
        })
    return pd.DataFrame(filas)


def elegir_umbral(curva):
    cumplen = curva[curva['menores_no_detectados_pct'] <= LIMITE_MENORES]
    if len(cumplen) > 0:
        return int(cumplen['umbral'].min()), 'principal'
    return int(curva.loc[curva['menores_no_detectados_pct'].idxmin(), 'umbral']), 'respaldo'


def intervalos_bootstrap(tabla, umbral):
    generador = np.random.default_rng(RANDOM_STATE)
    muestras = {'mae_global': [], 'menores_no_detectados_pct': [], 'adultos_con_identificacion_pct': []}
    muestras.update({f'mae_{grupo}': [] for grupo in GRUPOS})
    for _ in range(REPETICIONES_BOOTSTRAP):
        muestra = tabla.iloc[generador.integers(0, len(tabla), len(tabla))]
        muestras['mae_global'].append(mae(muestra))
        menores_no_detectados, adultos_con_identificacion = tasas_umbral(muestra, umbral)
        muestras['menores_no_detectados_pct'].append(menores_no_detectados)
        muestras['adultos_con_identificacion_pct'].append(adultos_con_identificacion)
        por_grupo = muestra.groupby('grupo', observed=False)['error'].apply(lambda e: e.abs().mean())
        for grupo in GRUPOS:
            muestras[f'mae_{grupo}'].append(float(por_grupo[grupo]))
    return {
        nombre: [float(np.nanpercentile(valores, 2.5)), float(np.nanpercentile(valores, 97.5))]
        for nombre, valores in muestras.items()
    }


def evaluar():
    modelos = {numero: leer_predicciones(numero) for numero in [0, 1, 2]}
    final = modelos[2]
    curva = curva_umbral(final)
    umbral, regla = elegir_umbral(curva)
    menores_no_detectados, adultos_con_identificacion = tasas_umbral(final, umbral)
    menores_validacion = int((final['real_age'] < 18).sum())

    metricas = {
        'n_validacion': int(len(final)),
        'menores_validacion': menores_validacion,
        'mae': {f'modelo_{numero}': mae(tabla) for numero, tabla in modelos.items()},
        'mediana_entrenamiento_modelo_0': float(modelos[0]['edad_estimada'].iloc[0]),
        'desviacion_estandar_estimaciones': {f'modelo_{numero}': float(tabla['edad_estimada'].std()) for numero, tabla in modelos.items()},
        'error_por_grupo': {
            f'modelo_{numero}': error_por_grupo(modelos[numero]).to_dict(orient='records')
            for numero in [0, 2]
        },
        'curva_umbral': curva.to_dict(orient='records'),
        'umbral': {
            'elegido': umbral,
            'regla': regla,
            'limite_menores_pct': LIMITE_MENORES,
            'menores_no_detectados_pct': menores_no_detectados,
            'menores_no_detectados_n': int(((final['real_age'] < 18) & (final['edad_estimada'] >= umbral)).sum()),
            'adultos_con_identificacion_pct': adultos_con_identificacion,
        },
        'intervalos_95_bootstrap': intervalos_bootstrap(final, umbral),
        'bootstrap': {'repeticiones': REPETICIONES_BOOTSTRAP, 'semilla': RANDOM_STATE},
    }

    (RAIZ / 'resultados' / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
    final.drop(columns=['grupo']).to_csv(RAIZ / 'resultados' / 'predicciones_validacion.csv', index=False)
    return metricas


if __name__ == '__main__':
    resultado = evaluar()
    print(json.dumps({clave: resultado[clave] for clave in ['n_validacion', 'menores_validacion', 'mae', 'umbral']}, indent=2, ensure_ascii=False))
