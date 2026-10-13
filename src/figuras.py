import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / 'figures'
FIRMA = 'Belem Cisneros Díaz'
FUENTE = 'Fuente: APPA-REAL (ChaLearn LAP), validación de 1,897 fotos'

for ruta in sorted((RAIZ / 'assets' / 'fuentes').glob('*.ttf')):
    font_manager.fontManager.addfont(str(ruta))
plt.rcParams['font.family'] = 'Raleway'


def leer_metricas():
    return json.loads((RAIZ / 'resultados' / 'metricas.json').read_text(encoding='utf-8'))


def estilo_ejes(ax, colores):
    ax.set_facecolor(colores['fondo'])
    for lado in ['top', 'right']:
        ax.spines[lado].set_visible(False)
    for lado in ['left', 'bottom']:
        ax.spines[lado].set_color(colores['ejes'])
    ax.tick_params(colors=colores['secundario'], labelsize=10)
    ax.grid(color=colores['rejilla'], linewidth=0.8)
    ax.set_axisbelow(True)


def cerrar(fig, colores, titulo, subtitulo, fuente, nombre):
    fig.text(0.04, 0.955, titulo, fontsize=17, fontweight='bold', color=colores['texto'],
             fontfamily='Bricolage Grotesque', va='top')
    fig.text(0.04, 0.955 - 0.62 / fig.get_figheight(), subtitulo, fontsize=11, color=colores['secundario'], va='top')
    fig.text(0.04, 0.03, fuente, fontsize=9, color=colores['secundario'])
    fig.text(0.97, 0.022, FIRMA, ha='right', fontsize=18, color=colores['texto'], fontfamily='Zeyada')
    CARPETA.mkdir(exist_ok=True)
    fig.savefig(CARPETA / nombre, dpi=200, facecolor=colores['fondo'])
    plt.close(fig)


def figura_distribucion():
    colores = {
        'fondo': '#F8F4F1', 'banda_menores': '#E8D9D6', 'rejilla': '#DCDCDE', 'ejes': '#A7A8A1',
        'texto': '#1D434F', 'secundario': '#4A7B8C', 'adultos': '#376F78', 'menores': '#B82B66',
        'mediana': '#9B1D52',
    }
    labels = pd.read_csv(RAIZ / 'data' / 'labels.csv')
    edades = labels['real_age']
    conteos = edades.value_counts().sort_index()
    mediana = edades.median()
    menores = int((edades < 18).sum())
    tope = conteos.max()

    fig, ax = plt.subplots(figsize=(13, 7.2))
    fig.patch.set_facecolor(colores['fondo'])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.14)
    estilo_ejes(ax, colores)
    ax.grid(axis='x', visible=False)
    ax.axvspan(-0.5, 17.5, color=colores['banda_menores'], zorder=0)
    ax.bar(conteos.index, conteos.values, width=0.85, zorder=2,
           color=[colores['menores'] if edad < 18 else colores['adultos'] for edad in conteos.index])
    ax.text(8.5, tope * 1.05, 'Menores de edad\n(0 a 17 años)', ha='center', va='center',
            color=colores['menores'], fontsize=10, fontweight='bold')
    ax.axvline(mediana, color=colores['mediana'], linewidth=1.8, linestyle='--', zorder=3)
    ax.text(mediana + 1.2, tope * 0.74, f'Mediana: {mediana:.0f} años', va='top',
            color=colores['mediana'], fontsize=11, fontweight='bold')
    ax.text(42, tope * 1.02, f'Picos en edades redondas: 25 años ({conteos[25]} fotos) y 30 años ({conteos[30]})',
            fontsize=10, color=colores['texto'], va='center')
    for edad in [25, 30]:
        ax.annotate('', xy=(edad, conteos[edad] + 2), xytext=(41.5, tope * 1.02),
                    arrowprops={'arrowstyle': '-', 'color': colores['ejes'], 'linewidth': 0.8})
    ax.set_xlim(-0.5, edades.max() + 1)
    ax.set_ylim(0, tope * 1.13)
    ax.set_xlabel('Edad real (años)', color=colores['texto'], fontsize=11)
    ax.set_ylabel('Número de fotos', color=colores['texto'], fontsize=11)
    ax.legend(handles=[Patch(color=colores['menores'], label='Menores de 18 años'),
                       Patch(color=colores['adultos'], label='18 años o más')],
              frameon=False, loc='upper right', fontsize=10, labelcolor=colores['texto'])
    cerrar(fig, colores,
           f'Los menores de 18 años son {menores:,} fotos ({menores / len(edades) * 100:.1f}%); '
           f'la mitad del conjunto tiene {mediana:.0f} años o menos',
           'Distribución de la edad real, año por año. Los picos en 25 y 30 años son compatibles con edades redondeadas.',
           f'Fuente: APPA-REAL (ChaLearn LAP), labels.csv · n = {len(edades):,} fotos',
           '01_distribucion_edades.png')


def figura_comparacion(metricas):
    colores = {
        'fondo': '#F8F4F1', 'rejilla': '#DCDCDE', 'ejes': '#A7A8A1', 'texto': '#1D434F',
        'secundario': '#4A7B8C', 'modelo_0': '#BBC2CC', 'modelo_1': '#4A7B8C', 'modelo_2': '#376F78',
        'meta': '#D83B80',
    }
    mae = metricas['mae']
    filas = [
        ('Modelo 0 · Constante (mediana)', mae['modelo_0'], colores['modelo_0']),
        ('Modelo 1 · ResNet50 congelada', mae['modelo_1'], colores['modelo_1']),
        ('Modelo 2 · ResNet50 con ajuste fino', mae['modelo_2'], colores['modelo_2']),
    ]
    reduccion = (1 - mae['modelo_2'] / mae['modelo_0']) * 100

    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.patch.set_facecolor(colores['fondo'])
    fig.subplots_adjust(left=0.27, right=0.95, top=0.76, bottom=0.17)
    estilo_ejes(ax, colores)
    ax.grid(axis='y', visible=False)
    posiciones = np.arange(len(filas))[::-1]
    for y, (nombre, valor, color) in zip(posiciones, filas):
        ax.barh(y, valor, height=0.55, color=color, zorder=2)
        ax.text(valor + 0.25, y, f'{valor:.2f} años', va='center', fontsize=12,
                fontweight='bold', color=colores['texto'])
    ax.set_yticks(posiciones, [f[0] for f in filas], fontsize=11, color=colores['texto'])
    ax.axvline(8, color=colores['meta'], linewidth=1.6, linestyle='--', zorder=3)
    ax.text(8.1, posiciones.max() + 0.42, 'Meta: MAE ≤ 8', color=colores['meta'], fontsize=10.5, fontweight='bold')
    ax.set_xlim(0, max(v for _, v, _ in filas) * 1.18)
    ax.set_ylim(-0.6, posiciones.max() + 0.7)
    ax.set_xlabel('MAE de validación (años): error absoluto medio entre la edad real y la estimada',
                  color=colores['texto'], fontsize=11)
    cerrar(fig, colores,
           f'Solo el ajuste fino cumple la meta: MAE de {mae["modelo_2"]:.2f} años, '
           f'{reduccion:.0f}% menos que predecir la mediana',
           'La red congelada con pesos de ImageNet no supera al modelo constante: los filtros genéricos no distinguen edades.',
           FUENTE, '02_comparacion_modelos.png')


def figura_curvas():
    colores = {
        'fondo': '#F8F4F1', 'rejilla': '#DCDCDE', 'ejes': '#A7A8A1', 'texto': '#1D434F',
        'secundario': '#4A7B8C', 'validacion': '#376F78', 'entrenamiento': '#2B595D',
        'meta': '#D83B80', 'mejor': '#9B1D52',
    }
    registros = {
        numero: pd.read_csv(RAIZ / 'resultados' / f'registro_modelo_{numero}.csv').assign(epoca=lambda t: t['epoch'] + 1)
        for numero in [1, 2]
    }
    nombres = {1: 'Modelo 1 · ResNet50 congelada', 2: 'Modelo 2 · ResNet50 con ajuste fino'}
    tope = max(t[['mae', 'val_mae']].max().max() for t in registros.values()) * 1.08

    fig, ejes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True,
                             gridspec_kw={'width_ratios': [len(registros[1]) + 2, len(registros[2]) + 2]})
    fig.patch.set_facecolor(colores['fondo'])
    fig.subplots_adjust(left=0.06, right=0.96, top=0.76, bottom=0.15, wspace=0.08)
    for ax, (numero, tabla) in zip(ejes, registros.items()):
        estilo_ejes(ax, colores)
        ax.plot(tabla['epoca'], tabla['mae'], color=colores['entrenamiento'], linewidth=2,
                linestyle='--', marker='o', markersize=5, label='Entrenamiento')
        ax.plot(tabla['epoca'], tabla['val_mae'], color=colores['validacion'], linewidth=2.4,
                marker='o', markersize=6, label='Validación')
        mejor = tabla.loc[tabla['val_mae'].idxmin()]
        ax.scatter(mejor['epoca'], mejor['val_mae'], s=140, facecolor='none',
                   edgecolor=colores['mejor'], linewidth=2, zorder=4)
        ax.annotate(f'Mejor época: {int(mejor["epoca"])}\nMAE {mejor["val_mae"]:.2f}',
                    xy=(mejor['epoca'], mejor['val_mae']),
                    xytext=(40, -48) if numero == 1 else (0, 34), textcoords='offset points',
                    ha='left' if numero == 1 else 'center', fontsize=10, fontweight='bold', color=colores['mejor'])
        ax.axhline(8, color=colores['meta'], linewidth=1.4, linestyle=':')
        ax.text(tabla['epoca'].max() + 0.4, 8.3, 'Meta 8', color=colores['meta'], fontsize=9.5, ha='right')
        ax.set_title(nombres[numero], loc='left', fontsize=12, color=colores['texto'], fontweight='bold')
        ax.set_xticks(tabla['epoca'])
        ax.set_xlim(0.5, tabla['epoca'].max() + 0.6)
        ax.set_xlabel('Época', color=colores['texto'], fontsize=11)
    ejes[0].set_ylim(0, tope)
    ejes[0].set_ylabel('MAE (años)', color=colores['texto'], fontsize=11)
    ejes[1].legend(frameon=False, fontsize=10, labelcolor=colores['texto'], loc='upper right')
    final = registros[2]
    mejor = final.loc[final['val_mae'].idxmin()]
    cerrar(fig, colores,
           f'El ajuste fino baja el MAE de validación de {final["val_mae"].iloc[0]:.1f} a {mejor["val_mae"]:.2f} años; '
           f'la red congelada se estanca en {registros[1]["val_mae"].min():.1f}',
           f'La parada temprana detuvo el modelo 2 en la época {len(final)} y restauró la época {int(mejor["epoca"])}. '
           f'Entrenamiento {mejor["mae"]:.2f} contra validación {mejor["val_mae"]:.2f}: sobreajuste moderado.',
           'Fuente: registros por época (resultados/registro_modelo_1.csv y registro_modelo_2.csv) · validación: 1,897 fotos',
           '03_curvas_aprendizaje.png')


def figura_error_por_grupo(metricas):
    colores = {
        'fondo': '#F8F4F1', 'rejilla': '#DCDCDE', 'ejes': '#A7A8A1', 'texto': '#1D434F',
        'secundario': '#4A7B8C', 'grupo': '#4A7B8C', 'destacado': '#B82B66', 'cero': '#1D434F',
    }
    tabla = pd.DataFrame(metricas['error_por_grupo']['modelo_2'])
    destacados = ['13-17', '18-24']
    color_barras = [colores['destacado'] if g in destacados else colores['grupo'] for g in tabla['grupo']]
    adolescentes = tabla.set_index('grupo').loc['13-17']

    fig, (ax_mae, ax_sesgo) = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.patch.set_facecolor(colores['fondo'])
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.17, wspace=0.18)
    x = np.arange(len(tabla))
    etiquetas = [f'{g}\n(n={n})' for g, n in zip(tabla['grupo'], tabla['fotos'])]

    estilo_ejes(ax_mae, colores)
    ax_mae.grid(axis='x', visible=False)
    ax_mae.bar(x, tabla['mae'], width=0.62, color=color_barras, zorder=2)
    for xi, valor in zip(x, tabla['mae']):
        ax_mae.text(xi, valor + 0.2, f'{valor:.1f}', ha='center', fontsize=10, color=colores['texto'], fontweight='bold')
    ax_mae.set_xticks(x, etiquetas, fontsize=9.5)
    ax_mae.set_ylim(0, tabla['mae'].max() * 1.15)
    ax_mae.set_ylabel('MAE (años)', color=colores['texto'], fontsize=11)
    ax_mae.set_title('Error absoluto medio', loc='left', fontsize=12, color=colores['texto'], fontweight='bold')

    estilo_ejes(ax_sesgo, colores)
    ax_sesgo.grid(axis='x', visible=False)
    ax_sesgo.bar(x, tabla['sesgo'], width=0.62, color=color_barras, zorder=2)
    ax_sesgo.axhline(0, color=colores['cero'], linewidth=1)
    for xi, valor in zip(x, tabla['sesgo']):
        ax_sesgo.text(xi, valor + (0.35 if valor >= 0 else -0.35), f'{valor:+.1f}', ha='center',
                      va='bottom' if valor >= 0 else 'top', fontsize=10, color=colores['texto'], fontweight='bold')
    ax_sesgo.set_xticks(x, etiquetas, fontsize=9.5)
    limite = np.abs(tabla['sesgo']).max() * 1.25
    ax_sesgo.set_ylim(-limite, limite)
    ax_sesgo.set_ylabel('Sesgo: edad estimada − edad real (años)', color=colores['texto'], fontsize=11)
    ax_sesgo.set_title('Sesgo: positivo = sobreestima, negativo = subestima', loc='left', fontsize=12,
                       color=colores['texto'], fontweight='bold')
    ax_sesgo.legend(handles=[Patch(color=colores['destacado'], label='Zona de decisión de venta (13-24)'),
                             Patch(color=colores['grupo'], label='Resto de los grupos')],
                    frameon=False, fontsize=9.5, labelcolor=colores['texto'], loc='lower left')
    cerrar(fig, colores,
           f'El modelo hace parecer mayores a los adolescentes: {adolescentes["sesgo"]:+.1f} años de sesgo '
           f'en 13-17, la dirección de mayor riesgo legal',
           'Modelo 2 por grupo de edad. Las estimaciones se acercan a las edades más frecuentes: '
           'sobreestima a los jóvenes y subestima a los mayores.',
           FUENTE, '04_error_por_grupo.png')


def figura_real_vs_estimada(metricas):
    colores = {
        'fondo': '#F8F4F1', 'banda_menores': '#E8D9D6', 'rejilla': '#DCDCDE', 'ejes': '#A7A8A1',
        'texto': '#1D434F', 'secundario': '#4A7B8C', 'adultos': '#376F78', 'menores': '#B82B66',
        'diagonal': '#1D434F', 'limite': '#D83B80',
    }
    tabla = pd.read_csv(RAIZ / 'resultados' / 'predicciones_validacion.csv')
    menores = tabla[tabla['real_age'] < 18]
    adultos = tabla[tabla['real_age'] >= 18]
    menores_como_adultos = int((menores['edad_estimada'] >= 18).sum())
    tope = 100

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor(colores['fondo'])
    fig.subplots_adjust(left=0.09, right=0.96, top=0.83, bottom=0.11)
    estilo_ejes(ax, colores)
    ax.axvspan(0, 18, color=colores['banda_menores'], zorder=0)
    ax.scatter(adultos['real_age'], adultos['edad_estimada'], s=14, alpha=0.45, color=colores['adultos'],
               edgecolor='none', label=f'18 años o más (n = {len(adultos):,})', zorder=2)
    ax.scatter(menores['real_age'], menores['edad_estimada'], s=18, alpha=0.75, color=colores['menores'],
               edgecolor='none', marker='D', label=f'Menores de 18 años (n = {len(menores):,})', zorder=3)
    ax.plot([0, tope], [0, tope], color=colores['diagonal'], linewidth=1.2, linestyle='--', zorder=4)
    ax.text(tope - 2, tope - 6, 'Acierto perfecto', rotation=45, ha='right', fontsize=10, color=colores['diagonal'])
    ax.axvline(18, color=colores['limite'], linewidth=1.2)
    ax.axhline(18, color=colores['limite'], linewidth=1.2)
    ax.text(97, 19.5, 'Edad estimada: 18 años', ha='right', fontsize=9.5, color=colores['limite'])
    ax.text(9, tope - 3, 'Zona de menores', ha='center', fontsize=10, color=colores['menores'], fontweight='bold')
    ax.text(2, 72, f'{menores_como_adultos} de {len(menores)} menores\nestimados ≥ 18 años',
            ha='left', fontsize=11, fontweight='bold', color=colores['menores'])
    ax.set_xlim(0, tope)
    ax.set_ylim(0, tope)
    ax.set_aspect('equal')
    ax.set_xlabel('Edad real (años)', color=colores['texto'], fontsize=11)
    ax.set_ylabel('Edad estimada por el modelo 2 (años)', color=colores['texto'], fontsize=11)
    ax.legend(frameon=False, fontsize=10, labelcolor=colores['texto'], loc='lower right', markerscale=1.6)
    cerrar(fig, colores,
           f'{menores_como_adultos} de {len(menores)} menores ({menores_como_adultos / len(menores) * 100:.0f}%) '
           'son estimados como mayores de edad',
           f'Edad real contra estimada (MAE {metricas["mae"]["modelo_2"]:.2f}). Los puntos sobre la diagonal '
           'sobreestiman; debajo, subestiman.',
           FUENTE, '05_real_vs_estimada.png')


def figura_umbral(metricas):
    colores = {
        'fondo': '#F8F4F1', 'colchon': '#DEC997', 'rejilla': '#DCDCDE', 'ejes': '#A7A8A1',
        'texto': '#1D434F', 'secundario': '#4A7B8C', 'menores': '#B82B66', 'adultos': '#376F78',
        'umbral': '#D83B80', 'limite': '#9B1D52',
    }
    curva = pd.DataFrame(metricas['curva_umbral'])
    umbral = metricas['umbral']
    elegido = umbral['elegido']
    intervalo = metricas['intervalos_95_bootstrap']['menores_no_detectados_pct']

    fig, ax = plt.subplots(figsize=(13, 7.2))
    fig.patch.set_facecolor(colores['fondo'])
    fig.subplots_adjust(left=0.07, right=0.82, top=0.78, bottom=0.14)
    estilo_ejes(ax, colores)
    ax.axvspan(18, elegido, color=colores['colchon'], alpha=0.35, zorder=0)
    ax.text((18 + elegido) / 2, 96, f'Colchón de seguridad: de 18 a {elegido} años', ha='center',
            fontsize=10, color=colores['texto'])
    ax.plot(curva['umbral'], curva['menores_no_detectados_pct'], color=colores['menores'], linewidth=2.4,
            marker='D', markersize=7, zorder=3)
    ax.plot(curva['umbral'], curva['adultos_con_identificacion_pct'], color=colores['adultos'], linewidth=2.4,
            linestyle='--', marker='o', markersize=7, zorder=3)
    final = curva.iloc[-1]
    ax.text(final['umbral'] + 0.3, final['menores_no_detectados_pct'], 'Menores no detectados\n(riesgo legal)',
            va='center', fontsize=10, color=colores['menores'], fontweight='bold')
    ax.text(final['umbral'] + 0.3, final['adultos_con_identificacion_pct'],
            'Adultos a los que se pide\nidentificación (fricción)', va='center', fontsize=10,
            color=colores['adultos'], fontweight='bold')
    ax.axhline(umbral['limite_menores_pct'], color=colores['limite'], linewidth=1.2, linestyle=':')
    ax.text(25.2, umbral['limite_menores_pct'] + 1.2, f'Límite: {umbral["limite_menores_pct"]:.0f}% de menores',
            fontsize=9.5, color=colores['limite'])
    ax.axvline(elegido, color=colores['umbral'], linewidth=1.8)
    ax.errorbar(elegido, umbral['menores_no_detectados_pct'],
                yerr=[[umbral['menores_no_detectados_pct'] - intervalo[0]], [intervalo[1] - umbral['menores_no_detectados_pct']]],
                color=colores['umbral'], capsize=5, linewidth=1.6, zorder=4)
    ax.annotate(f'T = {elegido}: {umbral["menores_no_detectados_pct"]:.1f}% de menores\n'
                f'(IC 95%: {intervalo[0]:.1f}% a {intervalo[1]:.1f}%)',
                xy=(elegido, umbral['menores_no_detectados_pct']), xytext=(-150, 60), textcoords='offset points',
                fontsize=10, fontweight='bold', color=colores['umbral'],
                arrowprops={'arrowstyle': '-', 'color': colores['umbral'], 'linewidth': 0.9})
    ax.set_xticks(curva['umbral'])
    ax.set_xlim(17.6, 30.4)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Umbral T: se pide identificación si la edad estimada es menor que T (años)',
                  color=colores['texto'], fontsize=11)
    ax.set_ylabel('Porcentaje', color=colores['texto'], fontsize=11)
    cerrar(fig, colores,
           f'Ningún umbral de 18 a 30 años deja pasar 1% o menos de menores: con T = {elegido} pasa el '
           f'{umbral["menores_no_detectados_pct"]:.1f}%',
           f'Y se pediría identificación al {umbral["adultos_con_identificacion_pct"]:.1f}% de los adultos. '
           'El modelo sirve como apoyo para pedir identificación, no como filtro autónomo.',
           f'Fuente: APPA-REAL (ChaLearn LAP), validación de 1,897 fotos, {metricas["menores_validacion"]} menores · '
           'intervalo de confianza por bootstrap (1,000 remuestreos)',
           '06_compensacion_umbral.png')


def generar():
    metricas = leer_metricas()
    figura_distribucion()
    figura_comparacion(metricas)
    figura_curvas()
    figura_error_por_grupo(metricas)
    figura_real_vs_estimada(metricas)
    figura_umbral(metricas)


if __name__ == '__main__':
    generar()
    print('Figuras generadas en', CARPETA)
