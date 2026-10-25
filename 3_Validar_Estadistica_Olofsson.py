import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import confusion_matrix, cohen_kappa_score

RUTA_BASE = r"F:\Clasificacion_Azul"
NOMBRE_MOSAICO = "A4_Sep_2026"

CARPETA_MOSAICO = os.path.join(RUTA_BASE, "RESULTADOS_CARACTERIZACION", f"Caracterizacion_{NOMBRE_MOSAICO}")
RUTA_TIF_CLASIFICADO = os.path.join(CARPETA_MOSAICO, f"Mapa_{NOMBRE_MOSAICO}_Clasificado.tif")
RUTA_CSV_HECTAREAS = os.path.join(CARPETA_MOSAICO, f"Resumen_Hectareas_{NOMBRE_MOSAICO}.csv")

CARPETA_VALIDACION = os.path.join(CARPETA_MOSAICO, "Validacion_Olofsson_480pts")
os.makedirs(CARPETA_VALIDACION, exist_ok=True)
RUTA_SHP_PUNTOS = os.path.join(CARPETA_VALIDACION, f"Puntos_Validacion_480_{NOMBRE_MOSAICO}.shp")

NOMBRES_OFICIALES = {
    1: "1. Bosque primario",
    2: "2. Bosque secundario",
    3: "3. Regeneracion positiva",
    4: "4. Regeneracion negativa",
    5: "5. Pozas mineras",
    6: "6. Rios Quebradas",
    7: "7. Arena_Inundable/Carcaba",
    8: "8. Arenales"
}

if not os.path.exists(RUTA_SHP_PUNTOS):
    with rasterio.open(RUTA_TIF_CLASIFICADO) as src:
        mapa = src.read(1)
        transform_grid = src.transform
        crs_mapa = src.crs

    rng = np.random.default_rng(2026)
    puntos = []
    id_pt = 1
    for cid in range(1, 9):
        fy, fx = np.where(mapa == cid)
        if len(fy) == 0:
            continue
        idx = rng.choice(len(fy), size=min(60, len(fy)), replace=False)
        for r_i, c_i in zip(fy[idx], fx[idx]):
            x_u, y_u = transform_grid * (c_i + 0.5, r_i + 0.5)
            puntos.append({
                'ID_Punto': id_pt,
                'Clase_Mapa': cid,
                'Nom_Mapa': NOMBRES_OFICIALES[cid],
                'Clase_Real': cid,
                'X_UTM': round(x_u, 2),
                'Y_UTM': round(y_u, 2),
                'geometry': Point(x_u, y_u)
            })
            id_pt += 1

    gpd.GeoDataFrame(puntos, crs=crs_mapa).to_file(RUTA_SHP_PUNTOS, encoding='utf-8')
    print(f"-> 480 puntos creados en: {RUTA_SHP_PUNTOS}")
else:
    gdf_pts = gpd.read_file(RUTA_SHP_PUNTOS)
    df_areas = pd.read_csv(RUTA_CSV_HECTAREAS).sort_values('Clase_Pred')

    area_total_ha = df_areas['Area_ha'].sum()
    W_i = (df_areas['Area_ha'] / area_total_ha).values

    y_map = gdf_pts['Clase_Mapa'].astype(int).values
    y_ref = gdf_pts['Clase_Real'].astype(int).values
    clases = list(range(1, 9))

    cm = confusion_matrix(y_map, y_ref, labels=clases)
    n_i = cm.sum(axis=1)
    p_ij = np.zeros_like(cm, dtype=float)
    for i in range(8):
        if n_i[i] > 0:
            p_ij[i, :] = W_i[i] * (cm[i, :] / n_i[i])

    acc_usr = np.diag(p_ij) / np.maximum(p_ij.sum(axis=1), 1e-9)
    acc_prod = np.diag(p_ij) / np.maximum(p_ij.sum(axis=0), 1e-9)
    f1_sc = 2 * (acc_usr * acc_prod) / np.maximum(acc_usr + acc_prod, 1e-9)

    oa_pond = np.trace(p_ij)
    kappa = cohen_kappa_score(y_ref, y_map)
    var_oa = np.sum([(W_i[i]**2) * acc_usr[i] * (1.0 - acc_usr[i]) / max(n_i[i] - 1, 1) for i in range(8)])
    ic95_oa = 1.96 * np.sqrt(var_oa)

    area_ajustada = p_ij.sum(axis=0) * area_total_ha
    se_prop = np.zeros(8)
    for j in range(8):
        s_v = sum((W_i[i]**2) * ((cm[i, j]/n_i[i]) * (1.0 - cm[i, j]/n_i[i])) / (n_i[i] - 1.0) for i in range(8) if n_i[i] > 1)
        se_prop[j] = np.sqrt(s_v)
    ic95_ha = 1.96 * se_prop * area_total_ha

    df_res = pd.DataFrame({
        'Clase Oficial': [NOMBRES_OFICIALES[c] for c in clases],
        'Puntos (n_i)': n_i,
        'Área Mapa (ha)': df_areas['Area_ha'].round(2).values,
        'Peso W_i (%)': (W_i * 100).round(2),
        'Prec. Usuario (%)': (acc_usr * 100).round(2),
        'Prec. Productor (%)': (acc_prod * 100).round(2),
        'F1-Score (%)': (f1_sc * 100).round(2),
        'Área Ajustada (ha)': area_ajustada.round(2),
        'IC 95% (± ha)': ic95_ha.round(2)
    })

    print(df_res.to_string(index=False))
    print(f"PRECISIÓN PONDERADA: {oa_pond*100:.2f}% (± {ic95_oa*100:.2f}%) | KAPPA: {kappa:.4f}")
    df_res.to_csv(os.path.join(CARPETA_VALIDACION, f"Cuadro_Olofsson_{NOMBRE_MOSAICO}.csv"), index=False, encoding='utf-8-sig')
