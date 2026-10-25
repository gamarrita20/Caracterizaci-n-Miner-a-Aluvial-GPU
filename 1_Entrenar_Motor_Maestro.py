import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point, box
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    IsolationForest
)
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

RUTA_BASE = r"F:\Clasificacion_Azul"
CARPETA_ENTRENAMIENTO = os.path.join(RUTA_BASE, "1_MODELO_ENTRENADO_LOCAL")
CARPETA_MOTOR = os.path.join(RUTA_BASE, "0_MOTOR_IA_CARACTERIZACION")
os.makedirs(CARPETA_ENTRENAMIENTO, exist_ok=True)
os.makedirs(CARPETA_MOTOR, exist_ok=True)

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

MAPA_HOMOLOGACION = {
    'bosque': 1, 'bosque primario': 1, 'bosque_primario': 1, '1. bosque primario': 1, '1': 1,
    'bosque secundario': 2, 'bosque_secundario': 2, '2. bosque secundario': 2, '2': 2,
    'regeneracion positiva': 3, 'regeneración positiva': 3, 'regeneracion_positiva': 3, '3. regeneracion positiva': 3, '3': 3,
    'regeneracion negativa': 4, 'regeneración negativa': 4, 'regeneracion_negativa': 4, '4. regeneracion negativa': 4, '4': 4,
    'pozas': 5, 'poza': 5, 'pozas mineras': 5, 'pozas_mineras': 5, '5. pozas mineras': 5, '5': 5,
    'rios': 6, 'rio': 6, 'ríos': 6, 'quebradas': 6, 'quebrada': 6,
    'rios quebradas': 6, 'ríos quebradas': 6, 'rios_quebradas': 6, '6. rios quebradas': 6, '6': 6,
    'playa': 7, 'playas': 7, 'arena_inundable/carcaba': 7, 'arena inundable': 7,
    'arenales inundables': 7, 'carcaba': 7, 'carcabas': 7, 'cárcava': 7, 'cárcavas': 7,
    '7. arena_inundable/carcaba': 7, '7': 7,
    'suelo desnudo': 8, 'suelo_desnudo': 8, 'arenales': 8, 'arenal': 8, '8. arenales': 8, '8': 8
}

CLASES_IGNORAR = {'cultivos', 'cultivo', 'deforestacion natural', 'deforestación natural', 'sindatos', 'sin datos', 'unclassified', 'none', 'nan'}

COLUMNAS_FEATURES = [
    'B1_Mean', 'B2_Mean', 'B3_Mean', 'B1_Std', 'B2_Std', 'B3_Std',
    'B1_P10', 'B1_P90', 'B2_P10', 'B2_P90', 'B3_P10', 'B3_P90',
    'Brillo', 'Brillo_P10', 'Brillo_P50', 'Brillo_P90', 'Rango_Luz_Sombra',
    'VDVI', 'VDVI_Std', 'NGRDI', 'ExG', 'NDWI_RGB', 'RI', 'CIVE',
    'Saturacion', 'Contraste_RB', 'Textura_CV', 'Gradiente_Rugosidad',
    'Ratio_RG', 'Ratio_GB', 'ExR', 'VARI',
    'Med_B1', 'Med_B2', 'Med_B3', 'Med_Brillo', 'Med_Std_Brillo',
    'Med_VDVI', 'Med_Std_VDVI', 'Med_NDWI', 'Diff_VDVI_Med', 'Diff_Brillo_Med',
    'Macro_Brillo', 'Macro_Std_Brillo', 'Macro_VDVI', 'Macro_Std_VDVI',
    'Macro_Rango_Sombra_Luz', 'Macro_Veg_Pct', 'Macro_Sombra_Pct',
    'Macro_Arena_Pct', 'Macro_Agua_Pct'
]


def extraer_51_variables_multiescala(patch_128):
    eps = 1e-6
    r_128, g_128, b_128 = patch_128[0], patch_128[1], patch_128[2]
    brillo_128 = (r_128 + g_128 + b_128) / 3.0

    if np.mean((brillo_128 > 3.0) & (brillo_128 < 252.0)) < 0.85:
        return None

    r_obj, g_obj, b_obj = r_128[56:72, 56:72], g_128[56:72, 56:72], b_128[56:72, 56:72]
    br_obj = brillo_128[56:72, 56:72]

    if np.mean(br_obj <= 3.0) > 0.05 or np.mean(br_obj >= 252.0) > 0.05:
        return None

    b1_m, b2_m, b3_m = float(np.mean(r_obj)), float(np.mean(g_obj)), float(np.mean(b_obj))
    b1_s, b2_s, b3_s = float(np.std(r_obj)), float(np.std(g_obj)), float(np.std(b_obj))

    b1_p10, b1_p90 = np.percentile(r_obj, [10, 90])
    b2_p10, b2_p90 = np.percentile(g_obj, [10, 90])
    b3_p10, b3_p90 = np.percentile(b_obj, [10, 90])
    br_p10, br_p50, br_p90 = np.percentile(br_obj, [10, 50, 90])
    rango_luz_sombra = float(br_p90 - br_p10)

    suma_rgb = b1_m + b2_m + b3_m + eps
    rn, gn, bn = b1_m / suma_rgb, b2_m / suma_rgb, b3_m / suma_rgb

    vdvi_map_obj = (2.0 * g_obj - r_obj - b_obj) / (2.0 * g_obj + r_obj + b_obj + eps)
    vdvi = float(np.mean(vdvi_map_obj))
    vdvi_std = float(np.std(vdvi_map_obj))

    ngrdi = (b2_m - b1_m) / (b2_m + b1_m + eps)
    exg = 2.0 * gn - rn - bn
    ndwi_rgb = (b2_m - b3_m) / (b2_m + b3_m + eps)
    ri = (b1_m - b2_m) / (b1_m + b2_m + eps)
    cive = 0.441 * rn - 0.811 * gn + 0.385 * bn + 18.78745

    brillo = float(np.mean(br_obj))
    max_rgb, min_rgb = max(b1_m, b2_m, b3_m), min(b1_m, b2_m, b3_m)
    saturacion = (max_rgb - min_rgb) / (max_rgb + eps)
    contraste_rb = b1_m - b3_m
    textura_cv = ((b1_s + b2_s + b3_s) / 3.0) / (brillo + 1.0)

    grad_x = np.abs(np.diff(br_obj, axis=1)).mean()
    grad_y = np.abs(np.diff(br_obj, axis=0)).mean()
    gradiente_rugosidad = float((grad_x + grad_y) * 0.5)

    ratio_rg = b1_m / (b2_m + 1.0)
    ratio_gb = b2_m / (b3_m + 1.0)
    exr = 1.4 * rn - gn
    vari = (b2_m - b1_m) / (b2_m + b1_m - b3_m + eps)

    r_med, g_med, b_med = r_128[40:88, 40:88], g_128[40:88, 40:88], b_128[40:88, 40:88]
    br_med = brillo_128[40:88, 40:88]

    med_b1, med_b2, med_b3 = float(np.mean(r_med)), float(np.mean(g_med)), float(np.mean(b_med))
    med_brillo = float(np.mean(br_med))
    med_std_brillo = float(np.std(br_med))

    vdvi_med_map = (2.0 * g_med - r_med - b_med) / (2.0 * g_med + r_med + b_med + eps)
    med_vdvi = float(np.mean(vdvi_med_map))
    med_std_vdvi = float(np.std(vdvi_med_map))
    med_ndwi = (med_b2 - med_b3) / (med_b2 + med_b3 + eps)

    diff_vdvi_med = vdvi - med_vdvi
    diff_brillo_med = brillo - med_brillo

    r_mac, g_mac, b_mac = r_128[::2, ::2], g_128[::2, ::2], b_128[::2, ::2]
    br_mac = brillo_128[::2, ::2]

    vdvi_mac_map = (2.0 * g_mac - r_mac - b_mac) / (2.0 * g_mac + r_mac + b_mac + eps)
    ndwi_mac_map = (g_mac - b_mac) / (g_mac + b_mac + eps)

    macro_brillo = float(np.mean(br_mac))
    macro_std_brillo = float(np.std(br_mac))
    macro_vdvi = float(np.mean(vdvi_mac_map))
    macro_std_vdvi = float(np.std(vdvi_mac_map))

    mac_p10, mac_p90 = np.percentile(br_mac, [10, 90])
    macro_rango_sombra_luz = float(mac_p90 - mac_p10)

    macro_veg_pct = float(np.mean(vdvi_mac_map > 0.07))
    macro_sombra_pct = float(np.mean(br_mac < max(macro_brillo * 0.68, 50.0)))
    macro_arena_pct = float(np.mean((br_mac > 130.0) & (vdvi_mac_map < 0.03)))
    macro_agua_pct = float(np.mean((ndwi_mac_map > 0.01) & (vdvi_mac_map < 0.04) & (br_mac < 135.0)))

    return [
        b1_m, b2_m, b3_m, b1_s, b2_s, b3_s,
        float(b1_p10), float(b1_p90), float(b2_p10), float(b2_p90), float(b3_p10), float(b3_p90),
        brillo, float(br_p10), float(br_p50), float(br_p90), rango_luz_sombra,
        vdvi, vdvi_std, ngrdi, exg, ndwi_rgb, ri, cive,
        saturacion, contraste_rb, textura_cv, gradiente_rugosidad,
        ratio_rg, ratio_gb, exr, vari,
        med_b1, med_b2, med_b3, med_brillo, med_std_brillo,
        med_vdvi, med_std_vdvi, med_ndwi, diff_vdvi_med, diff_brillo_med,
        macro_brillo, macro_std_brillo, macro_vdvi, macro_std_vdvi,
        macro_rango_sombra_luz, macro_veg_pct, macro_sombra_pct,
        macro_arena_pct, macro_agua_pct
    ]


def calcular_18_variables_fisicas_arena(
    brillo, br_p10, br_p50, br_p90, grad_rug, tex_cv,
    ndwi, contr_rb, sat, ri,
    med_br, med_std_br, diff_br_med,
    mac_br, mac_std_br, mac_rango_sl, mac_agua, mac_arena
):
    def norm_robusta(arr):
        med = np.median(arr)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        escala = max(float(iqr), float(np.std(arr)), 1e-4)
        return np.clip((arr - med) / escala, -4.5, 4.5)

    z_br_p50 = norm_robusta(br_p50)
    z_br_p10 = norm_robusta(br_p10)
    z_br_p90 = norm_robusta(br_p90)
    z_ndwi = norm_robusta(ndwi)
    z_contr_rb = norm_robusta(contr_rb)
    z_sat = norm_robusta(sat)
    z_ri = norm_robusta(ri)

    depresion_med = diff_br_med / (med_br + 5.0)
    depresion_mac = (brillo - mac_br) / (mac_br + 5.0)

    rango_rel_grieta = (br_p90 - br_p10) / (br_p50 + 5.0)
    rugosidad_rel = grad_rug / (br_p50 + 5.0)
    indice_carcava = rango_rel_grieta * rugosidad_rel * 10.0
    asimetria_grieta = (br_p50 - br_p10) / (br_p90 - br_p10 + 1.0)

    rugosidad_med = med_std_br / (med_br + 5.0)
    rugosidad_mac = mac_std_br / (mac_br + 5.0)
    rango_sombra_mac = mac_rango_sl / (mac_br + 5.0)

    X_esp = np.column_stack([
        z_br_p50, z_br_p10, z_br_p90,
        z_ndwi, z_contr_rb, z_sat, z_ri,
        depresion_med, depresion_mac,
        rango_rel_grieta, rugosidad_rel, indice_carcava,
        asimetria_grieta, tex_cv,
        rugosidad_med, rugosidad_mac, rango_sombra_mac,
        mac_agua - mac_arena
    ]).astype(np.float32)

    return np.nan_to_num(X_esp, nan=0.0, posinf=0.0, neginf=0.0)


def homologar_gdf(ruta_shp):
    gdf = gpd.read_file(ruta_shp)
    col_clase = None
    for cand in ['Class_name', 'Clase', 'Class', 'Nombre', 'Tipo', 'Layer']:
        for c_real in gdf.columns:
            if c_real.lower() == cand.lower():
                col_clase = c_real
                break
        if col_clase:
            break
    clases_cod = []
    for val in gdf[col_clase].astype(str).str.strip():
        v_low = val.lower()
        clases_cod.append(None if v_low in CLASES_IGNORAR else MAPA_HOMOLOGACION.get(v_low, None))
    gdf['Clase_ID'] = clases_cod
    gdf = gdf.dropna(subset=['Clase_ID', 'geometry']).copy()
    gdf['Clase_ID'] = gdf['Clase_ID'].astype(int)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf[['Clase_ID', 'geometry']]


if __name__ == "__main__":
    t0 = time.time()
    ruta_csv = os.path.join(CARPETA_ENTRENAMIENTO, "Base_IslaOtorongo_51Vars_Multiescala.csv")

    if os.path.exists(ruta_csv):
        df_final = pd.read_csv(ruta_csv)
    else:
        ruta_tif_isla = os.path.join(RUTA_BASE, "Isla_130826.tif")
        ruta_v2 = os.path.join(RUTA_BASE, "Class_Oto", "Clasificacion_V2.shp")
        ruta_v1 = os.path.join(RUTA_BASE, "Class_Oto", "Clasificacion_Oto.shp")

        with rasterio.open(ruta_tif_isla) as src:
            crs_tif = src.crs
            transform_tif = src.transform
            H_tif, W_tif = src.height, src.width
            bbox_tif = box(*src.bounds)
            img_rgb_ram = src.read([1, 2, 3])

        gdf_v2 = homologar_gdf(ruta_v2).to_crs(crs_tif)
        if os.path.exists(ruta_v1):
            gdf_v1 = homologar_gdf(ruta_v1).to_crs(crs_tif)
            pts_v1 = gdf_v1.copy()
            pts_v1['geometry'] = pts_v1.geometry.representative_point()
            cruce = gpd.sjoin(pts_v1, gdf_v2, how='inner', predicate='within')
            idx_acuerdo = cruce[cruce['Clase_ID_left'] == cruce['Clase_ID_right']].index
            gdf_limpio = gdf_v1.loc[idx_acuerdo].copy()
        else:
            gdf_limpio = gdf_v2.copy()

        gdf_limpio = gdf_limpio[gdf_limpio.geometry.intersects(bbox_tif)].reset_index(drop=True)
        gdf_limpio['area_m2'] = gdf_limpio.geometry.area
        areas_por_clase = gdf_limpio.groupby('Clase_ID')['area_m2'].sum().to_dict()

        registros, celdas_ocupadas = [], set()
        rng = np.random.default_rng(42)

        for _, row in gdf_limpio.iterrows():
            cid = int(row['Clase_ID'])
            geom = row.geometry
            area_pol = float(row['area_m2'])
            if area_pol < 6.0:
                continue

            geom_interna = geom.buffer(-2.0)
            if geom_interna.is_empty or geom_interna.area < 4.0:
                geom_interna = geom.buffer(-0.6)
            if geom_interna.is_empty:
                geom_interna = geom

            densidad_m2 = max(areas_por_clase.get(cid, 10000.0) / 4800.0, 18.0)
            n_pts = int(np.clip(round(area_pol / densidad_m2), 1, 90))

            puntos = [geom_interna.representative_point()]
            if n_pts > 1:
                minx, miny, maxx, maxy = geom_interna.bounds
                intentos = 0
                while len(puntos) < n_pts and intentos < n_pts * 6:
                    intentos += 1
                    p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                    if geom_interna.contains(p):
                        puntos.append(p)

            for pt in puntos:
                col_px, row_py = ~transform_tif * (pt.x, pt.y)
                px, py = int(col_px), int(row_py)
                if py < 64 or px < 64 or py >= H_tif - 64 or px >= W_tif - 64:
                    continue
                clave_celda = (py // 10, px // 10)
                if clave_celda in celdas_ocupadas:
                    continue
                celdas_ocupadas.add(clave_celda)

                patch = img_rgb_ram[:, py - 64 : py + 64, px - 64 : px + 64].astype(np.float32)
                feats = extraer_51_variables_multiescala(patch)
                if feats is not None:
                    registros.append(feats + [cid, NOMBRES_OFICIALES[cid]])

        del img_rgb_ram
        df_real = pd.DataFrame(registros, columns=COLUMNAS_FEATURES + ['Clase_ID', 'Class_name'])

        dfs_limpios = []
        for cid in sorted(df_real['Clase_ID'].unique()):
            sub = df_real[df_real['Clase_ID'] == cid].copy()
            if len(sub) > 60:
                iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
                sub = sub[iso.fit_predict(sub[COLUMNAS_FEATURES]) == 1]
            dfs_limpios.append(sub)

        df_final = pd.concat(dfs_limpios, ignore_index=True)
        df_final.to_csv(ruta_csv, index=False, encoding='utf-8-sig')

    X = df_final[COLUMNAS_FEATURES].values.astype(np.float32)
    y = df_final['Clase_ID'].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    clf_rf = RandomForestClassifier(n_estimators=700, max_depth=32, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    clf_et = ExtraTreesClassifier(n_estimators=700, max_depth=35, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    clf_hgb = HistGradientBoostingClassifier(max_iter=650, learning_rate=0.045, max_leaf_nodes=95, min_samples_leaf=10, l2_regularization=1.2, class_weight='balanced', random_state=42)

    ensamble = VotingClassifier(estimators=[('rf', clf_rf), ('et', clf_et), ('hgb', clf_hgb)], voting='soft', weights=[2, 2, 3], n_jobs=-1)
    ensamble.fit(X_train, y_train)

    y_pred = ensamble.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    nombres_ord = [NOMBRES_OFICIALES[i] for i in sorted(NOMBRES_OFICIALES.keys())]
    print(f"\nPRECISIÓN VALIDACIÓN ETAPA 1 (51 VARIABLES): {acc:.2f}%")
    print(classification_report(y_test, y_pred, target_names=nombres_ord, digits=3))

    ruta_motor_1 = os.path.join(CARPETA_MOTOR, "Motor_Maestro_51Vars.pkl")
    joblib.dump({'modelo': ensamble, 'columnas': COLUMNAS_FEATURES, 'nombres_clases': NOMBRES_OFICIALES, 'accuracy_validacion': acc}, ruta_motor_1)

    df_ar = df_final[df_final['Clase_ID'].isin([7, 8])].copy().reset_index(drop=True)
    X_ar = calcular_18_variables_fisicas_arena(
        df_ar['Brillo'].values, df_ar['Brillo_P10'].values, df_ar['Brillo_P50'].values, df_ar['Brillo_P90'].values,
        df_ar['Gradiente_Rugosidad'].values, df_ar['Textura_CV'].values,
        df_ar['NDWI_RGB'].values, df_ar['Contraste_RB'].values, df_ar['Saturacion'].values, df_ar['RI'].values,
        df_ar['Med_Brillo'].values, df_ar['Med_Std_Brillo'].values, df_ar['Diff_Brillo_Med'].values,
        df_ar['Macro_Brillo'].values, df_ar['Macro_Std_Brillo'].values, df_ar['Macro_Rango_Sombra_Luz'].values,
        df_ar['Macro_Agua_Pct'].values, df_ar['Macro_Arena_Pct'].values
    )
    y_ar = (df_ar['Clase_ID'].values == 8).astype(int)

    rf_a = RandomForestClassifier(n_estimators=500, max_depth=22, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    et_a = ExtraTreesClassifier(n_estimators=500, max_depth=25, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    hgb_a = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, class_weight='balanced', random_state=42)

    esp_arena = VotingClassifier(estimators=[('rf', rf_a), ('et', et_a), ('hgb', hgb_a)], voting='soft', weights=[2, 2, 3], n_jobs=-1)
    esp_arena.fit(X_ar, y_ar)

    ruta_motor_2 = os.path.join(CARPETA_MOTOR, "SubMotor_Especialista_Arenas_7vs8.pkl")
    joblib.dump(esp_arena, ruta_motor_2)
    print(f"-> ¡Motores guardados exitosamente en {CARPETA_MOTOR} en {(time.time() - t0)/60:.1f} min!")
