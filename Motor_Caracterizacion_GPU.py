import os
import shutil
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import cupy as cp
from cupyx.scipy.ndimage import uniform_filter as gpu_uniform_filter
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes, sieve
import geopandas as gpd
from shapely.geometry import shape
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    label,
    distance_transform_edt,
    grey_closing,
    uniform_filter
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier

warnings.filterwarnings('ignore')

RUTA_BASE = r"F:\Clasificacion_Azul"
CARPETA_MOTOR = os.path.join(RUTA_BASE, "0_MOTOR_IA_CARACTERIZACION")
os.makedirs(CARPETA_MOTOR, exist_ok=True)

RUTA_MODELO_ORIGEN = os.path.join(RUTA_BASE, "1_MODELO_ENTRENADO_LOCAL", "Modelo_Ensamble_51Vars_Multiescala.pkl")
RUTA_CSV_ENTRENAMIENTO = os.path.join(RUTA_BASE, "1_MODELO_ENTRENADO_LOCAL", "Base_IslaOtorongo_51Vars_Multiescala.csv")

RUTA_MOTOR_BLINDADO = os.path.join(CARPETA_MOTOR, "Motor_Maestro_51Vars.pkl")
RUTA_ESPECIALISTA_ARENAS = os.path.join(CARPETA_MOTOR, "SubMotor_Especialista_Arenas_7vs8.pkl")

PALETA_COLORES = {
    0: (0, 0, 0, 0),
    1: (0, 100, 0, 255),
    2: (34, 139, 34, 255),
    3: (124, 252, 0, 255),
    4: (189, 183, 107, 255),
    5: (0, 191, 255, 255),
    6: (0, 0, 205, 255),
    7: (210, 180, 140, 255),
    8: (255, 215, 0, 255)
}

COLORES_HEX = {
    1: ("#006400", "1. Bosque primario"),
    2: ("#228B22", "2. Bosque secundario"),
    3: ("#7CFC00", "3. Regeneracion positiva"),
    4: ("#BDB76B", "4. Regeneracion negativa"),
    5: ("#00BFFF", "5. Pozas mineras"),
    6: ("#0000CD", "6. Rios Quebradas"),
    7: ("#D2B48C", "7. Arena_Inundable/Carcaba"),
    8: ("#FFD700", "8. Arenales")
}


def liberar_memoria_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


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


def inicializar_motor():
    if not os.path.exists(RUTA_MOTOR_BLINDADO):
        if os.path.exists(RUTA_MODELO_ORIGEN):
            shutil.copy2(RUTA_MODELO_ORIGEN, RUTA_MOTOR_BLINDADO)
        else:
            raise FileNotFoundError(f"No se encontro el modelo en: {RUTA_MODELO_ORIGEN}")

    pack = joblib.load(RUTA_MOTOR_BLINDADO)

    if not os.path.exists(RUTA_ESPECIALISTA_ARENAS) and os.path.exists(RUTA_CSV_ENTRENAMIENTO):
        df = pd.read_csv(RUTA_CSV_ENTRENAMIENTO)
        df_ar = df[df['Clase_ID'].isin([7, 8])].copy().reset_index(drop=True)

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

        esp_arena = VotingClassifier(
            estimators=[('rf', rf_a), ('et', et_a), ('hgb', hgb_a)],
            voting='soft', weights=[2, 2, 3], n_jobs=-1
        )
        esp_arena.fit(X_ar, y_ar)
        joblib.dump(esp_arena, RUTA_ESPECIALISTA_ARENAS)

    pack['especialista_arenas'] = joblib.load(RUTA_ESPECIALISTA_ARENAS) if os.path.exists(RUTA_ESPECIALISTA_ARENAS) else None
    return pack


def guardar_estilo_qml(ruta_qml):
    cats = [f'      <category symbol="{i-1}" value="{i}" label="{lbl}" render="true"/>' for i, (_, lbl) in COLORES_HEX.items()]
    syms = []
    for i, (hc, _) in COLORES_HEX.items():
        r_v, g_v, b_v = int(hc[1:3], 16), int(hc[3:5], 16), int(hc[5:7], 16)
        syms.append(
            f'      <symbol alpha="0.85" type="fill" name="{i-1}">\n'
            f'        <layer locked="0" pass="0" class="SimpleFill">\n'
            f'          <prop k="color" v="{r_v},{g_v},{b_v},255"/>\n'
            f'          <prop k="outline_color" v="{r_v},{g_v},{b_v},255"/>\n'
            f'          <prop k="outline_style" v="no"/>\n'
            f'          <prop k="style" v="solid"/>\n'
            f'        </layer>\n'
            f'      </symbol>'
        )
    contenido = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.28.0" styleCategories="Symbology">\n'
        '  <renderer-v2 attr="Clase_Pred" type="categorizedSymbol" forceraster="0" enableorderby="0" symbollevels="0">\n'
        "    <categories>\n" + "\n".join(cats) + "\n    </categories>\n"
        "    <symbols>\n" + "\n".join(syms) + "\n    </symbols>\n"
        "  </renderer-v2>\n"
        "</qgis>"
    )
    with open(ruta_qml, "w", encoding="utf-8") as f:
        f.write(contenido)


def caracterizar_mosaico(ruta_tif, carpeta_destino_base=None, tamano_celda_m=0.90):
    t0 = time.time()
    liberar_memoria_gpu()

    if not os.path.isabs(ruta_tif):
        ruta_tif = os.path.join(RUTA_BASE, ruta_tif)
    if not os.path.exists(ruta_tif):
        raise FileNotFoundError(f"No se encontro el ortomosaico: {ruta_tif}")

    nombre_limpio = os.path.splitext(os.path.basename(ruta_tif))[0]
    if carpeta_destino_base is None:
        carpeta_destino_base = os.path.join(RUTA_BASE, "RESULTADOS_CARACTERIZACION")

    carpeta_salida = os.path.join(carpeta_destino_base, f"Caracterizacion_{nombre_limpio}")
    os.makedirs(carpeta_salida, exist_ok=True)

    print("=" * 85)
    print(f" INICIANDO MOTOR DE CARACTERIZACION GPU (V2 ESPECIALISTA ARENAS): {os.path.basename(ruta_tif)}")
    print("=" * 85)

    pack = inicializar_motor()
    modelo = pack['modelo']
    esp_arenas = pack.get('especialista_arenas', None)
    NOMBRES_OFICIALES = pack['nombres_clases']

    with rasterio.open(ruta_tif) as src:
        crs_orig = src.crs
        crs_wkt = crs_orig.to_wkt() if crs_orig is not None else ""
        if "32719" in crs_wkt:
            crs_salida = rasterio.crs.CRS.from_epsg(32719)
        elif "32718" in crs_wkt:
            crs_salida = rasterio.crs.CRS.from_epsg(32718)
        else:
            crs_salida = crs_orig

        transform_orig = src.transform
        H_orig, W_orig = src.height, src.width

        es_geografico = crs_salida.is_geographic if crs_salida is not None else False
        res_x_unid = abs(transform_orig.a)
        res_m = res_x_unid * 111132.0 if es_geografico else res_x_unid

        paso_celda = int(np.clip(round(tamano_celda_m / max(res_m, 0.005)), 4, 32))
        lado_celda_m = paso_celda * res_m
        area_celda_m2 = lado_celda_m * lado_celda_m

        H_grid = H_orig // paso_celda
        W_grid = W_orig // paso_celda
        W_use = W_grid * paso_celda

        probs_grid_u8 = np.zeros((8, H_grid, W_grid), dtype=np.uint8)
        mascara_valida_grid = np.zeros((H_grid, W_grid), dtype=bool)

        br_p10_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        br_p50_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        br_p90_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        rug_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        tex_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        ndwi_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        crb_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        sat_grid = np.zeros((H_grid, W_grid), dtype=np.float16)
        ri_grid = np.zeros((H_grid, W_grid), dtype=np.float16)

        ALTO_FRANJA = 320
        HALO = 16
        SUB_LOTE_GPU = 64
        eps = 1e-6
        n_elem = paso_celda * paso_celda
        idx_p10 = int(0.10 * (n_elem - 1))
        idx_p50 = int(0.50 * (n_elem - 1))
        idx_p90 = int(0.90 * (n_elem - 1))

        total_franjas = (H_grid + ALTO_FRANJA - 1) // ALTO_FRANJA

        for idx_f, gy_ini in enumerate(range(0, H_grid, ALTO_FRANJA), 1):
            gy_fin = min(H_grid, gy_ini + ALTO_FRANJA)
            p_ini = max(0, gy_ini - HALO)
            p_fin = min(H_grid, gy_fin + HALO)
            h_pad = p_fin - p_ini

            feats_pad = np.zeros((h_pad, W_grid, 51), dtype=np.float32)
            mask_pad = np.zeros((h_pad, W_grid), dtype=bool)

            for s_ini in range(0, h_pad, SUB_LOTE_GPU):
                s_fin = min(h_pad, s_ini + SUB_LOTE_GPU)
                h_sub = s_fin - s_ini

                py_0 = (p_ini + s_ini) * paso_celda
                py_1 = (p_ini + s_fin) * paso_celda

                ventana = Window(0, py_0, W_use, py_1 - py_0)
                raw_u8 = src.read([1, 2, 3], window=ventana)

                sub_rgb = cp.asarray(raw_u8, dtype=cp.uint8).astype(cp.float32)
                del raw_u8

                bloques = sub_rgb.reshape(3, h_sub, paso_celda, W_grid, paso_celda).transpose(0, 1, 3, 2, 4)
                br_2d = (bloques[0] + bloques[1] + bloques[2]) / 3.0
                grad_x = cp.mean(cp.abs(cp.diff(br_2d, axis=3)), axis=(2, 3))
                grad_y = cp.mean(cp.abs(cp.diff(br_2d, axis=2)), axis=(2, 3))
                gradiente_rugosidad = (grad_x + grad_y) * 0.5

                bloques_flat = bloques.reshape(3, h_sub, W_grid, n_elem)
                r_b, g_b, b_b = bloques_flat[0], bloques_flat[1], bloques_flat[2]
                br_b = (r_b + g_b + b_b) / 3.0

                val_mask = cp.mean((br_b > 3.0) & (br_b < 252.0), axis=-1) > 0.80
                mask_pad[s_ini:s_fin, :] = cp.asnumpy(val_mask)

                b1_m, b2_m, b3_m = cp.mean(r_b, axis=-1), cp.mean(g_b, axis=-1), cp.mean(b_b, axis=-1)
                b1_s, b2_s, b3_s = cp.std(r_b, axis=-1), cp.std(g_b, axis=-1), cp.std(b_b, axis=-1)

                r_sort, g_sort = cp.sort(r_b, axis=-1), cp.sort(g_b, axis=-1)
                b_sort, br_sort = cp.sort(b_b, axis=-1), cp.sort(br_b, axis=-1)

                b1_p10, b1_p90 = r_sort[..., idx_p10], r_sort[..., idx_p90]
                b2_p10, b2_p90 = g_sort[..., idx_p10], g_sort[..., idx_p90]
                b3_p10, b3_p90 = b_sort[..., idx_p10], b_sort[..., idx_p90]
                br_p10, br_p50, br_p90 = br_sort[..., idx_p10], br_sort[..., idx_p50], br_sort[..., idx_p90]
                rango_luz_sombra = br_p90 - br_p10

                suma_rgb = b1_m + b2_m + b3_m + eps
                rn, gn, bn = b1_m / suma_rgb, b2_m / suma_rgb, b3_m / suma_rgb
                vdvi_pix = (2.0 * g_b - r_b - b_b) / (2.0 * g_b + r_b + b_b + eps)
                vdvi = cp.mean(vdvi_pix, axis=-1)
                vdvi_std = cp.std(vdvi_pix, axis=-1)

                ngrdi = (b2_m - b1_m) / (b2_m + b1_m + eps)
                exg = 2.0 * gn - rn - bn
                ndwi_rgb = (b2_m - b3_m) / (b2_m + b3_m + eps)
                ri = (b1_m - b2_m) / (b1_m + b2_m + eps)
                cive = 0.441 * rn - 0.811 * gn + 0.385 * bn + 18.78745

                brillo = cp.mean(br_b, axis=-1)
                max_rgb = cp.maximum(cp.maximum(b1_m, b2_m), b3_m)
                min_rgb = cp.minimum(cp.minimum(b1_m, b2_m), b3_m)
                saturacion = (max_rgb - min_rgb) / (max_rgb + eps)
                contraste_rb = b1_m - b3_m
                textura_cv = ((b1_s + b2_s + b3_s) / 3.0) / (brillo + 1.0)
                ratio_rg = b1_m / (b2_m + 1.0)
                ratio_gb = b2_m / (b3_m + 1.0)
                exr = 1.4 * rn - gn
                vari = (b2_m - b1_m) / (b2_m + b1_m - b3_m + eps)

                lista_obj = [
                    b1_m, b2_m, b3_m, b1_s, b2_s, b3_s,
                    b1_p10, b1_p90, b2_p10, b2_p90, b3_p10, b3_p90,
                    brillo, br_p10, br_p50, br_p90, rango_luz_sombra,
                    vdvi, vdvi_std, ngrdi, exg, ndwi_rgb, ri, cive,
                    saturacion, contraste_rb, textura_cv, gradiente_rugosidad,
                    ratio_rg, ratio_gb, exr, vari
                ]
                stack_obj = cp.stack(lista_obj, axis=-1)
                feats_pad[s_ini:s_fin, :, :32] = cp.asnumpy(stack_obj)
                del sub_rgb, bloques, bloques_flat, r_sort, g_sort, b_sort, br_sort, vdvi_pix, stack_obj
                liberar_memoria_gpu()

            b1_m_gpu = cp.asarray(feats_pad[:, :, 0])
            b2_m_gpu = cp.asarray(feats_pad[:, :, 1])
            b3_m_gpu = cp.asarray(feats_pad[:, :, 2])
            br_m_gpu = cp.asarray(feats_pad[:, :, 12])
            br_s_gpu = (cp.asarray(feats_pad[:, :, 3]) + cp.asarray(feats_pad[:, :, 4]) + cp.asarray(feats_pad[:, :, 5])) / 3.0
            br_p10_gpu = cp.asarray(feats_pad[:, :, 13])
            br_p90_gpu = cp.asarray(feats_pad[:, :, 15])
            vdvi_m_gpu = cp.asarray(feats_pad[:, :, 17])
            vdvi_s_gpu = cp.asarray(feats_pad[:, :, 18])
            ndwi_m_gpu = cp.asarray(feats_pad[:, :, 21])

            k_med, k_mac = 6, 16
            med_b1 = gpu_uniform_filter(b1_m_gpu, size=k_med)
            med_b2 = gpu_uniform_filter(b2_m_gpu, size=k_med)
            med_b3 = gpu_uniform_filter(b3_m_gpu, size=k_med)
            med_brillo = gpu_uniform_filter(br_m_gpu, size=k_med)
            med_std_brillo = cp.sqrt(cp.maximum(gpu_uniform_filter(br_s_gpu**2 + br_m_gpu**2, size=k_med) - med_brillo**2, 0.0))
            med_vdvi = gpu_uniform_filter(vdvi_m_gpu, size=k_med)
            med_std_vdvi = cp.sqrt(cp.maximum(gpu_uniform_filter(vdvi_s_gpu**2 + vdvi_m_gpu**2, size=k_med) - med_vdvi**2, 0.0))
            med_ndwi = gpu_uniform_filter(ndwi_m_gpu, size=k_med)
            diff_vdvi_med = vdvi_m_gpu - med_vdvi
            diff_brillo_med = br_m_gpu - med_brillo

            macro_brillo = gpu_uniform_filter(br_m_gpu, size=k_mac)
            macro_std_brillo = cp.sqrt(cp.maximum(gpu_uniform_filter(br_s_gpu**2 + br_m_gpu**2, size=k_mac) - macro_brillo**2, 0.0))
            macro_vdvi = gpu_uniform_filter(vdvi_m_gpu, size=k_mac)
            macro_std_vdvi = cp.sqrt(cp.maximum(gpu_uniform_filter(vdvi_s_gpu**2 + vdvi_m_gpu**2, size=k_mac) - macro_vdvi**2, 0.0))
            macro_rango_sombra_luz = gpu_uniform_filter(br_p90_gpu, size=k_mac) - gpu_uniform_filter(br_p10_gpu, size=k_mac)

            es_veg = (vdvi_m_gpu > 0.07).astype(cp.float32)
            es_sombra = (br_m_gpu < cp.maximum(macro_brillo * 0.68, 50.0)).astype(cp.float32)
            es_arena = ((br_m_gpu > 130.0) & (vdvi_m_gpu < 0.03)).astype(cp.float32)
            es_agua = ((ndwi_m_gpu > 0.01) & (vdvi_m_gpu < 0.04) & (br_m_gpu < 135.0)).astype(cp.float32)

            macro_veg_pct = gpu_uniform_filter(es_veg, size=k_mac)
            macro_sombra_pct = gpu_uniform_filter(es_sombra, size=k_mac)
            macro_arena_pct = gpu_uniform_filter(es_arena, size=k_mac)
            macro_agua_pct = gpu_uniform_filter(es_agua, size=k_mac)

            lista_ctx = [
                med_b1, med_b2, med_b3, med_brillo, med_std_brillo,
                med_vdvi, med_std_vdvi, med_ndwi, diff_vdvi_med, diff_brillo_med,
                macro_brillo, macro_std_brillo, macro_vdvi, macro_std_vdvi,
                macro_rango_sombra_luz, macro_veg_pct, macro_sombra_pct,
                macro_arena_pct, macro_agua_pct
            ]
            stack_ctx = cp.stack(lista_ctx, axis=-1)
            feats_pad[:, :, 32:] = cp.asnumpy(stack_ctx)
            del b1_m_gpu, b2_m_gpu, b3_m_gpu, br_m_gpu, br_s_gpu, br_p10_gpu, br_p90_gpu, vdvi_m_gpu, vdvi_s_gpu, ndwi_m_gpu, stack_ctx
            liberar_memoria_gpu()

            corte_ini = gy_ini - p_ini
            corte_fin = corte_ini + (gy_fin - gy_ini)
            feats_franja = feats_pad[corte_ini:corte_fin]
            mask_franja = mask_pad[corte_ini:corte_fin]

            mascara_valida_grid[gy_ini:gy_fin, :] = mask_franja

            br_p10_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 13].astype(np.float16)
            br_p50_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 14].astype(np.float16)
            br_p90_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 15].astype(np.float16)
            ndwi_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 21].astype(np.float16)
            ri_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 22].astype(np.float16)
            sat_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 24].astype(np.float16)
            crb_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 25].astype(np.float16)
            tex_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 26].astype(np.float16)
            rug_grid[gy_ini:gy_fin, :] = feats_franja[:, :, 27].astype(np.float16)

            if np.any(mask_franja):
                X_val = np.nan_to_num(feats_franja[mask_franja], nan=0.0, posinf=0.0, neginf=0.0)
                probs_val = np.zeros((len(X_val), 8), dtype=np.float32)
                for b_i in range(0, len(X_val), 250000):
                    probs_val[b_i:b_i + 250000] = modelo.predict_proba(X_val[b_i:b_i + 250000])

                probs_u8_val = np.clip(np.rint(probs_val * 255.0), 0, 255).astype(np.uint8)
                for c in range(8):
                    plano_c = np.zeros(mask_franja.shape, dtype=np.uint8)
                    plano_c[mask_franja] = probs_u8_val[:, c]
                    probs_grid_u8[c, gy_ini:gy_fin, :] = plano_c

            del feats_pad, mask_pad, feats_franja, mask_franja
            print(f"   [Franja {idx_f:02d}/{total_franjas:02d}] Procesada y clasificada ({gy_fin/H_grid*100:.1f}%)")

    max_prob = np.zeros((H_grid, W_grid), dtype=np.float32)
    mapa = np.zeros((H_grid, W_grid), dtype=np.uint8)

    for c in range(8):
        plano_suave = gaussian_filter(probs_grid_u8[c].astype(np.float32), sigma=0.85)
        mejor = plano_suave > max_prob
        max_prob[mejor] = plano_suave[mejor]
        mapa[mejor] = c + 1
        del plano_suave

    mapa[~mascara_valida_grid] = 0
    mask_estrato_arena = np.isin(mapa, [7, 8]) & mascara_valida_grid

    if np.sum(mask_estrato_arena) > 100 and esp_arenas is not None:
        br_50_f = br_p50_grid.astype(np.float32)
        br_10_f = br_p10_grid.astype(np.float32)
        br_90_f = br_p90_grid.astype(np.float32)

        med_br_f = uniform_filter(br_50_f, size=6)
        med_std_br_f = np.sqrt(np.maximum(uniform_filter(br_50_f**2, size=6) - med_br_f**2, 0.0))
        diff_br_med_f = br_50_f - med_br_f

        mac_br_f = uniform_filter(br_50_f, size=16)
        mac_std_br_f = np.sqrt(np.maximum(uniform_filter(br_50_f**2, size=16) - mac_br_f**2, 0.0))
        mac_rango_sl_f = uniform_filter(br_90_f - br_10_f, size=16)

        es_agua_map = np.isin(mapa, [5, 6]).astype(np.float32)
        es_arena_br_map = ((mapa == 8) & (br_50_f > np.median(br_50_f[mask_estrato_arena]))).astype(np.float32)
        mac_agua_f = uniform_filter(es_agua_map, size=16)
        mac_arena_f = uniform_filter(es_arena_br_map, size=16)

        cierre_br = grey_closing(br_10_f, size=(5, 5))
        profundidad_grieta_2d = (cierre_br - br_10_f) / (br_50_f + 5.0)
        dist_agua_m = distance_transform_edt(~np.isin(mapa, [5, 6])) * lado_celda_m

        X_esp_val = calcular_18_variables_fisicas_arena(
            br_50_f[mask_estrato_arena], br_10_f[mask_estrato_arena],
            br_50_f[mask_estrato_arena], br_90_f[mask_estrato_arena],
            rug_grid[mask_estrato_arena].astype(np.float32), tex_grid[mask_estrato_arena].astype(np.float32),
            ndwi_grid[mask_estrato_arena].astype(np.float32), crb_grid[mask_estrato_arena].astype(np.float32),
            sat_grid[mask_estrato_arena].astype(np.float32), ri_grid[mask_estrato_arena].astype(np.float32),
            med_br_f[mask_estrato_arena], med_std_br_f[mask_estrato_arena], diff_br_med_f[mask_estrato_arena],
            mac_br_f[mask_estrato_arena], mac_std_br_f[mask_estrato_arena], mac_rango_sl_f[mask_estrato_arena],
            mac_agua_f[mask_estrato_arena], mac_arena_f[mask_estrato_arena]
        )

        prob_esp_7 = esp_arenas.predict_proba(X_esp_val)[:, 0]
        p7_orig = probs_grid_u8[6, mask_estrato_arena].astype(np.float32)
        p8_orig = probs_grid_u8[7, mask_estrato_arena].astype(np.float32)
        prob_orig_7 = p7_orig / (p7_orig + p8_orig + 1.0)

        z_br_local = X_esp_val[:, 0]
        ind_carcava_local = X_esp_val[:, 11]
        grieta_2d_local = profundidad_grieta_2d[mask_estrato_arena]
        dist_agua_local = dist_agua_m[mask_estrato_arena]

        score_7 = 0.55 * prob_esp_7 + 0.45 * prob_orig_7
        es_carcava_agrietada = (grieta_2d_local > 0.14) & (ind_carcava_local > np.percentile(ind_carcava_local, 72))
        score_7 = np.where(es_carcava_agrietada, score_7 + 0.25, score_7)

        es_arena_mojada = (z_br_local < -0.65) | ((z_br_local < -0.25) & (dist_agua_local < 18.0))
        score_7 = np.where(es_arena_mojada, score_7 + 0.22, score_7)

        es_arenal_seco_puro = (z_br_local > 0.35) & (grieta_2d_local < 0.09) & (dist_agua_local > 8.0)
        score_7 = np.where(es_arenal_seco_puro, score_7 - 0.28, score_7)

        plano_score7 = np.zeros((H_grid, W_grid), dtype=np.float32)
        plano_score7[mask_estrato_arena] = np.clip(score_7, 0.0, 1.0)
        plano_score7 = gaussian_filter(plano_score7, sigma=0.9)
        mapa[mask_estrato_arena] = np.where(plano_score7[mask_estrato_arena] >= 0.50, 7, 8).astype(np.uint8)

    del probs_grid_u8, max_prob, br_p10_grid, br_p50_grid, br_p90_grid
    del rug_grid, tex_grid, ndwi_grid, crb_grid, sat_grid, ri_grid

    min_celdas_mmu = max(8, int(round(13.0 / max(area_celda_m2, 0.1))))
    mapa = sieve(mapa, size=min_celdas_mmu, connectivity=8, mask=mascara_valida_grid.astype(np.uint8))
    mapa[~mascara_valida_grid] = 0

    mask_pozas = (mapa == 5)
    islas_pozas = binary_fill_holes(mask_pozas) & (~mask_pozas) & (mapa > 0)
    lbl_p, n_p = label(islas_pozas)
    if n_p > 0:
        tam_p = np.bincount(lbl_p.ravel())
        peq_p = (tam_p < int(350.0 / max(area_celda_m2, 0.1)))
        peq_p[0] = False
        mapa[peq_p[lbl_p]] = 5

    mask_arenal = (mapa == 8)
    islas_troncos = binary_fill_holes(mask_arenal) & (~mask_arenal) & np.isin(mapa, [2, 3, 4])
    lbl_a, n_a = label(islas_troncos)
    if n_a > 0:
        tam_a = np.bincount(lbl_a.ravel())
        peq_a = (tam_a < int(280.0 / max(area_celda_m2, 0.1)))
        peq_a[0] = False
        mapa[peq_a[lbl_a]] = 8

    transform_grid = rasterio.Affine(
        transform_orig.a * paso_celda, transform_orig.b, transform_orig.c,
        transform_orig.d, transform_orig.e * paso_celda, transform_orig.f
    )

    ruta_tif_out = os.path.join(carpeta_salida, f"Mapa_{nombre_limpio}_Clasificado.tif")
    with rasterio.open(ruta_tif_out, 'w', driver='GTiff', height=H_grid, width=W_grid, count=1, dtype=rasterio.uint8, crs=crs_salida, transform=transform_grid, compress='lzw', nodata=0) as dst:
        dst.write(mapa, 1)
        dst.write_colormap(1, PALETA_COLORES)

    geometrias, clases_id = [], []
    tol_simp = res_x_unid * paso_celda * 0.45
    for geom_dict, val in shapes(mapa, mask=(mapa > 0), transform=transform_grid, connectivity=8):
        cid = int(val)
        if cid in NOMBRES_OFICIALES:
            pol = shape(geom_dict).simplify(tol_simp, preserve_topology=True)
            if not pol.is_empty and pol.area > 0:
                geometrias.append(pol)
                clases_id.append(cid)

    gdf_finos = gpd.GeoDataFrame({'Clase_Pred': clases_id, 'geometry': geometrias}, crs=crs_salida)
    gdf_finos['Class_name'] = gdf_finos['Clase_Pred'].map(NOMBRES_OFICIALES)

    gdf_metrico = gdf_finos.to_crs(epsg=32719) if es_geografico else gdf_finos
    gdf_finos['Area_m2'] = gdf_metrico.geometry.area.round(2)
    gdf_finos['Area_ha'] = (gdf_finos['Area_m2'] / 10000.0).round(4)
    gdf_finos['ID_Pol'] = np.arange(1, len(gdf_finos) + 1)
    gdf_finos = gdf_finos[['ID_Pol', 'Clase_Pred', 'Class_name', 'Area_m2', 'Area_ha', 'geometry']]

    ruta_shp_finos = os.path.join(carpeta_salida, f"Poligonos_Finos_{nombre_limpio}.shp")
    gdf_finos.to_file(ruta_shp_finos, encoding='utf-8')
    guardar_estilo_qml(os.path.splitext(ruta_shp_finos)[0] + ".qml")

    gdf_disuelto = gdf_finos.dissolve(by=['Clase_Pred', 'Class_name'], as_index=False)
    conteo_pols = gdf_finos.groupby('Clase_Pred').size().to_dict()
    areas_ha_cl = gdf_finos.groupby('Clase_Pred')['Area_ha'].sum().round(2).to_dict()
    gdf_disuelto['Cantidad_P'] = gdf_disuelto['Clase_Pred'].map(conteo_pols)
    gdf_disuelto['Area_ha'] = gdf_disuelto['Clase_Pred'].map(areas_ha_cl)
    total_ha = max(gdf_disuelto['Area_ha'].sum(), 1e-6)
    gdf_disuelto['Porcentaje'] = ((gdf_disuelto['Area_ha'] / total_ha) * 100.0).round(2)
    gdf_disuelto = gdf_disuelto[['Clase_Pred', 'Class_name', 'Cantidad_P', 'Area_ha', 'Porcentaje', 'geometry']]

    ruta_shp_dis = os.path.join(carpeta_salida, f"Mapa_Disuelto_8Clases_{nombre_limpio}.shp")
    gdf_disuelto.to_file(ruta_shp_dis, encoding='utf-8')
    guardar_estilo_qml(os.path.splitext(ruta_shp_dis)[0] + ".qml")

    ruta_csv = os.path.join(carpeta_salida, f"Resumen_Hectareas_{nombre_limpio}.csv")
    gdf_disuelto.drop(columns='geometry').to_csv(ruta_csv, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    inicializar_motor()
