import os
from Motor_Caracterizacion_GPU import caracterizar_mosaico

RUTA_BASE = r"F:\Clasificacion_Azul"
CARPETA_SALIDA_GENERAL = os.path.join(RUTA_BASE, "RESULTADOS_CARACTERIZACION")

# Lista de ortomosaicos nuevos a clasificar (ruta relativa a F:\Clasificacion_Azul o ruta completa)
MOSAICOS_A_PROCESAR = [
    r"Imagen_Nueva\A4_Sep_2026.tif",
]

for ruta_item in MOSAICOS_A_PROCESAR:
    ruta_completa = ruta_item if os.path.isabs(ruta_item) else os.path.join(RUTA_BASE, ruta_item)

    if not os.path.exists(ruta_completa) and os.path.exists(ruta_completa + ".tif"):
        ruta_completa = ruta_completa + ".tif"
    elif not os.path.exists(ruta_completa) and os.path.exists(ruta_completa + ".TIF"):
        ruta_completa = ruta_completa + ".TIF"

    caracterizar_mosaico(
        ruta_tif=ruta_completa,
        carpeta_destino_base=CARPETA_SALIDA_GENERAL,
        tamano_celda_m=0.90
    )
