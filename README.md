# Caracterización Automatizada de Paisajes Post-Minería Aluvial Amazónica mediante OBIA Multiescala y Ensambles de Machine Learning Acelerados por GPU

Sistema de clasificación supervisada de alta resolución espacial (~10 - 11 cm/píxel) para ortomosaicos RGB capturados con vehículos aéreos no tripulados (UAV/Drones). Combina un enfoque **OBIA (Object-Based Image Analysis) Multiescala**, un **Super-Ensamble Jerárquico de Machine Learning** (Random Forest + ExtraTrees + HistGradientBoosting), un **Sub-Motor Físico-Morfológico Especialista en Arenas y Cárcavas**, reglas topológicas de limpieza ecológica y validación estadística ponderada por área bajo el estándar de **Olofsson et al. (2014)**.

---

## 1. Leyenda Oficial de Coberturas (8 Clases)

| ID | Clase Oficial | Descripción Ecológica y Fisonómica en Ortomosaico RGB | Color Oficial (RGB / Hex) |
| :-: | :--- | :--- | :--- |
| **1** | **1. Bosque primario** | Dosel arbóreo denso, estratificado, emergentes de gran diámetro y heterogeneidad lumínica interna profunda (sombras inter-copa marcadas). | `(0, 100, 0)` `#006400` |
| **2** | **2. Bosque secundario** | Dosel arbóreo en transición sucesional avanzada, copas de tamaño medio más homogéneas y menor contraste de sombra profunda. | `(34, 139, 34)` `#228B22` |
| **3** | **3. Regeneracion positiva** | Colonización vegetal arbustiva/arbórea joven vigorosa (VDVI > 0.08), cobertura continua sobre sustrato en recuperación. | `(124, 252, 0)` `#7CFC00` |
| **4** | **4. Regeneracion negativa** | Vegetación rala, estresada, herbácea degradada o restos leñosos sobre sustrato perturbado con baja actividad fotosintética. | `(189, 183, 107)` `#BDB76B` |
| **5** | **5. Pozas mineras** | Cuerpos de agua lénticos artificiales originados por dragado/succión minera, con variabilidad de turbidez (sedimentos en suspensión o aguas claras). | `(0, 191, 255)` `#00BFFF` |
| **6** | **6. Rios Quebradas** | Cuerpos de agua lóticos naturales (cauces activos y drenajes secundarios). | `(0, 0, 205)` `#0000CD` |
| **7** | **7. Arena_Inundable/Carcaba** | Depósitos aluviales/relaves con saturación hídrica intersticial (arena húmeda/mojada de menor albedo) y áreas de erosión hídrica severa (cárcavas/arenas agrietadas). | `(210, 180, 140)` `#D2B48C` |
| **8** | **8. Arenales** | Acumulaciones de arena seca y grava expuesta de alta reflectancia, topografía lisa y nula retención superficial de humedad. | `(255, 215, 0)` `#FFD700` |

---

## 2. Metodología de Entrenamiento e Ingeniería de Características

### 2.1. Depuración Espacial de Muestras de Entrenamiento
1. **Consenso Inter-Versión (Filtrado de Contradicciones):** Se cruzan espacialmente las segmentaciones de entrenamiento (`Clasificacion_Oto.shp` y `Clasificacion_V2.shp`), conservando exclusivamente los polígonos con acuerdo verificado al 100% entre el centroide representativo y la capa oficial depurada.
2. **Erosión Interna de Bordes (*Negative Buffering*):** Se aplica un margen de seguridad interno de `-2.0 m` a cada polígono para extraer muestras únicamente del núcleo puro de cada cobertura, eliminando el efecto de píxeles mixtos en las fronteras de transición.
3. **Poda de Valores Atípicos (*Outlier Pruning* con Isolation Forest):** Dentro de cada clase se aplica un bosque de aislamiento (`contamination=0.03`) que descarta el 3% de observaciones anómalas (sombras extremas de nubes o artefactos de ortorrectificación).

### 2.2. Extracción Multiescala OBIA (51 Variables en 3 Niveles Espaciales)
Para superar la limitación espectral de las cámaras RGB (ausencia de banda NIR) y evitar el efecto "sal y pimienta" del píxel aislado, cada unidad de análisis se evalúa integrando **tres escalas concéntricas**:

* **Escala 1 — Objeto / Superpíxel (16x16 px en entrenamiento; ~0.90 x 0.90 m en inferencia — 32 variables):**
  * **Estadísticos de Banda y Percentiles de Copa:** Media, desviación estándar y percentiles internos P10 (sombra interna), P50 (mediana) y P90 (superficie iluminada) en los canales Rojo (R), Verde (G), Azul (B) y Brillo.
  * **Índices Espectrales en el Visible:** `VDVI` (Media y desviación estándar local), `NGRDI`, `VARI`, `ExG`, `ExR`, `CIVE`, `NDWI_RGB` y `RI` (Redness Index).
  * **Textura y Rugosidad Micro-Topográfica:** Coeficiente de variación (`Textura_CV`), rango dinámico interno luz-sombra (`P90 - P10`) y **Gradiente Espacial de Rugosidad**.
* **Escala 2 — Vecindad Media de Dosel (48x48 px ~ 5.4 x 5.4 m — 10 variables):**
  * Medias y desviaciones estándar espaciales de R, G, B, Brillo, VDVI y NDWI_RGB, más las diferencias contextuales (`Diff_VDVI_Med` y `Diff_Brillo_Med`) calculadas mediante convolución en GPU aplicando la Ley de la Varianza Total.
* **Escala 3 — Contexto Macro-Ecológico del Paisaje (128x128 px ~ 14.5 x 14.5 m — 9 variables):**
  * Estructura macro del paisaje circundante: media y desviación estándar de Brillo y VDVI, rango estructural de sombras emergentes (`Macro_Rango_Sombra_Luz`) y **fracciones territoriales de cobertura en el radio de 14.5 m**: porcentaje de matriz vegetal (`Macro_Veg_Pct`), porcentaje de sombra de dosel (`Macro_Sombra_Pct`), porcentaje de arena desnuda (`Macro_Arena_Pct`) y porcentaje de lámina de agua (`Macro_Agua_Pct`).

---

## 3. Arquitectura del Clasificador Jerárquico en 2 Etapas

### Etapa 1: Super-Ensamble Multiclase (8 Clases)
Combina tres familias de algoritmos complementarios mediante votación probabilística ponderada (*Soft Voting*, pesos `[2, 2, 3]`):
1. **Random Forest (`RandomForestClassifier`):** 700 árboles, profundidad máxima 32, balanceo por submuestra (`balanced_subsample`).
2. **Extremely Randomized Trees (`ExtraTreesClassifier`):** 700 árboles, profundidad máxima 35, particiones aleatorias que reducen la varianza en bordes complejos.
3. **Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier`):** 650 iteraciones, tasa de aprendizaje 0.045, regularización L2 = 1.2, optimizado para fronteras no lineales entre sucesiones vegetales.

### Etapa 2: Sub-Motor Físico-Morfológico Especialista en Arenas (Clase 7 vs. Clase 8)
Dado que la reflectancia absoluta de las arenas varía según la hora solar de cada vuelo de dron, todas las celdas preclasificadas en el estrato de arenas (Clases 7 y 8) pasan por un segundo ensamble especialista que opera sobre **18 variables físicas auto-calibradas por ortomosaico**:
1. **Calibración Radiométrica Adaptativa (Z-Score Robusto):** Normaliza el brillo P50, P10, P90 y el NDWI_RGB respecto a la mediana y rango intercuartílico (IQR) de las arenas del propio vuelo, detectando la caída del 25% - 40% en reflectancia causada por la humedad intersticial en la **arena inundable**.
2. **Detección Morfológica de Cárcavas (Arenas Agrietadas):** Combina un filtro morfológico **Black-Hat 2D** con el **Índice de Grieta** y la distancia euclidiana hidrológica a cuerpos de agua (Clases 5 y 6).

---

## 4. Post-Procesamiento Ecológico y Reglas Topológicas (OBIA)
1. **Suavizado Probabilístico Preservador de Bordes:** Filtro Gaussiano (sigma = 0.85) sobre los 8 mapas de probabilidad antes de la decisión final.
2. **Unidad Mínima Cartografiable (`Sieve` MMU):** Fusión de micro-fragmentos aislados menores a ~13 m2 hacia la cobertura dominante adyacente.
3. **Absorción Topológica de Islas (Envolvente Ecológica):**
   * *Palizadas y troncos flotantes en Pozas Mineras:* Islas internas < 350 m2 encerradas dentro de `5. Pozas mineras` se reasignan al cuerpo de agua.
   * *Ár
