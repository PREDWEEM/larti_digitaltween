# Procedencia científica del gemelo de Lartigau

El motor y sus activos provienen de [PREDWEEM/LOLIUM_LARTIGAU-2026](https://github.com/PREDWEEM/LOLIUM_LARTIGAU-2026/tree/afc421c52695a3a721e3327bcde2ef72ebaef9e0), revisión `afc421c52695a3a721e3327bcde2ef72ebaef9e0`.
El repositorio original se conserva sin modificaciones.

La interfaz, persistencia, asimilación, cobertura variable y calibración externa
se adaptaron del gemelo [Balcarce](https://github.com/PREDWEEM/balca_digitaltween/tree/66717c23cc0716a13deb541be3b9fba70f4784d3).
Los parámetros fisiológicos, pesos y meteorología de Lartigau se toman del
repositorio original indicado arriba.

## Activos originales sin cambios

| Archivo | SHA-256 |
|---|---|
| `models/IW.npy` | `8614f90cd5f1337ae746690e474587b6fb22cf81652e694573cbfe4573f406d5` |
| `models/LW.npy` | `13cb012d7f4fe8e9e8399b31226e160ed60690cd4fb225a2de40375d250eb97b` |
| `models/bias_IW.npy` | `69423ba136a4caad97bed6b3aae2e7a387d87851a32eb5b2ab8de74dbcae3788` |
| `models/bias_out.npy` | `53451c25cc92da6bff25404a1e47815e38dfe58f7d83bb87c2d471298ec8a12d` |
| `models/modelo_clusters_k3.pkl` | `29f0508543bdda4b2520038c678a9df80ff9be555e0c15d5fa1f4da16a499d30` |

## Motor conservado

Se extrajeron las funciones científicas de `app_emergencia_core.py`, junto con
el parche activo `modelo_decaimiento_15abril.py`, al módulo `predweem_twin/core.py`.
Las pruebas comparan la simulación del gemelo con una extracción independiente
de esas funciones en `tests/fixtures/lartigau_original.py`, incluyendo distintas
coberturas, Wmax y exponente Kr.

- ANN con día juliano, TMAX, TMIN y precipitación. La cobertura no altera sus entradas.
- Latitud −38,6166 y longitud −61,7000.
- Cobertura operativa 75 % y Wmax 18,816 mm.
- Latencia hasta JD 25; primer pico válido mayor que 0,20.
- Termoinhibición: media móvil de cinco días mayor o igual que 24 °C.
- Choque hídrico de tres días, 45 mm, hasta JD 110.
- ET0 Hargreaves, balance superficial y Kr predeterminado igual a cero.
- Desde el 15/04: techo inicial del 50 % del máximo de emergencia previo a esa fecha,
  con factor `(1 − I) + I × exp(−(días/τ)^β)`, τ=60 días, β=1 e I=0,75.
  Si no existe emergencia previa al 15/04, el original no impone un techo.
- Tiempo térmico triangular 2–20–30 °C; banda operativa 600–800 °Cd.

La normalización de una serie parcial utiliza las 11 campañas del clasificador
original que quedan al excluir 2010 y 2015. No existe en el archivo una campaña
identificada como Lartigau. Se informa explícitamente como referencia compartida.
La normalización parcial es una extensión del gemelo, distinta de la
normalización del período completo del modelo original.

## Meteorología y datos de campo

`meteo_daily.csv`, el histórico MeteoBahía y `update_meteo.py` proceden de la
misma revisión original. El histórico corresponde a pronósticos archivados de
Coronel Falcón, no a observaciones meteorológicas de estación. Se conserva el
cierre inclusivo del 01/10/2026.

La calibración utiliza una copia fija de la meteorología del 01/01 al 30/08/2026
(242 días), el archivo original aportado y su CSV de 15 fechas. La hoja sólo
identifica FECHA y PLM2; la localidad se asigna por indicación del usuario.
Los hashes y el origen se registran en `data/calibration/lartigau_2026_source.json`
y en el perfil `lartigau_2026.json`.

La capa de calibración transforma la salida normalizada; no cambia los pesos
neuronales ni el reloj térmico. Los resultados del ajuste y de la evaluación
temporal se conservan por separado. El perfil es experimental de una campaña.
