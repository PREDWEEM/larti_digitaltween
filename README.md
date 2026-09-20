# PREDWEEM Digital Twin · Lartigau

Gemelo digital de *Lolium multiflorum* basado en
[PREDWEEM/LOLIUM_LARTIGAU-2026](https://github.com/PREDWEEM/LOLIUM_LARTIGAU-2026).
Integra meteorología, observaciones por lote y calibración local 2026 externa
a la red neuronal. El repositorio original y sus pesos se conservan sin cambios.

**PREDWEEM by Guillermo R. Chantre.** Copyright © 2026 Guillermo R. Chantre /
PREDWEEM. Todos los derechos reservados. Consulte [COPYRIGHT.md](COPYRIGHT.md).

## Ejecutar

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud, seleccione `PREDWEEM/larti_digitaltween`,
rama `main` y archivo principal `app.py`.

## Funcionamiento

- **Estado del lote:** emergencia acumulada, flujo diario con barras azules,
  curva base, curva calibrada, estado actualizado y banda amarilla 600–800 °Cd.
  El eje horizontal muestra fechas calendario.
- **Calibración local 2026:** activada por defecto. El interruptor sobre el
  gráfico principal permite desactivarla y reactivarla. La selección persiste
  durante la sesión y actualiza estado, gráfico, escenarios y exportación.
- **Observaciones:** carga CSV/XLS/XLSX de flujos en plantas/m² o emergencia
  acumulada. Admite tres repeticiones y media por m² cuando están disponibles.
  Permite eliminar registros seleccionados del lote.
- **Cobertura:** constante o serie FECHA + COBERTURA_PCT (0–100 %), incluso en
  el mismo archivo de emergencia. Interpola entre mediciones y conserva el último
  valor; antes de la primera medición utiliza el respaldo.
- **Escenarios:** cambios exploratorios de meteorología y comparación con el
  estado del lote, sin modificar las observaciones guardadas.
- **Trazabilidad:** parámetros, procedencia meteorológica, asimilación y
  descarga de la trayectoria diaria auditable en CSV.

Observaciones y cobertura se guardan por lote en `data/twin_state.db`, que no
se versiona. En alojamientos con disco efímero, conserve los archivos originales
para recuperar los registros tras un reinicio o redespliegue.

## Motor de Lartigau

La configuración inicial reproduce el original: cobertura 75 %, Wmax 18,816 mm,
latitud −38,6166, longitud −61,7000, latencia JD 25, termoinhibición de cinco días
a 24 °C y primer pico mayor que 0,20.

Desde el 15 de abril aplica el techo del 50 % del máximo previo, con decaimiento
τ=60 días, β=1 e intensidad 0,75. Conserva el balance hídrico y el reloj térmico
2–20–30 °C. [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) registra la revisión
de origen, las ecuaciones y los hashes de los activos.

Para series meteorológicas parciales utiliza las nueve campañas del clasificador
original que quedan al excluir 2010, 2015, Balcarce y San Pedro. Esta referencia es compartida:
el archivo no contiene una campaña identificada como Lartigau. La calibración
local 2026 se aplica sobre esa trayectoria. El total emergido durante el período
muestreado no se supone igual al potencial estacional completo. Al cargar
conteos, el potencial se estima a partir de sus intervalos o se utiliza un
valor previo aportado por el usuario.

La selección conserva 2008, 2009, 2011, 2012, 2013, 2014, 2023 y 2024
(archivos identificados sólo por año), y Tres Arroyos 2025. No se atribuyen
todas estas series a la localidad del gemelo. Los nombres utilizados y
excluidos se muestran en Trazabilidad y en el perfil de calibración.
La referencia se recarga en cada ejecución para evitar curvas o columnas
obsoletas en la caché de Streamlit.

El perfil 2026 y sus diagnósticos se regeneraron con esta selección, conservando
los conteos, meteorología fija, fechas de corte, ANN y parámetros fisiológicos.
Aplicación, escenarios y ajuste utilizan los mismos filtros. El cálculo
conserva su anclaje a la mediana histórica; esta revisión modifica la selección
de referencias. Los resultados siguientes corresponden a las nueve curvas.

## Meteorología 2026

La fuente predeterminada es MeteoBahía para Coronel Falcón:

- Histórico: `METEOBAHIA_XML_ARCHIVADO / Historico_pronostico`.
  Son pronósticos archivados, **no observaciones de estación**.
- Tramo vigente: `METEOBAHIA_XML_CORONEL_FALCON / Pronostico`.
- `meteo_daily.csv` conserva Fuente, TipoDato, CalidadDato y Emision_UTC.

El gemelo recorta la trayectoria al estado histórico más siete días disponibles.
Al consultar un corte pasado, reconstruye los días posteriores con el archivo
actual; no constituye una evaluación con la emisión disponible en aquella fecha.

El workflow `update_meteo.yml` actualiza histórico y XML diariamente a las
08:00 de Argentina. El cierre meteorológico es inclusivo al **01/10/2026**.
Después no consulta nuevos pronósticos y conserva la campaña cerrada.
Open-Meteo y un archivo aportado son opciones adicionales explícitas en la
interfaz; no reemplazan la fuente predeterminada.

## Calibración local incorporada

Se utilizó `VALIDA (1) (4)(1).xlsx`, hoja `Hoja1`, columnas FECHA y PLM2:

| Dato | Valor |
|---|---|
| Fechas de muestreo | 15, del 01/02 al 30/08/2026 |
| Intervalos reales de ajuste | 14, de 5 a 21 días |
| Total registrado | 3.933,5 plantas/m² |
| Repeticiones | No informadas |
| Meteorología fija del ajuste | 242 días, 01/01–30/08/2026 |
| Transformación | `logistic(0.300 + 0.725 × logit(F))` |

El cero del 01/02 delimita el primer intervalo hasta el 19/02 (600 plantas/m²).
No se infiere ausencia de emergencia antes de ese registro. La calibración
compara conteos con sumas del modelo sobre cada intervalo real, sin interpolar
conteos diarios ni suponer intervalos semanales.

La transformación es monótona, conserva 0 y 1 y los días sin flujo. No modifica
los pesos ANN, el decaimiento original ni el tiempo térmico. Cobertura y Wmax
son los valores operativos originales; el adjunto no informa esos parámetros.
Sin repeticiones se utiliza un piso común de ponderación, no un error de
muestreo medido.

| Evaluación | RMSE base | RMSE calibrado |
|---|---:|---:|
| Ajuste retrospectivo, 14 intervalos | 306,76 | 269,05 |
| Evaluación temporal, 8 intervalos posteriores | 199,43 | 384,81 |

Los RMSE se expresan en plantas/m² por intervalo. El ajuste retrospectivo
reduce el error en 12,3 %, pero la evaluación temporal **empeora**: mejora 3 de
8 intervalos. El perfil se presenta como **experimental**, sin mejora predictiva
demostrada. Los cortes temporales utilizan sólo conteos hasta el corte; la
meteorología archivada no garantiza una emisión disponible en cada fecha.
Falta validar el pico principal y la transferencia entre campañas.

La calibración sólo se aplica al seleccionar Lartigau, desde el 30/08/2026 y
con el mismo motor y referencia del ajuste. Si se asimilan conteos de 2026,
utiliza la base para evitar reutilizar esa evidencia en calibración y asimilación.
El motivo de aplicación aparece junto al interruptor. El ajuste no reduce
automáticamente la incertidumbre.

Los conteos adjuntos se incluyen como referencia de calibración. Para asimilarlos
en un lote, descargue el CSV desde **Calibración por sitio** y cárguelo en
**Observaciones**. No se incorporan automáticamente a la base SQLite.

`data/calibration/` conserva el Excel original, conteos CSV, meteorología fija,
perfil JSON, resultados por intervalo y procedencia con hashes. La actualización
meteorológica diaria no altera esa copia fija.

Para reproducir el ajuste sin consultas de red:

```bash
python scripts/calibrate_site.py
```

## Verificación

```bash
python -m pytest -q
python -m compileall -q app.py predweem_twin scripts update_meteo.py
```

Las pruebas incluyen equivalencia con el motor original, decaimiento desde
15/04, serie parcial, integridad del adjunto, reproducción del perfil, prevención
de doble uso de conteos, asimilación, cobertura, almacenamiento, procedencia
meteorológica y cierre de campaña. Se ejecutan en GitHub Actions.
