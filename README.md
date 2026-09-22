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

## Visitas periódicas para reducir la hibernación

El workflow [mantener_activo.yml](.github/workflows/mantener_activo.yml) abre
[la aplicación de Lartigau](https://citqf36upunnthfnngj7sy.streamlit.app/)
con Chromium cada cuatro horas: 00:53, 04:53, 08:53, 12:53, 16:53 y 20:53 UTC
(01:53, 05:53, 09:53, 13:53, 17:53 y 21:53 de Argentina).
También permite ejecución manual desde
**Actions → Mantener activo el gemelo Lartigau → Run workflow** y se ejecuta
al modificar el workflow o su script.

Cuando aparece **Yes, get this app back up!**, la tarea hace clic en el botón
y espera la apertura, con un límite total de cinco minutos. Comprueba el
encabezado de Lartigau, el indicador de emergencia, el panel principal y su
gráfico, incluso si están dentro de un iframe. Una respuesta HTTP 200 por sí
sola no cuenta como éxito. Si la app muestra una excepción o no termina de
cargar, la ejecución queda fallida; los avisos dependen de las preferencias
de notificaciones de GitHub Actions.

La tarea no requiere secretos ni modifica observaciones o parámetros del modelo.
Playwright se instala solamente en el ejecutor de Actions. Esto reduce el riesgo
de hibernación, pero **no garantiza disponibilidad continua**:
[Streamlit suspende las apps sin visitas durante 12 horas](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation)
y [GitHub puede demorar tareas o desactivarlas tras 60 días sin actividad en un repositorio público](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
Si ocurre lo último, vuelva a habilitar el workflow desde Actions.

## Funcionamiento

- **Estado del lote:** gráficos de flujo y acumulado a la par, selector
  Semanal/Diario, fondo histórico orientativo y eje hasta el 1 de octubre.
  Mantiene las curvas base, calibrada y actualizada y la banda 600–800 °Cd.
- **Configuración:** integrada en el cuerpo mediante un desplegable, sin panel lateral.
- **Indicadores:** intensidad de emergencia a siete días y semáforo térmico.
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

Para series meteorológicas parciales, el acumulado base se ancla a la mediana
histórica en la fecha del estado. El total disponible al final del pronóstico
no se interpreta como el 100 % de la emergencia. Los conteos actualizan ese
estado y permiten estimar el potencial estacional por intervalos o utilizar
un valor previo aportado por el usuario.

El pool reúne **nueve referencias**: **2008, 2009, 2011, 2012, 2013, 2014,
2023, 2024 y Lartigau 2026**. Las primeras ocho son archivos identificados
sólo por año; esos nombres no acreditan que todas procedan de Lartigau.
Una lista explícita impide incorporar automáticamente otras curvas. Se excluyen
2010, 2015, Balcarce, San Pedro y Tres Arroyos 2025 **antes** de calcular
P10, mediana y P90. El clasificador original permanece intacto.

**Disponibilidad temporal:** el total de Lartigau 2026 se conoce desde el
último conteo del **30/08/2026**. Los cortes anteriores usan sólo las ocho
series previas. Desde esa fecha, incluidas las consultas de 2027, participan
las nueve referencias. La aplicación, los escenarios y cada corte de las
evaluaciones de calibración utilizan el mismo criterio.

Para 2026 se emplean los 15 registros de
`data/calibration/lartigau_2026_counts.csv`, del 01/02 al 30/08. Se suma
`PLM2`, se divide por el total registrado (**3933,5 plantas/m²**) y se
interpola el acumulado entre visitas. Cada una de las nueve campañas tiene
**igual peso**, independientemente de su densidad. No se promedia primero el
resumen de las ocho curvas con 2026 como dos grupos de igual peso.

Antes del 01/02, la curva 2026 queda desconocida y el resumen usa las ocho
curvas previas. Para evitar retrocesos al cambiar la composición disponible,
se conserva el máximo acumulado de cada cuantil. Las columnas `*_Empirico`
permiten auditar los cuantiles sin esa regularización. Después del último
conteo, 2026 mantiene el total de su ventana como supuesto de referencia:
no son nuevas observaciones ni prueba de cierre biológico de la campaña.
Los percentiles describen el pool; no son intervalos de confianza.

La interfaz y el perfil JSON registran campañas utilizadas y excluidas.
**Trazabilidad → Curvas de la referencia local** permite descargar el pool,
incluyendo las nueve curvas normalizadas y el número de campañas por día.
Incorporar 2026 al pool no carga esos conteos como observaciones de un lote.

Sin siete días futuros, el sistema conserva el estado disponible y muestra una
advertencia de horizonte incompleto; no presupone que la campaña terminó.

### Gráficos y configuración

La configuración aparece en el cuerpo, sin panel lateral. La vista principal
presenta **dos gráficos a la par**: flujo y emergencia acumulada, con eje
temporal del 1 de enero al **1 de octubre**. El selector **Semanal/Diario**
se inicia en Semanal. El fondo tenue muestra únicamente el **pool histórico
orientativo**, sin curvas individuales de años; continúa después de la fecha
del estado para visualizar la trayectoria de referencia del resto del período.
El gemelo sólo se extiende hasta la meteorología disponible, como máximo siete
días después del corte. El fondo histórico no crea meteorología ni pronósticos.

Los dos flujos se representan en **% del total por día o por semana**. Un 2 %
equivale a dos puntos porcentuales del acumulado. El histórico usa los totales
de sus ventanas registradas; el gemelo usa su total estacional estimado. El
flujo histórico se deriva de diferencias del acumulado mediano, no de conteos
diarios. La interpolación y la combinación de campañas suavizan sus picos.
Las semanas son de lunes a domingo, sin renormalizar; las barras parciales
aparecen rayadas y especifican sus días disponibles. El acumulado no cambia
al alternar la frecuencia. Los períodos sin referencia se mantienen desconocidos.

### Intensidad de emergencia a siete días

`Índice = flujo del gemelo de mañana a siete días después / máximo semanal del pool`.

El numerador suma siete flujos diarios futuros. El denominador utiliza el mismo
pool que los gráficos, trasladado al calendario consultado, y sólo semanas
completas de lunes a domingo dentro del eje enero–1 de octubre. No es el máximo
diario ni el máximo individual de una campaña. Ambas magnitudes se calculan
en la misma escala fraccional y se muestran como porcentajes.

| Intensidad | Condición |
| --- | --- |
| 🔴 Alta | Más del 75 % del máximo semanal histórico |
| 🟠 Media | Del 25 al 75 %, inclusive |
| 🟡 Baja | Flujo positivo y menor al 25 % del máximo |
| 🟢 Nula | Flujo semanal exactamente igual a cero |

Se requieren siete fechas futuras válidas: un horizonte incompleto se indica
en gris, sin clasificarlo como Bajo o Nulo. Con flujo positivo pero sin un
máximo histórico válido se muestra «Sin referencia». El intervalo futuro
móvil puede abarcar partes de dos semanas calendario. El selector del gráfico
no cambia este cálculo. Es intensidad relativa, no probabilidad de emergencia.

### Semáforo del tiempo térmico desde el primer pico

| Indicador | TT acumulado |
| --- | --- |
| 🔴 FUERA DE CONTROL | >800 °Cd |
| 🟠 ULTIMO PLAZO | >700 y ≤800 °Cd |
| 🟡 CONTROL A TIEMPO | ≥600 y ≤700 °Cd |
| 🟢 AUN NO CONTROLAR | <600 °Cd |

La categoría se determina con el valor sin redondear en la fecha del estado.
Se conservan los parámetros fisiológicos, pesos y meteorología de Lartigau;
se mantiene el techo y decaimiento propio desde el 15/04.


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
| Evaluación temporal, 8 intervalos posteriores | 199,43 | 447,54 |

Los RMSE se expresan en plantas/m² por intervalo. El ajuste retrospectivo
reduce el error en 12,3 %, pero la evaluación temporal **empeora**: mejora 3 de
8 intervalos. El perfil se presenta como **experimental**, sin mejora predictiva
demostrada. Los cortes temporales utilizan sólo conteos hasta el corte; la
meteorología archivada no garantiza una emisión disponible en cada fecha.
Falta validar el pico principal y la transferencia entre campañas.

La calibración sólo se aplica al seleccionar Lartigau, desde el 30/08/2026 y
con el mismo motor y referencia del ajuste. Si se asimilan conteos de 2026,
utiliza la base para evitar reutilizar esa evidencia en calibración y asimilación.
El motivo de aplicación aparece junto al interruptor. Desactivarlo conserva el
pool histórico: normalización estacional y calibración son capas distintas.
El ajuste no reduce automáticamente la incertidumbre.

El perfil y sus diagnósticos se regeneraron el 22/09/2026 con la nueva selección.
El ajuste final usa nueve referencias; cada evaluación anterior al 30/08 utiliza
sólo las ocho series previas, sin el total 2026 conocido después. La huella del
perfil incluye ahora el CSV 2026. Se conservan los conteos, la meteorología
fija, las fechas de corte y los parámetros del modelo.

El pool y los gráficos admiten consultas de 2027. La operación con meteorología
2027 requiere habilitar esa campaña y cargar su serie; el actualizador mantiene
el cierre configurado del 01/10/2026.

Los conteos adjuntos integran el pool histórico y la calibración. Para asimilarlos
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
de doble uso de conteos, composición y peso del pool, disponibilidad temporal,
conservación del flujo semanal, límites de los semáforos, asimilación, cobertura,
almacenamiento, procedencia
meteorológica y cierre de campaña. Se ejecutan en GitHub Actions.
