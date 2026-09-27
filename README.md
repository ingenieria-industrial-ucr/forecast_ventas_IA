# Guía didáctica: Forecast de ventas con IA y aplicación en Tkinter

2026-09-27 · @Mauricio Andrés Zamora Hernández

## Introducción

Al terminar esta guía, cada estudiante habrá creado, entrenado y desplegado un modelo de IA que pronostica ventas mensuales por producto, cliente y zona, dentro de una aplicación de escritorio en Python con Tkinter.

**Objetivos de aprendizaje**

- Entender qué es un modelo de aprendizaje automático (Machine Learning) y en qué se diferencia de un método clásico de pronóstico como la media móvil.
- Preparar datos de ventas para que un modelo los pueda "leer": variables de rezago, calendario y codificación de clientes y zonas.
- Entrenar, evaluar y comparar modelos con métricas de ingeniería (MAE, RMSE, MAPE).
- Guardar el modelo entrenado y usarlo dentro de una aplicación con carga y exportación a Excel y gráficos en Seaborn.

**¿Qué es un modelo de IA, en palabras simples?** Es una función matemática que aprende patrones a partir de ejemplos del pasado. Le mostramos miles de filas del tipo "en marzo, el cliente X de la zona Norte compró 120 unidades del producto A, y el mes anterior había comprado 110", y el modelo descubre por sí mismo las reglas que relacionan esas entradas con la venta. Después le damos datos nuevos y nos devuelve una predicción.

**El ciclo de trabajo** tiene tres fases que recorreremos en orden: crear (datos y variables), entrenar (ajustar y evaluar el modelo) y desplegar (ponerlo a disposición de un usuario final a través de la aplicación Tkinter).

**Modelo elegido.** Usaremos un **Random Forest Regressor** de scikit-learn y lo compararemos con un **Gradient Boosting** y con una línea base ingenua. Ambos son modelos de árboles de decisión: funcionan bien con datos tabulares, no exigen escalar variables, se entrenan en segundos en cualquier portátil y permiten ver qué variables pesan más, lo que facilita explicarlos en clase.

**Duración sugerida:** 4 sesiones de 2 horas (Pasos 0–2, Pasos 3–4, Paso 5, Pasos 6–7).

## Paso 0: Preparar el entorno de trabajo

Necesitamos Python 3.10 o superior, un editor (VS Code recomendado) y ocho librerías. Todo el proyecto cabe en una carpeta.

**0.1 Instalar Python.** Descargar desde python.org. En Windows, marcar la casilla *Add Python to PATH* durante la instalación. Tkinter viene incluido en Windows y macOS; en Linux se instala con `sudo apt install python3-tk`.

**0.2 Crear la carpeta del proyecto y un entorno virtual.** Un entorno virtual es una "caja" aislada donde instalamos las librerías de este proyecto sin afectar a otros. Abrir una terminal (en VS Code: *Terminal → New Terminal*):

```bash
mkdir forecast_ia
cd forecast_ia
python -m venv venv

# Activar el entorno
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

Cuando el entorno está activo, la terminal muestra `(venv)` al inicio de la línea.

**0.3 Instalar las librerías.** Crear un archivo `requirements.txt` con este contenido:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
seaborn>=0.13
matplotlib>=3.7
openpyxl>=3.1
joblib>=1.3
pillow>=10.0
```

y ejecutar:

```bash
pip install -r requirements.txt
```

| Librería | Para qué la usamos |
| --- | --- |
| pandas | Leer, limpiar y transformar tablas (CSV y Excel) |
| numpy | Cálculos numéricos |
| scikit-learn | Crear, entrenar y evaluar los modelos de IA |
| seaborn / matplotlib | Gráficas estadísticas |
| openpyxl | Leer y escribir archivos Excel .xlsx |
| joblib | Guardar y cargar el modelo entrenado |
| pillow | Insertar la gráfica como imagen dentro del Excel exportado |

**0.4 Estructura final del proyecto.** Al terminar la guía, la carpeta quedará así:

```text
forecast_ia/
├── requirements.txt
├── generar_dataset.py     ← Paso 1: crea los datos de ejemplo
├── eda.py                 ← Paso 2: análisis exploratorio
├── motor_ia.py            ← Pasos 3-5: el "cerebro" de IA (funciones reutilizables)
├── entrenar_modelo.py     ← Pasos 4-5: entrena, evalúa y guarda el modelo
├── app_pronostico.py      ← Paso 6: la aplicación Tkinter (deployment)
├── datos/                 ← ventas_historicas.csv y .xlsx
├── graficas/              ← imágenes PNG generadas
├── modelos/               ← modelo_ventas.pkl
└── resultados/            ← métricas y pronósticos en Excel
```

**Comprobación:** ejecutar `python -c "import sklearn, seaborn, tkinter; print('Todo listo')"`. Si aparece *Todo listo*, se puede continuar.

## Paso 1: El dataset de ventas

El dataset tiene 1.728 filas: 36 meses (enero 2023 a diciembre 2025) × 4 productos × 12 clientes repartidos en 4 zonas. Cada fila es la venta mensual de un producto a un cliente.

**1.1 Diccionario de datos**

| Columna | Tipo | Descripción | Ejemplo |
| --- | --- | --- | --- |
| fecha | fecha | Primer día del mes | 2023-01-01 |
| anio, mes | entero | Año y mes de la venta | 2023, 1 |
| producto | texto | Filtro de aire, Rodamiento, Válvula de control, Correa de transmisión | Rodamiento |
| cliente | texto | 12 clientes, código C01 a C12 | C07 Minera del Sur |
| tipo\_cliente | texto | Mayorista, Minorista o Distribuidor | Mayorista |
| zona | texto | Norte, Centro, Sur, Occidente | Sur |
| precio\_unitario | decimal | Precio en $, sube \~3 % al año | 40.00 |
| promocion | 0/1 | 1 si hubo promoción ese mes | 0 |
| unidades | entero | **Variable objetivo**: lo que queremos predecir | 145 |
| ventas\_total | decimal | unidades × precio\_unitario | 5800.00 |

**1.2 Patrones que "escondimos" en los datos.** Los datos son sintéticos a propósito: sabemos qué debe encontrar el modelo y podemos verificar si lo aprende.

| Patrón | Cómo se construyó |
| --- | --- |
| Tendencia | Cada producto crece entre 4 % y 12 % anual |
| Estacionalidad | Cada producto tiene un mes pico: Rodamiento en marzo, Filtro de aire en junio, Válvula en octubre, Correa en diciembre |
| Efecto zona | Norte ×1,20; Centro ×1,00; Sur ×0,85; Occidente ×0,70 |
| Efecto cliente | Tamaño entre ×0,5 (minoristas pequeños) y ×1,6 (grandes mayoristas) |
| Promoción | 10 % de los meses, aumenta las ventas un 25 % |
| Ruido | Variación aleatoria de ±10 %, como en la vida real |

**1.3 Crear el archivo `generar_dataset.py`** y ejecutarlo con `python generar_dataset.py`:

```python
"""
generar_dataset.py
Genera un dataset sintético de ventas mensuales (ene-2023 a dic-2025)
de 4 productos industriales, vendidos a 12 clientes en 4 zonas geográficas.

Ejecutar:  python generar_dataset.py
Salida:    datos/ventas_historicas.csv  y  datos/ventas_historicas.xlsx
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)  # Semilla: todos obtendrán exactamente los mismos datos

# 1. Catálogo de productos: precio, venta base mensual, crecimiento anual,
#    amplitud estacional y mes de mayor venta.
productos = {
    "Filtro de aire":        {"precio": 25.0, "base": 120, "crec": 0.10, "amp": 0.25, "pico": 6},
    "Rodamiento":            {"precio": 40.0, "base": 90,  "crec": 0.06, "amp": 0.10, "pico": 3},
    "Válvula de control":    {"precio": 85.0, "base": 45,  "crec": 0.12, "amp": 0.15, "pico": 10},
    "Correa de transmisión": {"precio": 18.0, "base": 150, "crec": 0.04, "amp": 0.20, "pico": 12},
}

# 2. Zonas geográficas con un factor de tamaño de mercado.
zonas = {"Norte": 1.20, "Centro": 1.00, "Sur": 0.85, "Occidente": 0.70}

# 3. Clientes: cada uno pertenece a una zona y tiene un tipo y un tamaño.
clientes = [
    ("C01 Metalúrgica Andina", "Norte", "Mayorista", 1.6),
    ("C02 Ensambles del Norte", "Norte", "Minorista", 0.7),
    ("C03 Agroindustrias Río", "Norte", "Distribuidor", 1.1),
    ("C04 Plásticos Centrales", "Centro", "Mayorista", 1.4),
    ("C05 Talleres Unidos", "Centro", "Minorista", 0.6),
    ("C06 Logística Capital", "Centro", "Distribuidor", 1.0),
    ("C07 Minera del Sur", "Sur", "Mayorista", 1.5),
    ("C08 Ferretería Austral", "Sur", "Minorista", 0.5),
    ("C09 Alimentos Patagonia", "Sur", "Distribuidor", 0.9),
    ("C10 Textiles Pacífico", "Occidente", "Mayorista", 1.3),
    ("C11 Mantenimiento Costa", "Occidente", "Minorista", 0.6),
    ("C12 Energía Occidental", "Occidente", "Distribuidor", 1.0),
]

fechas = pd.date_range("2023-01-01", "2025-12-01", freq="MS")  # 36 meses

filas = []
for i, fecha in enumerate(fechas):
    anios = i / 12
    for prod, p in productos.items():
        # Estacionalidad: una onda coseno con máximo en el mes "pico"
        estacional = 1 + p["amp"] * np.cos(2 * np.pi * (fecha.month - p["pico"]) / 12)
        tendencia = (1 + p["crec"]) ** anios
        # El precio sube ~3 % al año (inflación)
        precio = round(p["precio"] * (1.03 ** anios), 2)
        for cliente, zona, tipo, tamano in clientes:
            promocion = int(np.random.rand() < 0.10)        # 10 % de los meses
            efecto_promo = 1.25 if promocion else 1.0
            ruido = np.random.normal(1, 0.10)
            unidades = p["base"] * tendencia * estacional * zonas[zona] * tamano \
                       * efecto_promo * ruido
            unidades = max(0, int(round(unidades)))
            filas.append({
                "fecha": fecha.strftime("%Y-%m-%d"),
                "anio": fecha.year,
                "mes": fecha.month,
                "producto": prod,
                "cliente": cliente,
                "tipo_cliente": tipo,
                "zona": zona,
                "precio_unitario": precio,
                "promocion": promocion,
                "unidades": unidades,
                "ventas_total": round(unidades * precio, 2),
            })

df = pd.DataFrame(filas)
os.makedirs("datos", exist_ok=True)
df.to_csv("datos/ventas_historicas.csv", index=False, encoding="utf-8-sig")
df.to_excel("datos/ventas_historicas.xlsx", index=False)

print(f"Dataset creado: {len(df)} filas, {df.shape[1]} columnas")
print(df.head())
```

**Resultado esperado:** `Dataset creado: 1728 filas, 11 columnas`. Las primeras filas del CSV:

```csv
fecha,anio,mes,producto,cliente,tipo_cliente,zona,precio_unitario,promocion,unidades,ventas_total
2023-01-01,2023,1,Filtro de aire,C01 Metalúrgica Andina,Mayorista,Norte,25.0,0,160,4000.0
2023-01-01,2023,1,Filtro de aire,C02 Ensambles del Norte,Minorista,Norte,25.0,0,81,2025.0
2023-01-01,2023,1,Filtro de aire,C03 Agroindustrias Río,Distribuidor,Norte,25.0,1,159,3975.0
```

**Nota para el estudiante:** el `encoding="utf-8-sig"` hace que Excel muestre bien las tildes y la ñ al abrir el CSV. Con datos reales de una empresa, este paso se reemplaza por una exportación del ERP con las mismas columnas.

## Paso 2: Análisis exploratorio con Seaborn

Antes de entrenar cualquier modelo hay que mirar los datos: el análisis exploratorio (EDA) confirma que existen tendencia, estacionalidad y diferencias entre zonas y clientes, que son justo lo que el modelo debe aprender.

**2.1 Crear `eda.py`** y ejecutarlo con `python eda.py`. Guarda cinco gráficas en la carpeta `graficas/`.

```python
"""
eda.py - Análisis exploratorio con Seaborn.
Ejecutar: python eda.py   (guarda las gráficas en la carpeta graficas/)
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="deep")
os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("datos/ventas_historicas.csv", parse_dates=["fecha"])

# 1. Revisión rápida
print(df.info())
print(df.describe())
print("Valores nulos por columna:\n", df.isna().sum())

# 2. Evolución mensual por producto (tendencia + estacionalidad)
mensual = df.groupby(["fecha", "producto"], as_index=False)["unidades"].sum()
plt.figure(figsize=(12, 5))
sns.lineplot(data=mensual, x="fecha", y="unidades", hue="producto", marker="o")
plt.title("Unidades vendidas por mes y producto")
plt.tight_layout(); plt.savefig("graficas/01_tendencia_productos.png", dpi=120); plt.close()

# 3. Ventas totales por zona y producto
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="zona", y="ventas_total", hue="producto", estimator=sum, errorbar=None)
plt.title("Ventas totales ($) por zona y producto, 2023-2025")
plt.tight_layout(); plt.savefig("graficas/02_ventas_zona.png", dpi=120); plt.close()

# 4. Estacionalidad: distribución de unidades por mes
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x="mes", y="unidades", hue="producto")
plt.title("Estacionalidad: unidades por mes del año")
plt.tight_layout(); plt.savefig("graficas/03_estacionalidad.png", dpi=120); plt.close()

# 5. Mapa de calor cliente x producto
tabla = df.pivot_table(index="cliente", columns="producto", values="unidades", aggfunc="sum")
plt.figure(figsize=(10, 6))
sns.heatmap(tabla, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Unidades totales por cliente y producto")
plt.tight_layout(); plt.savefig("graficas/04_heatmap_clientes.png", dpi=120); plt.close()

# 6. Efecto de la promoción
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="producto", y="unidades", hue="promocion")
plt.title("Unidades con y sin promoción")
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig("graficas/05_promocion.png", dpi=120); plt.close()

print("Gráficas guardadas en la carpeta graficas/")
```

**2.2 Qué debe observar el estudiante en cada gráfica**

| Gráfica | Pregunta guía | Respuesta esperada |
| --- | --- | --- |
| 01 Tendencia | ¿Las ventas crecen? ¿Se repiten picos cada año? | Sí: crecimiento leve y picos anuales distintos por producto |
| 02 Zona | ¿Qué zona vende más? | Norte, seguida de Centro; Occidente es la menor |
| 03 Estacionalidad | ¿En qué mes vende más cada producto? | Filtro en junio, Correa en diciembre |
| 04 Mapa de calor | ¿Qué clientes son los más grandes? | C01, C07, C04 y C10 (mayoristas) |
| 05 Promoción | ¿La promoción funciona? | Sí: la caja con promoción = 1 está más arriba |

**Conceptos Seaborn usados:** `lineplot` (series de tiempo), `barplot` con `estimator=sum` (totales por categoría), `boxplot` (distribución y valores atípicos), `heatmap` (tablas cruzadas). El parámetro `hue` colorea por una categoría, lo que permite comparar productos en un solo gráfico.

## Paso 3: Ingeniería de variables

Un modelo de IA no entiende "marzo" ni "cliente C07": hay que traducir cada fila a números que describan el contexto de la venta. Esa traducción se llama ingeniería de variables (*feature engineering*) y suele decidir la calidad del pronóstico más que el algoritmo elegido.

**3.1 Del pronóstico clásico al pronóstico con IA.** En un método clásico (media móvil, suavizamiento exponencial) se ajusta una serie a la vez, y con 48 combinaciones producto-cliente-zona serían 48 modelos separados. Con IA entrenamos **un solo modelo global** con todas las series: aprende a la vez lo que es común (estacionalidad del producto) y lo que es particular (tamaño del cliente, mercado de la zona).

**3.2 Las variables que construiremos**

| Grupo | Variables | Qué le aportan al modelo |
| --- | --- | --- |
| Identidad | producto, cliente, zona, tipo\_cliente | Quién compra y dónde; se convierten a columnas 0/1 (one-hot) |
| Calendario | mes, trimestre, t (índice de tiempo) | Estacionalidad y posición en el tiempo |
| Rezagos (*lags*) | lag\_1, lag\_2, lag\_3, lag\_12 | Ventas de hace 1, 2, 3 y 12 meses de esa misma serie |
| Medias móviles | media\_movil\_3, media\_movil\_6 | Nivel reciente, suavizando el ruido |
| Negocio | precio\_unitario, promocion | Factores comerciales que mueven la demanda |

**3.3 Codificación one-hot.** La columna `zona` con valores Norte/Sur/Centro/Occidente se transforma en cuatro columnas: `zona_Norte`, `zona_Sur`, etc., con 1 en la zona de la fila y 0 en las demás. Así el modelo puede aprender "si zona\_Norte = 1, vender más" sin suponer que Norte > Sur numéricamente.

**3.4 Regla de oro: no mirar el futuro.** Todas las variables se calculan con `shift()`, es decir, solo con meses anteriores. Si la media móvil incluyera el mes actual, el modelo "vería la respuesta" durante el entrenamiento y parecería excelente, pero fallaría con datos nuevos. Este error se llama **fuga de información** (*data leakage*).

**3.5 Crear `motor_ia.py` (parte 1).** Este archivo es el "cerebro" del proyecto: lo usarán el script de entrenamiento y la aplicación, así ambos hacen exactamente lo mismo. Se construye en tres partes (Pasos 3, 4 y 5); cada parte se pega a continuación de la anterior en el mismo archivo.

```python
"""
motor_ia.py
"Cerebro" del proyecto: preparación de datos, ingeniería de variables,
entrenamiento, evaluación y pronóstico. Lo usan tanto el script de
entrenamiento como la aplicación Tkinter, así ambos hacen exactamente lo mismo.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------ #
# Configuración: qué columnas usa el modelo
# ------------------------------------------------------------------ #
COLUMNAS_REQUERIDAS = ["fecha", "producto", "cliente", "zona", "unidades"]
LLAVE = ["producto", "cliente", "zona"]                 # identifica una serie
CATEGORICAS = ["producto", "cliente", "zona", "tipo_cliente"]
NUMERICAS = ["mes", "trimestre", "t", "precio_unitario", "promocion",
             "lag_1", "lag_2", "lag_3", "lag_12",
             "media_movil_3", "media_movil_6"]
VARIABLES = CATEGORICAS + NUMERICAS


# ------------------------------------------------------------------ #
# 1. Limpieza y estandarización
# ------------------------------------------------------------------ #
def preparar_datos(df):
    """Valida columnas, normaliza fechas a inicio de mes, agrega duplicados
    y rellena con 0 los meses sin venta de cada serie."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {faltan}")

    df["fecha"] = pd.to_datetime(df["fecha"]).dt.to_period("M").dt.to_timestamp()
    if "tipo_cliente" not in df.columns:
        df["tipo_cliente"] = "Sin tipo"
    if "precio_unitario" not in df.columns:
        df["precio_unitario"] = 0.0
    if "promocion" not in df.columns:
        df["promocion"] = 0

    # Si hay varias filas del mismo mes y serie, se suman
    df = (df.groupby(LLAVE + ["fecha"], as_index=False)
            .agg(unidades=("unidades", "sum"),
                 precio_unitario=("precio_unitario", "mean"),
                 promocion=("promocion", "max"),
                 tipo_cliente=("tipo_cliente", "first")))

    # Rejilla completa: todas las series x todos los meses
    meses = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="MS")
    series = df[LLAVE + ["tipo_cliente"]].drop_duplicates(LLAVE)
    rejilla = series.merge(pd.DataFrame({"fecha": meses}), how="cross")
    df = rejilla.merge(df.drop(columns="tipo_cliente"), on=LLAVE + ["fecha"], how="left")
    df["unidades"] = df["unidades"].fillna(0)
    df["promocion"] = df["promocion"].fillna(0).astype(int)
    df["precio_unitario"] = df.groupby(LLAVE)["precio_unitario"].transform(
        lambda s: s.ffill().bfill()).fillna(0)
    return df.sort_values(LLAVE + ["fecha"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
# 2. Ingeniería de variables
# ------------------------------------------------------------------ #
def crear_variables(df):
    """Agrega variables de calendario y de 'memoria' (rezagos y medias
    móviles). Todas miran SOLO al pasado para evitar fuga de información."""
    df = df.sort_values(LLAVE + ["fecha"]).copy()
    g = df.groupby(LLAVE)["unidades"]

    df["mes"] = df["fecha"].dt.month
    df["trimestre"] = df["fecha"].dt.quarter
    df["t"] = (df["fecha"].dt.year - 2000) * 12 + df["fecha"].dt.month  # índice de tiempo

    for k in [1, 2, 3, 12]:
        df[f"lag_{k}"] = g.shift(k)
    df["media_movil_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    df["media_movil_6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
    return df
```

**3.6 Probar lo construido.** En una terminal con el entorno activo:

```python
import pandas as pd, motor_ia as ia
datos = ia.preparar_datos(pd.read_csv("datos/ventas_historicas.csv"))
var = ia.crear_variables(datos)
print(var[["fecha", "producto", "cliente", "unidades", "lag_1", "lag_12", "media_movil_3"]].head(15))
```

Los primeros 12 meses de cada serie tendrán `lag_12` vacío (NaN), porque no existe un año anterior. Por eso el modelo se entrena desde enero 2024 en adelante: 24 de los 36 meses son utilizables.

## Paso 4: Entrenar y evaluar el modelo

Entrenar es mostrarle al modelo ejemplos con la respuesta conocida para que ajuste sus reglas internas; evaluar es medir cuánto se equivoca con datos que nunca vio. En nuestros datos, el Random Forest reduce el error de la línea base de 14,0 % a 11,8 % (MAPE).

**4.1 División temporal de los datos.** En series de tiempo **nunca** se mezclan los datos al azar: se entrena con el pasado y se prueba con el futuro. Reservamos los últimos 6 meses (julio a diciembre 2025) como examen.

| Conjunto | Periodo | Uso |
| --- | --- | --- |
| Entrenamiento | ene-2024 a jun-2025 (18 meses) | El modelo aprende aquí |
| Prueba | jul-2025 a dic-2025 (6 meses) | Se mide el error aquí |

**4.2 Los tres competidores**

- **Línea base ingenua:** "este mes se venderá lo mismo que el mismo mes del año anterior" (`lag_12`). Si la IA no le gana a esto, no sirve.
- **Random Forest:** 300 árboles de decisión, cada uno entrenado con una muestra distinta de datos; la predicción es el promedio de todos. Robusto y difícil de sobreajustar.
- **Gradient Boosting:** árboles que se construyen en secuencia, cada uno corrigiendo los errores del anterior. Suele ser muy preciso, pero es más sensible a sus parámetros.

**4.3 Métricas de error** (en todas, menor es mejor salvo R²):

| Métrica | Significado para el ingeniero |
| --- | --- |
| MAE | Error promedio en unidades: "nos equivocamos en ±14 unidades por serie y mes" |
| RMSE | Como el MAE, pero castiga más los errores grandes |
| MAPE % | Error promedio en porcentaje; fácil de comunicar a gerencia |
| R² | Proporción de la variación explicada (1 = perfecto) |

**4.4 Agregar a `motor_ia.py` (parte 2)**, a continuación del código del Paso 3:

```python
# ------------------------------------------------------------------ #
# 3. Modelos
# ------------------------------------------------------------------ #
def construir_modelo(tipo="random_forest"):
    """Crea un Pipeline: codificación one-hot + regresor."""
    preprocesador = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("num", "passthrough", NUMERICAS),
    ])
    if tipo == "random_forest":
        regresor = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                         random_state=42, n_jobs=-1)
    elif tipo == "gradient_boosting":
        regresor = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                             max_depth=4, random_state=42)
    else:
        raise ValueError("tipo debe ser 'random_forest' o 'gradient_boosting'")
    return Pipeline([("preprocesador", preprocesador), ("modelo", regresor)])


def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    mascara = y_real > 0
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_real[mascara] - y_pred[mascara]) / y_real[mascara])) * 100),
        "R2": r2_score(y_real, y_pred),
    }


def dividir_entrenamiento_prueba(df_var, meses_prueba=6):
    """División temporal: los últimos N meses se reservan para prueba."""
    df_var = df_var.dropna(subset=["lag_12", "media_movil_6"])
    corte = df_var["fecha"].max() - pd.DateOffset(months=meses_prueba - 1)
    return df_var[df_var["fecha"] < corte], df_var[df_var["fecha"] >= corte]


def entrenar_y_evaluar(df_limpio, meses_prueba=6):
    """Entrena la línea base y dos modelos de IA, devuelve una tabla de
    métricas, el mejor modelo (re-entrenado con TODOS los datos) y su nombre."""
    df_var = crear_variables(df_limpio)
    train, test = dividir_entrenamiento_prueba(df_var, meses_prueba)

    resultados = {"Línea base (mismo mes año anterior)":
                  calcular_metricas(test["unidades"], test["lag_12"])}
    modelos = {}
    for tipo in ["random_forest", "gradient_boosting"]:
        m = construir_modelo(tipo)
        m.fit(train[VARIABLES], train["unidades"])
        resultados[tipo] = calcular_metricas(test["unidades"], m.predict(test[VARIABLES]))
        modelos[tipo] = m

    tabla = pd.DataFrame(resultados).T.round(2)
    mejor = min(modelos, key=lambda k: resultados[k]["MAE"])

    # Re-entrenar el ganador con todo el historial disponible
    completo = pd.concat([train, test])
    modelo_final = construir_modelo(mejor).fit(completo[VARIABLES], completo["unidades"])
    return modelo_final, tabla, mejor, test.assign(prediccion=modelos[mejor].predict(test[VARIABLES]))


def importancia_variables(modelo, top=15):
    nombres = modelo.named_steps["preprocesador"].get_feature_names_out()
    nombres = [n.split("__", 1)[1] for n in nombres]
    imp = modelo.named_steps["modelo"].feature_importances_
    return (pd.DataFrame({"variable": nombres, "importancia": imp})
              .sort_values("importancia", ascending=False).head(top))
```

**¿Qué es un Pipeline?** Es una "línea de producción" que encadena el preprocesamiento (one-hot) y el modelo. Al guardarlo, se guarda todo junto, así la aplicación no tiene que repetir la codificación a mano.

**4.5 Resultados obtenidos con el dataset de ejemplo** (los estudiantes deben obtener los mismos gracias a `random_state=42`):

| Modelo | MAE | RMSE | MAPE % | R² |
| --- | --- | --- | --- | --- |
| Línea base (mismo mes año anterior) | 18,41 | 29,14 | 14,02 | 0,85 |
| **Random Forest** | **14,49** | **22,69** | **11,76** | **0,91** |
| Gradient Boosting | 15,10 | 22,90 | 12,02 | 0,91 |

**4.6 ¿En qué se fija el modelo?** Las variables más importantes son `lag_12` (0,42) y `lag_1` (0,42), seguidas de `media_movil_3` (0,09) y `promocion` (0,02). Traducido: el modelo combina "lo que pasó el mismo mes del año pasado" (estacionalidad) con "lo que pasó el mes pasado" (nivel actual), exactamente los patrones que pusimos en el Paso 1. La zona y el cliente pesan poco por sí solos porque su efecto ya está contenido en los rezagos de cada serie.

## Paso 5: Pronosticar el futuro y guardar el modelo

Para predecir enero 2026 el modelo necesita `lag_1` = diciembre 2025, que sí conocemos; pero para febrero 2026 necesita enero 2026, que aún no existe. La solución es el **pronóstico recursivo**: se predice un mes, esa predicción se agrega al historial como si fuera real y se usa para calcular los rezagos del mes siguiente.

**5.1 Agregar a `motor_ia.py` (parte 3)**, al final del archivo:

```python
# ------------------------------------------------------------------ #
# 4. Pronóstico recursivo
# ------------------------------------------------------------------ #
def pronosticar(modelo, df_limpio, meses=6, promocion_futura=0):
    """Predice mes a mes: cada predicción se agrega al historial y sirve
    de rezago para el mes siguiente."""
    historial = df_limpio[LLAVE + ["fecha", "tipo_cliente", "unidades",
                                   "precio_unitario", "promocion"]].copy()
    ultima_fecha = historial["fecha"].max()
    base = (historial.sort_values("fecha").groupby(LLAVE).last()
                     .reset_index()[LLAVE + ["tipo_cliente", "precio_unitario"]])
    salidas = []
    for h in range(1, meses + 1):
        fecha = ultima_fecha + pd.DateOffset(months=h)
        nuevas = base.assign(fecha=fecha, promocion=promocion_futura, unidades=np.nan)
        temp = crear_variables(pd.concat([historial, nuevas], ignore_index=True))
        fila = temp[temp["fecha"] == fecha].copy()
        fila["unidades"] = np.clip(np.round(modelo.predict(fila[VARIABLES])), 0, None)
        historial = pd.concat([historial, fila[historial.columns]], ignore_index=True)
        salidas.append(fila)

    pron = pd.concat(salidas, ignore_index=True)
    pron["ventas_estimadas"] = (pron["unidades"] * pron["precio_unitario"]).round(2)
    return pron[["fecha", "producto", "cliente", "tipo_cliente", "zona",
                 "precio_unitario", "unidades", "ventas_estimadas"]] \
        .rename(columns={"unidades": "unidades_pronosticadas"})
```

Supuestos del pronóstico que el estudiante debe declarar en su informe: el precio futuro es el último precio conocido, y la promoción futura es 0 (o 1 si se marca la opción en la aplicación). El error crece con el horizonte, porque cada mes se apoya en predicciones anteriores; por eso limitamos el horizonte a 12 meses.

**5.2 Crear `entrenar_modelo.py`.** Este script orquesta todo: carga, entrena, compara, guarda el modelo, dibuja gráficas de validación y genera un primer pronóstico. Ejecutarlo con `python entrenar_modelo.py` (tarda unos 10 a 30 segundos).

```python
"""
entrenar_modelo.py - Entrena, evalúa y guarda el modelo de IA.
Ejecutar: python entrenar_modelo.py
Salida:   modelos/modelo_ventas.pkl, resultados/metricas.xlsx, gráficas
"""
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import motor_ia as ia

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

# 1. Cargar y limpiar
crudo = pd.read_csv("datos/ventas_historicas.csv")
datos = ia.preparar_datos(crudo)
print(f"Datos listos: {len(datos)} filas, "
      f"{datos.groupby(ia.LLAVE).ngroups} series (producto-cliente-zona)")

# 2. Entrenar y comparar
modelo, metricas, mejor, prueba = ia.entrenar_y_evaluar(datos, meses_prueba=6)
print("\nMétricas en los últimos 6 meses (datos que el modelo NO vio):")
print(metricas)
print(f"\nModelo ganador: {mejor}")

# 3. Guardar el modelo junto con información útil
joblib.dump({"modelo": modelo, "nombre": mejor, "metricas": metricas,
             "variables": ia.VARIABLES}, "modelos/modelo_ventas.pkl")
metricas.to_excel("resultados/metricas.xlsx")
print("Modelo guardado en modelos/modelo_ventas.pkl")

# 4. Gráfica: real vs predicho en el periodo de prueba
sns.set_theme(style="whitegrid")
comp = prueba.groupby(["fecha", "producto"], as_index=False)[["unidades", "prediccion"]].sum()
g = sns.relplot(data=comp.melt(id_vars=["fecha", "producto"], var_name="serie", value_name="valor"),
                x="fecha", y="valor", hue="serie", col="producto", col_wrap=2,
                kind="line", marker="o", height=3.5, aspect=1.4, facet_kws={"sharey": False})
g.figure.suptitle("Real vs. predicho (periodo de prueba)", y=1.02)
g.savefig("graficas/06_real_vs_predicho.png", dpi=120); plt.close("all")

# 5. Gráfica: importancia de variables
imp = ia.importancia_variables(modelo)
plt.figure(figsize=(8, 5))
sns.barplot(data=imp, x="importancia", y="variable", color="steelblue")
plt.title("¿En qué se fija el modelo? Importancia de variables")
plt.tight_layout(); plt.savefig("graficas/07_importancia.png", dpi=120); plt.close()

# 6. Primer pronóstico de prueba
pron = ia.pronosticar(modelo, datos, meses=6)
pron.to_excel("resultados/pronostico_6_meses.xlsx", index=False)
print(pron.groupby(["fecha", "producto"])["unidades_pronosticadas"].sum().unstack())
```

**5.3 ¿Qué es el archivo `.pkl`?** Es el modelo "congelado": contiene los 300 árboles con todas sus reglas aprendidas. Cualquier programa puede cargarlo con `joblib.load()` y predecir sin volver a entrenar. Este archivo es el puente entre el entrenamiento y el deployment.

**Resultado esperado:** el script imprime la tabla del Paso 4.5, crea `modelos/modelo_ventas.pkl` y un pronóstico de enero a junio 2026 con un total de 34.987 unidades.

## Paso 6: Deployment en una aplicación Tkinter

Hacer *deployment* significa poner el modelo en manos del usuario final, que no sabe programar. Nuestra aplicación de escritorio permite cargar un Excel o CSV, entrenar o cargar el modelo, generar el pronóstico, filtrar por producto, zona o cliente, ver gráficas Seaborn y exportar todo a Excel.

**6.1 Diseño de la interfaz**

| Zona de la ventana | Elementos | Función |
| --- | --- | --- |
| Barra superior | 6 botones | Cargar datos, Entrenar, Cargar modelo, Guardar modelo, Generar pronóstico, Exportar a Excel |
| Parámetros y filtros | Spinbox, Checkbutton, 3 Combobox | Horizonte (1–12 meses), promoción futura, filtros por producto, zona y cliente |
| Pestaña 1 | Treeview | Tabla de datos históricos |
| Pestaña 2 | Texto + gráfica | Métricas del modelo e importancia de variables |
| Pestaña 3 | Treeview | Tabla del pronóstico |
| Pestaña 4 | Lienzo Matplotlib | 4 gráficas Seaborn seleccionables y botón Guardar PNG |
| Barra inferior | Label + Progressbar | Mensajes de estado y barra de progreso |

**6.2 Conceptos clave de Tkinter usados en el código**

- `tk.Tk` es la ventana principal; la aplicación es una clase que hereda de ella, lo que ordena el código.
- `ttk.Notebook` crea pestañas; `ttk.Treeview` muestra tablas; `ttk.Combobox` muestra listas desplegables.
- `filedialog` abre las ventanas estándar de "Abrir" y "Guardar como"; `messagebox` muestra avisos.
- `FigureCanvasTkAgg` incrusta una figura de Matplotlib/Seaborn dentro de la ventana.
- `threading` ejecuta el entrenamiento en segundo plano para que la ventana no se congele; `self.after()` revisa cada 200 ms si terminó.

**6.3 Crear `app_pronostico.py`:**

```python
"""
app_pronostico.py - Aplicación de escritorio para pronosticar ventas con IA.
Ejecutar: python app_pronostico.py

Flujo: Cargar datos (CSV/Excel) -> Entrenar o cargar modelo -> Generar
pronóstico -> Ver gráficas (Seaborn) -> Exportar a Excel.
"""
import os
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import joblib
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import motor_ia as ia

sns.set_theme(style="whitegrid", palette="deep")
TODOS = "(Todos)"


class AppPronostico(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pronóstico de Ventas con IA - Ingeniería Industrial")
        self.geometry("1250x780")
        self.minsize(1000, 650)

        # Estado de la aplicación
        self.datos = None          # datos limpios (preparar_datos)
        self.modelo = None         # Pipeline de scikit-learn
        self.nombre_modelo = ""
        self.metricas = None
        self.pronostico = None

        self._construir_barra()
        self._construir_filtros()
        self._construir_pestanas()
        self._construir_estado()

    # ============================================================ #
    # Construcción de la interfaz
    # ============================================================ #
    def _construir_barra(self):
        barra = ttk.Frame(self, padding=6)
        barra.pack(fill="x")
        botones = [
            ("📂 Cargar datos", self.cargar_datos),
            ("🧠 Entrenar modelo", self.entrenar_modelo),
            ("📥 Cargar modelo", self.cargar_modelo),
            ("💾 Guardar modelo", self.guardar_modelo),
            ("📈 Generar pronóstico", self.generar_pronostico),
            ("📤 Exportar a Excel", self.exportar_excel),
        ]
        for texto, comando in botones:
            ttk.Button(barra, text=texto, command=comando).pack(side="left", padx=3)

    def _construir_filtros(self):
        marco = ttk.LabelFrame(self, text="Parámetros y filtros", padding=6)
        marco.pack(fill="x", padx=6)

        ttk.Label(marco, text="Meses a pronosticar:").grid(row=0, column=0, sticky="w")
        self.horizonte = tk.IntVar(value=6)
        ttk.Spinbox(marco, from_=1, to=12, textvariable=self.horizonte, width=5)\
            .grid(row=0, column=1, padx=(2, 15))

        self.promo = tk.IntVar(value=0)
        ttk.Checkbutton(marco, text="Suponer promoción futura", variable=self.promo)\
            .grid(row=0, column=2, padx=(0, 15))

        self.filtros = {}
        for i, campo in enumerate(["producto", "zona", "cliente"]):
            ttk.Label(marco, text=f"{campo.capitalize()}:").grid(row=1, column=2 * i, sticky="w", pady=(4, 0))
            combo = ttk.Combobox(marco, values=[TODOS], state="readonly", width=28)
            combo.set(TODOS)
            combo.grid(row=1, column=1 + 2 * i, padx=(2, 15), pady=(4, 0), sticky="w")
            combo.bind("<<ComboboxSelected>>", lambda e: self.actualizar_vistas())
            self.filtros[campo] = combo

    def _construir_pestanas(self):
        self.pestanas = ttk.Notebook(self)
        self.pestanas.pack(fill="both", expand=True, padx=6, pady=6)

        # Pestaña 1: datos
        f1 = ttk.Frame(self.pestanas)
        self.pestanas.add(f1, text="1. Datos históricos")
        self.lbl_resumen = ttk.Label(f1, text="Aún no se han cargado datos.", padding=4)
        self.lbl_resumen.pack(anchor="w")
        self.tabla_datos = self._crear_tabla(f1)

        # Pestaña 2: modelo
        f2 = ttk.Frame(self.pestanas)
        self.pestanas.add(f2, text="2. Modelo y métricas")
        self.txt_modelo = tk.Text(f2, font=("Consolas", 11), height=12, wrap="none")
        self.txt_modelo.pack(fill="x", padx=4, pady=4)
        self.txt_modelo.insert("end", "Entrene o cargue un modelo para ver sus métricas.")
        self.marco_imp = ttk.Frame(f2)
        self.marco_imp.pack(fill="both", expand=True)

        # Pestaña 3: pronóstico (tabla)
        f3 = ttk.Frame(self.pestanas)
        self.pestanas.add(f3, text="3. Pronóstico")
        self.tabla_pron = self._crear_tabla(f3)

        # Pestaña 4: gráficas
        f4 = ttk.Frame(self.pestanas)
        self.pestanas.add(f4, text="4. Gráficas")
        sup = ttk.Frame(f4, padding=4)
        sup.pack(fill="x")
        ttk.Label(sup, text="Tipo de gráfica:").pack(side="left")
        self.tipo_grafica = ttk.Combobox(sup, state="readonly", width=35, values=[
            "Histórico + pronóstico",
            "Pronóstico por zona",
            "Pronóstico por cliente",
            "Mapa de calor producto x zona",
        ])
        self.tipo_grafica.set("Histórico + pronóstico")
        self.tipo_grafica.pack(side="left", padx=4)
        self.tipo_grafica.bind("<<ComboboxSelected>>", lambda e: self.dibujar_grafica())
        ttk.Button(sup, text="🖼 Guardar PNG", command=self.guardar_png).pack(side="left", padx=4)

        self.figura = plt.Figure(figsize=(10, 5), dpi=100)
        self.lienzo = FigureCanvasTkAgg(self.figura, master=f4)
        NavigationToolbar2Tk(self.lienzo, f4).update()
        self.lienzo.get_tk_widget().pack(fill="both", expand=True)

    def _construir_estado(self):
        barra = ttk.Frame(self, padding=(6, 2))
        barra.pack(fill="x", side="bottom")
        self.estado = ttk.Label(barra, text="Listo.")
        self.estado.pack(side="left")
        self.progreso = ttk.Progressbar(barra, mode="indeterminate", length=180)
        self.progreso.pack(side="right")

    def _crear_tabla(self, padre):
        marco = ttk.Frame(padre)
        marco.pack(fill="both", expand=True)
        tabla = ttk.Treeview(marco, show="headings")
        sy = ttk.Scrollbar(marco, orient="vertical", command=tabla.yview)
        sx = ttk.Scrollbar(marco, orient="horizontal", command=tabla.xview)
        tabla.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tabla.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        marco.rowconfigure(0, weight=1)
        marco.columnconfigure(0, weight=1)
        return tabla

    # ============================================================ #
    # Utilidades
    # ============================================================ #
    def llenar_tabla(self, tabla, df, max_filas=1000):
        tabla.delete(*tabla.get_children())
        tabla["columns"] = list(df.columns)
        for col in df.columns:
            tabla.heading(col, text=col)
            tabla.column(col, width=130, anchor="center")
        for fila in df.head(max_filas).itertuples(index=False):
            valores = [v.strftime("%Y-%m") if isinstance(v, pd.Timestamp) else v for v in fila]
            tabla.insert("", "end", values=valores)

    def aplicar_filtros(self, df):
        for campo, combo in self.filtros.items():
            if combo.get() != TODOS:
                df = df[df[campo] == combo.get()]
        return df

    def en_segundo_plano(self, tarea, al_terminar, mensaje):
        """Ejecuta tareas largas sin congelar la ventana."""
        self.estado.config(text=mensaje)
        self.progreso.start(10)
        resultado = {}

        def trabajo():
            try:
                resultado["ok"] = tarea()
            except Exception as e:  # noqa: BLE001
                resultado["error"] = e

        hilo = threading.Thread(target=trabajo, daemon=True)
        hilo.start()

        def revisar():
            if hilo.is_alive():
                self.after(200, revisar)
                return
            self.progreso.stop()
            if "error" in resultado:
                self.estado.config(text="Ocurrió un error.")
                messagebox.showerror("Error", str(resultado["error"]))
            else:
                al_terminar(resultado["ok"])
        revisar()

    # ============================================================ #
    # Acciones de los botones
    # ============================================================ #
    def cargar_datos(self):
        ruta = filedialog.askopenfilename(
            title="Seleccione el archivo de ventas",
            filetypes=[("Excel o CSV", "*.xlsx *.xls *.csv"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            crudo = pd.read_csv(ruta) if ruta.lower().endswith(".csv") else pd.read_excel(ruta)
            self.datos = ia.preparar_datos(crudo)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error al cargar", str(e))
            return

        self.pronostico = None
        for campo, combo in self.filtros.items():
            combo["values"] = [TODOS] + sorted(self.datos[campo].unique())
            combo.set(TODOS)
        d = self.datos
        self.lbl_resumen.config(text=(
            f"Archivo: {os.path.basename(ruta)}  |  {len(d)} filas  |  "
            f"{d['producto'].nunique()} productos, {d['cliente'].nunique()} clientes, "
            f"{d['zona'].nunique()} zonas  |  "
            f"{d['fecha'].min():%Y-%m} a {d['fecha'].max():%Y-%m}"))
        self.actualizar_vistas()
        self.estado.config(text="Datos cargados. Siguiente paso: entrenar o cargar un modelo.")

    def entrenar_modelo(self):
        if self.datos is None:
            messagebox.showwarning("Atención", "Primero cargue los datos.")
            return
        if self.datos["fecha"].nunique() < 20:
            messagebox.showwarning("Atención", "Se necesitan al menos 20 meses de historia.")
            return

        def listo(res):
            self.modelo, self.metricas, self.nombre_modelo, _ = res
            self.mostrar_modelo()
            self.estado.config(text=f"Modelo entrenado: {self.nombre_modelo}.")

        self.en_segundo_plano(lambda: ia.entrenar_y_evaluar(self.datos), listo,
                              "Entrenando modelos... (puede tardar unos segundos)")

    def cargar_modelo(self):
        ruta = filedialog.askopenfilename(title="Seleccione el modelo",
                                          filetypes=[("Modelo", "*.pkl")])
        if not ruta:
            return
        paquete = joblib.load(ruta)
        self.modelo = paquete["modelo"]
        self.nombre_modelo = paquete.get("nombre", "modelo cargado")
        self.metricas = paquete.get("metricas")
        self.mostrar_modelo()
        self.estado.config(text=f"Modelo cargado desde {os.path.basename(ruta)}.")

    def guardar_modelo(self):
        if self.modelo is None:
            messagebox.showwarning("Atención", "No hay un modelo para guardar.")
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".pkl",
                                            initialfile="modelo_ventas.pkl",
                                            filetypes=[("Modelo", "*.pkl")])
        if ruta:
            joblib.dump({"modelo": self.modelo, "nombre": self.nombre_modelo,
                         "metricas": self.metricas, "variables": ia.VARIABLES}, ruta)
            self.estado.config(text=f"Modelo guardado en {ruta}")

    def generar_pronostico(self):
        if self.datos is None or self.modelo is None:
            messagebox.showwarning("Atención", "Necesita datos y un modelo.")
            return
        meses = int(self.horizonte.get())

        def listo(pron):
            self.pronostico = pron
            self.actualizar_vistas()
            self.pestanas.select(3)
            total = pron["unidades_pronosticadas"].sum()
            self.estado.config(text=f"Pronóstico listo: {meses} meses, {total:,.0f} unidades en total.")

        self.en_segundo_plano(
            lambda: ia.pronosticar(self.modelo, self.datos, meses, self.promo.get()),
            listo, "Generando pronóstico...")

    def exportar_excel(self):
        if self.pronostico is None:
            messagebox.showwarning("Atención", "Primero genere un pronóstico.")
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            initialfile="pronostico_ventas.xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not ruta:
            return
        p = self.pronostico
        with pd.ExcelWriter(ruta, engine="openpyxl") as w:
            p.to_excel(w, sheet_name="Pronostico_detalle", index=False)
            p.pivot_table(index="fecha", columns="producto", values="unidades_pronosticadas",
                          aggfunc="sum").to_excel(w, sheet_name="Resumen_producto")
            p.pivot_table(index="fecha", columns="zona", values="ventas_estimadas",
                          aggfunc="sum").to_excel(w, sheet_name="Resumen_zona")
            if self.metricas is not None:
                self.metricas.to_excel(w, sheet_name="Metricas_modelo")
            self.datos.to_excel(w, sheet_name="Historico", index=False)

            # Insertar la gráfica actual como imagen
            try:
                from openpyxl.drawing.image import Image as XLImage
                png = os.path.join(tempfile.gettempdir(), "grafica_pronostico.png")
                self.dibujar_grafica()
                self.figura.savefig(png, dpi=110, bbox_inches="tight")
                hoja = w.book.create_sheet("Grafica")
                hoja.add_image(XLImage(png), "A1")
            except Exception:  # noqa: BLE001  (si falta Pillow, se omite la imagen)
                pass
        messagebox.showinfo("Exportado", f"Archivo guardado:\n{ruta}")

    def guardar_png(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("Imagen PNG", "*.png")])
        if ruta:
            self.figura.savefig(ruta, dpi=150, bbox_inches="tight")

    # ============================================================ #
    # Vistas
    # ============================================================ #
    def actualizar_vistas(self):
        if self.datos is not None:
            self.llenar_tabla(self.tabla_datos, self.aplicar_filtros(self.datos))
        if self.pronostico is not None:
            self.llenar_tabla(self.tabla_pron, self.aplicar_filtros(self.pronostico))
        self.dibujar_grafica()

    def mostrar_modelo(self):
        self.txt_modelo.delete("1.0", "end")
        self.txt_modelo.insert("end", f"Modelo activo: {self.nombre_modelo}\n\n")
        if self.metricas is not None:
            self.txt_modelo.insert("end", "Métricas en los últimos 6 meses (prueba):\n")
            self.txt_modelo.insert("end", self.metricas.to_string())
        # Gráfica de importancia de variables
        for w in self.marco_imp.winfo_children():
            w.destroy()
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        sns.barplot(data=ia.importancia_variables(self.modelo), x="importancia",
                    y="variable", color="steelblue", ax=ax)
        ax.set_title("Importancia de variables")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=self.marco_imp).get_tk_widget().pack(fill="both", expand=True)

    def dibujar_grafica(self):
        self.figura.clear()
        ax = self.figura.add_subplot(111)
        tipo = self.tipo_grafica.get()

        if self.datos is None:
            ax.text(0.5, 0.5, "Cargue datos para ver gráficas", ha="center", va="center")
            self.lienzo.draw()
            return

        hist = self.aplicar_filtros(self.datos)
        pron = self.aplicar_filtros(self.pronostico) if self.pronostico is not None else None

        if tipo == "Histórico + pronóstico":
            h = hist.groupby(["fecha", "producto"], as_index=False)["unidades"].sum()
            h["tipo"] = "Histórico"
            if pron is not None:
                p = pron.groupby(["fecha", "producto"], as_index=False)["unidades_pronosticadas"] \
                        .sum().rename(columns={"unidades_pronosticadas": "unidades"})
                p["tipo"] = "Pronóstico"
                h = pd.concat([h, p])
            sns.lineplot(data=h, x="fecha", y="unidades", hue="producto", style="tipo",
                         markers=True, dashes={"Histórico": "", "Pronóstico": (4, 2)}, ax=ax)
            ax.set_title("Unidades mensuales: histórico y pronóstico")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        elif pron is None:
            ax.text(0.5, 0.5, "Genere un pronóstico para ver esta gráfica", ha="center", va="center")
        elif tipo == "Pronóstico por zona":
            sns.barplot(data=pron, x="zona", y="ventas_estimadas", hue="producto",
                        estimator="sum", errorbar=None, ax=ax)
            ax.set_title("Ventas estimadas ($) por zona en el horizonte")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        elif tipo == "Pronóstico por cliente":
            orden = pron.groupby("cliente")["ventas_estimadas"].sum().sort_values(ascending=False).index
            sns.barplot(data=pron, y="cliente", x="ventas_estimadas", order=orden,
                        estimator="sum", errorbar=None, color="teal", ax=ax)
            ax.set_title("Ventas estimadas ($) por cliente")
        elif tipo == "Mapa de calor producto x zona":
            tabla = pron.pivot_table(index="producto", columns="zona",
                                     values="unidades_pronosticadas", aggfunc="sum")
            sns.heatmap(tabla, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
            ax.set_title("Unidades pronosticadas por producto y zona")

        self.figura.tight_layout()
        self.lienzo.draw()


if __name__ == "__main__":
    app = AppPronostico()
    app.mainloop()
```

**6.4 Usar la aplicación.** Ejecutar `python app_pronostico.py` y seguir este orden:

1. **Cargar datos:** seleccionar `datos/ventas_historicas.xlsx` (o el CSV). La pestaña 1 muestra 1.728 filas, 4 productos, 12 clientes y 4 zonas, de 2023-01 a 2025-12.
2. **Entrenar modelo** (o **Cargar modelo** si ya existe `modelos/modelo_ventas.pkl`). La pestaña 2 muestra las métricas y la importancia de variables.
3. Elegir el horizonte (por ejemplo, 6 meses) y pulsar **Generar pronóstico**. La barra inferior indica el total de unidades.
4. En la pestaña 4, cambiar el tipo de gráfica y usar los filtros (por ejemplo, Zona = Norte) para analizar segmentos.
5. **Exportar a Excel:** crea un libro con seis hojas: Pronostico\_detalle, Resumen\_producto, Resumen\_zona, Metricas\_modelo, Historico y Grafica (con la imagen de la gráfica actual).

**6.5 Formato mínimo para cargar datos propios.** El archivo debe tener, al menos, las columnas `fecha`, `producto`, `cliente`, `zona` y `unidades`. Las columnas `tipo_cliente`, `precio_unitario` y `promocion` son opcionales: si faltan, se rellenan con valores por defecto. Se necesitan al menos 20 meses de historia, porque el modelo usa el rezago de 12 meses más 6 meses de prueba.

## Paso 7: Ejecutable, errores comunes, ejercicios y evaluación

Con la aplicación funcionando, el último paso del deployment es entregarla como un ejecutable que funcione en un computador sin Python, y consolidar lo aprendido con ejercicios.

**7.1 Crear un ejecutable (.exe en Windows)** con PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name PronosticoVentasIA app_pronostico.py
```

El ejecutable queda en la carpeta `dist/`. La opción `--windowed` evita que se abra una consola negra detrás de la ventana. El primer arranque tarda unos segundos porque descomprime las librerías. Si falta algún módulo de scikit-learn al ejecutar, agregar `--collect-all sklearn` al comando.

**7.2 Errores comunes y solución**

| Mensaje de error | Causa | Solución |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'sklearn'` | El entorno virtual no está activo | Activar `venv` y repetir `pip install -r requirements.txt` |
| `No module named 'tkinter'` | Linux sin Tk | `sudo apt install python3-tk` |
| `Faltan columnas obligatorias` | El Excel tiene otros nombres de columna | Renombrar a fecha, producto, cliente, zona, unidades |
| `FileNotFoundError: datos/ventas_historicas.csv` | Se ejecutó desde otra carpeta | Abrir la terminal dentro de `forecast_ia/` |
| Tildes raras (Ã³) al abrir el CSV en Excel | Codificación | Usar el .xlsx o importar el CSV como UTF-8 |
| Error al cargar un .pkl de un compañero | Versiones distintas de scikit-learn | Usar las mismas versiones o re-entrenar con el botón Entrenar |
| La ventana se queda "No responde" | Datos muy grandes | Esperar a la barra de progreso; reducir `n_estimators` |

**7.3 Ejercicios propuestos** (de menor a mayor dificultad)

1. Cambiar la semilla `np.random.seed(42)` por el número de lista del estudiante, regenerar los datos y comparar las métricas obtenidas.
2. Agregar una quinta zona ("Oriente", factor 0,9) con dos clientes nuevos y verificar que la aplicación la muestra en los filtros.
3. Agregar la variable `lag_6` en `crear_variables` y en `NUMERICAS`. ¿Mejora el MAPE?
4. Cambiar `meses_prueba` a 3 y a 12. ¿Cómo cambia el error? ¿Por qué?
5. Incorporar un tercer modelo (`ExtraTreesRegressor` o `LinearRegression`) en `construir_modelo` y compararlo.
6. Agregar a la aplicación una gráfica nueva: pronóstico por `tipo_cliente` con `sns.barplot`.
7. Calcular el inventario de seguridad por producto con el RMSE del modelo: SS = Z × RMSE × √(lead time), con Z = 1,65 para un 95 % de nivel de servicio, y exportarlo en una hoja nueva del Excel.
8. Proyecto final: aplicar el flujo completo a datos reales (de una empresa o un dataset público) y presentar un informe con métricas, supuestos y recomendaciones.

**7.4 Rúbrica de evaluación sugerida (100 puntos)**

| Criterio | Puntos | Qué se evalúa |
| --- | --- | --- |
| Entorno y dataset | 10 | Entorno funcionando y dataset generado correctamente |
| Análisis exploratorio | 15 | Gráficas Seaborn e interpretación de tendencia, estacionalidad y zonas |
| Ingeniería de variables | 15 | Explica rezagos, one-hot y por qué se evita la fuga de información |
| Entrenamiento y evaluación | 20 | Compara con la línea base e interpreta MAE, RMSE y MAPE |
| Aplicación Tkinter | 25 | Carga, pronóstico, filtros, gráficas y exportación a Excel funcionando |
| Informe y presentación | 15 | Supuestos, limitaciones y recomendación de negocio |

**7.5 Recursos entregados con esta guía.** El proyecto completo y probado está en el archivo `proyecto_forecast_ia.zip`: los cinco scripts, requirements.txt, el dataset en CSV y Excel, las siete gráficas de ejemplo, las métricas y un Excel de ejemplo exportado desde la aplicación. El modelo `.pkl` no se incluye para que cada estudiante lo entrene.
