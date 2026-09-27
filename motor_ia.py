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
