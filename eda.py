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
