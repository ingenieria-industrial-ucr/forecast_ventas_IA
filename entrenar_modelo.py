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
