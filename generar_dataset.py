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
