Desde el Paso 4 hay dos tipos de acciones: **pegar código en archivos** (en VS Code) y **ejecutar comandos en la terminal**. Antes de cada comando, confirme dos cosas: que la terminal esté dentro de la carpeta `forecast_ia` y que el entorno virtual esté activo (se ve `(venv)` al inicio de la línea). Si no lo está:

```bash
cd forecast_ia
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

## Paso 4: entrenar y evaluar

**Archivo:** abra `motor_ia.py` (el que creó en el Paso 3) y pegue la **parte 2** del código (desde `# 3. Modelos` hasta `importancia_variables`) al final del archivo, debajo de `crear_variables`. Guarde con Ctrl+S.

Este paso no tiene un script propio, porque `motor_ia.py` solo contiene funciones. Para comprobar que funciona, ejecute en la terminal:

```bash
python -c "import pandas as pd, motor_ia as ia; d=ia.preparar_datos(pd.read_csv('datos/ventas_historicas.csv')); m,t,n,_=ia.entrenar_y_evaluar(d); print(t); print('Ganador:', n)"
```

Tras 10 a 30 segundos debe aparecer la tabla de métricas, con el Random Forest cerca de MAE 14,49 y MAPE 11,76 %, y el mensaje `Ganador: random_forest`.

## Paso 5: pronóstico y guardar el modelo

**Archivo 1:** pegue la **parte 3** (`# 4. Pronóstico recursivo`, la función `pronosticar`) al final de `motor_ia.py` y guarde. Con esto el archivo queda completo.

**Archivo 2:** cree un archivo nuevo llamado `entrenar_modelo.py` en la misma carpeta, pegue el código de la sección 5.2 y guarde.

**Comando:**

```bash
python entrenar_modelo.py
```

Al terminar, la terminal muestra el número de series (48), la tabla de métricas, el modelo ganador y el pronóstico de enero a junio 2026 por producto. Revise que se hayan creado estos archivos:

- `modelos/modelo_ventas.pkl`, que es el modelo entrenado.
- `resultados/metricas.xlsx` y `resultados/pronostico_6_meses.xlsx`.
- `graficas/06_real_vs_predicho.png` y `graficas/07_importancia.png`.

## Paso 6: la aplicación Tkinter

**Archivo:** cree `app_pronostico.py`, pegue el código de la sección 6.3 y guarde. Debe estar en la misma carpeta que `motor_ia.py`, porque la aplicación lo importa.

**Comando:**

```bash
python app_pronostico.py
```

Se abre la ventana. Úsela en este orden:

1. **📂 Cargar datos:** elija `datos/ventas_historicas.xlsx`. Arriba de la tabla debe decir 1728 filas, 4 productos, 12 clientes y 4 zonas.
2. **🧠 Entrenar modelo:** espere a que la barra de progreso se detenga. También puede usar **📥 Cargar modelo** y elegir `modelos/modelo_ventas.pkl`, que es más rápido.
3. Ajuste **Meses a pronosticar** (por ejemplo, 6) y pulse **📈 Generar pronóstico**.
4. En la pestaña **4. Gráficas**, cambie el tipo de gráfica y pruebe los filtros de producto, zona y cliente.
5. **📤 Exportar a Excel:** elija dónde guardar. El libro tendrá seis hojas, incluida la gráfica.

Para cerrar la aplicación, cierre la ventana. La terminal vuelve a quedar disponible.

## Paso 7: crear el ejecutable (opcional)

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name PronosticoVentasIA app_pronostico.py
```

El resultado queda en `dist/PronosticoVentasIA.exe` (en Windows). Si al abrirlo aparece un error de módulo faltante de scikit-learn, repita el comando agregando `--collect-all sklearn`.

**Resumen de comandos en orden:** la prueba de una línea del Paso 4, `python entrenar_modelo.py`, `python app_pronostico.py` y los dos de PyInstaller. Si alguno falla, la tabla 7.2 de la guía tiene los errores más frecuentes. También puede pegarme aquí el mensaje de error y lo revisamos.