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
