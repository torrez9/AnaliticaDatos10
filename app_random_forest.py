import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import time

# ==========================================================
# COMPATIBILIDAD SCIKIT-LEARN
# ==========================================================

try:
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList

except Exception:
    pass


# ==========================================================
# CONFIGURACIÓN STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="Sistema Predictivo Retail Inteligente",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CSS PERSONALIZADO
# ==========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #292236 0%, #3d2b50 45%, #4a2d52 100%);
        padding: 35px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        color: #ff6f91;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #d8d8d8;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.28);
        min-height: 245px;
    }

    .prediction-label {
        font-size: 13px;
        color: #b9b9b9;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .prediction-value {
        font-size: 42px;
        color: #4c83ff;
        font-weight: 900;
        margin-top: 10px;
    }

    .prediction-sub {
        font-size: 16px;
        color: #d0d0d0;
        margin-top: 10px;
    }

    .info-box-green {
        background: rgba(21, 128, 61, 0.20);
        border-left: 5px solid #22c55e;
        padding: 18px;
        border-radius: 12px;
        color: #d1fae5;
        margin-top: 15px;
        line-height: 1.6;
    }

    .info-box-yellow {
        background: rgba(202, 138, 4, 0.22);
        border-left: 5px solid #facc15;
        padding: 18px;
        border-radius: 12px;
        color: #fef9c3;
        margin-top: 15px;
        line-height: 1.6;
    }

    .info-box-red {
        background: rgba(185, 28, 28, 0.22);
        border-left: 5px solid #ef4444;
        padding: 18px;
        border-radius: 12px;
        color: #fee2e2;
        margin-top: 15px;
        line-height: 1.6;
    }

    .info-box-blue {
        background: rgba(37, 99, 235, 0.18);
        border-left: 5px solid #3b82f6;
        padding: 18px;
        border-radius: 12px;
        color: #dbeafe;
        margin-top: 15px;
        line-height: 1.6;
    }

    div[data-testid="stMetricValue"] {
        font-size: 30px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Sistema Predictivo Retail Inteligente</div>
        <div class="hero-subtitle">
            Predicción de ventas, What If dinámico, análisis ABC, treemap y distribución de ingresos
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CARGAR MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_predictor_ventas_random_forest.pkl")


try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error(
        """
        No se encontró el archivo del modelo.

        Verifica que esté en la misma carpeta que este script:

        modelo_predictor_ventas_random_forest.pkl
        """
    )
    st.stop()
except Exception as e:
    st.error("Ocurrió un error al cargar el modelo.")
    st.exception(e)
    st.stop()


# ==========================================================
# COLUMNAS DEL MODELO
# ==========================================================

COLUMNAS_MODELO = [
    "Cantidad",
    "Descuento",
    "Precio",
    "Precio_costo",
    "Precio_venta",
    "Precio_descuento",
    "Precio_Mayorista",
    "Idbodega",
    "Id_Medida",
    "Estado"
]


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def preparar_datos(datos):
    datos = datos.copy()

    for col in COLUMNAS_MODELO:
        if col not in datos.columns:
            datos[col] = 0

    datos = datos[COLUMNAS_MODELO]

    numericas = [
        "Cantidad",
        "Descuento",
        "Precio",
        "Precio_costo",
        "Precio_venta",
        "Precio_descuento",
        "Precio_Mayorista",
        "Idbodega",
        "Id_Medida"
    ]

    for col in numericas:
        datos[col] = pd.to_numeric(datos[col], errors="coerce").fillna(0)

    datos["Idbodega"] = datos["Idbodega"].astype(int)
    datos["Id_Medida"] = datos["Id_Medida"].astype(int)
    datos["Estado"] = datos["Estado"].astype(str)

    return datos


def crear_fila_producto(
    cantidad,
    precio,
    descuento,
    precio_costo,
    precio_venta,
    precio_mayorista,
    idbodega,
    id_medida,
    estado
):
    precio_descuento = precio_venta - descuento

    if precio_descuento < 0:
        precio_descuento = 0

    datos = pd.DataFrame({
        "Cantidad": [cantidad],
        "Descuento": [descuento],
        "Precio": [precio],
        "Precio_costo": [precio_costo],
        "Precio_venta": [precio_venta],
        "Precio_descuento": [precio_descuento],
        "Precio_Mayorista": [precio_mayorista],
        "Idbodega": [idbodega],
        "Id_Medida": [id_medida],
        "Estado": [estado]
    })

    return datos


def calcular_indicadores(fila):
    cantidad = float(fila["Cantidad"])
    precio_costo = float(fila["Precio_costo"])
    precio_venta = float(fila["Precio_venta"])
    descuento = float(fila["Descuento"])

    precio_descuento = precio_venta - descuento

    if precio_descuento < 0:
        precio_descuento = 0

    costo_total = precio_costo * cantidad
    venta_calculada = precio_descuento * cantidad
    ganancia = venta_calculada - costo_total

    if venta_calculada > 0:
        margen = (ganancia / venta_calculada) * 100
    else:
        margen = 0

    return precio_descuento, costo_total, venta_calculada, ganancia, margen


def generar_estado_comercial(ganancia, margen):
    if ganancia > 0 and margen >= 25:
        return "Rentable", "Alta"
    elif ganancia > 0 and margen < 25:
        return "Rentabilidad moderada", "Media"
    elif ganancia == 0:
        return "Punto de equilibrio", "Media"
    else:
        return "Pérdida", "Baja"


def generar_explicacion(cantidad, precio_venta, descuento, precio_descuento, ganancia, margen, resultado):
    if ganancia > 0 and margen >= 25:
        return f"""
        <div class="info-box-green">
        El modelo estima un resultado de <b>{resultado:.2f}</b>. 
        Con <b>{cantidad:.0f} unidades</b>, un precio de venta de <b>{precio_venta:.2f}</b> 
        y un descuento de <b>{descuento:.2f}</b>, el producto mantiene una ganancia positiva. 
        El precio final queda en <b>{precio_descuento:.2f}</b> y el margen estimado es de 
        <b>{margen:.2f}%</b>. La operación puede considerarse favorable.
        </div>
        """

    elif ganancia > 0 and margen < 25:
        return f"""
        <div class="info-box-yellow">
        El modelo genera una predicción positiva de <b>{resultado:.2f}</b>, pero el margen comercial es moderado. 
        El margen estimado es de <b>{margen:.2f}%</b>. 
        La venta puede realizarse, pero conviene revisar el descuento o el precio de venta.
        </div>
        """

    elif ganancia == 0:
        return f"""
        <div class="info-box-yellow">
        La operación queda en punto de equilibrio. 
        El ingreso calculado cubre el costo, pero no genera ganancia. 
        Se recomienda aumentar el precio de venta o reducir el descuento.
        </div>
        """

    else:
        return f"""
        <div class="info-box-red">
        El análisis comercial indica una pérdida estimada de <b>{abs(ganancia):.2f}</b>. 
        Esto ocurre porque el costo total supera el ingreso generado por la venta. 
        Se recomienda aumentar el precio de venta, reducir el descuento o revisar el costo del producto.
        </div>
        """


def grafico_gauge(valor, titulo):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=valor,
            delta={"reference": 50},
            title={"text": titulo},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e"},
                "steps": [
                    {"range": [0, 40], "color": "#7f1d1d"},
                    {"range": [40, 70], "color": "#854d0e"},
                    {"range": [70, 100], "color": "#14532d"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.75,
                    "value": valor,
                },
            },
        )
    )

    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig


def convertir_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")

    return output.getvalue()


def lista_rango_seguro(inicio, fin, paso):
    inicio = int(inicio)
    fin = int(fin)
    paso = int(paso)

    if paso <= 0:
        paso = 1

    if fin < inicio:
        return []

    return list(range(inicio, fin + 1, paso))


def obtener_importancias_modelo(modelo):
    try:
        if hasattr(modelo, "feature_importances_"):
            importancias = modelo.feature_importances_
            nombres = COLUMNAS_MODELO

            if len(importancias) != len(nombres):
                nombres = [f"Variable_{i+1}" for i in range(len(importancias))]

            return pd.DataFrame({
                "Variable": nombres,
                "Importancia": importancias
            }).sort_values("Importancia", ascending=False)

        if hasattr(modelo, "named_steps"):
            ultimo_paso = list(modelo.named_steps.values())[-1]

            if hasattr(ultimo_paso, "feature_importances_"):
                importancias = ultimo_paso.feature_importances_
                nombres = COLUMNAS_MODELO

                if len(importancias) != len(nombres):
                    nombres = [f"Variable_{i+1}" for i in range(len(importancias))]

                return pd.DataFrame({
                    "Variable": nombres,
                    "Importancia": importancias
                }).sort_values("Importancia", ascending=False)

    except Exception:
        pass

    return None


def procesar_csv_con_progreso(modelo, df_modelo, barra_progreso=None, texto_progreso=None):
    predicciones = []
    total = len(df_modelo)
    errores = 0

    for idx, fila in df_modelo.iterrows():
        if barra_progreso is not None:
            barra_progreso.progress((idx + 1) / total)

        if texto_progreso is not None:
            texto_progreso.write(f"Procesando registro {idx + 1} de {total}")

        fila_df = pd.DataFrame([fila])
        fila_df = preparar_datos(fila_df)

        try:
            pred = float(modelo.predict(fila_df)[0])
        except Exception:
            pred = 0
            errores += 1

        predicciones.append(pred)

    return predicciones, errores


def generar_what_if_individual(
    modelo,
    cantidad_base,
    precio_base,
    precio_costo_base,
    precio_venta_base,
    precio_mayorista_base,
    idbodega_base,
    id_medida_base,
    estado_base,
    descuento_min,
    descuento_max,
    paso_descuento,
    variacion_precio_min,
    variacion_precio_max,
    paso_variacion_precio,
    cantidad_min,
    cantidad_max,
    paso_cantidad,
    barra_progreso=None,
    texto_progreso=None
):
    escenarios = []

    descuentos = lista_rango_seguro(descuento_min, descuento_max, paso_descuento)
    variaciones_precio = lista_rango_seguro(variacion_precio_min, variacion_precio_max, paso_variacion_precio)
    cantidades = lista_rango_seguro(cantidad_min, cantidad_max, paso_cantidad)

    total_escenarios = len(descuentos) * len(variaciones_precio) * len(cantidades)

    if total_escenarios == 0:
        return pd.DataFrame(), 0, 0

    contador = 0
    errores = 0

    for cant in cantidades:
        for d in descuentos:
            for variacion in variaciones_precio:
                contador += 1

                if barra_progreso is not None:
                    barra_progreso.progress(contador / total_escenarios)

                if texto_progreso is not None:
                    texto_progreso.write(
                        f"Procesando escenario {contador} de {total_escenarios} | "
                        f"Cantidad: {cant} | Descuento: {d} | Variación precio: {variacion}%"
                    )

                nuevo_precio_venta = precio_venta_base * (1 + variacion / 100)
                nuevo_precio_descuento = nuevo_precio_venta - d

                if nuevo_precio_descuento < 0:
                    nuevo_precio_descuento = 0

                fila = pd.DataFrame({
                    "Cantidad": [cant],
                    "Descuento": [d],
                    "Precio": [precio_base],
                    "Precio_costo": [precio_costo_base],
                    "Precio_venta": [nuevo_precio_venta],
                    "Precio_descuento": [nuevo_precio_descuento],
                    "Precio_Mayorista": [precio_mayorista_base],
                    "Idbodega": [idbodega_base],
                    "Id_Medida": [id_medida_base],
                    "Estado": [estado_base]
                })

                fila_modelo = preparar_datos(fila)

                try:
                    pred = float(modelo.predict(fila_modelo)[0])
                except Exception:
                    pred = 0
                    errores += 1

                precio_descuento_calc, costo_total, venta_calculada, ganancia, margen = calcular_indicadores(fila.iloc[0])
                estado_comercial, confianza = generar_estado_comercial(ganancia, margen)

                escenarios.append({
                    "Servicio": f"Bodega {idbodega_base} - Medida {id_medida_base}",
                    "Categoria": estado_comercial,
                    "Cantidad": cant,
                    "Variacion_precio_%": variacion,
                    "Descuento": d,
                    "Precio_base": precio_base,
                    "Precio_costo": precio_costo_base,
                    "Precio_venta_simulado": nuevo_precio_venta,
                    "Precio_final": precio_descuento_calc,
                    "Precio_Mayorista": precio_mayorista_base,
                    "Prediccion_Modelo": pred,
                    "Costo_total": costo_total,
                    "Venta_calculada": venta_calculada,
                    "Ganancia_estimada": ganancia,
                    "Margen_estimado": margen,
                    "Estado_Comercial": estado_comercial,
                    "Idbodega": idbodega_base,
                    "Id_Medida": id_medida_base,
                    "Estado": estado_base
                })

    return pd.DataFrame(escenarios), total_escenarios, errores


def preparar_columnas_visuales(df):
    df = df.copy()

    if "Servicio" not in df.columns:
        df["Servicio"] = (
            "Bodega " + df["Idbodega"].astype(str) +
            " - Medida " + df["Id_Medida"].astype(str)
        )

    if "Categoria" not in df.columns:
        if "Estado_Comercial" in df.columns:
            df["Categoria"] = df["Estado_Comercial"]
        else:
            df["Categoria"] = df["Estado"].astype(str)

    return df


def agregar_ranking_abc(df, columna_valor="Venta_calculada"):
    df = df.copy()
    resumen = df.groupby("Servicio", as_index=False)[columna_valor].sum()
    resumen = resumen.sort_values(columna_valor, ascending=False)

    total = resumen[columna_valor].sum()

    if total > 0:
        resumen["Porcentaje"] = resumen[columna_valor] / total * 100
        resumen["Porcentaje_acumulado"] = resumen["Porcentaje"].cumsum()
    else:
        resumen["Porcentaje"] = 0
        resumen["Porcentaje_acumulado"] = 0

    def clasificar_abc(p):
        if p <= 80:
            return "A"
        elif p <= 95:
            return "B"
        else:
            return "C"

    resumen["Ranking_ABC"] = resumen["Porcentaje_acumulado"].apply(clasificar_abc)

    return resumen


def mostrar_graficos_avanzados(df, titulo="Análisis visual avanzado"):
    df = preparar_columnas_visuales(df)

    st.markdown(f'<div class="section-title">{titulo}</div>', unsafe_allow_html=True)

    resumen_servicio = df.groupby(["Categoria", "Servicio"], as_index=False).agg({
        "Venta_calculada": "sum",
        "Ganancia_estimada": "sum",
        "Cantidad": "sum",
        "Prediccion_Modelo": "mean"
    })

    resumen_servicio = resumen_servicio[resumen_servicio["Venta_calculada"] > 0]

    if resumen_servicio.empty:
        st.warning("No hay datos positivos suficientes para generar treemap o dona.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        fig_treemap = px.treemap(
            resumen_servicio,
            path=["Categoria", "Servicio"],
            values="Venta_calculada",
            color="Venta_calculada",
            title="Treemap de ingresos por servicio",
            hover_data={
                "Ganancia_estimada": ":.2f",
                "Cantidad": ":.0f",
                "Prediccion_Modelo": ":.2f"
            },
            color_continuous_scale="Blues"
        )

        fig_treemap.update_layout(
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_treemap, use_container_width=True)

        st.caption(
            "Este treemap muestra qué servicios o grupos aportan más ingresos. "
            "Los bloques más grandes representan mayor participación económica."
        )

    with col_b:
        top_dona = resumen_servicio.sort_values("Venta_calculada", ascending=False).head(20)

        fig_dona = px.pie(
            top_dona,
            names="Servicio",
            values="Venta_calculada",
            hole=0.55,
            title="Distribución porcentual de ingresos"
        )

        fig_dona.update_traces(
            textposition="inside",
            textinfo="percent"
        )

        fig_dona.update_layout(
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend_title_text="Servicio"
        )

        st.plotly_chart(fig_dona, use_container_width=True)

        st.caption(
            "La dona permite observar la participación porcentual de cada servicio o grupo "
            "sobre el ingreso total."
        )

    st.markdown('<div class="section-title">Ranking y ABC</div>', unsafe_allow_html=True)

    ranking = agregar_ranking_abc(df, "Venta_calculada")

    col_c, col_d = st.columns(2)

    with col_c:
        fig_ranking = px.bar(
            ranking.head(15),
            x="Servicio",
            y="Venta_calculada",
            color="Ranking_ABC",
            text="Venta_calculada",
            title="Ranking de ingresos por servicio"
        )

        fig_ranking.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_ranking.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig_ranking, use_container_width=True)

    with col_d:
        fig_abc = px.bar(
            ranking.head(20),
            x="Servicio",
            y="Porcentaje_acumulado",
            color="Ranking_ABC",
            text="Porcentaje_acumulado",
            title="Participación acumulada ABC"
        )

        fig_abc.add_hline(y=80, line_dash="dash")
        fig_abc.add_hline(y=95, line_dash="dash")

        fig_abc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_abc.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig_abc, use_container_width=True)

    with st.expander("Ver tabla de ranking ABC"):
        st.dataframe(ranking, use_container_width=True)


# ==========================================================
# PANEL DE CONTROL
# ==========================================================

st.sidebar.title("Panel de Control")

modo = st.sidebar.radio(
    "Modo de trabajo",
    [
        "Predicción individual",
        "What If individual",
        "Predicción por CSV",
        "Análisis del modelo"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Columnas requeridas para CSV**

    - Cantidad  
    - Descuento  
    - Precio  
    - Precio_costo  
    - Precio_venta  
    - Precio_descuento  
    - Precio_Mayorista  
    - Idbodega  
    - Id_Medida  
    - Estado  

    **Opcional para mejores gráficos**

    - Servicio  
    - Categoria  
    """
)


# ==========================================================
# MODO 1: PREDICCIÓN INDIVIDUAL
# ==========================================================

if modo == "Predicción individual":

    st.markdown('<div class="section-title">Parámetros de Entrada</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        cantidad = st.number_input("Cantidad", min_value=1, value=10, step=1)

    with col2:
        precio = st.number_input("Precio general", min_value=0.0, value=100.0, step=1.0)

    with col3:
        descuento = st.number_input("Descuento aplicado", min_value=0.0, value=0.0, step=1.0)

    col4, col5, col6 = st.columns(3)

    with col4:
        precio_costo = st.number_input("Precio costo", min_value=0.0, value=75.0, step=1.0)

    with col5:
        precio_venta = st.number_input("Precio venta", min_value=0.0, value=100.0, step=1.0)

    with col6:
        precio_mayorista = st.number_input("Precio mayorista", min_value=0.0, value=90.0, step=1.0)

    col7, col8, col9 = st.columns(3)

    with col7:
        idbodega = st.number_input("ID Bodega", min_value=1, value=1, step=1)

    with col8:
        id_medida = st.number_input("ID Medida", min_value=1, value=1, step=1)

    with col9:
        estado = st.selectbox("Estado del producto", ["Activo", "Desconocido"])

    precio_descuento = precio_venta - descuento

    if precio_descuento < 0:
        precio_descuento = 0

    st.info(f"Precio con descuento calculado: {precio_descuento:.2f}")

    if st.button("Clasificar / Predecir Operación", use_container_width=True):

        datos = crear_fila_producto(
            cantidad,
            precio,
            descuento,
            precio_costo,
            precio_venta,
            precio_mayorista,
            idbodega,
            id_medida,
            estado
        )

        datos_modelo = preparar_datos(datos)

        try:
            barra = st.progress(0)
            texto = st.empty()

            for i in range(101):
                barra.progress(i / 100)
                texto.write(f"Analizando producto... {i}%")
                time.sleep(0.003)

            prediccion = modelo.predict(datos_modelo)
            resultado = float(prediccion[0])

            texto.success("Predicción completada correctamente.")

            if resultado < 0:
                resultado = 0

            precio_descuento, costo_total, venta_calculada, ganancia, margen = calcular_indicadores(datos.iloc[0])
            estado_comercial, confianza_texto = generar_estado_comercial(ganancia, margen)
            confianza_visual = max(0, min(100, margen + 50))

            st.divider()

            st.markdown('<div class="section-title">Resultado de la Predicción</div>', unsafe_allow_html=True)

            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="prediction-label">Clasificación Comercial</div>
                        <div class="prediction-value">{estado_comercial}</div>
                        <div class="prediction-sub">Resultado estimado por el modelo: <b>{resultado:.2f}</b></div>
                        <div class="prediction-sub">Nivel interpretativo: <b>{confianza_texto}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_res2:
                st.plotly_chart(
                    grafico_gauge(confianza_visual, "Confianza del análisis"),
                    use_container_width=True
                )

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                st.metric("Costo total", f"{costo_total:.2f}")

            with col_m2:
                st.metric("Venta calculada", f"{venta_calculada:.2f}")

            with col_m3:
                st.metric("Ganancia estimada", f"{ganancia:.2f}")

            with col_m4:
                st.metric("Margen estimado", f"{margen:.2f}%")

            st.markdown('<div class="section-title">Interpretación del Resultado</div>', unsafe_allow_html=True)

            st.markdown(
                generar_explicacion(
                    cantidad,
                    precio_venta,
                    descuento,
                    precio_descuento,
                    ganancia,
                    margen,
                    resultado
                ),
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-title">Análisis de la Predicción</div>', unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                df_comercial = pd.DataFrame({
                    "Indicador": ["Costo total", "Venta calculada", "Ganancia estimada"],
                    "Valor": [costo_total, venta_calculada, ganancia]
                })

                fig1 = px.bar(
                    df_comercial,
                    x="Indicador",
                    y="Valor",
                    text="Valor",
                    title="Costo, venta y ganancia"
                )

                fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig1.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white"
                )

                st.plotly_chart(fig1, use_container_width=True)

            with col_g2:
                df_precios = pd.DataFrame({
                    "Concepto": ["Precio costo", "Precio venta", "Descuento", "Precio final"],
                    "Valor": [precio_costo, precio_venta, descuento, precio_descuento]
                })

                fig2 = px.bar(
                    df_precios,
                    x="Valor",
                    y="Concepto",
                    orientation="h",
                    text="Valor",
                    title="Composición del precio"
                )

                fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white"
                )

                st.plotly_chart(fig2, use_container_width=True)

            df_visual = pd.DataFrame({
                "Servicio": [
                    "Costo total",
                    "Venta calculada",
                    "Ganancia estimada",
                    "Predicción modelo"
                ],
                "Categoria": [
                    "Costo",
                    "Ingreso",
                    "Ganancia",
                    "Modelo"
                ],
                "Venta_calculada": [
                    max(costo_total, 0),
                    max(venta_calculada, 0),
                    max(ganancia, 0),
                    max(resultado, 0)
                ],
                "Ganancia_estimada": [
                    0,
                    ganancia,
                    ganancia,
                    ganancia
                ],
                "Cantidad": [
                    cantidad,
                    cantidad,
                    cantidad,
                    cantidad
                ],
                "Prediccion_Modelo": [
                    resultado,
                    resultado,
                    resultado,
                    resultado
                ]
            })

            mostrar_graficos_avanzados(df_visual, "Treemap y distribución visual del resultado")

            with st.expander("Ver datos enviados al modelo"):
                st.dataframe(datos_modelo, use_container_width=True)

        except Exception as e:
            st.error("Ocurrió un error al realizar la predicción.")
            st.exception(e)


# ==========================================================
# MODO 2: WHAT IF INDIVIDUAL
# ==========================================================

elif modo == "What If individual":

    st.markdown('<div class="section-title">What If Individual Dinámico</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box-blue">
        Este módulo simula múltiples escenarios para un producto. 
        Permite cambiar cantidad, descuento y variación del precio para analizar cómo cambia la ganancia,
        el margen y la predicción del modelo.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Datos base del producto")

    col1, col2, col3 = st.columns(3)

    with col1:
        cantidad_base = st.number_input("Cantidad base", min_value=1, value=10, step=1)

    with col2:
        precio_base = st.number_input("Precio general base", min_value=0.0, value=100.0, step=1.0)

    with col3:
        precio_costo_base = st.number_input("Precio costo base", min_value=0.0, value=75.0, step=1.0)

    col4, col5, col6 = st.columns(3)

    with col4:
        precio_venta_base = st.number_input("Precio venta base", min_value=0.0, value=100.0, step=1.0)

    with col5:
        precio_mayorista_base = st.number_input("Precio mayorista base", min_value=0.0, value=90.0, step=1.0)

    with col6:
        descuento_actual = st.number_input("Descuento actual", min_value=0.0, value=0.0, step=1.0)

    col7, col8, col9 = st.columns(3)

    with col7:
        idbodega_base = st.number_input("ID Bodega", min_value=1, value=1, step=1)

    with col8:
        id_medida_base = st.number_input("ID Medida", min_value=1, value=1, step=1)

    with col9:
        estado_base = st.selectbox("Estado del producto", ["Activo", "Desconocido"])

    st.subheader("Rangos de simulación")

    col10, col11, col12 = st.columns(3)

    with col10:
        descuento_min = st.number_input("Descuento mínimo", min_value=0, value=0, step=1)

    with col11:
        descuento_max = st.number_input("Descuento máximo", min_value=0, value=30, step=1)

    with col12:
        paso_descuento = st.number_input("Paso de descuento", min_value=1, value=5, step=1)

    col13, col14, col15 = st.columns(3)

    with col13:
        variacion_precio_min = st.number_input("Variación precio mínima (%)", value=-20, step=5)

    with col14:
        variacion_precio_max = st.number_input("Variación precio máxima (%)", value=30, step=5)

    with col15:
        paso_variacion_precio = st.number_input("Paso variación precio (%)", min_value=1, value=5, step=1)

    col16, col17, col18 = st.columns(3)

    with col16:
        cantidad_min = st.number_input("Cantidad mínima", min_value=1, value=1, step=1)

    with col17:
        cantidad_max = st.number_input("Cantidad máxima", min_value=1, value=30, step=1)

    with col18:
        paso_cantidad = st.number_input("Paso cantidad", min_value=1, value=5, step=1)

    descuentos_preview = lista_rango_seguro(descuento_min, descuento_max, paso_descuento)
    variaciones_preview = lista_rango_seguro(variacion_precio_min, variacion_precio_max, paso_variacion_precio)
    cantidades_preview = lista_rango_seguro(cantidad_min, cantidad_max, paso_cantidad)

    total_preview = len(descuentos_preview) * len(variaciones_preview) * len(cantidades_preview)

    st.info(f"Escenarios que se van a generar: {total_preview}")

    if total_preview > 5000:
        st.warning("Estás intentando generar más de 5,000 escenarios. Reduce los rangos o aumenta los pasos.")

    if descuento_max < descuento_min:
        st.error("El descuento máximo no puede ser menor que el descuento mínimo.")
        st.stop()

    if variacion_precio_max < variacion_precio_min:
        st.error("La variación máxima del precio no puede ser menor que la variación mínima.")
        st.stop()

    if cantidad_max < cantidad_min:
        st.error("La cantidad máxima no puede ser menor que la cantidad mínima.")
        st.stop()

    if st.button("Ejecutar What If Individual", use_container_width=True):

        if total_preview == 0:
            st.error("No se generaron escenarios. Revisa los rangos configurados.")
            st.stop()

        if total_preview > 5000:
            st.error("Demasiados escenarios. Baja el rango o aumenta el paso para continuar.")
            st.stop()

        st.markdown('<div class="section-title">Procesando What If</div>', unsafe_allow_html=True)

        barra_progreso = st.progress(0)
        texto_progreso = st.empty()

        df_what_if, total_escenarios, errores = generar_what_if_individual(
            modelo=modelo,
            cantidad_base=cantidad_base,
            precio_base=precio_base,
            precio_costo_base=precio_costo_base,
            precio_venta_base=precio_venta_base,
            precio_mayorista_base=precio_mayorista_base,
            idbodega_base=idbodega_base,
            id_medida_base=id_medida_base,
            estado_base=estado_base,
            descuento_min=descuento_min,
            descuento_max=descuento_max,
            paso_descuento=paso_descuento,
            variacion_precio_min=variacion_precio_min,
            variacion_precio_max=variacion_precio_max,
            paso_variacion_precio=paso_variacion_precio,
            cantidad_min=cantidad_min,
            cantidad_max=cantidad_max,
            paso_cantidad=paso_cantidad,
            barra_progreso=barra_progreso,
            texto_progreso=texto_progreso
        )

        barra_progreso.progress(1.0)
        texto_progreso.success(
            f"What If finalizado. Escenarios procesados: {total_escenarios}. Errores de predicción: {errores}."
        )

        if df_what_if.empty:
            st.error("No se pudieron generar resultados para el What If.")
            st.stop()

        st.divider()

        st.markdown('<div class="section-title">Resultado del What If Individual</div>', unsafe_allow_html=True)

        mejor_escenario = df_what_if.sort_values("Ganancia_estimada", ascending=False).iloc[0]
        peor_escenario = df_what_if.sort_values("Ganancia_estimada", ascending=True).iloc[0]
        promedio_ganancia = df_what_if["Ganancia_estimada"].mean()
        promedio_prediccion = df_what_if["Prediccion_Modelo"].mean()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("Escenarios generados", total_escenarios)

        with col_m2:
            st.metric("Mejor ganancia", f"{mejor_escenario['Ganancia_estimada']:.2f}")

        with col_m3:
            st.metric("Ganancia promedio", f"{promedio_ganancia:.2f}")

        with col_m4:
            st.metric("Predicción promedio", f"{promedio_prediccion:.2f}")

        st.markdown(
            f"""
            <div class="info-box-green">
            El mejor escenario encontrado tiene una cantidad de 
            <b>{mejor_escenario['Cantidad']:.0f}</b> unidades, una variación de precio de 
            <b>{mejor_escenario['Variacion_precio_%']:.0f}%</b> y un descuento de 
            <b>{mejor_escenario['Descuento']:.2f}</b>. 
            Con esta combinación, la ganancia estimada sería de 
            <b>{mejor_escenario['Ganancia_estimada']:.2f}</b>, con un margen de 
            <b>{mejor_escenario['Margen_estimado']:.2f}%</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-box-red">
            El escenario menos conveniente presenta una ganancia estimada de 
            <b>{peor_escenario['Ganancia_estimada']:.2f}</b>. 
            Esto permite identificar combinaciones de precio, descuento o cantidad que afectan negativamente la rentabilidad.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">Gráficos What If</div>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            df_what_if,
            x="Descuento",
            y="Ganancia_estimada",
            size="Cantidad",
            color="Variacion_precio_%",
            hover_data=[
                "Precio_venta_simulado",
                "Precio_final",
                "Margen_estimado",
                "Prediccion_Modelo",
                "Estado_Comercial"
            ],
            title="Escenarios What If: descuento, cantidad y ganancia"
        )

        fig_scatter.add_hline(y=0, line_dash="dash")
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_heatmap = px.density_heatmap(
            df_what_if,
            x="Descuento",
            y="Variacion_precio_%",
            z="Ganancia_estimada",
            histfunc="avg",
            title="Mapa de calor: ganancia según descuento y variación de precio",
            text_auto=True
        )

        fig_heatmap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        resumen_estado = df_what_if["Estado_Comercial"].value_counts().reset_index()
        resumen_estado.columns = ["Estado_Comercial", "Cantidad"]

        fig_pie = px.pie(
            resumen_estado,
            names="Estado_Comercial",
            values="Cantidad",
            hole=0.45,
            title="Distribución de escenarios por estado comercial"
        )

        fig_pie.update_traces(textposition="inside", textinfo="percent+label")

        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        mostrar_graficos_avanzados(df_what_if, "Treemap y dona del What If individual")

        st.markdown('<div class="section-title">Tabla de Escenarios Simulados</div>', unsafe_allow_html=True)

        st.dataframe(df_what_if, use_container_width=True)

        csv_what_if = df_what_if.to_csv(index=False).encode("utf-8")
        excel_what_if = convertir_excel(df_what_if)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                label="Descargar What If en CSV",
                data=csv_what_if,
                file_name="what_if_individual.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_d2:
            st.download_button(
                label="Descargar What If en Excel",
                data=excel_what_if,
                file_name="what_if_individual.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# ==========================================================
# MODO 3: PREDICCIÓN POR CSV
# ==========================================================

elif modo == "Predicción por CSV":

    st.markdown('<div class="section-title">Carga de Base de Datos CSV</div>', unsafe_allow_html=True)

    st.markdown(
        """
        En este apartado se puede cargar una base de datos en formato **CSV** para probar el modelo con varios registros.
        """
    )

    archivo_csv = st.file_uploader(
        "Seleccione el archivo CSV",
        type=["csv"]
    )

    if archivo_csv is not None:

        try:
            df_csv = pd.read_csv(archivo_csv)

            st.success("Archivo CSV cargado correctamente.")

            st.subheader("Vista previa de la base cargada")
            st.dataframe(df_csv.head(20), use_container_width=True)

            columnas_faltantes = [col for col in COLUMNAS_MODELO if col not in df_csv.columns]

            if columnas_faltantes:
                st.warning(
                    f"""
                    El archivo no contiene todas las columnas requeridas.
                    El sistema intentará completarlas con valores por defecto.

                    Columnas faltantes: {columnas_faltantes}
                    """
                )

            df_modelo = preparar_datos(df_csv)

            df_modelo["Precio_descuento"] = df_modelo["Precio_venta"] - df_modelo["Descuento"]
            df_modelo["Precio_descuento"] = df_modelo["Precio_descuento"].apply(lambda x: max(0, x))

            if "Servicio" in df_csv.columns:
                df_modelo["Servicio"] = df_csv["Servicio"].astype(str)
            else:
                df_modelo["Servicio"] = (
                    "Bodega " + df_modelo["Idbodega"].astype(str) +
                    " - Medida " + df_modelo["Id_Medida"].astype(str)
                )

            if "Categoria" in df_csv.columns:
                df_modelo["Categoria"] = df_csv["Categoria"].astype(str)
            else:
                df_modelo["Categoria"] = df_modelo["Estado"].astype(str)

            if st.button("Procesar CSV con el modelo", use_container_width=True):

                st.markdown('<div class="section-title">Procesando CSV</div>', unsafe_allow_html=True)

                barra_csv = st.progress(0)
                texto_csv = st.empty()

                predicciones, errores_csv = procesar_csv_con_progreso(
                    modelo=modelo,
                    df_modelo=df_modelo[COLUMNAS_MODELO],
                    barra_progreso=barra_csv,
                    texto_progreso=texto_csv
                )

                barra_csv.progress(1.0)
                texto_csv.success(
                    f"Procesamiento finalizado. Registros: {len(df_modelo)}. Errores: {errores_csv}."
                )

                df_resultados = df_modelo.copy()
                df_resultados["Prediccion_Modelo"] = predicciones

                indicadores = df_resultados.apply(calcular_indicadores, axis=1, result_type="expand")
                indicadores.columns = [
                    "Precio_descuento_calculado",
                    "Costo_total",
                    "Venta_calculada",
                    "Ganancia_estimada",
                    "Margen_estimado"
                ]

                df_resultados = pd.concat([df_resultados, indicadores], axis=1)

                df_resultados["Estado_Comercial"] = df_resultados.apply(
                    lambda fila: generar_estado_comercial(
                        fila["Ganancia_estimada"],
                        fila["Margen_estimado"]
                    )[0],
                    axis=1
                )

                st.divider()

                st.markdown('<div class="section-title">Resultados Generales del CSV</div>', unsafe_allow_html=True)

                total_registros = len(df_resultados)
                venta_total = df_resultados["Venta_calculada"].sum()
                ganancia_total = df_resultados["Ganancia_estimada"].sum()
                promedio_prediccion = df_resultados["Prediccion_Modelo"].mean()
                margen_promedio = df_resultados["Margen_estimado"].mean()

                col_csv1, col_csv2, col_csv3, col_csv4 = st.columns(4)

                with col_csv1:
                    st.metric("Registros procesados", total_registros)

                with col_csv2:
                    st.metric("Venta total", f"{venta_total:.2f}")

                with col_csv3:
                    st.metric("Ganancia total", f"{ganancia_total:.2f}")

                with col_csv4:
                    st.metric("Margen promedio", f"{margen_promedio:.2f}%")

                if ganancia_total > 0:
                    st.markdown(
                        f"""
                        <div class="info-box-green">
                        La base cargada presenta una ganancia total positiva de 
                        <b>{ganancia_total:.2f}</b>. En términos generales, los productos evaluados muestran
                        un comportamiento comercial favorable.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="info-box-red">
                        La base cargada presenta una pérdida total estimada de 
                        <b>{abs(ganancia_total):.2f}</b>. Se recomienda revisar descuentos, precios de venta 
                        y costos de los productos incluidos en el archivo.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown('<div class="section-title">Dashboard del Archivo CSV</div>', unsafe_allow_html=True)

                col_csv_g1, col_csv_g2 = st.columns(2)

                with col_csv_g1:
                    resumen_estado = df_resultados["Estado_Comercial"].value_counts().reset_index()
                    resumen_estado.columns = ["Estado_Comercial", "Cantidad"]

                    fig_estado = px.pie(
                        resumen_estado,
                        names="Estado_Comercial",
                        values="Cantidad",
                        title="Distribución de resultados comerciales",
                        hole=0.45
                    )

                    fig_estado.update_traces(textposition="inside", textinfo="percent+label")

                    fig_estado.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white"
                    )

                    st.plotly_chart(fig_estado, use_container_width=True)

                with col_csv_g2:
                    fig_ganancia = px.histogram(
                        df_resultados,
                        x="Ganancia_estimada",
                        nbins=20,
                        title="Distribución de ganancias estimadas"
                    )

                    fig_ganancia.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white"
                    )

                    st.plotly_chart(fig_ganancia, use_container_width=True)

                fig_scatter = px.scatter(
                    df_resultados,
                    x="Precio_venta",
                    y="Ganancia_estimada",
                    size="Cantidad",
                    color="Estado_Comercial",
                    title="Relación entre precio de venta, cantidad y ganancia",
                    hover_data=[
                        "Servicio",
                        "Cantidad",
                        "Descuento",
                        "Precio_costo",
                        "Prediccion_Modelo"
                    ]
                )

                fig_scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white"
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

                mostrar_graficos_avanzados(df_resultados, "Treemap, dona y análisis ABC por ingresos")

                st.markdown('<div class="section-title">What If Global del CSV</div>', unsafe_allow_html=True)

                ajuste_descuento = st.slider(
                    "Simular aumento o reducción global del descuento",
                    min_value=-20,
                    max_value=50,
                    value=0,
                    step=5
                )

                ajuste_precio = st.slider(
                    "Simular aumento o reducción global del precio de venta (%)",
                    min_value=-30,
                    max_value=50,
                    value=0,
                    step=5
                )

                df_escenario = df_modelo.copy()

                df_escenario["Descuento"] = df_escenario["Descuento"] + ajuste_descuento
                df_escenario["Descuento"] = df_escenario["Descuento"].apply(lambda x: max(0, x))

                df_escenario["Precio_venta"] = df_escenario["Precio_venta"] * (1 + ajuste_precio / 100)
                df_escenario["Precio_descuento"] = df_escenario["Precio_venta"] - df_escenario["Descuento"]
                df_escenario["Precio_descuento"] = df_escenario["Precio_descuento"].apply(lambda x: max(0, x))

                barra_if_csv = st.progress(0)
                texto_if_csv = st.empty()

                pred_escenario, errores_if_csv = procesar_csv_con_progreso(
                    modelo=modelo,
                    df_modelo=df_escenario[COLUMNAS_MODELO],
                    barra_progreso=barra_if_csv,
                    texto_progreso=texto_if_csv
                )

                barra_if_csv.progress(1.0)
                texto_if_csv.success(f"What If global procesado. Errores: {errores_if_csv}.")

                df_escenario["Prediccion_Modelo"] = pred_escenario

                indicadores_escenario = df_escenario.apply(calcular_indicadores, axis=1, result_type="expand")
                indicadores_escenario.columns = [
                    "Precio_descuento_calculado",
                    "Costo_total",
                    "Venta_calculada",
                    "Ganancia_estimada",
                    "Margen_estimado"
                ]

                df_escenario = pd.concat([df_escenario, indicadores_escenario], axis=1)

                df_escenario["Estado_Comercial"] = df_escenario.apply(
                    lambda fila: generar_estado_comercial(
                        fila["Ganancia_estimada"],
                        fila["Margen_estimado"]
                    )[0],
                    axis=1
                )

                venta_total_escenario = df_escenario["Venta_calculada"].sum()
                ganancia_total_escenario = df_escenario["Ganancia_estimada"].sum()

                col_if1, col_if2, col_if3 = st.columns(3)

                with col_if1:
                    st.metric("Venta original", f"{venta_total:.2f}")

                with col_if2:
                    st.metric("Venta simulada", f"{venta_total_escenario:.2f}")

                with col_if3:
                    st.metric("Ganancia simulada", f"{ganancia_total_escenario:.2f}")

                comparativo = pd.DataFrame({
                    "Escenario": ["Original", "What If"],
                    "Venta total": [venta_total, venta_total_escenario],
                    "Ganancia total": [ganancia_total, ganancia_total_escenario]
                })

                fig_comp = px.bar(
                    comparativo,
                    x="Escenario",
                    y=["Venta total", "Ganancia total"],
                    barmode="group",
                    title="Comparación Original vs What If"
                )

                fig_comp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white"
                )

                st.plotly_chart(fig_comp, use_container_width=True)

                mostrar_graficos_avanzados(df_escenario, "Treemap y dona del escenario What If global")

                st.subheader("Tabla final con predicciones")
                st.dataframe(df_resultados, use_container_width=True)

                csv_descarga = df_resultados.to_csv(index=False).encode("utf-8")
                excel_descarga = convertir_excel(df_resultados)

                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    st.download_button(
                        label="Descargar resultados en CSV",
                        data=csv_descarga,
                        file_name="resultados_prediccion.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col_d2:
                    st.download_button(
                        label="Descargar resultados en Excel",
                        data=excel_descarga,
                        file_name="resultados_prediccion.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        except Exception as e:
            st.error("No se pudo procesar el archivo CSV.")
            st.exception(e)


# ==========================================================
# MODO 4: ANÁLISIS DEL MODELO
# ==========================================================

else:

    st.markdown('<div class="section-title">Análisis General del Modelo</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box-blue">
        Esta sección muestra información interpretativa sobre el modelo cargado, las variables utilizadas
        y la lógica general del Random Forest.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Información técnica del modelo")

    st.write("Tipo de objeto cargado:")
    st.code(str(type(modelo)))

    st.write("Columnas esperadas por el modelo:")
    st.dataframe(pd.DataFrame({"Columnas": COLUMNAS_MODELO}), use_container_width=True)

    importancias = obtener_importancias_modelo(modelo)

    if importancias is not None:

        st.subheader("Importancia de variables")

        fig_imp = px.bar(
            importancias,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Variables con mayor peso en el modelo"
        )

        fig_imp.update_layout(
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_imp, use_container_width=True)

        variable_top = importancias.iloc[0]["Variable"]

        st.markdown(
            f"""
            <div class="info-box-green">
            La variable con mayor peso relativo en el modelo es <b>{variable_top}</b>. 
            Esto indica que dicha característica tuvo mayor influencia en las decisiones internas del Random Forest.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div class="info-box-yellow">
            No fue posible extraer las importancias internas del modelo. 
            Esto puede ocurrir si el archivo PKL contiene un Pipeline con transformaciones internas 
            o si el estimador no expone directamente el atributo feature_importances_.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Interpretación del funcionamiento")

    st.markdown(
        """
        El modelo Random Forest trabaja mediante un conjunto de árboles de decisión. 
        Cada árbol evalúa variables como cantidad, precio, descuento, costo, bodega, medida y estado. 
        Luego, el resultado final se obtiene combinando las respuestas de todos los árboles.

        En este sistema, el modelo se complementa con reglas comerciales para interpretar si la operación 
        es rentable, moderada, está en punto de equilibrio o representa una posible pérdida.
        """
    )

    st.subheader("Variables utilizadas")

    df_variables = pd.DataFrame({
        "Variable": COLUMNAS_MODELO,
        "Descripción": [
            "Cantidad de unidades del producto.",
            "Descuento aplicado al producto.",
            "Precio general o base del producto.",
            "Costo de adquisición del producto.",
            "Precio de venta definido para el cliente.",
            "Precio final después de aplicar descuento.",
            "Precio utilizado para ventas mayoristas.",
            "Identificador de la bodega.",
            "Identificador de la unidad de medida.",
            "Estado del producto dentro del sistema."
        ]
    })

    st.dataframe(df_variables, use_container_width=True)

    st.markdown(
        """
        <div class="info-box-blue">
        Este análisis puede utilizarse en una presentación para explicar que el modelo fue entrenado con datos
        procesados mediante ETL, limpieza y transformación, y luego exportado como archivo PKL para integrarse
        en Streamlit.
        </div>
        """,
        unsafe_allow_html=True
    )