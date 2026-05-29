import streamlit as st
import pandas as pd
import joblib

# ==========================================
# CARGAR MODELO Y COLUMNAS
# ==========================================

modelo = joblib.load("modelo_predictor_ventas.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# ==========================================
# CONFIGURACIÓN STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Predicción de Demanda",
    layout="centered"
)

st.title("Sistema Predictivo de Demanda")

st.write(
    """
    Predicción de demanda futura basada en:
    producto, precio, descuentos y estacionalidad mensual.
    """
)

# ==========================================
# INPUTS PRINCIPALES
# ==========================================

st.subheader("Información del Producto")

id_producto = st.number_input(
    "ID Producto",
    min_value=1,
    step=1
)

precio_unitario = st.number_input(
    "Precio Unitario",
    min_value=0.0,
    value=100.0
)

descuento_porcentaje = st.slider(
    "Descuento (%)",
    min_value=0,
    max_value=100,
    value=0
)

# ==========================================
# FECHA FUTURA
# ==========================================

st.subheader("Período a Predecir")

año = st.selectbox(
    "Año",
    [2026, 2027, 2028, 2029, 2030]
)

mes_nombre = st.selectbox(
    "Mes",
    [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre"
    ]
)

# ==========================================
# CONVERSIÓN DE MES
# ==========================================

meses = {
    "Enero": 1,
    "Febrero": 2,
    "Marzo": 3,
    "Abril": 4,
    "Mayo": 5,
    "Junio": 6,
    "Julio": 7,
    "Agosto": 8,
    "Septiembre": 9,
    "Octubre": 10,
    "Noviembre": 11,
    "Diciembre": 12
}

mes = meses[mes_nombre]

# ==========================================
# VARIABLES TEMPORALES
# ==========================================

trimestre = (mes - 1) // 3 + 1

# Valores promedio/fijos
dia_semana = 0
hora = 12
monto_descuento = 0

# ==========================================
# VALORES FIJOS NECESARIOS
# ==========================================

# IMPORTANTE:
# Deben existir en el entrenamiento.
# Usa valores comunes/representativos.

sucursal = "Managua"
categoria_producto = "Electronica"

# ==========================================
# BOTÓN PREDICCIÓN
# ==========================================

if st.button("Predecir Demanda"):

    # ==========================================
    # CREAR DATAFRAME
    # ==========================================

    datos = pd.DataFrame({
        'id_producto': [id_producto],
        'precio_unitario': [precio_unitario],
        'descuento_porcentaje': [descuento_porcentaje],
        'monto_descuento': [monto_descuento],
        'mes': [mes],
        'trimestre': [trimestre],
        'año': [año],
        'dia_semana': [dia_semana],
        'hora': [hora],
        'sucursal': [sucursal],
        'categoria_producto': [categoria_producto]
    })

    # ==========================================
    # ONE HOT ENCODING
    # ==========================================

    datos = pd.get_dummies(
        datos,
        columns=[
            'sucursal',
            'categoria_producto'
        ]
    )

    # ==========================================
    # ASEGURAR COLUMNAS EXACTAS
    # ==========================================

    datos = datos.reindex(
        columns=columnas_modelo,
        fill_value=0
    )

    # ==========================================
    # PREDICCIÓN
    # ==========================================

    prediccion = modelo.predict(datos)

    demanda = max(0, prediccion[0])

    # ==========================================
    # RESULTADOS
    # ==========================================

    st.success(
        f"""
        Demanda estimada para {mes_nombre} de {año}:
        {demanda:.2f} unidades
        """
    )

    # ==========================================
    # RECOMENDACIÓN DE STOCK
    # ==========================================

    stock_recomendado = demanda * 1.15

    st.info(
        f"""
        Stock recomendado:
        {stock_recomendado:.0f} unidades
        (incluyendo margen de seguridad del 15%)
        """
    )