# Sistema Predictivo Retail Inteligente

Aplicación web desarrollada con **Streamlit** para realizar predicción de ventas, análisis comercial, simulaciones **What If**, carga de datos por archivo CSV y visualización de resultados mediante gráficos interactivos.

El sistema integra un modelo de Machine Learning entrenado previamente con datos procesados mediante limpieza, transformación y preparación de datos. El modelo se exporta como archivo `.pkl` y se utiliza dentro de la aplicación para generar predicciones a partir de variables comerciales del producto.

---

## Descripción del proyecto

Este proyecto tiene como finalidad apoyar la toma de decisiones comerciales mediante un sistema predictivo capaz de analizar productos, precios, costos, descuentos, cantidades, bodegas, medidas y estado del producto.

La aplicación permite evaluar si una operación comercial puede considerarse rentable, moderada, en punto de equilibrio o con pérdida, combinando la predicción del modelo con indicadores financieros calculados automáticamente.

---

## Funcionalidades principales

### Predicción individual

Permite ingresar manualmente los datos de un producto y obtener una predicción del modelo. Además, calcula indicadores comerciales como:

- Costo total.
- Venta calculada.
- Ganancia estimada.
- Margen estimado.
- Clasificación comercial.
- Explicación automática del resultado.

### What If individual

Permite simular múltiples escenarios para un producto modificando:

- Cantidad.
- Descuento.
- Variación porcentual del precio de venta.

Esta sección ayuda a responder preguntas como:

- ¿Qué pasa si aumento el precio?
- ¿Qué pasa si aplico mayor descuento?
- ¿Qué combinación genera mejor ganancia?
- ¿Qué escenario provoca pérdida?

### Predicción por CSV

Permite cargar una base de datos en formato `.csv` para procesar varios registros de productos de forma masiva.

El sistema genera:

- Predicción por cada registro.
- Indicadores comerciales por producto.
- Clasificación comercial.
- Dashboard de resultados.
- Descarga de resultados en CSV.
- Descarga de resultados en Excel.

### Análisis del modelo

Incluye una sección para visualizar información técnica del modelo, columnas utilizadas, interpretación del funcionamiento y, cuando es posible, la importancia de variables.

---

## Visualizaciones incluidas

La aplicación incorpora gráficos interactivos con Plotly, entre ellos:

- Gráfico de barras de costo, venta y ganancia.
- Gráfico de composición de precio.
- Velocímetro de confianza del análisis.
- Gráfico de dispersión de escenarios.
- Mapa de calor para análisis What If.
- Gráfico de dona para distribución porcentual.
- Treemap de ingresos por servicio.
- Ranking ABC por ingresos.
- Histograma de ganancias estimadas.
- Comparación entre escenario original y escenario What If.

---

## Flujo general del sistema

1. Se parte de una base de datos inicial con registros comerciales.
2. Los datos son limpiados, transformados y organizados mediante procesos ETL.
3. La base limpia se utiliza para entrenamiento del modelo predictivo.
4. El modelo entrenado se exporta como archivo `.pkl`.
5. El archivo `.pkl` se integra en una aplicación Streamlit.
6. El usuario puede realizar predicciones individuales o por lote.
7. El sistema muestra métricas, gráficos y recomendaciones para la toma de decisiones.

---

## Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Plotly
- OpenPyXL
- Machine Learning
- Random Forest
- ETL
- Data Mining

---

## Estructura sugerida del proyecto

```text
analiticadatos10/
│
├── app_random_forest.py
├── modelo_predictor_ventas_random_forest.pkl
├── requirements.txt
├── README.md
├── LICENSE
└── data/
    └── prueba_productos_masiva_random_forest.csv