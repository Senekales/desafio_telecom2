# Proyecto de Predicción de Churn de Clientes

## Propósito del Análisis

El objetivo principal de este proyecto es desarrollar y evaluar modelos de aprendizaje automático para predecir la **renuncia (churn)** de clientes en una empresa de telecomunicaciones. Al identificar a los clientes con alta probabilidad de abandonar, la empresa puede implementar estrategias de retención proactivas y personalizadas para minimizar la pérdida de clientes.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

*   `Notebook_Principal.ipynb`: Este cuaderno de Jupyter contiene todo el código para la carga de datos, preprocesamiento, análisis exploratorio de datos (EDA), modelado y evaluación de los modelos.
*   `TablaTratadaTelecom.csv`: Archivo CSV que contiene los datos de los clientes después de un tratamiento inicial (este archivo se utiliza como entrada en el cuaderno principal).
*   `TelecomX_diccionario.md`: Diccionario de datos que describe las variables presentes en el dataset.
*   `visualizaciones/`: (Opcional) Carpeta para almacenar gráficos y visualizaciones generadas durante el EDA y la evaluación del modelo.

## Preparación de los Datos

El proceso de preparación de los datos incluyó las siguientes etapas:

1.  **Carga y Exploración Inicial:** Se cargaron los datos desde `TablaTratadaTelecom.csv` en un DataFrame de pandas. Se realizó una exploración inicial para entender la estructura de los datos, identificar valores faltantes y visualizar la información de cada columna.
2.  **Manejo de Valores Faltantes:** Se identificó la presencia de valores faltantes en las columnas `Total.Day` y `account.Charges.Total`. Al analizar las filas con datos faltantes, se observó que correspondían a clientes con una permanencia (`customer.tenure`) de 0 meses. Dado que representan un pequeño porcentaje del total de registros, se optó por eliminar estas filas para mantener la integridad de los datos para el análisis de clientes activos.
3.  **Eliminación de Columnas Irrelevantes:** La columna `customerID` se eliminó ya que es un identificador único de cliente y no aporta información relevante para la predicción del churn.
4.  **Clasificación de Variables:** Las variables se clasificaron en numéricas (`customer.SeniorCitizen`, `customer.tenure`, `account.Charges.Monthly`, `account.Charges.Total`) y categóricas (el resto de las columnas, de tipo `object`).
5.  **Codificación de Variables Categóricas:** Se aplicó **One-Hot Encoding** a las variables categóricas para convertirlas en un formato numérico que pudiera ser utilizado por los modelos de aprendizaje automático. Se eligió One-Hot Encoding porque las variables categóricas no tienen un orden intrínseco y el número de categorías por columna es relativamente bajo.
6.  **Manejo de Colinearidad:** Se analizó la matriz de correlación para identificar variables altamente correlacionadas. Se observó una fuerte correlación entre `Total.Day` y `account.Charges.Monthly`. Para evitar la colinearidad, se eliminó la columna `Total.Day`, manteniendo la dimensión mensual de los cargos. Posteriormente, se identificaron y eliminaron columnas resultantes del One-Hot Encoding que presentaban correlaciones absolutas unitarias, dejando una sola columna para representar la información categórica original (ej. se eliminó 'Churn\_No' manteniendo 'Churn\_Yes').
7.  **Separación en Conjuntos de Entrenamiento y Prueba:** Los datos procesados se dividieron en conjuntos de entrenamiento (70%) y prueba (30%) utilizando `train_test_split` de scikit-learn. Se utilizó `stratify=y` para asegurar que la proporción de la variable objetivo (`Churn_Yes`) fuera similar en ambos conjuntos, lo cual es importante dado el desbalance de clases.
8.  **Balanceo de Clases (Submuestreo):** Para abordar el desbalance en la variable objetivo (`Churn_Yes`), se aplicó **Random Under-Sampling** al conjunto de entrenamiento. Esto reduce el número de instancias de la clase mayoritaria (clientes que no renuncian) para igualar el número de instancias de la clase minoritaria (clientes que renuncian), ayudando a los modelos a aprender mejor las características de la clase minoritaria.

## Modelado y Justificaciones

Se entrenaron y evaluaron tres modelos:

1.  **DummyRegressor (Línea de Base):** Un modelo simple que predice la media de la variable objetivo. Sirve como punto de referencia para comparar el rendimiento de modelos más complejos. La justificación para incluir este modelo es establecer un umbral de rendimiento mínimo.
2.  **RandomForestRegressor (Sin optimizar):** Se entrenó un modelo inicial de Bosque Aleatorio para predecir la probabilidad de churn. La elección de este modelo se basa en su capacidad para manejar datos no lineales y su robustez. Se evaluó su rendimiento inicial para establecer una base para la optimización.
3.  **RandomForestClassifier (Regresión Logística y Optimizado):** Se exploraron modelos de clasificación como Regresión Logística y Bosque Aleatorio optimizado. Se utilizó `GridSearchCV` con validación cruzada para encontrar los mejores hiperparámetros del modelo de Bosque Aleatorio, optimizando la métrica **recall** para la clase "Yes" (clientes que renuncian). La justificación para optimizar el recall es que, en un problema de predicción de churn, es más crítico identificar correctamente a los clientes que van a abandonar (reducir falsos negativos) que tener una precisión general muy alta. Un alto recall permite a la empresa llegar a la mayor cantidad posible de clientes en riesgo.

## Análisis Exploratorio de Datos (EDA) e Insights

Durante el EDA, se obtuvieron insights importantes sobre las características de los clientes y su relación con el churn. Algunos ejemplos de visualizaciones y hallazgos incluyen:

*   **Distribución de Churn:** Se visualizó la proporción de clientes que renunciaron versus los que no, confirmando el desbalance de clases.
*   **Correlación de Variables Numéricas:** Se generó un mapa de calor de la matriz de correlación de las variables numéricas, mostrando la alta correlación entre `Total.Day` y `account.Charges.Monthly`, lo que llevó a la decisión de eliminar una de ellas.
*   **Correlación de Variables Categóricas:** Se visualizó la matriz de correlación de las variables categóricas codificadas, identificando las correlaciones perfectas (1.0 o -1.0) que guiaron la eliminación de columnas para evitar la colinearidad.

## Cómo Ejecutar el Cuaderno

Para ejecutar el cuaderno `Notebook_Principal.ipynb`, sigue estos pasos:

1.  **Clonar el Repositorio:** Clona este repositorio en tu máquina local o abre el cuaderno en Google Colab.
2.  **Instalar Bibliotecas:** Asegúrate de tener las siguientes bibliotecas de Python instaladas. Puedes instalarlas usando pip:
