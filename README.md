# Tennis Predictor

Este repositorio contiene un pipeline completo para predecir partidos de tenis de la ATP. El proceso incluye preprocesado de datos, cálculo de características avanzadas (como ELO y rachas), entrenamiento de un modelo de Machine Learning y una interfaz visual para realizar predicciones.

## Características Principales

- **Preprocesado Automático:** Limpia y transforma el dataset original, manejando fechas, valores nulos y codificando variables categóricas.
- **Ingeniería de Características Avanzada:**
  - **Estadísticas Históricas:** Calcula rachas de victorias (últimos 5 partidos), tanto generales como por superficie.
  - **Head-to-Head (H2H):** Ratio de victorias histórico entre los dos jugadores.
  - **ELO Dinámico por Superficie:** Implementa un sistema de calificación ELO que se ajusta después de cada partido y es específico para cada superficie (Hard, Clay, Grass).
  - **Features de Ranking y Puntos:** Diferencias y ratios de puntos y ranking ATP.
- **Entrenamiento Robusto:**
  - **Modelo:** Utiliza un `RandomForestClassifier`.
  - **Optimización:** Búsqueda de hiperparámetros con `RandomizedSearchCV`.
  - **Validación Temporal:** Emplea `TimeSeriesSplit` para validar el modelo de forma cronológica, evitando fugas de datos del futuro.
- **Interfaz Visual:** Una aplicación creada con Streamlit para explorar el dataset y realizar predicciones de forma interactiva.
- **Predicción por CLI:** Un script para hacer predicciones directamente desde la terminal.

## Estructura del Repositorio

```
├─── app/
│    └─── streamlit_app.py       # Interfaz visual con Streamlit
├─── data/
│    ├─── atp_preprocessed.csv   # Datos procesados (salida de preprocess.py)
│    └─── elo_ratings.json       # Calificaciones ELO finales (salida de preprocess.py)
├─── models/
│    └─── rf_model.joblib        # Modelo RandomForest entrenado
├─── outputs/
│    ├─── metrics.txt            # Métricas de entrenamiento y mejores parámetros
│    └─── cv_results.csv         # Resultados completos de la validación cruzada
├─── scripts/
│    ├─── preprocess.py          # Script para limpiar y generar features
│    ├─── train_model.py         # Script para entrenar el modelo
│    └─── predict_match.py       # Script para predicciones por terminal
├─── requirements.txt             # Dependencias del proyecto
└─── README.md
```

## Flujo de Trabajo (Instrucciones)

Estas instrucciones asumen que trabajas desde la raíz del repositorio (`e:\coding\tennis-predictor`) en una terminal de PowerShell.

### 1. Instalación

Primero, crea un entorno virtual y activa las dependencias.

```powershell
# 1. Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Preprocesado de Datos

Este script lee el `atp_tennis.csv` (no incluido en el repo), lo procesa y genera dos archivos en la carpeta `data/`:
- `atp_preprocessed.pkl`: El dataset listo para el entrenamiento.
- `elo_ratings.json`: Un diccionario con las puntuaciones ELO finales de cada jugador por superficie.

```powershell
python .\scripts\preprocess.py
```

### 3. Entrenamiento del Modelo

El script carga los datos preprocesados, busca los mejores hiperparámetros usando validación cruzada temporal y entrena un `RandomForestClassifier` final.

```powershell
python .\scripts\train_model.py
```

**Salidas generadas:**
- `models/rf_model.joblib`: El modelo entrenado.
- `outputs/metrics.txt`: Resumen del rendimiento, mejores parámetros y features utilizadas.
- `outputs/cv_results.csv`: Log detallado de la búsqueda de hiperparámetros.

### 4. Realizar Predicciones (Terminal)

Puedes usar `predict_match.py` para obtener una predicción rápida.

**Ejemplo (modo demo, usa un partido aleatorio del dataset):**
```powershell
python .\scripts\predict_match.py --mode demo
```

**Ejemplo (partido personalizado):**
```powershell
python .\scripts\predict_match.py --mode custom --player1 "Djokovic N." --player2 "Nadal R." --date "2025-05-26" --surface "Clay" --rank1 1 --rank2 5 --pts1 12000 --pts2 8000
```

### 5. Lanzar la Interfaz Visual

Para una experiencia más interactiva, utiliza la aplicación de Streamlit.

```powershell
streamlit run .\app\streamlit_app.py
```

La aplicación se abrirá en tu navegador y te permitirá:
- **Modo Demo:** Predecir un partido aleatorio del dataset.
- **Modo Dataset:** Elegir un partido histórico y ver la predicción del modelo.
- **Modo Custom:** Introducir los datos de un partido nuevo para obtener una predicción.

```