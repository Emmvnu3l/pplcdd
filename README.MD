# Proyecto de Análisis de Cáncer de Piel - Chile

Proyecto de Machine Learning para análisis y clasificación de lesiones de piel usando un dataset de pacientes chilenos.

## Estructura del Proyecto

```
pplcdd/
├── data/
│   ├── raw/                    # Dataset original
│   │   └── dataset_chile_cancer_piel.csv
│   └── processed/              # Datos procesados
│       └── df_processed.csv
├── notebooks/                  # Notebooks Jupyter
│   ├── 01_exploratory_analysis.ipynb      # Análisis exploratorio
│   ├── 02_supervised_modeling.ipynb       # Modelado supervisado
│   ├── 03_model_evaluation.ipynb          # Evaluación de modelos
│   ├── 04_hyperparameter_optimization.ipynb  # Optimización
│   └── 05_final_analysis.ipynb            # Análisis final
├── src/                        # Módulos Python reutilizables
│   ├── data_preprocessing.py   # Preprocesamiento
│   ├── model_training.py       # Entrenamiento
│   ├── model_evaluation.py     # Evaluación
│   └── hyperparameter_tuning.py # Optimización
├── models/                     # Modelos entrenados
│   └── trained_models/         # Modelos serializados
├── results/                    # Resultados
│   ├── metrics/               # Métricas CSV
│   ├── plots/                 # Visualizaciones
│   └── reports/               # Reportes
└── README.md                   # Este archivo
```

## Requisitos

- Python 3.8 o superior

## Configuración del Entorno Virtual

Sigue estos pasos para configurar el entorno virtual y usarlo en todo el proyecto:

### 1. Crear el Entorno Virtual

Abre una terminal en la carpeta raíz del proyecto (`pplcdd/`) y ejecuta:

**Windows:**
```bash
python -m venv venv
```

**Linux/Mac:**
```bash
python3 -m venv venv
```

Esto creará una carpeta `venv/` con el entorno virtual aislado.

### 2. Activar el Entorno Virtual

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

Verás que el prompt cambia a algo como: `(venv) C:\Users\...\pplcdd>`

### 3. Instalar Dependencias

Con el entorno activado, instala todas las dependencias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 4. Configurar VS Code para usar el Entorno

Para que VS Code use el entorno virtual en todos los archivos:

1. Abre la paleta de comandos (`Ctrl+Shift+P`)
2. Escribe y selecciona: `Python: Select Interpreter`
3. Selecciona la ruta que apunta a `venv\Scripts\python.exe`

O presiona `Ctrl+Shift+P` → `Preferences: Open Settings (JSON)` y añade:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"
}
```

### 5. Verificar la Configuración

Crea un archivo de prueba o abre un notebook y ejecuta:

```python
import sys
print(sys.executable)
```

Debe mostrar la ruta al Python del entorno virtual, por ejemplo:
`C:\Users\...\pplcdd\venv\Scripts\python.exe`

### 6. Desactivar el Entorno

Cuando termines de trabajar:

```bash
deactivate
```

### Tips Importantes

- **Siempre activa el entorno** antes de ejecutar notebooks o scripts
- Si instalas nuevas librerías, hazlo con el entorno activado
- El entorno aísla las dependencias del proyecto del sistema global

## Uso

### Ejecución Secuencial

Los notebooks deben ejecutarse en orden:

1. **`01_exploratory_analysis.ipynb`**
   - Carga y análisis exploratorio del dataset
   - Visualizaciones descriptivas
   - Creación del target (diagnóstico)
   - Guarda datos procesados en `data/processed/`

2. **`02_supervised_modeling.ipynb`**
   - Entrenamiento de 7 modelos de clasificación
   - Regresión Logística, Árbol, Random Forest, etc.
   - Guarda modelos en `models/trained_models/`

3. **`03_model_evaluation.ipynb`**
   - Evaluación comparativa de modelos
   - Matrices de confusión
   - Curvas ROC y métricas avanzadas

4. **`04_hyperparameter_optimization.ipynb`**
   - Optimización con GridSearchCV y RandomizedSearchCV
   - Guarda modelos optimizados

5. **`05_final_analysis.ipynb`**
   - Integración de resultados
   - Conclusiones y trabajo futuro

### Uso de Módulos

Los módulos en `src/` pueden importarse en cualquier notebook:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

from data_preprocessing import load_processed_data, encode_categorical_features
from model_training import get_models_dict, train_and_evaluate_model
from model_evaluation import load_all_models, calculate_advanced_metrics
from hyperparameter_tuning import get_param_grids, run_grid_search
```

## Dataset

- **Nombre**: `dataset_chile_cancer_piel.csv`
- **Registros**: 1000 pacientes
- **Variables**: 27 características (numéricas y categóricas)
- **Target**: Variable binaria `diagnostico` (0: bajo riesgo, 1: alto riesgo)

### Variables Principales

**Demográficas:**
- Edad (años)
- Sexo
- Peso, Estatura, IMC

**Factores de riesgo:**
- Tabaquismo
- Consumo de alcohol
- Exposición solar crónica
- Inmunosupresión

**Características de la lesión:**
- Localización
- Tamaño máximo
- Tiempo de evolución
- Signos de alarma (úlceración, bordes irregulares, etc.)

## Modelos Implementados

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **SVM** (Support Vector Machine)
6. **K-Nearest Neighbors**
7. **Naive Bayes**

## Métricas de Evaluación

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Matthews Correlation Coefficient
- Cohen's Kappa

## Consideraciones Importantes

⚠️ **Desbalance del Target**: El dataset tiene un desbalance extremo (99% clase 1, 1% clase 0). Las métricas estándar como accuracy pueden ser engañosas. Se recomienda usar F1-Score y ROC-AUC para evaluación.

## Resultados

Los resultados se guardan en:
- **Métricas**: `results/metrics/*.csv`
- **Visualizaciones**: `results/plots/*.png`
- **Reportes**: `results/reports/*.txt`
- **Modelos**: `models/trained_models/*.joblib`

## Recomendaciones para Trabajo Futuro

1. Aplicar técnicas de resampling (SMOTE, undersampling)
2. Probar enfoques de cost-sensitive learning
3. Explorar ensemble methods más avanzados
4. Considerar validación cruzada estratificada
5. Investigar técnicas de anomaly detection para la clase minoritaria

