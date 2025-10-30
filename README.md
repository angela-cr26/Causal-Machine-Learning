# Implementación de causal UberML

## 📋 Descripción del proyecto

Este repositorio contiene una guía completa y práctica sobre la implementación de técnicas de **Causal Machine Learning** utilizando el paquete **CausalML** desarrollado por Uber. 
El material está diseñado para servir como la parte aplicativa de un informe sobre Causal ML.

## 🎯 Objetivos

- Proporcionar una comprensión profunda de los conceptos de Causal ML
- Explicar la implementación práctica de meta-learners (S, T, X, R-Learner)
- Demostrar casos de uso reales del mundo empresarial
- Ofrecer código ejecutable y bien documentado
- Incluir visualizaciones para facilitar la comprensión

## 📁 Estructura del Proyecto

```
.
├── README.md                           # Este archivo
├── causalml_implementacion.py          # Guía completa de implementación
├── causalml_ejemplos_practicos.py      # Casos de uso reales
├── causalml_guia_visual.py             # Generación de visualizaciones
└── visualizaciones/
    ├── viz_01_ml_vs_causal.png         # ML tradicional vs Causal ML
    ├── viz_02_metalearners.png         # Comparación de meta-learners
    ├── viz_03_uplift_dist.png          # Distribución de uplift
    ├── viz_04_uplift_curves.png        # Gain y Qini curves
    ├── viz_05_architecture.png         # Arquitectura de CausalML
    └── viz_06_timeline.png             # Timeline de implementación
```

## 🚀 Contenido Principal

### 1. **causalml_implementacion.py**

Documento principal que cubre:

#### Conceptos Fundamentales
- ¿Qué es Causal Machine Learning?
- CATE (Conditional Average Treatment Effect)
- ITE (Individual Treatment Effect)
- Diferencia con ML tradicional

#### Meta-Learners
- **S-Learner** (Single Learner)
  - Concepto: Un solo modelo con tratamiento como feature
  - Implementación paso a paso
  - Ventajas y desventajas
  - Código con ejemplos

- **T-Learner** (Two Learner)
  - Concepto: Dos modelos separados (control y tratamiento)
  - Implementación detallada
  - Cuándo usar este enfoque
  - Ejemplos prácticos

- **X-Learner** (Cross Learner)
  - Concepto: Algoritmo en 3 etapas
  - Imputación contrafactual
  - Mejor para datos desbalanceados
  - Implementación completa

- **R-Learner** (Robinson Learner)
  - Concepto: Orthogonalization
  - Residualización
  - Propiedades teóricas
  - Código explicado

#### Técnicas Avanzadas
- Propensity Score Estimation
- Propensity Score Matching
- Uplift Trees y Random Forests
- Múltiples tratamientos
- Optimización con costos

#### Evaluación
- Métricas específicas (Qini, AUUC)
- Gain Charts
- Sensitivity Analysis
- Feature Selection para Uplift

### 2. **causalml_ejemplos_practicos.py**

Tres casos de uso completos del mundo real:

#### Caso 1: Optimización de Campaña de Email Marketing
```python
Objetivo: Identificar clientes con alta probabilidad de responder
Features: edad, ingresos, engagement, compras previas
Resultado: Mejora de ROI mediante targeting selectivo
```

**Resultados del Ejemplo:**
- ATE Observado: 12.15 puntos porcentuales
- Mejora en targeting: 13.96 pp vs targeting aleatorio
- Ahorro de costos: 80% con targeting selectivo

#### Caso 2: Optimización de Descuentos
```python
Objetivo: Determinar a quién ofrecer descuentos
Segmentos identificados:
  - Sleeping Dogs: No dar descuento (efecto negativo)
  - Lost Causes: No responden
  - Persuadables: TARGET principal
  - Sure Things: Comprarían de todas formas
```

**Insight Clave:** No todos necesitan descuento para comprar

#### Caso 3: Personalización de Canales
```python
Objetivo: Asignar canal óptimo (Email, SMS, Push)
Análisis: Efectos heterogéneos por demografía
Resultado: Recomendaciones personalizadas por usuario
```

### 3. **causalml_guia_visual.py**

Genera 6 visualizaciones explicativas:

1. **ML vs Causal ML**: Diferencias conceptuales
2. **Meta-Learners**: Comparación visual de S/T/X/R-Learners
3. **Uplift Distribution**: Segmentación de usuarios
4. **Uplift Curves**: Gain y Qini curves
5. **Arquitectura CausalML**: Estructura del paquete
6. **Timeline**: Fases de implementación

## 📊 Visualizaciones Generadas

### 1. ML Tradicional vs Causal ML
Explica visualmente la diferencia entre predecir outcomes y estimar efectos causales.

### 2. Comparación de Meta-Learners
Diagrama de flujo de cada meta-learner con sus ventajas y desventajas.

### 3. Distribución de Uplift
Muestra la distribución heterogénea de efectos y segmentación de usuarios.

### 4. Curvas de Evaluación
Gain Curve y Qini Curve para evaluar performance del modelo.

### 5. Arquitectura de CausalML
Diagrama completo de módulos y métodos disponibles.

### 6. Timeline de Implementación
Guía de 12 semanas para implementación completa.

## 💻 Instalación y Uso

### Instalación de CausalML

```bash
# Instalación básica
pip install causalml

# Con todas las dependencias
pip install causalml[tf,plotting]

# Dependencias necesarias
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Ejecución de Ejemplos

```bash
# Generar visualizaciones
python causalml_guia_visual.py

# Ejecutar casos de uso
python causalml_ejemplos_practicos.py

# Ver implementación completa
# (causalml_implementacion.py es un documento educativo)
```

## 📖 Explicación de Código Clave

### Ejemplo: T-Learner Simple

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class TLearner:
    def __init__(self):
        self.model_0 = GradientBoostingRegressor()
        self.model_1 = GradientBoostingRegressor()
    
    def fit(self, X, treatment, y):
        # Separar datos por grupo
        idx_control = treatment == 0
        idx_treated = treatment == 1
        
        # Entrenar modelo para control
        self.model_0.fit(X[idx_control], y[idx_control])
        
        # Entrenar modelo para tratamiento
        self.model_1.fit(X[idx_treated], y[idx_treated])
        
        return self
    
    def predict_cate(self, X):
        # Predecir efecto individual
        y0 = self.model_0.predict(X)
        y1 = self.model_1.predict(X)
        
        # CATE = diferencia entre predicciones
        return y1 - y0
```

### Uso con CausalML Real

```python
from causalml.inference.meta import XGBTRegressor
from causalml.dataset import synthetic_data

# Generar datos sintéticos
y, X, treatment, tau, b, e = synthetic_data(
    mode=1, n=10000, p=10, sigma=1.0
)

# Entrenar T-Learner con XGBoost
learner = XGBTRegressor(random_state=42)
learner.fit(X, treatment, y)

# Predecir CATE
cate = learner.predict(X)

# Estimar ATE con intervalo de confianza
ate, ate_lb, ate_ub = learner.estimate_ate(X, treatment, y)
print(f'ATE: {ate[0]:.3f} (95% CI: {ate_lb[0]:.3f}, {ate_ub[0]:.3f})')
```

## 🎓 Conceptos Clave Explicados

### CATE (Conditional Average Treatment Effect)
```
τ(x) = E[Y(1) - Y(0) | X = x]
```
El efecto promedio del tratamiento para individuos con características X.

### ITE (Individual Treatment Effect)
```
τᵢ = Y(1)ᵢ - Y(0)ᵢ
```
El efecto del tratamiento para un individuo específico i.

### Propensity Score
```
e(x) = P(T=1 | X=x)
```
Probabilidad de recibir tratamiento dadas características observadas.

### Uplift
```
Uplift = P(Y=1 | T=1, X) - P(Y=1 | T=0, X)
```
Incremento en probabilidad de outcome debido al tratamiento.

## 📈 Métricas de Evaluación

### Qini Score
Mide la ganancia acumulada vs tratamiento aleatorio:
```python
from causalml.metrics import qini_score

qini = qini_score(y_true, uplift_pred, treatment)
```

### AUUC (Area Under Uplift Curve)
Área bajo la curva de uplift:
```python
from causalml.metrics import auuc_score

auuc = auuc_score(y_true, uplift_pred, treatment)
```

### Gain Chart
Visualiza ganancia incremental:
```python
from causalml.metrics import plot_gain

fig, ax = plot_gain(y_true, uplift_pred, treatment)
```

## 🔧 Workflow Completo

### 1. Preparación de Datos
```python
# Cargar y limpiar datos
df = pd.read_csv('data.csv')

# Definir features, treatment, outcome
X = df[feature_columns]
treatment = df['treatment']
y = df['outcome']

# Split train/test
X_train, X_test, t_train, t_test, y_train, y_test = \
    train_test_split(X, treatment, y, test_size=0.3)
```

### 2. Estimar Propensity Score
```python
from causalml.propensity import ElasticNetPropensityModel

pm = ElasticNetPropensityModel(n_fold=5, random_state=42)
propensity = pm.fit_predict(X_train, t_train)
```

### 3. Entrenar Modelos
```python
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor

# X-Learner con XGBoost
x_learner = BaseXRegressor(learner=XGBRegressor())
x_learner.fit(X_train, t_train, y_train, p=propensity)
```

### 4. Predecir CATE
```python
# Predecir efectos individuales
cate_pred = x_learner.predict(X_test)
```

### 5. Evaluar
```python
from causalml.metrics import qini_score, plot_gain

# Calcular métricas
qini = qini_score(y_test, cate_pred, t_test)

# Visualizar
fig, ax = plot_gain(y_test, cate_pred, t_test)
```

### 6. Definir Estrategia
```python
# Identificar usuarios de alto uplift
threshold = np.percentile(cate_pred, 80)
high_uplift = cate_pred >= threshold

# Aplicar targeting selectivo
target_users = df_test[high_uplift]
```

## 🎯 Casos de Uso Empresariales

### Marketing
- **Optimización de campañas**: Target usuarios que responderán
- **Personalización**: Mensaje/canal óptimo por usuario
- **Presupuesto**: Maximizar ROI con targeting selectivo

### E-commerce
- **Descuentos**: A quién dar descuento para maximizar profit
- **Promociones**: Identificar "persuadables"
- **Retención**: Intervenciones efectivas por segmento

### Producto
- **Features**: Qué features tienen mayor impacto
- **A/B testing**: Identificar heterogeneidad en efectos
- **Personalización**: Experiencia óptima por usuario
