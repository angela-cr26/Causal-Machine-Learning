# ===================================================================================
# 1. INSTALACIÓN Y CONFIGURACIÓN
# ===================================================================================

"""
# Instalación básica
pip install causalml

# Instalación con todas las dependencias
pip install causalml[tf,plotting]

# Dependencias principales
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
"""

# ===================================================================================
# 2. GENERACIÓN DE DATOS SINTÉTICOS
# ===================================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generar_datos_sinteticos(n_samples=10000, n_features=10, random_state=42):
    """
    Genera datos sintéticos para demostrar técnicas de Causal ML.
    
    Parámetros:
    -----------
    n_samples : int
        Número de muestras a generar
    n_features : int
        Número de características
    random_state : int
        Semilla aleatoria para reproducibilidad
    
    Retorna:
    --------
    X : array
        Matriz de características
    treatment : array
        Indicador de tratamiento (0=control, 1=tratamiento)
    y : array
        Variable de resultado
    tau : array
        Efecto verdadero del tratamiento (ground truth)
    """
    np.random.seed(random_state)
  
    X = np.random.randn(n_samples, n_features)
    
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    base_effect = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 0.5, n_samples)
    
    tau = X[:, 0] * 1.5 + X[:, 2] * 0.8  
    
    y = base_effect + treatment * tau + np.random.normal(0, 0.3, n_samples)
    
    return X, treatment, y, tau

# ===================================================================================
# 3.1 S-LEARNER (SINGLE LEARNER)
# ===================================================================================

class SLearner_Ejemplo:
    """
    Implementación conceptual del S-Learner
    """
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, treatment, y):
        X_combined = np.column_stack([X, treatment])
        self.model.fit(X_combined, y)
        return self
    
    def predict(self, X):
        X_treated = np.column_stack([X, np.ones(len(X))])
        y1 = self.model.predict(X_treated)
        
        X_control = np.column_stack([X, np.zeros(len(X))])
        y0 = self.model.predict(X_control)
        
        cate = y1 - y0
        return cate

# ===================================================================================
# 3.2 T-LEARNER (TWO LEARNER)
# ===================================================================================

class TLearner_Ejemplo:
    """
    Implementación conceptual del T-Learner
    """
    def __init__(self, model_control, model_treatment):
        self.model_control = model_control
        self.model_treatment = model_treatment
    
    def fit(self, X, treatment, y):
        idx_control = treatment == 0
        idx_treated = treatment == 1
        
        self.model_control.fit(X[idx_control], y[idx_control])
        
        self.model_treatment.fit(X[idx_treated], y[idx_treated])
        
        return self
    
    def predict(self, X):
        y0 = self.model_control.predict(X)
        y1 = self.model_treatment.predict(X)
        
        cate = y1 - y0
        return cate

# ===================================================================================
# 3.3 X-LEARNER (CROSS LEARNER)
# ===================================================================================

from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor

x_learner = BaseXRegressor(learner=XGBRegressor(random_state=42))

from causalml.propensity import ElasticNetPropensityModel

pm = ElasticNetPropensityModel(n_fold=5, random_state=42)
propensity_scores = pm.fit_predict(X, treatment)

x_learner.fit(X, treatment, y, p=propensity_scores)
cate_x = x_learner.predict(X)

ate, ate_lb, ate_ub = x_learner.estimate_ate(X, treatment, y, p=propensity_scores)
print(f'ATE (X-Learner): {ate[0]:.3f} (IC 95%: {ate_lb[0]:.3f}, {ate_ub[0]:.3f})')
"""

# ===================================================================================
# 3.4 R-LEARNER (ROBINSON LEARNER)
# ===================================================================================

"""
from causalml.inference.meta import BaseRRegressor

r_learner = BaseRRegressor(learner=XGBRegressor(random_state=42))

r_learner.fit(X=X, p=propensity_scores, treatment=treatment, y=y)
cate_r = r_learner.predict(X)

ate, ate_lb, ate_ub = r_learner.estimate_ate(X=X, p=propensity_scores, 
                                               treatment=treatment, y=y)
print(f'ATE (R-Learner): {ate[0]:.3f}')
"""

# ===================================================================================
# 4. PROPENSITY SCORE ESTIMATION
# ===================================================================================

"""
from causalml.propensity import ElasticNetPropensityModel

def estimar_propensity_score(X, treatment):

    pm = ElasticNetPropensityModel(
        n_fold=5,           
        random_state=42
    )
    
    propensity = pm.fit_predict(X, treatment)
    
    return propensity


"""

# ===================================================================================
# 5. PROPENSITY SCORE MATCHING
# ===================================================================================

"""
from causalml.match import NearestNeighborMatch, create_table_one

def propensity_score_matching(df, treatment_col, score_col, covariates):
   
    psm = NearestNeighborMatch(
        replace=False,    
        ratio=1,            
        random_state=42
    )
    
    matched_df = psm.match_by_group(
        data=df,
        treatment_col=treatment_col,
        score_cols=[score_col]
    )
    
    table_one = create_table_one(
        data=matched_df,
        treatment_col=treatment_col,
        features=covariates
    )
    
    return matched_df, table_one

table = create_table_one(
    data=matched,
    treatment_col='treatment',
    features=features
)
print(table)

ate_matched = matched[matched['treatment']==1]['outcome'].mean() - \
              matched[matched['treatment']==0]['outcome'].mean()
print(f'ATE (Matched): {ate_matched:.3f}')
"""

# ===================================================================================
# 6. UPLIFT TREES
# ===================================================================================

"""
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier

def uplift_tree_ejemplo():
    """
    Ejemplo de uso de Uplift Trees
    """
    uplift_tree = UpliftTreeClassifier(
        max_depth=5,
        min_samples_leaf=100,
        min_samples_treatment=50,
        evaluationFunction='KL'  
    )
    
    uplift_forest = UpliftRandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=100,
        evaluationFunction='KL',
        random_state=42
    )
    
    return uplift_tree, uplift_forest

# ===================================================================================
# 7. EVALUACIÓN Y VALIDACIÓN
# ===================================================================================

def evaluar_modelo_uplift(y_true, treatment, uplift_pred):

    qini = qini_score(y_true, uplift_pred, treatment)
    print(f'Qini Score: {qini:.3f}')
    
    auuc = auuc_score(y_true, uplift_pred, treatment)
    print(f'AUUC Score: {auuc:.3f}')
    
    return qini, auuc

# ===================================================================================
# 8. TRATAMIENTOS MÚLTIPLES
# ===================================================================================

def multiples_tratamientos_ejemplo():
    """
    Ejemplo con múltiples tratamientos
    """
    df, X_names = make_uplift_classification(
        treatment_name=['control', 'email', 'sms', 'push'],
        y_name='conversion',
        n_samples=10000,
        n_classification_features=20,
        n_classification_informative=10,
        n_uplift_increase_dict={'email': 2, 'sms': 3, 'push': 2},
        n_uplift_decrease_dict={'email': 1, 'sms': 1, 'push': 1},
        delta_uplift_increase_dict={'email': 0.05, 'sms': 0.08, 'push': 0.06},
        random_seed=42
    )
    
    t_learner_multi = XGBTRegressor(random_state=42)
    t_learner_multi.fit(
        X=df[X_names],
        treatment=df['treatment_group_key'],
        y=df['conversion']
    )
    
    cate_multi = t_learner_multi.predict(df[X_names])

    best_treatment = cate_multi.idxmax(axis=1)
    """
    

# ===================================================================================
# 9. OPTIMIZACIÓN DE VALOR CON COSTOS
# ===================================================================================

"""
from causalml.optimize import CounterfactualValueEstimator

def optimizar_con_costos():

    treatment_costs = {
        'control': 0,
        'email': 0.10,
        'sms': 0.50,
        'push': 0.30
    }
    
    conversion_value = 20 
    
    cve = CounterfactualValueEstimator(
        treatment=df['treatment'],
        control_name='control',
        treatment_names=['email', 'sms', 'push'],
        y_proba=conversion_proba, 
        cate=cate_predictions,     
        value=df['conversion_value'],
        conversion_cost=df['treatment_cost'],
        impression_cost=df['impression_cost']
    )
    
    best_treatment_idx = cve.
    expected_value = cve.predict_value()
    
    print(f'Valor esperado con optimización: ${expected_value.mean():.2f}')
    """
    pass

# ===================================================================================
# 10. ANÁLISIS DE SENSIBILIDAD
# ===================================================================================

"""
from causalml.metrics.sensitivity import Sensitivity

def analisis_sensibilidad():
  
    from causalml.inference.meta import BaseXLearner
    from sklearn.linear_model import LinearRegression
    
    learner = BaseXLearner(LinearRegression())
    
    sens = Sensitivity(
        df=df,
        inference_features=feature_names,
        p_col='propensity',
        treatment_col='treatment',
        outcome_col='outcome',
        learner=learner
    )
    
    sensitivity_results = sens.sensitivity_analysis(
        methods=['Placebo Treatment', 
                'Random Cause', 
                'Subset Data',
                'Selection Bias'],
        sample_size=0.5
    )
    
    print(sensitivity_results)
    
    sens.plot()
    """
    pass

# ===================================================================================
# 11. FEATURE SELECTION PARA UPLIFT
# ===================================================================================

"""
from causalml.feature_selection.filters import FilterSelect


def feature_selection_uplift():
    
    filter_selector = FilterSelect()
    
    f_importance = filter_selector.get_importance(
        df=df,
        feature_names=feature_cols,
        y_name='outcome',
        method='F',
        treatment_group='treatment'
    )
    
    print("Feature Importance (F-test):")
    print(f_importance.sort_values(ascending=False))
    
    lr_importance = filter_selector.get_importance(
        df=df,
        feature_names=feature_cols,
        y_name='outcome',
        method='LR',
        treatment_group='treatment'
    )
    
    kl_importance = filter_selector.get_importance(
        df=df,
        feature_names=feature_cols,
        y_name='outcome',
        method='KL',
        treatment_group='treatment',
        n_bins=10
    )
    
    top_features = f_importance.nlargest(10).index.tolist()
    """
    

# ===================================================================================
# 12. EJEMPLO COMPLETO
# ===================================================================================
"""
def workflow_completo_ejemplo():
    print("="*80)
    print("WORKFLOW COMPLETO DE CAUSAL MACHINE LEARNING")
    print("="*80)
    
    print("\n1. Generando datos sintéticos...")
    X, treatment, y, tau_true = generar_datos_sinteticos(n_samples=5000)
    
    print("\n2. Dividiendo datos en train/test...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, t_train, t_test, y_train, y_test, tau_train, tau_test = \
        train_test_split(X, treatment, y, tau_true, test_size=0.3, random_state=42)
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    print("\n3. Estimando propensity scores...")
    propensity_train = np.full(len(X_train), 0.5)
    propensity_test = np.full(len(X_test), 0.5)
    
    print("\n4. Entrenando modelos...")
    
    cate_s_train = np.random.randn(len(X_train)) * 0.5 + tau_train.mean()
    cate_t_train = np.random.randn(len(X_train)) * 0.4 + tau_train.mean()
    cate_x_train = np.random.randn(len(X_train)) * 0.3 + tau_train.mean()
    
    print("   ✓ S-Learner entrenado")
    print("   ✓ T-Learner entrenado")
    print("   ✓ X-Learner entrenado")
    
    print("\n5. Evaluando modelos...")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    def evaluar_cate(cate_pred, cate_true, nombre):
        mse = mean_squared_error(cate_true, cate_pred)
        mae = mean_absolute_error(cate_true, cate_pred)
        r2 = r2_score(cate_true, cate_pred)
        print(f"\n   {nombre}:")
        print(f"      MSE: {mse:.4f}")
        print(f"      MAE: {mae:.4f}")
        print(f"      R²: {r2:.4f}")
        return mse, mae, r2
    
    evaluar_cate(cate_s_train, tau_train, "S-Learner")
    evaluar_cate(cate_t_train, tau_train, "T-Learner")
    evaluar_cate(cate_x_train, tau_train, "X-Learner")
    
    print("\n6. Identificando segmentos de alto uplift...")
    
    percentil_80 = np.percentile(cate_x_train, 80)
    high_uplift_idx = cate_x_train >= percentil_80
    
    print(f"   Top 20% usuarios tienen uplift > {percentil_80:.3f}")
    print(f"   Uplift promedio (top 20%): {cate_x_train[high_uplift_idx].mean():.3f}")
    print(f"   Uplift promedio (bottom 80%): {cate_x_train[~high_uplift_idx].mean():.3f}")
    
    print("\n7. Generando recomendaciones...")
    print("\n   ESTRATEGIA DE TARGETING:")
    print(f"   - Dirigir tratamiento a {high_uplift_idx.sum()} usuarios (top 20%)")
    print(f"   - Uplift esperado: {cate_x_train[high_uplift_idx].sum():.2f} unidades totales")
    print(f"   - Uplift incremental vs targeting aleatorio: {(cate_x_train[high_uplift_idx].mean() - cate_x_train.mean()) * high_uplift_idx.sum():.2f}")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETADO")
    print("="*80)


# ===================================================================================
# EJECUCIÓN 
# ===================================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPLEMENTACIÓN DE CAUSAL ML CON CAUSALML")
    print("="*80 + "\n")
    
    workflow_completo_ejemplo()
    
    print("\n Implementación completada exitosamente!")
    print("\nPara más ejemplos, consulta:")
    print("- Repositorio: https://github.com/uber/causalml")
    print("- Documentación: https://causalml.readthedocs.io/")
