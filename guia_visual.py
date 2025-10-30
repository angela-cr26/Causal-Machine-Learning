"""
===================================================================================
GUÍA VISUAL DE CAUSALML - DIAGRAMAS Y EXPLICACIONES
===================================================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===================================================================================
# 1. DIAGRAMA: DIFERENCIA ENTRE ML TRADICIONAL Y CAUSAL ML
# ===================================================================================

def plot_ml_vs_causal_ml():
    """
    Visualiza la diferencia entre ML tradicional y Causal ML
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.text(0.5, 0.9, 'MACHINE LEARNING TRADICIONAL', 
             ha='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.8, 'Pregunta: ¿Qué pasará?', ha='center', fontsize=11)
    
    ax1.add_patch(FancyBboxPatch((0.2, 0.6), 0.6, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='blue', facecolor='lightblue'))
    ax1.text(0.5, 0.65, 'Features (X)', ha='center', va='center', fontweight='bold')
    
    ax1.arrow(0.5, 0.58, 0, -0.08, head_width=0.03, head_length=0.02, fc='black')
    
    ax1.add_patch(FancyBboxPatch((0.25, 0.35), 0.5, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='green', facecolor='lightgreen'))
    ax1.text(0.5, 0.4, 'Modelo ML', ha='center', va='center', fontweight='bold')
    
    ax1.arrow(0.5, 0.33, 0, -0.08, head_width=0.03, head_length=0.02, fc='black')
    
    ax1.add_patch(FancyBboxPatch((0.3, 0.1), 0.4, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='red', facecolor='lightcoral'))
    ax1.text(0.5, 0.15, 'Predicción (Ŷ)', ha='center', va='center', fontweight='bold')
    
    ax1.text(0.1, 0.03, 'Objetivo: Predecir outcome', fontsize=9, style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    ax2.text(0.5, 0.9, 'CAUSAL MACHINE LEARNING', 
             ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.8, 'Pregunta: ¿Qué pasará SI...?', ha='center', fontsize=11)
    
    ax2.add_patch(FancyBboxPatch((0.15, 0.6), 0.3, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='blue', facecolor='lightblue'))
    ax2.text(0.3, 0.65, 'Features (X)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.add_patch(FancyBboxPatch((0.55, 0.6), 0.3, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='purple', facecolor='plum'))
    ax2.text(0.7, 0.65, 'Treatment (T)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.arrow(0.3, 0.58, 0.1, -0.13, head_width=0.03, head_length=0.02, fc='black')
    ax2.arrow(0.7, 0.58, -0.1, -0.13, head_width=0.03, head_length=0.02, fc='black')
    
    ax2.add_patch(FancyBboxPatch((0.25, 0.35), 0.5, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='green', facecolor='lightgreen'))
    ax2.text(0.5, 0.4, 'Causal Model', ha='center', va='center', fontweight='bold')
    
    ax2.arrow(0.5, 0.33, 0, -0.08, head_width=0.03, head_length=0.02, fc='black')
    
    ax2.add_patch(FancyBboxPatch((0.25, 0.1), 0.5, 0.1, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='orange', facecolor='lightyellow'))
    ax2.text(0.5, 0.15, 'Efecto Causal (τ)', ha='center', va='center', fontweight='bold')
    
    ax2.text(0.05, 0.03, 'Objetivo: Estimar efecto del tratamiento', fontsize=9, style='italic')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# ===================================================================================
# 2. DIAGRAMA: META-LEARNERS COMPARISON
# ===================================================================================

def plot_metalearners_comparison():
    """
    Visualiza cómo funcionan los diferentes meta-learners
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    ax.text(0.5, 0.95, 'S-LEARNER (Single Model)', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.15, 0.75), 0.25, 0.1, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.275, 0.8, 'X', ha='center', va='center', fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.6, 0.75), 0.25, 0.1, 
                                boxstyle="round", edgecolor='purple', facecolor='plum'))
    ax.text(0.725, 0.8, 'T', ha='center', va='center', fontweight='bold')
    
    ax.arrow(0.275, 0.73, 0.175, -0.18, head_width=0.03, head_length=0.02, fc='black')
    ax.arrow(0.725, 0.73, -0.175, -0.18, head_width=0.03, head_length=0.02, fc='black')
    
    ax.add_patch(FancyBboxPatch((0.3, 0.45), 0.4, 0.1, 
                                boxstyle="round", edgecolor='green', facecolor='lightgreen'))
    ax.text(0.5, 0.5, 'f(X, T)', ha='center', va='center', fontweight='bold')
    
    ax.text(0.5, 0.3, 'τ(x) = f(x, 1) - f(x, 0)', ha='center', fontsize=10)
    
    ax.text(0.5, 0.15, '✓ Simple\n✗ Puede no capturar\n   heterogeneidad compleja', 
            ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.text(0.5, 0.95, 'T-LEARNER (Two Models)', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.05, 0.75), 0.25, 0.08, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.175, 0.79, 'X | T=0', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.arrow(0.175, 0.73, 0, -0.08, head_width=0.03, head_length=0.02, fc='black')
    
    ax.add_patch(FancyBboxPatch((0.05, 0.55), 0.25, 0.08, 
                                boxstyle="round", edgecolor='green', facecolor='lightgreen'))
    ax.text(0.175, 0.59, 'μ₀(x)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.7, 0.75), 0.25, 0.08, 
                                boxstyle="round", edgecolor='purple', facecolor='plum'))
    ax.text(0.825, 0.79, 'X | T=1', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.arrow(0.825, 0.73, 0, -0.08, head_width=0.03, head_length=0.02, fc='black')
    
    ax.add_patch(FancyBboxPatch((0.7, 0.55), 0.25, 0.08, 
                                boxstyle="round", edgecolor='green', facecolor='lightgreen'))
    ax.text(0.825, 0.59, 'μ₁(x)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.arrow(0.175, 0.53, 0.275, -0.18, head_width=0.03, head_length=0.02, fc='black')
    ax.arrow(0.825, 0.53, -0.275, -0.18, head_width=0.03, head_length=0.02, fc='black')
    
    ax.text(0.5, 0.3, 'τ(x) = μ₁(x) - μ₀(x)', ha='center', fontsize=10)
    
    ax.text(0.5, 0.15, '✓ Captura heterogeneidad\n✓ Flexible\n✗ Alta varianza', 
            ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax = axes[1, 0]
    ax.text(0.5, 0.95, 'X-LEARNER (3 Stages)', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.text(0.5, 0.85, 'Stage 1: Como T-Learner', ha='center', fontsize=9)
    ax.text(0.5, 0.75, 'Stage 2: Imputar efectos', ha='center', fontsize=9)
    ax.text(0.25, 0.65, 'τ₁ = Y₁ - μ₀(X₁)', ha='center', fontsize=8)
    ax.text(0.75, 0.65, 'τ₀ = μ₁(X₀) - Y₀', ha='center', fontsize=8)
    
    ax.text(0.5, 0.55, 'Stage 3: Entrenar τ̂₁(x) y τ̂₀(x)', ha='center', fontsize=9)
    
    ax.add_patch(FancyBboxPatch((0.25, 0.35), 0.5, 0.08, 
                                boxstyle="round", edgecolor='orange', facecolor='lightyellow'))
    ax.text(0.5, 0.39, 'Combinar con propensity', ha='center', va='center', fontsize=9)
    
    ax.text(0.5, 0.25, 'τ(x) = e(x)·τ̂₀(x) + (1-e(x))·τ̂₁(x)', ha='center', fontsize=9)
    
    ax.text(0.5, 0.12, '✓ Mejor con desbalance\n✓ Menor varianza\n✗ Más complejo', 
            ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.text(0.5, 0.95, 'R-LEARNER (Orthogonalization)', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.text(0.5, 0.82, 'Step 1: Residualización', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.73, 'Ỹ = Y - E[Y|X]', ha='center', fontsize=9)
    ax.text(0.5, 0.66, 'T̃ = T - E[T|X] = T - e(X)', ha='center', fontsize=9)
    
    ax.text(0.5, 0.55, 'Step 2: Resolver', ha='center', fontsize=10, fontweight='bold')
    ax.add_patch(FancyBboxPatch((0.15, 0.42), 0.7, 0.08, 
                                boxstyle="round", edgecolor='orange', facecolor='lightyellow'))
    ax.text(0.5, 0.46, 'τ(x) = argmin E[(Ỹ - τ(X)·T̃)²]', ha='center', va='center', fontsize=9)
    
    ax.text(0.5, 0.3, 'Elimina sesgo de confounders', ha='center', fontsize=9, style='italic')
    
    ax.text(0.5, 0.15, '✓ Robusto\n✓ Propiedades teóricas\n✗ Requiere buen e(x)', 
            ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# ===================================================================================
# 3. VISUALIZACIÓN: UPLIFT DISTRIBUTION
# ===================================================================================

def plot_uplift_distribution():
    """
    Visualiza distribución de uplift y segmentación
    """
    np.random.seed(42)
    n = 1000
    
    uplift = np.concatenate([
        np.random.normal(-0.05, 0.02, int(n*0.2)),  # Sleeping dogs
        np.random.normal(0.02, 0.03, int(n*0.3)),   # Lost causes
        np.random.normal(0.12, 0.04, int(n*0.35)),  # Persuadables
        np.random.normal(0.25, 0.05, int(n*0.15))   # Sure things
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(uplift, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Uplift = 0')
    ax1.axvline(uplift.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Media = {uplift.mean():.3f}')
    ax1.set_xlabel('Uplift Predicho', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Uplift Individual', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    segments = pd.DataFrame({'uplift': uplift})
    segments['segmento'] = 'Unknown'
    segments.loc[segments['uplift'] < 0, 'segmento'] = 'Sleeping Dogs\n(uplift < 0)'
    segments.loc[(segments['uplift'] >= 0) & (segments['uplift'] < 0.08), 'segmento'] = 'Lost Causes\n(0 ≤ uplift < 0.08)'
    segments.loc[(segments['uplift'] >= 0.08) & (segments['uplift'] < 0.18), 'segmento'] = 'Persuadables\n(0.08 ≤ uplift < 0.18)'
    segments.loc[segments['uplift'] >= 0.18, 'segmento'] = 'Sure Things\n(uplift ≥ 0.18)'
    
    segment_counts = segments['segmento'].value_counts()
    colors = ['red', 'gray', 'green', 'gold']
    
    wedges, texts, autotexts = ax2.pie(segment_counts, labels=segment_counts.index, 
                                        autopct='%1.1f%%', colors=colors,
                                        startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Segmentación de Usuarios\npor Uplift', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ===================================================================================
# 4. VISUALIZACIÓN: GAIN CURVE Y QINI CURVE
# ===================================================================================

def plot_uplift_curves():
    """
    Visualiza Gain Curve y Qini Curve
    """
    np.random.seed(42)
    n = 1000
    
    uplift_true = np.random.beta(2, 5, n) * 0.3
    uplift_pred = uplift_true + np.random.normal(0, 0.05, n)
    
    sorted_idx = np.argsort(-uplift_pred)
    uplift_sorted = uplift_true[sorted_idx]
    
    cumulative_gain = np.cumsum(uplift_sorted)
    cumulative_gain_pct = cumulative_gain / cumulative_gain[-1]
    population_pct = np.arange(1, n+1) / n
    
    random_baseline = population_pct
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(population_pct * 100, cumulative_gain_pct * 100, 
             linewidth=2, label='Modelo Uplift', color='blue')
    ax1.plot(population_pct * 100, random_baseline * 100, 
             '--', linewidth=2, label='Targeting Aleatorio', color='red')
    ax1.fill_between(population_pct * 100, cumulative_gain_pct * 100, 
                     random_baseline * 100, alpha=0.3, color='green', 
                     label='Ganancia del Modelo')
    ax1.set_xlabel('Población Targetted (%)', fontsize=12)
    ax1.set_ylabel('Ganancia Acumulada (%)', fontsize=12)
    ax1.set_title('Gain Curve\n(Evalúa performance de targeting)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    
    treatment = np.random.binomial(1, 0.5, n)
    outcome = (uplift_true * treatment + np.random.binomial(1, 0.1, n)).clip(0, 1)
    
    sorted_idx = np.argsort(-uplift_pred)
    treatment_sorted = treatment[sorted_idx]
    outcome_sorted = outcome[sorted_idx]
    
    n_treated = np.cumsum(treatment_sorted)
    n_control = np.arange(1, n+1) - n_treated
    outcome_treated = np.cumsum(outcome_sorted * treatment_sorted)
    outcome_control = np.cumsum(outcome_sorted * (1 - treatment_sorted))
    
    qini = outcome_treated - outcome_control * (n_treated / (n_control + 1e-10))
    qini_pct = (qini / (qini[-1] + 1e-10)) * 100
    
    ax2.plot(population_pct * 100, qini_pct, linewidth=2, label='Modelo Uplift', color='blue')
    ax2.plot(population_pct * 100, random_baseline * 100, '--', linewidth=2, 
             label='Random', color='red')
    ax2.fill_between(population_pct * 100, qini_pct, random_baseline * 100, 
                     alpha=0.3, color='green', label='AUUC (Area Under Uplift Curve)')
    ax2.set_xlabel('Población Targetted (%)', fontsize=12)
    ax2.set_ylabel('Qini Coefficient (%)', fontsize=12)
    ax2.set_title('Qini Curve\n(Métrica estándar de Uplift)', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    return fig

# ===================================================================================
# 5. ARQUITECTURA DE CAUSALML
# ===================================================================================

def plot_causalml_architecture():
    """
    Diagrama de la arquitectura del paquete CausalML
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.text(0.5, 0.98, 'ARQUITECTURA DE CAUSALML', 
            ha='center', fontsize=16, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.1, 0.88), 0.8, 0.07, 
                                boxstyle="round", edgecolor='black', 
                                facecolor='lightblue', linewidth=2))
    ax.text(0.5, 0.915, 'DATOS: Features (X), Treatment (T), Outcome (Y)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.arrow(0.5, 0.87, 0, -0.03, head_width=0.02, head_length=0.01, 
             fc='black', linewidth=2)
    
    ax.add_patch(FancyBboxPatch((0.1, 0.78), 0.35, 0.06, 
                                boxstyle="round", edgecolor='green', 
                                facecolor='lightgreen'))
    ax.text(0.275, 0.81, 'Propensity Score\nEstimation', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.55, 0.78), 0.35, 0.06, 
                                boxstyle="round", edgecolor='green', 
                                facecolor='lightgreen'))
    ax.text(0.725, 0.81, 'Feature\nSelection', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    ax.arrow(0.275, 0.77, 0, -0.03, head_width=0.02, head_length=0.01, fc='black')
    ax.arrow(0.725, 0.77, 0, -0.03, head_width=0.02, head_length=0.01, fc='black')
    
    ax.add_patch(FancyBboxPatch((0.05, 0.42), 0.9, 0.28, 
                                boxstyle="round", edgecolor='purple', 
                                facecolor='lavender', linewidth=2))
    ax.text(0.5, 0.68, 'MÉTODOS DE ESTIMACIÓN CAUSAL', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.08, 0.54), 0.18, 0.1, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.17, 0.59, 'Meta-Learners\n• S-Learner\n• T-Learner\n• X-Learner\n• R-Learner', 
            ha='center', va='center', fontsize=7)
    
    ax.add_patch(FancyBboxPatch((0.28, 0.54), 0.18, 0.1, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.37, 0.59, 'Tree-Based\n• Uplift Trees\n• Uplift RF\n• Causal Trees', 
            ha='center', va='center', fontsize=7)
    
    ax.add_patch(FancyBboxPatch((0.48, 0.54), 0.18, 0.1, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.57, 0.59, 'Neural Nets\n• DragonNet\n• CEVAE\n• TARNet', 
            ha='center', va='center', fontsize=7)
    
    ax.add_patch(FancyBboxPatch((0.68, 0.54), 0.18, 0.1, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.77, 0.59, 'Econometric\n• IV (2SLS)\n• DR-Learner\n• TMLE', 
            ha='center', va='center', fontsize=7)
    
    ax.add_patch(FancyBboxPatch((0.18, 0.44), 0.64, 0.06, 
                                boxstyle="round", edgecolor='blue', facecolor='lightblue'))
    ax.text(0.5, 0.47, 'Propensity Score Matching (PSM)', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.arrow(0.5, 0.41, 0, -0.03, head_width=0.02, head_length=0.01, 
             fc='black', linewidth=2)
    
    ax.add_patch(FancyBboxPatch((0.1, 0.28), 0.35, 0.08, 
                                boxstyle="round", edgecolor='orange', 
                                facecolor='peachpuff'))
    ax.text(0.275, 0.32, 'Validation\n• Qini Score\n• AUUC\n• Gain Chart', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.55, 0.28), 0.35, 0.08, 
                                boxstyle="round", edgecolor='orange', 
                                facecolor='peachpuff'))
    ax.text(0.725, 0.32, 'Optimization\n• Targeting\n• Cost-aware\n• Multi-treatment', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.arrow(0.275, 0.27, 0.1, -0.06, head_width=0.02, head_length=0.01, fc='black')
    ax.arrow(0.725, 0.27, -0.1, -0.06, head_width=0.02, head_length=0.01, fc='black')
    
    ax.add_patch(FancyBboxPatch((0.1, 0.12), 0.8, 0.07, 
                                boxstyle="round", edgecolor='red', 
                                facecolor='lightcoral', linewidth=2))
    ax.text(0.5, 0.155, 'OUTPUT: CATE/ITE Predictions, ATE, Policy Recommendations', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((0.88, 0.45), 0.1, 0.15, 
                                boxstyle="round", edgecolor='darkgreen', 
                                facecolor='lightgreen'))
    ax.text(0.93, 0.525, 'Sensitivity\nAnalysis', ha='center', va='center', 
            fontsize=7, fontweight='bold', rotation=0)
    
    ax.arrow(0.87, 0.525, -0.02, 0, head_width=0.02, head_length=0.01, fc='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# ===================================================================================
# 6. TIMELINE DE IMPLEMENTACIÓN
# ===================================================================================

def plot_implementation_timeline():
    """
    Visualiza el timeline recomendado de implementación
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    phases = [
        {
            'name': 'Fase 1:\nPreparación',
            'tasks': ['• Definir objetivos\n• Recolectar datos\n• A/B test setup\n• Feature engineering'],
            'duration': 2,
            'color': 'lightblue'
        },
        {
            'name': 'Fase 2:\nModelado Básico',
            'tasks': ['• Propensity scores\n• Meta-learners básicos\n• Validación inicial\n• Análisis exploratorio'],
            'duration': 3,
            'color': 'lightgreen'
        },
        {
            'name': 'Fase 3:\nOptimización',
            'tasks': ['• Modelos avanzados\n• Feature selection\n• Hiperparámetros\n• Comparación modelos'],
            'duration': 3,
            'color': 'lightyellow'
        },
        {
            'name': 'Fase 4:\nValidación',
            'tasks': ['• Sensitivity analysis\n• Holdout validation\n• A/B test validación\n• Documentación'],
            'duration': 2,
            'color': 'peachpuff'
        },
        {
            'name': 'Fase 5:\nDeployment',
            'tasks': ['• Integración\n• Monitoreo\n• Iteración\n• Scaling'],
            'duration': 2,
            'color': 'lightcoral'
        }
    ]
    
    start_x = 0
    y_pos = 0.5
    
    for i, phase in enumerate(phases):
        width = phase['duration'] * 0.15
        
        ax.add_patch(FancyBboxPatch((start_x, y_pos), width, 0.25, 
                                    boxstyle="round,pad=0.01", 
                                    edgecolor='black', facecolor=phase['color'], 
                                    linewidth=2))
        
        ax.text(start_x + width/2, y_pos + 0.2, phase['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.text(start_x + width/2, y_pos + 0.1, phase['tasks'][0], 
                ha='center', va='center', fontsize=7)
        
        ax.text(start_x + width/2, y_pos - 0.05, f"{phase['duration']} semanas", 
                ha='center', va='top', fontsize=8, style='italic')
        
        if i < len(phases) - 1:
            ax.arrow(start_x + width, y_pos + 0.125, 0.02, 0, 
                    head_width=0.03, head_length=0.015, fc='black')
        
        start_x += width + 0.03
    
    ax.text(0.5, 0.9, 'TIMELINE DE IMPLEMENTACIÓN DE CAUSALML', 
            ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.05, 'Duración total estimada: 12 semanas', 
            ha='center', fontsize=11, style='italic', transform=ax.transAxes)
    
    ax.set_xlim(-0.05, 1.3)
    ax.set_ylim(0.3, 0.95)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

"""
# ===================================================================================
# GENERACIÓN DE VISUALIZACIONES
# ===================================================================================
"""
def generar_todas_visualizaciones():
    """
    Genera todas las visualizaciones y las guarda
    """
    print("Generando visualizaciones de CausalML...")
    
    print("\n1. ML vs Causal ML...")
    fig1 = plot_ml_vs_causal_ml()
    fig1.savefig('/home/user/viz_01_ml_vs_causal.png', dpi=300, bbox_inches='tight')
    
    print("2. Meta-Learners Comparison...")
    fig2 = plot_metalearners_comparison()
    fig2.savefig('/home/user/viz_02_metalearners.png', dpi=300, bbox_inches='tight')
    
    print("3. Uplift Distribution...")
    fig3 = plot_uplift_distribution()
    fig3.savefig('/home/user/viz_03_uplift_dist.png', dpi=300, bbox_inches='tight')
    
    print("4. Uplift Curves...")
    fig4 = plot_uplift_curves()
    fig4.savefig('/home/user/viz_04_uplift_curves.png', dpi=300, bbox_inches='tight')
    
    print("5. CausalML Architecture...")
    fig5 = plot_causalml_architecture()
    fig5.savefig('/home/user/viz_05_architecture.png', dpi=300, bbox_inches='tight')
    
    print("6. Implementation Timeline...")
    fig6 = plot_implementation_timeline()
    fig6.savefig('/home/user/viz_06_timeline.png', dpi=300, bbox_inches='tight')
    
    print("\n Todas las visualizaciones generadas exitosamente!")
    print("\nArchivos guardados:")
    print("  - viz_01_ml_vs_causal.png")
    print("  - viz_02_metalearners.png")
    print("  - viz_03_uplift_dist.png")
    print("  - viz_04_uplift_curves.png")
    print("  - viz_05_architecture.png")
    print("  - viz_06_timeline.png")

if __name__ == "__main__":
    generar_todas_visualizaciones()
