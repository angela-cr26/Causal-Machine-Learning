"""
===================================================================================
EJEMPLOS PRÁCTICOS DE CAUSALML - CASOS DE USO REALES
===================================================================================

Este documento presenta ejemplos prácticos y ejecutables de CausalML aplicados
a casos de uso reales del mundo empresarial.

===================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===================================================================================
# CASO DE USO 1: OPTIMIZACIÓN DE CAMPAÑA DE EMAIL MARKETING
# ===================================================================================

class EmailMarketingCampaign:
    """
    Caso de uso: Una empresa quiere optimizar su campaña de email marketing.
    Objetivo: Identificar qué clientes tienen mayor probabilidad de responder
    positivamente al email (hacer una compra) para maximizar ROI.
    """
    
    def __init__(self):
        self.name = "Email Marketing Campaign Optimization"
        
    def generar_datos(self, n_customers=10000):
        """
        Genera datos sintéticos simulando un experimento A/B de email marketing
        
        Features:
        - edad: edad del cliente
        - ingresos: ingresos anuales estimados
        - compras_previas: número de compras en último año
        - tiempo_cliente: años como cliente
        - engagement_score: score de engagement (0-100)
        
        Treatment:
        - 0: No recibió email (control)
        - 1: Recibió email (tratamiento)
        
        Outcome:
        - compra: 1 si realizó compra, 0 si no
        - valor_compra: valor de la compra en dólares
        """
        np.random.seed(42)
        
        edad = np.random.normal(40, 15, n_customers).clip(18, 80)
        ingresos = np.random.lognormal(11, 0.5, n_customers).clip(20000, 200000)
        compras_previas = np.random.poisson(3, n_customers)
        tiempo_cliente = np.random.exponential(2, n_customers).clip(0, 20)
        engagement_score = np.random.beta(2, 5, n_customers) * 100
        
        df = pd.DataFrame({
            'edad': edad,
            'ingresos': ingresos,
            'compras_previas': compras_previas,
            'tiempo_cliente': tiempo_cliente,
            'engagement_score': engagement_score
        })
        
        df['tratamiento'] = np.random.binomial(1, 0.5, n_customers)
        
        prob_base = 0.05 + \
                    0.001 * df['engagement_score'] + \
                    0.01 * df['compras_previas'] + \
                    0.002 * (50 - df['edad']).clip(0, None)
        
        efecto_email = 0.02 + \
                      0.002 * df['engagement_score'] + \
                      0.015 * df['compras_previas'] - \
                      0.001 * (df['edad'] - 40).abs()
        
        df['efecto_verdadero'] = efecto_email
        
        df['prob_compra'] = prob_base + df['tratamiento'] * efecto_email
        df['prob_compra'] = df['prob_compra'].clip(0, 1)
        
        df['compra'] = np.random.binomial(1, df['prob_compra'])
        
        df['valor_compra'] = np.where(
            df['compra'] == 1,
            np.random.lognormal(4, 0.8, n_customers) * (1 + df['engagement_score']/100),
            0
        )
        
        return df
    
    def analisis_exploratorio(self, df):
        """
        Realiza análisis exploratorio de los datos
        """
        print("\n" + "="*80)
        print("ANÁLISIS EXPLORATORIO - EMAIL MARKETING CAMPAIGN")
        print("="*80)
        
        print(f"\nNúmero de clientes: {len(df)}")
        print(f"Clientes en tratamiento: {df['tratamiento'].sum()} ({df['tratamiento'].mean()*100:.1f}%)")
        print(f"Clientes en control: {(df['tratamiento']==0).sum()} ({(df['tratamiento']==0).mean()*100:.1f}%)")
        
        print("\n--- Outcomes ---")
        print(f"\nTasa de conversión general: {df['compra'].mean()*100:.2f}%")
        print(f"Tasa de conversión (Control): {df[df['tratamiento']==0]['compra'].mean()*100:.2f}%")
        print(f"Tasa de conversión (Tratamiento): {df[df['tratamiento']==1]['compra'].mean()*100:.2f}%")
        
        ate_observado = df[df['tratamiento']==1]['compra'].mean() - \
                       df[df['tratamiento']==0]['compra'].mean()
        print(f"\n📊 ATE Observado: {ate_observado*100:.2f} puntos porcentuales")
        
        ate_verdadero = df['efecto_verdadero'].mean()
        print(f"📊 ATE Verdadero: {ate_verdadero*100:.2f} puntos porcentuales")
        
        print(f"\nValor promedio de compra: ${df[df['compra']==1]['valor_compra'].mean():.2f}")
        print(f"Revenue total (Control): ${df[df['tratamiento']==0]['valor_compra'].sum():,.2f}")
        print(f"Revenue total (Tratamiento): ${df[df['tratamiento']==1]['valor_compra'].sum():,.2f}")
        
        print("\n--- Distribución de Features ---")
        print(df[['edad', 'ingresos', 'compras_previas', 'tiempo_cliente', 'engagement_score']].describe())
        
    def implementar_metalearners(self, df):
        """
        Implementa meta-learners manualmente para ilustrar conceptos
        """
        print("\n" + "="*80)
        print("IMPLEMENTACIÓN DE META-LEARNERS")
        print("="*80)
        
        features = ['edad', 'ingresos', 'compras_previas', 'tiempo_cliente', 'engagement_score']
        X = df[features].values
        treatment = df['tratamiento'].values
        y = df['compra'].values
        tau_true = df['efecto_verdadero'].values
        
        X_train, X_test, t_train, t_test, y_train, y_test, tau_train, tau_test = \
            train_test_split(X, treatment, y, tau_true, test_size=0.3, random_state=42)
        
        print(f"\nDatos de entrenamiento: {len(X_train)}")
        print(f"Datos de prueba: {len(X_test)}")
        
        from sklearn.ensemble import GradientBoostingRegressor
        
        print("\n--- S-LEARNER ---")
        s_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        X_s_train = np.column_stack([X_train, t_train])
        s_model.fit(X_s_train, y_train)
        
        X_t1 = np.column_stack([X_test, np.ones(len(X_test))])
        X_t0 = np.column_stack([X_test, np.zeros(len(X_test))])
        cate_s = s_model.predict(X_t1) - s_model.predict(X_t0)
        
        mse_s = mean_squared_error(tau_test, cate_s)
        print(f"MSE (S-Learner): {mse_s:.6f}")
        print(f"Correlación con efecto verdadero: {np.corrcoef(tau_test, cate_s)[0,1]:.4f}")
        
        print("\n--- T-LEARNER ---")
        t_model_0 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        t_model_1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        t_model_0.fit(X_train[t_train==0], y_train[t_train==0])
        t_model_1.fit(X_train[t_train==1], y_train[t_train==1])
        
        cate_t = t_model_1.predict(X_test) - t_model_0.predict(X_test)
        
        mse_t = mean_squared_error(tau_test, cate_t)
        print(f"MSE (T-Learner): {mse_t:.6f}")
        print(f"Correlación con efecto verdadero: {np.corrcoef(tau_test, cate_t)[0,1]:.4f}")
        
        print("\n--- COMPARACIÓN ---")
        print(f"Mejor modelo: {'T-Learner' if mse_t < mse_s else 'S-Learner'}")
        
        return {
            'X_test': X_test,
            'tau_test': tau_test,
            'cate_s': cate_s,
            'cate_t': cate_t,
            'features': features
        }
    
    def estrategia_targeting(self, df, cate_predictions, percentil=80):
        """
        Define estrategia de targeting basada en predicciones de CATE
        """
        print("\n" + "="*80)
        print("ESTRATEGIA DE TARGETING")
        print("="*80)
        
        threshold = np.percentile(cate_predictions, percentil)
        high_uplift = cate_predictions >= threshold
        
        print(f"\nTargeting al top {100-percentil}% de usuarios")
        print(f"Threshold de uplift: {threshold:.4f}")
        print(f"Usuarios seleccionados: {high_uplift.sum()} de {len(cate_predictions)}")
        
        uplift_promedio_general = cate_predictions.mean()
        uplift_promedio_targetted = cate_predictions[high_uplift].mean()
        
        print(f"\n📊 Uplift promedio (todos): {uplift_promedio_general*100:.2f} pp")
        print(f"📊 Uplift promedio (targetted): {uplift_promedio_targetted*100:.2f} pp")
        print(f"📈 Mejora: {(uplift_promedio_targetted - uplift_promedio_general)*100:.2f} pp")
        
        costo_email = 0.10 
        valor_conversion = 50 
        
        n_total = len(cate_predictions)
        costo_total_todos = n_total * costo_email
        conversiones_esperadas_todos = (cate_predictions * n_total).sum()
        valor_todos = conversiones_esperadas_todos * valor_conversion
        roi_todos = (valor_todos - costo_total_todos) / costo_total_todos
        
        n_targetted = high_uplift.sum()
        costo_targetted = n_targetted * costo_email
        conversiones_esperadas_targetted = (cate_predictions[high_uplift] * n_targetted).sum()
        valor_targetted = conversiones_esperadas_targetted * valor_conversion
        roi_targetted = (valor_targetted - costo_targetted) / costo_targetted
        
        print("\n--- ANÁLISIS DE ROI ---")
        print(f"\nEstrategia 1: Enviar email a TODOS")
        print(f"  Costo: ${costo_total_todos:,.2f}")
        print(f"  Conversiones esperadas: {conversiones_esperadas_todos:.1f}")
        print(f"  Valor esperado: ${valor_todos:,.2f}")
        print(f"  ROI: {roi_todos*100:.1f}%")
        
        print(f"\nEstrategia 2: Targeting selectivo (top {100-percentil}%)")
        print(f"  Costo: ${costo_targetted:,.2f}")
        print(f"  Conversiones esperadas: {conversiones_esperadas_targetted:.1f}")
        print(f"  Valor esperado: ${valor_targetted:,.2f}")
        print(f"  ROI: {roi_targetted*100:.1f}%")
        
        print(f"\n✅ Ahorro de costos: ${costo_total_todos - costo_targetted:,.2f} ({((costo_total_todos - costo_targetted)/costo_total_todos)*100:.1f}%)")
        print(f"✅ Mejora de ROI: {(roi_targetted - roi_todos)*100:.1f} puntos porcentuales")

# ===================================================================================
# CASO DE USO 2: OPTIMIZACIÓN DE DESCUENTOS/PROMOCIONES
# ===================================================================================

class DiscountOptimization:
    """
    Caso de uso: Una tienda online quiere optimizar su estrategia de descuentos.
    Pregunta: ¿A quién debemos ofrecer descuentos para maximizar profit?
    
    Insight clave: No todos los clientes necesitan descuento para comprar.
    Algunos comprarían de todas formas (no darles descuento = más profit).
    """
    
    def __init__(self):
        self.name = "Discount Optimization"
        
    def generar_datos(self, n_customers=8000):
        """
        Genera datos sintéticos de experimento de descuentos
        """
        np.random.seed(123)
        
        valor_carrito = np.random.lognormal(4, 0.8, n_customers)
        visitas_previas = np.random.poisson(5, n_customers)
        tiempo_sitio = np.random.exponential(10, n_customers).clip(1, 60)
        precio_sensibilidad = np.random.beta(2, 2, n_customers)  
        
        df = pd.DataFrame({
            'valor_carrito': valor_carrito,
            'visitas_previas': visitas_previas,
            'tiempo_sitio': tiempo_sitio,
            'precio_sensibilidad': precio_sensibilidad
        })
        
        df['descuento'] = np.random.binomial(1, 0.5, n_customers)
        
        prob_base = 0.3 + \
                   0.005 * df['visitas_previas'] + \
                   0.005 * df['tiempo_sitio']
        
        efecto_descuento = df['precio_sensibilidad'] * 0.4 - \
                          0.01 * (df['valor_carrito'] / 100)
        
        df['efecto_verdadero'] = efecto_descuento
        
        df['prob_compra'] = (prob_base + df['descuento'] * efecto_descuento).clip(0, 1)
        df['compra'] = np.random.binomial(1, df['prob_compra'])
        
        df['revenue'] = np.where(
            df['compra'] == 1,
            df['valor_carrito'] * (1 - 0.2 * df['descuento']),
            0
        )
        
        return df
    
    def analizar_segmentos(self, df, cate_predictions):
        """
        Analiza diferentes segmentos de clientes
        """
        print("\n" + "="*80)
        print("ANÁLISIS DE SEGMENTOS")
        print("="*80)
        
        df_analysis = df.copy()
        df_analysis['uplift_pred'] = cate_predictions
        
        df_analysis['segmento'] = 'Unknown'
        df_analysis.loc[df_analysis['uplift_pred'] < -0.05, 'segmento'] = 'Sleeping Dogs'
        df_analysis.loc[(df_analysis['uplift_pred'] >= -0.05) & 
                       (df_analysis['uplift_pred'] < 0.05), 'segmento'] = 'Lost Causes'
        df_analysis.loc[(df_analysis['uplift_pred'] >= 0.05) & 
                       (df_analysis['uplift_pred'] < 0.20), 'segmento'] = 'Persuadables'
        df_analysis.loc[df_analysis['uplift_pred'] >= 0.20, 'segmento'] = 'Sure Things'
        
        print("\nDistribución de segmentos:")
        print(df_analysis['segmento'].value_counts())
        
        print("\n--- Características por segmento ---")
        for segmento in ['Sleeping Dogs', 'Lost Causes', 'Persuadables', 'Sure Things']:
            if segmento in df_analysis['segmento'].values:
                seg_data = df_analysis[df_analysis['segmento'] == segmento]
                print(f"\n{segmento}:")
                print(f"  Uplift promedio: {seg_data['uplift_pred'].mean()*100:.2f}%")
                print(f"  Valor carrito promedio: ${seg_data['valor_carrito'].mean():.2f}")
                print(f"  Sensibilidad precio: {seg_data['precio_sensibilidad'].mean():.2f}")
        
        print("\n RECOMENDACIÓN: Dar descuento solo a 'Persuadables' y 'Sure Things'")

# ===================================================================================
# CASO DE USO 3: PERSONALIZACIÓN DE CANALES DE COMUNICACIÓN
# ===================================================================================

class ChannelPersonalization:
    """
    Caso de uso: Empresa con múltiples canales de comunicación
    (Email, SMS, Push notification, None)
    Objetivo: Asignar el mejor canal para cada usuario
    """
    
    def __init__(self):
        self.name = "Channel Personalization"
    
    def generar_datos(self, n_customers=12000):
        """
        Genera datos con múltiples tratamientos
        """
        np.random.seed(456)
        
        edad = np.random.normal(35, 12, n_customers).clip(18, 70)
        mobile_usage = np.random.beta(5, 2, n_customers) * 100 
        email_open_rate = np.random.beta(3, 7, n_customers)
        
        df = pd.DataFrame({
            'edad': edad,
            'mobile_usage': mobile_usage,
            'email_open_rate': email_open_rate
        })
        df['canal'] = np.random.choice([0, 1, 2, 3], n_customers)
        
        prob_base = 0.10
        
        efecto_email = df['email_open_rate'] * 0.15
        
        efecto_sms = 0.08 + 0.002 * df['edad'] - 0.0005 * df['mobile_usage']
        
        efecto_push = 0.15 - 0.001 * df['edad'] + 0.001 * df['mobile_usage']
        
        df['efecto_verdadero'] = 0
        df.loc[df['canal'] == 1, 'efecto_verdadero'] = efecto_email[df['canal'] == 1]
        df.loc[df['canal'] == 2, 'efecto_verdadero'] = efecto_sms[df['canal'] == 2]
        df.loc[df['canal'] == 3, 'efecto_verdadero'] = efecto_push[df['canal'] == 3]
        
        df['prob_conversion'] = (prob_base + df['efecto_verdadero']).clip(0, 1)
        df['conversion'] = np.random.binomial(1, df['prob_conversion'])
        
        return df
    
    def recomendar_canal(self, df):
        """
        Recomienda mejor canal para cada usuario
        """
        print("\n" + "="*80)
        print("RECOMENDACIÓN DE CANAL ÓPTIMO")
        print("="*80)
        
        print("\nEstrategia basada en características:")
        
        df['canal_recomendado'] = 'email'  
      
        df.loc[df['edad'] > 50, 'canal_recomendado'] = 'sms'
        
        df.loc[(df['edad'] < 30) & (df['mobile_usage'] > 70), 'canal_recomendado'] = 'push'
        
        df.loc[df['email_open_rate'] > 0.5, 'canal_recomendado'] = 'email'
        
        print("\nDistribución de canales recomendados:")
        print(df['canal_recomendado'].value_counts())
        
        return df

# ===================================================================================
# EJECUCIÓN DE EJEMPLOS
# ===================================================================================

def ejecutar_todos_casos():
    """
    Ejecuta todos los casos de uso
    """
    print("\n" + "="*80)
    print("EJEMPLOS PRÁCTICOS DE CAUSALML")
    print("Casos de Uso del Mundo Real")
    print("="*80)
    
    # CASO 1: Email Marketing
    print("\n\n" + "#"*80)
    print("# CASO 1: OPTIMIZACIÓN DE CAMPAÑA DE EMAIL MARKETING")
    print("#"*80)
    
    caso1 = EmailMarketingCampaign()
    df1 = caso1.generar_datos(n_customers=10000)
    caso1.analisis_exploratorio(df1)
    
    resultados1 = caso1.implementar_metalearners(df1)
    
    features_test = ['edad', 'ingresos', 'compras_previas', 'tiempo_cliente', 'engagement_score']
    df1_test = df1.iloc[-int(0.3*len(df1)):].reset_index(drop=True)
    df1_test['cate_pred'] = resultados1['cate_t']
    
    caso1.estrategia_targeting(df1_test, resultados1['cate_t'], percentil=80)
    
    # CASO 2: Optimización de Descuentos
    print("\n\n" + "#"*80)
    print("# CASO 2: OPTIMIZACIÓN DE DESCUENTOS")
    print("#"*80)
    
    caso2 = DiscountOptimization()
    df2 = caso2.generar_datos(n_customers=8000)
    
    print(f"\nNúmero de clientes: {len(df2)}")
    print(f"Tasa de conversión (sin descuento): {df2[df2['descuento']==0]['compra'].mean()*100:.2f}%")
    print(f"Tasa de conversión (con descuento): {df2[df2['descuento']==1]['compra'].mean()*100:.2f}%")
    
    caso2.analizar_segmentos(df2, df2['efecto_verdadero'])
    
    # CASO 3: Personalización de Canales
    print("\n\n" + "#"*80)
    print("# CASO 3: PERSONALIZACIÓN DE CANALES")
    print("#"*80)
    
    caso3 = ChannelPersonalization()
    df3 = caso3.generar_datos(n_customers=12000)
    
    print(f"\nNúmero de clientes: {len(df3)}")
    print("\nTasa de conversión por canal:")
    for canal in [0, 1, 2, 3]:
        canal_name = ['Control', 'Email', 'SMS', 'Push'][canal]
        tasa = df3[df3['canal']==canal]['conversion'].mean()
        print(f"  {canal_name}: {tasa*100:.2f}%")
    
    df3_recomendaciones = caso3.recomendar_canal(df3)
    
    print("\n" + "="*80)
    print("TODOS LOS CASOS DE USO COMPLETADOS")
    print("="*80)

if __name__ == "__main__":
    ejecutar_todos_casos()
    
    print("\n\nEjemplos ejecutados exitosamente!")
    print("\nPróximos pasos:")
    print("1. Implementar con datos reales de tu empresa")
    print("2. Usar librerías CausalML para modelos más sofisticados")
    print("3. Validar resultados con A/B tests")
    print("4. Iterar y mejorar modelos continuamente")
