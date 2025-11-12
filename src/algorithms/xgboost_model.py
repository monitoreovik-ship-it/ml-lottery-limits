"""
Algoritmo #5: XGBoost (Extreme Gradient Boosting)
Gradient boosting state-of-the-art para clasificación multi-label
"""

import numpy as np
import os
from collections import Counter

# Importar XGBoost (debe estar instalado: pip install xgboost)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no instalado. Ejecuta: pip install xgboost")


class XGBoostLottery:
    """
    XGBoost para predicción de lotería usando clasificación multi-label.
    
    Teoría:
    - Entrena 56 clasificadores binarios (uno por número 1-56)
    - Features: lag features (sorteos anteriores), rolling stats
    - Gradient boosting optimiza error residual iterativamente
    
    Arquitectura:
    - Input: Ventana de N sorteos previos + estadísticas rolling
    - Output: 56 probabilidades [0-1] → seleccionar top 6
    
    Hipótesis:
    - Si hay patrones no lineales, XGBoost los detectará
    - Esperamos: Overfitting en training, pobre performance en test
    """
    
    def __init__(self, n_lags=10, n_estimators=100, max_depth=4):
        """
        Args:
            n_lags: Número de sorteos previos a usar como features
            n_estimators: Número de árboles en ensemble
            max_depth: Profundidad máxima de cada árbol
        """
        self.name = f"XGBoost (lags={n_lags}, trees={n_estimators})"
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = {}  # Un modelo por cada número (1-56)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost no disponible. Instala con: pip install xgboost")
    
    def _create_lag_features(self, history, target_idx):
        """
        Crea features de lag para un índice específico del historial
        
        Args:
            history: Lista completa de sorteos
            target_idx: Índice del sorteo a predecir
        
        Returns:
            features: Array con lag features y rolling stats
        """
        features = []
        
        # LAG FEATURES: últimos n_lags sorteos aplanados
        start_idx = max(0, target_idx - self.n_lags)
        lag_window = history[start_idx:target_idx]
        
        # Aplanar números (cada sorteo → 6 features)
        for draw in lag_window:
            features.extend(sorted(draw['numbers']))
        
        # Padding si no hay suficientes lags
        if len(lag_window) < self.n_lags:
            padding = [0] * (6 * (self.n_lags - len(lag_window)))
            features = padding + features
        
        # ROLLING STATISTICS (últimos 5 sorteos)
        if target_idx >= 5:
            recent = history[target_idx-5:target_idx]
            all_numbers = []
            for draw in recent:
                all_numbers.extend(draw['numbers'])
            
            # Frecuencias de cada número en ventana reciente
            freq = Counter(all_numbers)
            freq_vector = [freq.get(i, 0) for i in range(1, 57)]
            
            # Agregar: suma, promedio, std de frecuencias
            features.append(sum(freq_vector))
            features.append(np.mean(freq_vector))
            features.append(np.std(freq_vector))
            
            # Promedio de sumas de sorteos recientes
            sums = [sum(draw['numbers']) for draw in recent]
            features.append(np.mean(sums))
            features.append(np.std(sums))
        else:
            # Padding para estadísticas
            features.extend([0] * (56 + 5))
        
        return np.array(features)
    
    def _prepare_dataset(self, history):
        """
        Prepara dataset completo para entrenamiento
        
        Returns:
            X: Features matrix [n_samples, n_features]
            y: Target matrix [n_samples, 56] (multi-label binario)
        """
        X = []
        y = []
        
        # Empezar desde índice n_lags para tener suficiente historia
        for i in range(self.n_lags, len(history)):
            # Features
            features = self._create_lag_features(history, i)
            X.append(features)
            
            # Target: vector binario de 56 posiciones
            target = np.zeros(56)
            for num in history[i]['numbers']:
                target[num - 1] = 1  # Marcar números presentes
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def fit(self, history):
        """
        Entrena 56 modelos XGBoost (uno por cada número)
        """
        if len(history) <= self.n_lags:
            raise ValueError(f"❌ Historial insuficiente. Necesitas al menos {self.n_lags + 1} sorteos.")
        
        # Preparar dataset
        X, y = self._prepare_dataset(history)
        
        print(f"✅ {self.name}: Preparando entrenamiento...")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        
        # Entrenar un modelo por cada número (1-56)
        for num in range(1, 57):
            # Target binario para este número
            y_num = y[:, num - 1]
            
            # Crear modelo XGBoost CON base_score=0.5
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                base_score=0.5,  # ←←← CORRECCIÓN CLAVE
                random_state=42,
                verbosity=0  # Silencioso
            )
            
            # Entrenar
            model.fit(X, y_num)
            self.models[num] = model
            
            # Progreso cada 10 modelos
            if num % 10 == 0:
                print(f"   ... entrenados {num}/56 modelos")
        
        print(f"✅ {self.name}: 56 modelos entrenados")
        return self
    
    def predict(self, history):
        """
        Predice siguiente sorteo
        
        Método:
        1. Extraer features del contexto actual
        2. Obtener probabilidad de cada número (1-56)
        3. Seleccionar top 6 con mayor probabilidad
        """
        if not self.models:
            raise ValueError("❌ Modelo no entrenado. Llama fit() primero.")
        
        # Crear features para siguiente predicción
        features = self._create_lag_features(history, len(history)).reshape(1, -1)
        
        # Obtener probabilidad de cada número
        probabilities = {}
        for num in range(1, 57):
            model = self.models[num]
            prob = model.predict_proba(features)[0][1]  # P(número presente)
            probabilities[num] = prob
        
        # Top 6 números con mayor probabilidad
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, prob in sorted_probs[:6]]
        
        return sorted(top_numbers)
    
    def get_feature_importance(self, top_n=10):
        """
        Analiza feature importance promediando sobre todos los modelos
        """
        if not self.models:
            raise ValueError("❌ Modelo no entrenado.")
        
        # Promediar importancias de todos los modelos
        avg_importance = None
        
        for num, model in self.models.items():
            importance = model.feature_importances_
            if avg_importance is None:
                avg_importance = importance
            else:
                avg_importance += importance
        
        avg_importance /= len(self.models)
        
        # Top N features
        top_indices = np.argsort(avg_importance)[-top_n:][::-1]
        
        print(f"\n📊 Top {top_n} Features Más Importantes:")
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. Feature {idx}: {avg_importance[idx]:.4f}")
        
        return avg_importance


# ==================== TEST ====================
if __name__ == "__main__":
    print("🚀 ALGORITMO XGBOOST - TEST")
    print("=" * 60)
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost no disponible. Instala con:")
        print("   pip install xgboost")
        exit(1)
    
    # Determinar ruta absoluta a data/raw/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    # Importar collector
    from src.data.collector import MelateCollector
    
    # Cargar sorteos
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    # Validar datos suficientes
    if len(history) < 15:
        print(f"⚠️ Datos insuficientes ({len(history)} sorteos).")
        print("   XGBoost requiere al menos 15 sorteos para entrenamiento confiable.")
        print("   Usando datos sintéticos para demo...")
        
        # Generar datos sintéticos
        np.random.seed(42)
        history = []
        for i in range(20):
            numbers = sorted(np.random.choice(56, size=6, replace=False) + 1)
            history.append({
                'date': f'2024-{10 + i//30:02d}-{(i%30)+1:02d}',
                'numbers': numbers.tolist()
            })
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: Entrenar XGBoost
    print("\n" + "="*60)
    print("🔮 TEST 1: Entrenamiento XGBoost")
    print("="*60)
    
    xgb_model = XGBoostLottery(n_lags=5, n_estimators=50, max_depth=3)
    xgb_model.fit(history)
    
    # Test 2: Predicción
    print("\n" + "="*60)
    print("🔮 TEST 2: Predicción")
    print("="*60)
    
    prediction = xgb_model.predict(history)
    print(f"\n🎯 Predicción: {prediction}")
    
    # Test 3: Feature Importance
    print("\n" + "="*60)
    print("🔮 TEST 3: Feature Importance")
    print("="*60)
    
    xgb_model.get_feature_importance(top_n=10)
    
    # Test 4: Validación walk-forward (últimos 3 sorteos)
    print("\n" + "="*60)
    print("📊 VALIDACIÓN WALK-FORWARD (últimos 3 sorteos)")
    print("="*60)
    
    if len(history) >= 13:
        results = []
        
        for i in range(3):
            # Entrenar con todos menos los últimos (3-i)
            train_data = history[:-(3-i)] if i < 2 else history[:-1]
            test_idx = -(3-i) if i < 2 else -1
            test_result = history[test_idx]['numbers']
            
            # Entrenar y predecir
            xgb_val = XGBoostLottery(n_lags=5, n_estimators=50, max_depth=3)
            xgb_val.fit(train_data)
            pred_val = xgb_val.predict(train_data)
            
            # Evaluar
            matches = len(set(pred_val) & set(test_result))
            results.append(matches)
            
            print(f"\nSorteo {history[test_idx]['date']}:")
            print(f"   Predicción:  {pred_val}")
            print(f"   Real:        {test_result}")
            print(f"   ✅ Aciertos: {matches}/6")
        
        # Resumen
        avg_matches = np.mean(results)
        print(f"\n📈 Resumen Walk-Forward:")
        print(f"   Aciertos promedio: {avg_matches:.2f}/6")
        print(f"   Esperado por azar: 0.64 ± 0.72")
        
        z_score = (avg_matches - 0.64) / 0.72
        print(f"   Z-score: {z_score:.2f}")
        print(f"   Conclusión: {'🎉 Significativo (p<0.05)' if abs(z_score) > 1.96 else '✅ No significativo'}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - Feature importance revela qué lags usa el modelo")
    print("   - Si avg_matches ≈ 0.64 → XGBoost no mejora sobre azar")
    print("   - Alta importancia en lags recientes → posible overfitting")
    print("   - Conclusión esperada: Gradient boosting no detecta patrones reales")