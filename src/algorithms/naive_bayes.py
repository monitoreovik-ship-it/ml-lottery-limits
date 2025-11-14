"""
Algoritmo #10: Naive Bayes
Clasificador probabilístico basado en teorema de Bayes con asunción de independencia
"""

import numpy as np
import os
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class NaiveBayesLottery:
    """
    Naive Bayes para predicción de lotería.
    
    Teoría:
    - Asume independencia condicional entre features (P(x1,x2|y) = P(x1|y)P(x2|y))
    - Esta asunción es CLARAMENTE VIOLADA en features de lag
    - Pero útil como baseline probabilístico
    
    Arquitectura:
    - 56 clasificadores Gaussian NB (uno por número)
    - Features: lag features + rolling stats
    - Output: P(número aparece | features)
    
    Hipótesis:
    - Asunción de independencia es incorrecta → performance limitada
    - Esperamos: similar a KNN (~1.0 aciertos)
    """
    
    def __init__(self, n_lags=10):
        """
        Args:
            n_lags: Número de sorteos previos como features
        """
        self.name = "Naive Bayes"
        self.n_lags = n_lags
        self.models = {}  # Un modelo por cada número (1-56)
        self.scaler = StandardScaler()
        
    def _create_lag_features(self, history, target_idx):
        """
        Crea features de lag (mismo que otros algoritmos)
        """
        features = []
        
        # LAG FEATURES
        start_idx = max(0, target_idx - self.n_lags)
        lag_window = history[start_idx:target_idx]
        
        for draw in lag_window:
            features.extend(sorted(draw['numbers']))
        
        # Padding
        if len(lag_window) < self.n_lags:
            padding = [0] * (6 * (self.n_lags - len(lag_window)))
            features = padding + features
        
        # ROLLING STATISTICS
        if target_idx >= 5:
            recent = history[target_idx-5:target_idx]
            all_numbers = []
            for draw in recent:
                all_numbers.extend(draw['numbers'])
            
            freq = Counter(all_numbers)
            
            # Estadísticas agregadas
            features.append(len(freq))  # Números únicos
            features.append(sum(freq.values()))  # Total números
            features.append(np.mean(list(freq.values())))  # Frecuencia promedio
            features.append(np.std(list(freq.values())) if len(freq) > 1 else 0)
            
            # Sumas
            sums = [sum(draw['numbers']) for draw in recent]
            features.extend([np.mean(sums), np.std(sums)])
        else:
            features.extend([0] * 6)
        
        return np.array(features)
    
    def _prepare_dataset(self, history):
        """
        Prepara dataset completo
        """
        X = []
        y = []
        
        for i in range(self.n_lags, len(history)):
            features = self._create_lag_features(history, i)
            X.append(features)
            
            # Target: vector binario
            target = np.zeros(56)
            for num in history[i]['numbers']:
                target[num - 1] = 1
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def fit(self, history):
        """
        Entrena 56 modelos Naive Bayes
        """
        if len(history) <= self.n_lags:
            raise ValueError(f"❌ Historial insuficiente. Necesitas al menos {self.n_lags + 1} sorteos.")
        
        # Preparar dataset
        X, y = self._prepare_dataset(history)
        
        # Normalizar features (mejora Gaussian NB)
        X = self.scaler.fit_transform(X)
        
        print(f"✅ {self.name}: Preparando entrenamiento...")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        
        # Entrenar modelo por cada número
        for num in range(1, 57):
            y_num = y[:, num - 1]
            
            # Crear Gaussian Naive Bayes
            model = GaussianNB(
                var_smoothing=1e-9  # Suavizado para estabilidad
            )
            
            # Entrenar
            model.fit(X, y_num)
            self.models[num] = model
            
            if num % 10 == 0:
                print(f"   ... entrenados {num}/56 modelos")
        
        print(f"✅ {self.name}: 56 modelos entrenados")
        return self
    
    def predict(self, history):
        """
        Predice siguiente sorteo
        """
        if not self.models:
            raise ValueError("❌ Modelo no entrenado. Llama fit() primero.")
        
        # Crear features
        features = self._create_lag_features(history, len(history)).reshape(1, -1)
        features = self.scaler.transform(features)
        
        # Obtener probabilidades
        probabilities = {}
        for num in range(1, 57):
            model = self.models[num]
            proba = model.predict_proba(features)[0]
            
            # Manejar caso: solo una clase observada durante el entrenamiento
            if len(proba) == 1:
                # Si la única clase es 0 → P(1) = 0
                # Si la única clase es 1 → P(1) = 1 (muy raro en lotería)
                if model.classes_[0] == 0:
                    prob = 0.0
                else:
                    prob = 1.0
            else:
                prob = proba[1]  # P(número=1|features)
                
            probabilities[num] = prob
        
        # Top 6
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, prob in sorted_probs[:6]]
        
        return sorted(top_numbers)


# ==================== TEST ====================
if __name__ == "__main__":
    print("📊 ALGORITMO NAIVE BAYES - TEST")
    print("=" * 60)
    
    # Cargar datos
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    from src.data.collector import MelateCollector
    
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: Entrenar
    print("\n" + "="*60)
    print("🔮 TEST 1: Entrenamiento Naive Bayes")
    print("="*60)
    
    nb_model = NaiveBayesLottery(n_lags=5)
    nb_model.fit(history)
    
    # Test 2: Predicción
    print("\n" + "="*60)
    print("🔮 TEST 2: Predicción")
    print("="*60)
    
    prediction = nb_model.predict(history)
    print(f"\n🎯 Predicción: {prediction}")
    
    # Test 3: Validación
    print("\n" + "="*60)
    print("📊 VALIDACIÓN (últimos 3 sorteos)")
    print("="*60)
    
    if len(history) >= 10:
        results = []
        
        for i in range(min(3, len(history) - 7)):
            train_data = history[:-(3-i)] if i < 2 else history[:-1]
            test_idx = -(3-i) if i < 2 else -1
            test_result = history[test_idx]['numbers']
            
            nb_val = NaiveBayesLottery(n_lags=5)
            nb_val.fit(train_data)
            pred_val = nb_val.predict(train_data)
            
            matches = len(set(pred_val) & set(test_result))
            results.append(matches)
            
            print(f"\nSorteo {history[test_idx]['date']}:")
            print(f"   Predicción:  {pred_val}")
            print(f"   Real:        {test_result}")
            print(f"   ✅ Aciertos: {matches}/6")
        
        if results:
            avg_matches = np.mean(results)
            print(f"\n📈 Promedio: {avg_matches:.2f}/6")
            print(f"   Esperado: 0.64 ± 0.72")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - Naive Bayes asume independencia entre features")
    print("   - Esta asunción es incorrecta (lag features correlacionados)")
    print("   - Performance esperada: ~1.0 aciertos")
    print("   - Conclusión: Asunción ingenua → predicción ingenua")