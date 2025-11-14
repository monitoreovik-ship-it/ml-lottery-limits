"""
Algoritmo #11: SVM (Support Vector Machine)
Clasificador de margen máximo con kernel trick para no-linealidad
"""

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMLottery:
    """
    SVM para predicción de lotería.
    
    Teoría:
    - Encuentra hiperplano óptimo que separa clases
    - Kernel RBF permite capturar relaciones no lineales
    - Maximiza margen → buena generalización (en teoría)
    
    Arquitectura:
    - 56 clasificadores SVM (uno por número)
    - Kernel: RBF (Radial Basis Function)
    - C=1.0 (regularización), γ='scale'
    
    Hipótesis:
    - Si hay separabilidad no lineal, SVM la encontrará
    - Esperamos: overfitting en training, falla en test
    - Performance: ~1.0-1.3 aciertos
    """
    
    def __init__(self, n_lags=10, C=1.0, kernel='rbf'):
        """
        Args:
            n_lags: Sorteos previos como features
            C: Parámetro de regularización (default: 1.0)
            kernel: Tipo de kernel ('rbf', 'linear', 'poly')
        """
        self.name = f"SVM ({kernel}, C={C})"
        self.n_lags = n_lags
        self.C = C
        self.kernel = kernel
        self.models = {}
        self.scaler = StandardScaler()
        
    def _create_lag_features(self, history, target_idx):
        """
        Features (mismo esquema que otros algoritmos)
        """
        features = []
        
        # Lags
        start_idx = max(0, target_idx - self.n_lags)
        lag_window = history[start_idx:target_idx]
        
        for draw in lag_window:
            features.extend(sorted(draw['numbers']))
        
        if len(lag_window) < self.n_lags:
            padding = [0] * (6 * (self.n_lags - len(lag_window)))
            features = padding + features
        
        # Rolling stats
        if target_idx >= 5:
            recent = history[target_idx-5:target_idx]
            sums = [sum(draw['numbers']) for draw in recent]
            features.extend([np.mean(sums), np.std(sums), np.min(sums), np.max(sums)])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def _prepare_dataset(self, history):
        """
        Prepara dataset
        """
        X = []
        y = []
        
        for i in range(self.n_lags, len(history)):
            features = self._create_lag_features(history, i)
            X.append(features)
            
            target = np.zeros(56)
            for num in history[i]['numbers']:
                target[num - 1] = 1
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def fit(self, history):
        """
        Entrena 56 SVMs (con manejo de una sola clase)
        """
        if len(history) <= self.n_lags:
            raise ValueError(f"❌ Historial insuficiente. Necesitas al menos {self.n_lags + 1} sorteos.")
        
        X, y = self._prepare_dataset(history)
        X = self.scaler.fit_transform(X)
        
        print(f"✅ {self.name}: Preparando entrenamiento...")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Kernel: {self.kernel}")
        
        for num in range(1, 57):
            y_num = y[:, num - 1]
            
            # Verificar si hay al menos dos clases
            unique_classes = np.unique(y_num)
            if len(unique_classes) < 2:
                # Solo una clase → usar modelo dummy
                print(f"   ⚠️  Número {num}: solo clase {unique_classes[0]} → usando predicción constante")
                self.models[num] = {'type': 'dummy', 'class': unique_classes[0]}
            else:
                # Entrenar SVM normal
                model = SVC(
                    C=self.C,
                    kernel=self.kernel,
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
                model.fit(X, y_num)
                self.models[num] = {'type': 'svm', 'model': model}
            
            if num % 10 == 0:
                print(f"   ... procesados {num}/56 modelos")
        
        print(f"✅ {self.name}: 56 modelos preparados")
        return self
    
    def predict(self, history):
        """
        Predice siguiente sorteo
        """
        if not self.models:
            raise ValueError("❌ Modelo no entrenado.")
        
        features = self._create_lag_features(history, len(history)).reshape(1, -1)
        features = self.scaler.transform(features)
        
        probabilities = {}
        for num in range(1, 57):
            model_info = self.models[num]
            
            if model_info['type'] == 'dummy':
                # Predicción constante
                prob = 1.0 if model_info['class'] == 1 else 0.0
            else:
                # SVM normal
                proba = model_info['model'].predict_proba(features)[0]
                # Manejar caso extremo (aunque raro en SVM con 2 clases)
                if len(proba) == 1:
                    prob = 1.0 if model_info['model'].classes_[0] == 1 else 0.0
                else:
                    prob = proba[1]  # P(clase=1)
            
            probabilities[num] = prob
        
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, prob in sorted_probs[:6]]
        
        return sorted(top_numbers)


# ==================== TEST ====================
if __name__ == "__main__":
    print("⚔️  ALGORITMO SVM - TEST")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    from src.data.collector import MelateCollector
    
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: SVM RBF
    print("\n" + "="*60)
    print("🔮 TEST 1: SVM con kernel RBF")
    print("="*60)
    
    svm_model = SVMLottery(n_lags=5, C=1.0, kernel='rbf')
    svm_model.fit(history)
    
    prediction = svm_model.predict(history)
    print(f"\n🎯 Predicción: {prediction}")
    
    # Test 2: Validación
    print("\n" + "="*60)
    print("📊 VALIDACIÓN (últimos 3 sorteos)")
    print("="*60)
    
    if len(history) >= 10:
        results = []
        
        for i in range(min(3, len(history) - 7)):
            train_data = history[:-(3-i)] if i < 2 else history[:-1]
            test_idx = -(3-i) if i < 2 else -1
            test_result = history[test_idx]['numbers']
            
            svm_val = SVMLottery(n_lags=5, C=1.0, kernel='rbf')
            svm_val.fit(train_data)
            pred_val = svm_val.predict(train_data)
            
            matches = len(set(pred_val) & set(test_result))
            results.append(matches)
            
            print(f"\nSorteo {history[test_idx]['date']}:")
            print(f"   Predicción:  {pred_val}")
            print(f"   Real:        {test_result}")
            print(f"   ✅ Aciertos: {matches}/6")
        
        if results:
            avg = np.mean(results)
            print(f"\n📈 Promedio: {avg:.2f}/6")
            print(f"   Esperado: 0.64 ± 0.72")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - SVM busca hiperplano de separación óptima")
    print("   - Kernel RBF captura no-linealidad")
    print("   - En datos aleatorios: hiperplano es arbitrario")
    print("   - Performance: ~1.0-1.3 aciertos (similar a otros ML)")