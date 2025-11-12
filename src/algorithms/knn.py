"""
Algoritmo #4: K-Nearest Neighbors (KNN)
Predice basándose en similitud con sorteos históricos más cercanos
"""

import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import os


class KNNLottery:
    """
    K-Nearest Neighbors para predicción de lotería.
    
    Teoría:
    - Busca los K sorteos históricos más "similares" al contexto actual
    - Define similitud mediante distancia Hamming o features engineeradas
    - Predice los números más frecuentes en esos K vecinos
    
    Hipótesis:
    - Si existe estructura, sorteos "similares" deberían tener resultados similares
    - Esperamos: NO hay estructura → vecinos no mejoran predicción
    """
    
    def __init__(self, k=5, metric='hamming'):
        """
        Args:
            k: Número de vecinos a considerar
            metric: 'hamming' (para sets de números) o 'euclidean' (para features)
        """
        self.name = f"KNN (k={k}, {metric})"
        self.k = k
        self.metric = metric
        self.model = None
        self.history_vectors = None
        self.history_results = None
        
    def _extract_features(self, draw):
        """
        Ingeniería de features desde un sorteo
        
        Features (17 dimensiones):
        - 6 números principales (ordenados)
        - Suma total
        - Promedio
        - Rango (max - min)
        - Desviación estándar
        - Cantidad pares
        - Cantidad impares
        - Cantidad primos
        - Cantidad en 1er tercio (1-18)
        - Cantidad en 2do tercio (19-37)
        - Cantidad en 3er tercio (38-56)
        - Gap promedio entre números consecutivos
        """
        numbers = sorted(draw['numbers'])
        
        features = []
        
        # Números principales (6 features)
        features.extend(numbers)
        
        # Estadísticas básicas (4 features)
        features.append(sum(numbers))  # Suma
        features.append(np.mean(numbers))  # Promedio
        features.append(max(numbers) - min(numbers))  # Rango
        features.append(np.std(numbers))  # Desviación estándar
        
        # Paridad (2 features)
        features.append(sum(1 for n in numbers if n % 2 == 0))  # Pares
        features.append(sum(1 for n in numbers if n % 2 == 1))  # Impares
        
        # Primos (1 feature)
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
        features.append(sum(1 for n in numbers if n in primes))
        
        # Distribución por tercios (3 features)
        features.append(sum(1 for n in numbers if 1 <= n <= 18))
        features.append(sum(1 for n in numbers if 19 <= n <= 37))
        features.append(sum(1 for n in numbers if 38 <= n <= 56))
        
        # Gap promedio (1 feature)
        gaps = [numbers[i+1] - numbers[i] for i in range(5)]
        features.append(np.mean(gaps))
        
        return np.array(features)
    
    def fit(self, history):
        """
        Entrena el modelo KNN con historial
        
        Args:
            history: Lista de dicts con 'numbers' key
        """
        # Extraer features de todos los sorteos
        self.history_vectors = np.array([
            self._extract_features(draw) for draw in history
        ])
        
        # Guardar resultados (números ganadores)
        self.history_results = [draw['numbers'] for draw in history]
        
        # Entrenar modelo KNN
        # Nota: usamos el doble de k para tener margen
        n_neighbors = min(self.k * 2, len(history) - 1)
        
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self.metric if self.metric != 'hamming' else 'euclidean'
        )
        self.model.fit(self.history_vectors)
        
        print(f"✅ {self.name}: Entrenado con {len(history)} sorteos")
        print(f"   Features: {self.history_vectors.shape[1]} dimensiones")
        print(f"   Vecinos: {n_neighbors}")
        
        return self
    
    def predict(self, history):
        """
        Predice siguiente sorteo basándose en K vecinos más cercanos
        
        Método:
        1. Extraer features del último sorteo (contexto actual)
        2. Encontrar K sorteos históricos más similares
        3. Agregar números frecuentes en esos K vecinos
        4. Retornar top 6
        """
        if self.model is None:
            raise ValueError("❌ Modelo no entrenado. Llama fit() primero.")
        
        # Features del último sorteo
        last_features = self._extract_features(history[-1]).reshape(1, -1)
        
        # Encontrar K vecinos más cercanos
        distances, indices = self.model.kneighbors(last_features, n_neighbors=self.k)
        
        # Agregar números de los vecinos
        neighbor_numbers = []
        for idx in indices[0]:
            neighbor_numbers.extend(self.history_results[idx])
        
        # Contar frecuencias
        freq = Counter(neighbor_numbers)
        
        # Top 6 más frecuentes
        top_numbers = [num for num, count in freq.most_common(6)]
        
        # Si faltan, completar con menos frecuentes (evitar aleatorios para reproducibilidad)
        if len(top_numbers) < 6:
            remaining = [n for n in range(1, 57) if n not in top_numbers]
            top_numbers.extend(remaining[:6 - len(top_numbers)])
        
        return sorted(top_numbers[:6])
    
    def explain_prediction(self, history):
        """
        Explica la predicción mostrando los vecinos más cercanos
        """
        if self.model is None:
            raise ValueError("❌ Modelo no entrenado.")
        
        last_features = self._extract_features(history[-1]).reshape(1, -1)
        distances, indices = self.model.kneighbors(last_features, n_neighbors=self.k)
        
        print(f"\n🔍 Explicación de Predicción (KNN k={self.k}):")
        print(f"   Último sorteo: {history[-1]['numbers']}")
        print(f"\n   Los {self.k} vecinos más cercanos:")
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            neighbor = history[idx]
            print(f"   {i+1}. {neighbor['date']} - {neighbor['numbers']} (distancia: {dist:.2f})")
        
        # Predicción final
        prediction = self.predict(history)
        print(f"\n   🎯 Predicción agregada: {prediction}")
        
        return prediction


class KNNEnsemble:
    """
    Ensemble de múltiples KNN con diferentes valores de K
    Combina predicciones por votación
    """
    
    def __init__(self, k_values=[3, 5, 7, 9]):
        self.name = f"KNN Ensemble (k={k_values})"
        self.k_values = k_values
        self.models = []
        
    def fit(self, history):
        """Entrena múltiples modelos KNN"""
        self.models = []
        for k in self.k_values:
            model = KNNLottery(k=k, metric='euclidean')
            model.fit(history)
            self.models.append(model)
        
        print(f"✅ {self.name}: {len(self.models)} modelos entrenados")
        return self
    
    def predict(self, history):
        """Combina predicciones de todos los modelos"""
        all_predictions = []
        
        for model in self.models:
            pred = model.predict(history)
            all_predictions.extend(pred)
        
        # Votación: top 6 más frecuentes
        freq = Counter(all_predictions)
        top_numbers = [num for num, count in freq.most_common(6)]
        
        # Completar si faltan
        if len(top_numbers) < 6:
            remaining = [n for n in range(1, 57) if n not in top_numbers]
            top_numbers.extend(remaining[:6 - len(top_numbers)])
        
        return sorted(top_numbers[:6])


# ==================== TEST ====================
if __name__ == "__main__":
    print("🎯 ALGORITMO KNN - TEST")
    print("=" * 60)
    
    # Determinar ruta absoluta a data/raw/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    # Importar collector
    from src.data.collector import MelateCollector
    
    # Cargar sorteos
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    if len(history) < 10:
        print("⚠️ Datos insuficientes. Usando datos sintéticos.")
        history = [
            {'date': '2024-10-01', 'numbers': [5, 12, 23, 34, 45, 56]},
            {'date': '2024-10-05', 'numbers': [3, 12, 25, 34, 47, 55]},
            {'date': '2024-10-10', 'numbers': [1, 12, 28, 34, 49, 53]},
            {'date': '2024-10-15', 'numbers': [7, 14, 23, 36, 45, 51]},
            {'date': '2024-10-20', 'numbers': [2, 15, 23, 38, 45, 50]},
            {'date': '2024-10-25', 'numbers': [8, 16, 24, 35, 46, 52]},
            {'date': '2024-11-01', 'numbers': [4, 17, 26, 37, 48, 54]},
            {'date': '2024-11-05', 'numbers': [9, 18, 27, 39, 49, 55]},
            {'date': '2024-11-10', 'numbers': [6, 19, 29, 40, 50, 56]},
            {'date': '2024-11-15', 'numbers': [10, 20, 30, 41, 51, 53]},
        ]
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: KNN básico (k=5)
    print("\n" + "="*60)
    print("🔮 TEST 1: KNN Básico (k=5)")
    print("="*60)
    
    knn5 = KNNLottery(k=5, metric='euclidean')
    knn5.fit(history)
    
    prediction5 = knn5.predict(history)
    print(f"\n🎯 Predicción: {prediction5}")
    
    # Explicación detallada
    knn5.explain_prediction(history)
    
    # Test 2: Comparar diferentes valores de K
    print("\n" + "="*60)
    print("🔮 TEST 2: Comparación de K valores")
    print("="*60)
    
    k_values = [3, 5, 7]
    for k in k_values:
        knn = KNNLottery(k=k, metric='euclidean')
        knn.fit(history)
        pred = knn.predict(history)
        print(f"   k={k}: {pred}")
    
    # Test 3: Ensemble
    print("\n" + "="*60)
    print("🔮 TEST 3: KNN Ensemble")
    print("="*60)
    
    ensemble = KNNEnsemble(k_values=[3, 5, 7, 9])
    ensemble.fit(history)
    
    pred_ensemble = ensemble.predict(history)
    print(f"\n🎯 Predicción Ensemble: {pred_ensemble}")
    
    # Test 4: Validación con último sorteo
    print("\n" + "="*60)
    print("📊 VALIDACIÓN vs Último Sorteo Real")
    print("="*60)
    
    if len(history) >= 10:
        # Entrenar con n-1 sorteos
        train_data = history[:-1]
        test_result = history[-1]['numbers']
        
        knn_val = KNNLottery(k=5, metric='euclidean')
        knn_val.fit(train_data)
        pred_val = knn_val.predict(train_data)
        
        matches = len(set(pred_val) & set(test_result))
        
        print(f"Predicción:      {pred_val}")
        print(f"Resultado Real:  {test_result}")
        print(f"✅ Aciertos: {matches}/6")
        print(f"   Esperado por azar: 0.64 ± 0.72")
        
        # Análisis estadístico
        z_score = (matches - 0.64) / 0.72
        print(f"   Z-score: {z_score:.2f}")
        print(f"   Performance: {'🎉 Significativo' if abs(z_score) > 2 else '✅ Dentro de lo esperado'}")
    
    # Test 5: Análisis de features
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE FEATURES")
    print("="*60)
    
    print("\nÚltimos 3 sorteos (features extraídas):")
    for draw in history[-3:]:
        features = knn5._extract_features(draw)
        print(f"\n{draw['date']}: {draw['numbers']}")
        print(f"   Suma: {features[6]:.0f}")
        print(f"   Promedio: {features[7]:.1f}")
        print(f"   Rango: {features[8]:.0f}")
        print(f"   Pares/Impares: {features[10]:.0f}/{features[11]:.0f}")
        print(f"   Primos: {features[12]:.0f}")
        print(f"   Distribución tercios: {features[13]:.0f}-{features[14]:.0f}-{features[15]:.0f}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - Si aciertos ~0-1 → KNN no encuentra estructura")
    print("   - Features no capturan patrones predictivos")
    print("   - Vecinos 'similares' no producen resultados similares")
    print("   - Conclusión esperada: Lotería es independiente del pasado")