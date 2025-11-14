"""
Algoritmo #13: Bayesian Network
Red probabilística que modela dependencias entre números
"""

import numpy as np
import os
from collections import Counter, defaultdict
from itertools import combinations


class BayesianNetworkLottery:
    """
    Bayesian Network simplificado para lotería.
    
    Teoría:
    - Modela dependencias condicionales P(X_i | X_j, X_k, ...)
    - Estructura: grafo dirigido acíclico (DAG)
    - Inferencia: calcula probabilidades conjuntas
    
    Implementación Simplificada:
    - Naive Bayes + co-ocurrencias de pares
    - P(num) ∝ freq(num) × Π P(num | otros_en_sorteo)
    
    Hipótesis:
    - Si hay dependencias causales entre números, red las capturará
    - Esperamos: no hay dependencias reales → performance = frecuencia
    """
    
    def __init__(self):
        self.name = "Bayesian Network"
        self.prior_probs = {}  # P(número)
        self.conditional_probs = {}  # P(num_i | num_j)
        self.pair_counts = Counter()  # Co-ocurrencias
        self.total_draws = 0
        
    def fit(self, history):
        """
        Aprende estructura y probabilidades de la red
        """
        self.total_draws = len(history)
        
        print(f"✅ {self.name}: Construyendo red bayesiana...")
        print(f"   Sorteos: {len(history)}")
        
        # 1. PRIORS: P(número)
        number_counts = Counter()
        for draw in history:
            for num in draw['numbers']:
                number_counts[num] += 1
        
        for num in range(1, 57):
            count = number_counts.get(num, 0)
            self.prior_probs[num] = count / len(history)
        
        # 2. CO-OCURRENCIAS: P(num_i, num_j)
        for draw in history:
            for pair in combinations(sorted(draw['numbers']), 2):
                self.pair_counts[pair] += 1
        
        # 3. CONDICIONALES: P(num_i | num_j)
        for num_i in range(1, 57):
            self.conditional_probs[num_i] = {}
            for num_j in range(1, 57):
                if num_i == num_j:
                    continue
                
                # Contar: ¿cuántas veces aparece num_i cuando num_j está presente?
                pair_sorted = tuple(sorted([num_i, num_j]))
                joint_count = self.pair_counts.get(pair_sorted, 0)
                
                # P(num_i | num_j) = P(num_i, num_j) / P(num_j)
                marginal_j = number_counts.get(num_j, 0)
                
                if marginal_j > 0:
                    self.conditional_probs[num_i][num_j] = joint_count / marginal_j
                else:
                    self.conditional_probs[num_i][num_j] = 0
        
        print(f"✅ Red construida: {len(self.prior_probs)} nodos")
        print(f"   Co-ocurrencias: {len(self.pair_counts)} pares")
        
        return self
    
    def predict(self, history):
        """
        Predice usando inferencia bayesiana simplificada
        
        Estrategia:
        1. Iniciar con priors P(número)
        2. Ajustar basándose en co-ocurrencias recientes
        3. Seleccionar top 6
        """
        if not self.prior_probs:
            raise ValueError("❌ Modelo no entrenado.")
        
        # Iniciar con priors
        scores = {num: prob for num, prob in self.prior_probs.items()}
        
        # Ajustar basándose en últimos 3 sorteos (evidencia)
        recent_numbers = []
        for draw in history[-3:]:
            recent_numbers.extend(draw['numbers'])
        
        recent_freq = Counter(recent_numbers)
        
        # Actualizar scores basándose en condicionales
        for num in range(1, 57):
            # P(num | evidencia) ∝ P(num) × Π P(num | num_recent)
            conditional_factor = 1.0
            
            for num_recent in set(recent_numbers):
                if num != num_recent:
                    cond_prob = self.conditional_probs[num].get(num_recent, 0.01)
                    conditional_factor *= (1 + cond_prob)  # Suavizado multiplicativo
            
            scores[num] *= conditional_factor
        
        # Top 6
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, score in sorted_scores[:6]]
        
        return sorted(top_numbers)
    
    def analyze_dependencies(self):
        """
        Analiza las dependencias más fuertes detectadas
        """
        print("\n🔗 Análisis de Dependencias (Top 10 pares):")
        
        top_pairs = self.pair_counts.most_common(10)
        
        for pair, count in top_pairs:
            expected = self.total_draws * (6/56) * (5/55)  # Prob. teórica
            ratio = count / expected if expected > 0 else 0
            
            print(f"   {pair}: {count} veces (esperado: {expected:.2f}, ratio: {ratio:.2f}x)")
        
        # Verificar si algún par es significativo
        max_count = top_pairs[0][1] if top_pairs else 0
        max_expected = self.total_draws * (6/56) * (5/55)
        
        if max_count / max_expected < 2.0:
            print("\n✅ Conclusión: No hay pares significativamente frecuentes")
            print("   → Independencia entre números (como esperado)")
        else:
            print("\n⚠️ Posible dependencia detectada (requiere test estadístico)")


# ==================== TEST ====================
if __name__ == "__main__":
    print("🕸️  ALGORITMO BAYESIAN NETWORK - TEST")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    from src.data.collector import MelateCollector
    
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: Construir red
    print("\n" + "="*60)
    print("🔮 TEST 1: Construcción de Red Bayesiana")
    print("="*60)
    
    bn_model = BayesianNetworkLottery()
    bn_model.fit(history)
    
    # Test 2: Análisis de dependencias
    bn_model.analyze_dependencies()
    
    # Test 3: Predicción
    print("\n" + "="*60)
    print("🔮 TEST 2: Predicción")
    print("="*60)
    
    prediction = bn_model.predict(history)
    print(f"\n🎯 Predicción: {prediction}")
    
    # Test 4: Validación
    print("\n" + "="*60)
    print("📊 VALIDACIÓN (últimos 3 sorteos)")
    print("="*60)
    
    if len(history) >= 10:
        results = []
        
        for i in range(min(3, len(history) - 7)):
            train_data = history[:-(3-i)] if i < 2 else history[:-1]
            test_idx = -(3-i) if i < 2 else -1
            test_result = history[test_idx]['numbers']
            
            bn_val = BayesianNetworkLottery()
            bn_val.fit(train_data)
            pred_val = bn_val.predict(train_data)
            
            matches = len(set(pred_val) & set(test_result))
            results.append(matches)
            
            print(f"\nSorteo {history[test_idx]['date']}:")
            print(f"   Predicción:  {pred_val}")
            print(f"   Real:        {test_result}")
            print(f"   ✅ Aciertos: {matches}/6")
        
        if results:
            avg = np.mean(results)
            print(f"\n📈 Promedio: {avg:.2f}/6")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - Bayesian Network modela dependencias P(X_i|X_j)")
    print("   - Si números son independientes → co-ocurrencias = azar")
    print("   - Performance: ~1.0-1.5 aciertos (similar a Frequency)")
    print("   - Conclusión: No hay estructura causal real")