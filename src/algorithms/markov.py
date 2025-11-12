"""
Algoritmo #3: Markov Chain First-Order
Predice números basándose en transiciones observadas entre sorteos consecutivos
"""

import numpy as np
from collections import defaultdict, Counter
import os


class MarkovChain:
    """
    Cadena de Markov de primer orden para predicción de números.
    
    Teoría:
    - Modela P(número_t | número_t-1)
    - Asume dependencia temporal entre sorteos consecutivos
    - Usa matriz de transición 56x56
    
    Hipótesis:
    - Si la lotería NO es aleatoria, ciertos números "llevan" a otros
    - Esperamos encontrar: NO hay dependencia → matriz uniforme
    """
    
    def __init__(self, order=1):
        self.name = "Markov Chain (1st Order)"
        self.order = order
        self.transition_matrix = None
        self.state_counts = None
        
    def fit(self, history):
        """
        Construye matriz de transición desde historial
        
        Args:
            history: Lista de dicts con 'numbers' key
        """
        # Inicializar contadores
        transitions = defaultdict(lambda: defaultdict(int))
        state_totals = defaultdict(int)
        
        # Contar transiciones número_i → número_j entre sorteos consecutivos
        for i in range(len(history) - 1):
            current_numbers = set(history[i]['numbers'])
            next_numbers = set(history[i + 1]['numbers'])
            
            # Para cada número en sorteo actual
            for num_from in current_numbers:
                state_totals[num_from] += 1
                
                # Contar a qué números "transiciona" en siguiente sorteo
                for num_to in next_numbers:
                    transitions[num_from][num_to] += 1
        
        # Construir matriz de probabilidades (56x56)
        self.transition_matrix = np.zeros((56, 56))
        
        for num_from in range(1, 57):
            if state_totals[num_from] > 0:
                for num_to in range(1, 57):
                    count = transitions[num_from][num_to]
                    self.transition_matrix[num_from-1, num_to-1] = \
                        count / state_totals[num_from]
            else:
                # Si nunca vimos este número, distribución uniforme
                self.transition_matrix[num_from-1, :] = 1/56
        
        # Suavizado Laplace (evitar probabilidades 0)
        alpha = 0.01
        self.transition_matrix = (self.transition_matrix + alpha) / \
                                 (self.transition_matrix.sum(axis=1, keepdims=True) + 56*alpha)
        
        self.state_counts = state_totals
        print(f"✅ {self.name}: Matriz de transición construida")
        print(f"   Estados observados: {len(state_totals)}/56")
        
        return self
    
    def predict(self, history):
        """
        Predice siguiente sorteo usando última observación como estado inicial
        
        Método:
        1. Tomar últimos números como estado inicial
        2. Promediar sus distribuciones de transición
        3. Samplear top 6 números con mayor probabilidad
        """
        if self.transition_matrix is None:
            raise ValueError("❌ Modelo no entrenado. Llama fit() primero.")
        
        # Usar último sorteo como estado inicial
        last_numbers = history[-1]['numbers']
        
        # Promediar probabilidades de transición desde cada número
        probs = np.zeros(56)
        for num in last_numbers:
            probs += self.transition_matrix[num - 1, :]
        probs /= len(last_numbers)
        
        # Top 6 números con mayor probabilidad
        top_indices = np.argsort(probs)[-6:]
        prediction = sorted(top_indices + 1)  # +1 porque indices son 0-55
        
        return prediction
    
    def get_transition_prob(self, num_from, num_to):
        """Obtiene P(num_to | num_from)"""
        return self.transition_matrix[num_from - 1, num_to - 1]
    
    def analyze_stationarity(self):
        """
        Analiza si la matriz converge a distribución estacionaria
        (si es verdaderamente aleatoria, debería ser uniforme)
        """
        # Iterar matriz varias veces
        M = self.transition_matrix.copy()
        for _ in range(100):
            M = M @ self.transition_matrix
        
        # Distribución estacionaria = primera fila después de convergencia
        stationary = M[0, :]
        
        # Medir desviación de uniformidad
        uniform = np.ones(56) / 56
        kl_divergence = np.sum(stationary * np.log(stationary / uniform + 1e-10))
        
        print(f"📊 Análisis de Estacionariedad:")
        print(f"   KL-Divergence vs Uniforme: {kl_divergence:.6f}")
        print(f"   Interpretación: {'Cercano a uniforme (aleatorio)' if kl_divergence < 0.1 else 'Posible patrón detectado'}")
        
        return stationary, kl_divergence


class MarkovSecondOrder(MarkovChain):
    """
    Extensión: Markov de segundo orden
    P(número_t | número_t-1, número_t-2)
    """
    
    def __init__(self):
        super().__init__(order=2)
        self.name = "Markov Chain (2nd Order)"
        self.bigram_transitions = None
    
    def fit(self, history):
        """
        Construye transiciones basadas en pares de sorteos previos
        """
        if len(history) < 3:
            print("⚠️ Datos insuficientes para 2nd order. Usando 1st order.")
            return super().fit(history)
        
        transitions = defaultdict(lambda: defaultdict(int))
        bigram_counts = defaultdict(int)
        
        # Contar transiciones (num_t-2, num_t-1) → num_t
        for i in range(len(history) - 2):
            prev_prev = tuple(sorted(history[i]['numbers']))
            prev = tuple(sorted(history[i + 1]['numbers']))
            current = history[i + 2]['numbers']
            
            state = (prev_prev, prev)
            bigram_counts[state] += 1
            
            for num in current:
                transitions[state][num] += 1
        
        self.bigram_transitions = transitions
        print(f"✅ {self.name}: {len(bigram_counts)} estados bigram")
        
        return self
    
    def predict(self, history):
        """Predice usando últimos 2 sorteos como contexto"""
        if len(history) < 2 or not self.bigram_transitions:
            # Entrenar y usar 1st order si no hay datos suficientes
            fallback = MarkovChain()
            fallback.fit(history)
            return fallback.predict(history)
        
        # Estado = (sorteo_t-2, sorteo_t-1)
        state = (
            tuple(sorted(history[-2]['numbers'])),
            tuple(sorted(history[-1]['numbers']))
        )
        
        if state in self.bigram_transitions:
            # Usar transiciones observadas
            freq = Counter(self.bigram_transitions[state])
            top_numbers = [num for num, _ in freq.most_common(6)]
        else:
            # Estado no visto → fallback a 1st order
            print("⚠️ Estado no observado, usando fallback")
            fallback = MarkovChain()
            fallback.fit(history)
            return fallback.predict(history)
        
        # Completar con aleatorios si faltan
        while len(top_numbers) < 6:
            random_num = np.random.randint(1, 57)
            if random_num not in top_numbers:
                top_numbers.append(random_num)
        
        return sorted(top_numbers[:6])


# ==================== TEST ====================
if __name__ == "__main__":
    print("🔗 ALGORITMO MARKOV CHAIN - TEST")
    print("=" * 60)
    
    # Determinar ruta absoluta a data/raw/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    # Importar collector
    from src.data.collector import MelateCollector
    
    # Cargar sorteos
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    if len(history) < 5:
        print("⚠️ Datos insuficientes. Usando datos sintéticos.")
        history = [
            {'date': '2024-10-01', 'numbers': [5, 12, 23, 34, 45, 56]},
            {'date': '2024-10-05', 'numbers': [3, 12, 25, 34, 47, 55]},
            {'date': '2024-10-10', 'numbers': [1, 12, 28, 34, 49, 53]},
            {'date': '2024-10-15', 'numbers': [7, 14, 23, 36, 45, 51]},
            {'date': '2024-10-20', 'numbers': [2, 15, 23, 38, 45, 50]},
        ]
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: Markov 1st Order
    print("\n" + "="*60)
    print("🔮 TEST 1: Markov Chain (1st Order)")
    print("="*60)
    
    markov1 = MarkovChain()
    markov1.fit(history)
    
    prediction1 = markov1.predict(history)
    print(f"\n🎯 Predicción: {prediction1}")
    
    # Analizar estacionariedad
    markov1.analyze_stationarity()
    
    # Mostrar algunas probabilidades de transición
    print(f"\n📈 Ejemplos de Probabilidades de Transición:")
    for num_from in [12, 23, 34, 45]:  # Números frecuentes
        probs = [(i+1, markov1.get_transition_prob(num_from, i+1)) 
                 for i in range(56)]
        top_3 = sorted(probs, key=lambda x: x[1], reverse=True)[:3]
        print(f"   {num_from} → {top_3[0][0]} ({top_3[0][1]:.3f}), "
              f"{top_3[1][0]} ({top_3[1][1]:.3f}), "
              f"{top_3[2][0]} ({top_3[2][1]:.3f})")
    
    # Test 2: Markov 2nd Order (si hay datos suficientes)
    if len(history) >= 5:
        print("\n" + "="*60)
        print("🔮 TEST 2: Markov Chain (2nd Order)")
        print("="*60)
        
        markov2 = MarkovSecondOrder()
        markov2.fit(history)
        
        prediction2 = markov2.predict(history)
        print(f"\n🎯 Predicción: {prediction2}")
    
    # Test 3: Comparar con último resultado real
    print("\n" + "="*60)
    print("📊 VALIDACIÓN vs Último Sorteo Real")
    print("="*60)
    
    if len(history) >= 2:
        # Entrenar con n-1 sorteos
        train_data = history[:-1]
        test_result = history[-1]['numbers']
        
        markov1_retrain = MarkovChain()
        markov1_retrain.fit(train_data)
        pred_retrain = markov1_retrain.predict(train_data)
        
        matches = len(set(pred_retrain) & set(test_result))
        
        print(f"Predicción:      {pred_retrain}")
        print(f"Resultado Real:  {test_result}")
        print(f"✅ Aciertos: {matches}/6")
        print(f"   Esperado por azar: 0.64 ± 0.72")
        print(f"   Performance: {'🎉 Sobre azar' if matches >= 2 else '✅ Dentro de lo esperado'}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - KL-Div pequeña → Lotería es realmente aleatoria")
    print("   - Aciertos ~0-1 → Markov no mejora sobre azar")
    print("   - Conclusión esperada: NO hay memoria en el sistema")