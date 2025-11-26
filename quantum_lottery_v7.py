#!/usr/bin/env python3
"""
Sistema Cuántico-Probabilístico V7.0 para Melate
Inspirado en principios de mecánica cuántica

Conceptos clave:
1. Superposición: Múltiples modelos simultáneos
2. Función de onda: Distribución de probabilidad
3. Entrelazamiento: Coocurrencias entre números
4. Colapso: Selección de combinación final
5. Incertidumbre: Límites probabilísticos aceptados
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from datetime import datetime
import itertools

class QuantumLotterySystem:
    """Sistema cuántico-probabilístico para predicción de lotería"""
    
    def __init__(self, min_num=1, max_num=56, combination_size=6):
        self.min_num = min_num
        self.max_num = max_num
        self.n_numbers = max_num - min_num + 1
        self.combination_size = combination_size
        
        # Estado cuántico inicial (superposición uniforme)
        self.psi = None  # Función de onda
        self.entanglement_matrix = None  # Matriz de entrelazamiento
        
    def load_historical_data(self, data_dir="data/raw"):
        """Carga datos históricos"""
        path = Path(data_dir)
        files = sorted(path.glob("*.json"))
        
        draws = []
        for file in files:
            with open(file, 'r') as f:
                draw = json.load(f)
                draws.append(draw['numbers'])
        
        print(f"📂 Cargados {len(draws)} sorteos históricos")
        return draws
    
    def initialize_wave_function(self, draws: List[List[int]]):
        """
        Inicializa función de onda |ψ⟩
        Representa probabilidad de cada número en superposición
        """
        print("\n🌊 INICIALIZANDO FUNCIÓN DE ONDA |ψ⟩")
        
        # Contar frecuencias
        all_numbers = [num for draw in draws for num in draw]
        freq = Counter(all_numbers)
        
        # Normalizar a probabilidades
        total = sum(freq.values())
        self.psi = np.zeros(self.n_numbers)
        
        for num in range(self.min_num, self.max_num + 1):
            idx = num - self.min_num
            self.psi[idx] = freq.get(num, 0) / total
        
        # Normalización cuántica (suma de cuadrados = 1)
        self.psi = self.psi / np.linalg.norm(self.psi)
        
        print(f"✅ |ψ⟩ inicializada: {self.n_numbers} estados cuánticos")
        print(f"   Norma: {np.linalg.norm(self.psi):.6f} (debe ser ~1.0)")
        
        return self.psi
    
    def compute_entanglement_matrix(self, draws: List[List[int]]):
        """
        Calcula matriz de entrelazamiento cuántico
        E[i,j] = probabilidad de que números i y j aparezcan juntos
        """
        print("\n🔗 CALCULANDO MATRIZ DE ENTRELAZAMIENTO")
        
        # Inicializar matriz
        E = np.zeros((self.n_numbers, self.n_numbers))
        
        # Contar coocurrencias
        for draw in draws:
            for num1, num2 in itertools.combinations(draw, 2):
                idx1 = num1 - self.min_num
                idx2 = num2 - self.min_num
                E[idx1, idx2] += 1
                E[idx2, idx1] += 1  # Simétrica
        
        # Normalizar
        max_cooccur = len(draws) * (self.combination_size - 1)
        E = E / max_cooccur
        
        self.entanglement_matrix = E
        
        # Estadísticas
        non_zero = np.count_nonzero(E)
        total_pairs = self.n_numbers * (self.n_numbers - 1) / 2
        density = non_zero / (2 * total_pairs) * 100
        
        print(f"✅ Matriz de entrelazamiento {self.n_numbers}x{self.n_numbers}")
        print(f"   Densidad: {density:.2f}% de pares entrelazados")
        print(f"   Max entrelazamiento: {E.max():.4f}")
        
        return E
    
    def quantum_oscillation_model(self, draws: List[List[int]], periods=[5, 10, 20]):
        """
        Modelo de oscilación cuántica
        Detecta ciclos periódicos en apariciones
        """
        print("\n〰️ MODELO DE OSCILACIÓN CUÁNTICA")
        
        oscillation_scores = np.zeros(self.n_numbers)
        
        for period in periods:
            if len(draws) < period:
                continue
            
            recent_window = draws[-period:]
            recent_freq = Counter([num for draw in recent_window for num in draw])
            
            for num in range(self.min_num, self.max_num + 1):
                idx = num - self.min_num
                recent_count = recent_freq.get(num, 0)
                expected = period * self.combination_size / self.n_numbers
                
                # Puntaje de oscilación
                oscillation = abs(recent_count - expected) / expected if expected > 0 else 0
                oscillation_scores[idx] += oscillation
        
        # Normalizar
        oscillation_scores = oscillation_scores / len(periods)
        
        print(f"✅ Oscilaciones calculadas para períodos {periods}")
        print(f"   Max oscilación: {oscillation_scores.max():.4f}")
        
        return oscillation_scores
    
    def anti_frequency_model(self, draws: List[List[int]], lookback=20):
        """
        Modelo anti-frecuencia
        Teoría: números que NO han salido recientemente tienen mayor probabilidad
        """
        print("\n🔄 MODELO ANTI-FRECUENCIA (NÚMEROS FRÍOS)")
        
        if len(draws) < lookback:
            lookback = len(draws)
        
        recent_draws = draws[-lookback:]
        recent_numbers = set([num for draw in recent_draws for num in draw])
        
        anti_freq_scores = np.ones(self.n_numbers)
        
        for num in range(self.min_num, self.max_num + 1):
            idx = num - self.min_num
            if num in recent_numbers:
                # Penalizar números recientes
                anti_freq_scores[idx] = 0.3
            else:
                # Bonificar números fríos
                anti_freq_scores[idx] = 1.5
        
        # Normalizar
        anti_freq_scores = anti_freq_scores / np.sum(anti_freq_scores)
        
        cold_nums = [num for num in range(self.min_num, self.max_num + 1) 
                     if num not in recent_numbers]
        
        print(f"✅ Números fríos (últimos {lookback} sorteos): {len(cold_nums)}")
        print(f"   Ejemplos: {cold_nums[:10]}")
        
        return anti_freq_scores
    
    def pattern_disruption_model(self, draws: List[List[int]]):
        """
        Modelo de disrupción de patrones
        Penaliza números que aparecen en patrones predecibles
        """
        print("\n💥 MODELO DE DISRUPCIÓN DE PATRONES")
        
        disruption_scores = np.ones(self.n_numbers)
        
        # Detectar números que salen en múltiplos
        if len(draws) >= 10:
            for num in range(self.min_num, self.max_num + 1):
                positions = [i for i, draw in enumerate(draws) if num in draw]
                
                if len(positions) >= 3:
                    # Calcular diferencias entre apariciones
                    diffs = np.diff(positions)
                    
                    # Si hay patrón regular, penalizar
                    if len(set(diffs)) <= 2:  # Muy regular
                        idx = num - self.min_num
                        disruption_scores[idx] = 0.5
        
        print(f"✅ Patrones analizados")
        print(f"   Números con patrón regular penalizados")
        
        return disruption_scores
    
    def quantum_superposition(self, models: Dict[str, np.ndarray], weights: Dict[str, float]):
        """
        Superpone múltiples modelos en un estado cuántico integrado
        |Ψ⟩ = Σ wᵢ |ψᵢ⟩
        """
        print("\n⚛️ SUPERPOSICIÓN CUÁNTICA DE MODELOS")
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Superponer
        superposition = np.zeros(self.n_numbers)
        
        for model_name, model_scores in models.items():
            weight = normalized_weights.get(model_name, 0)
            superposition += weight * model_scores
            print(f"   + {weight:.3f} × {model_name}")
        
        # Normalización cuántica
        superposition = superposition / np.linalg.norm(superposition)
        
        print(f"✅ Estado superpuesto |Ψ⟩ generado")
        print(f"   Norma: {np.linalg.norm(superposition):.6f}")
        
        return superposition, normalized_weights
    
    def quantum_collapse(self, superposition: np.ndarray, 
                        entanglement_matrix: np.ndarray,
                        n_combinations=3) -> List[List[int]]:
        """
        Colapso de función de onda a combinaciones concretas
        Maximiza probabilidad conjunta + entrelazamiento
        """
        print("\n💥 COLAPSO DE FUNCIÓN DE ONDA")
        
        # Probabilidades individuales
        probs = superposition ** 2  # |ψ|²
        
        combinations = []
        
        for combo_num in range(n_combinations):
            print(f"\n   Colapsando combinación #{combo_num + 1}...")
            
            # Método de Monte Carlo con entrelazamiento
            selected = []
            available = list(range(self.min_num, self.max_num + 1))
            
            for pick in range(self.combination_size):
                if pick == 0:
                    # Primer número: usar probabilidad pura
                    pick_probs = np.array([probs[n - self.min_num] for n in available])
                else:
                    # Números siguientes: considerar entrelazamiento
                    pick_probs = np.zeros(len(available))
                    
                    for i, num in enumerate(available):
                        idx = num - self.min_num
                        
                        # Probabilidad base
                        base_prob = probs[idx]
                        
                        # Bonus por entrelazamiento con ya seleccionados
                        entanglement_bonus = 0
                        for sel_num in selected:
                            sel_idx = sel_num - self.min_num
                            entanglement_bonus += entanglement_matrix[idx, sel_idx]
                        
                        pick_probs[i] = base_prob * (1 + entanglement_bonus)
                
                # Normalizar
                pick_probs = pick_probs / np.sum(pick_probs)
                
                # Seleccionar
                chosen_idx = np.random.choice(len(available), p=pick_probs)
                chosen_num = available[chosen_idx]
                
                selected.append(chosen_num)
                available.remove(chosen_num)
            
            combinations.append(sorted(selected))
            
            # Para siguiente combinación, reducir probabilidad de números ya usados
            for num in selected:
                probs[num - self.min_num] *= 0.5
            probs = probs / np.sum(probs)
        
        print(f"\n✅ {n_combinations} combinaciones colapsadas")
        
        return combinations
    
    def predict(self, draw_date: str, data_dir="data/raw", n_combinations=3):
        """Pipeline completo de predicción"""
        
        print("="*60)
        print("🌌 SISTEMA CUÁNTICO-PROBABILÍSTICO V7.0")
        print("="*60)
        print(f"\n📅 Predicción para sorteo: {draw_date}")
        
        # 1. Cargar datos
        draws = self.load_historical_data(data_dir)
        
        # 2. Inicializar función de onda
        wave_function = self.initialize_wave_function(draws)
        
        # 3. Calcular entrelazamiento
        entanglement = self.compute_entanglement_matrix(draws)
        
        # 4. Modelos múltiples (superposición)
        models = {
            'wave_function': wave_function,
            'oscillation': self.quantum_oscillation_model(draws),
            'anti_frequency': self.anti_frequency_model(draws),
            'disruption': self.pattern_disruption_model(draws)
        }
        
        # 5. Pesos de modelos (ajustables)
        weights = {
            'wave_function': 0.25,      # Frecuencias base
            'oscillation': 0.25,        # Ciclos
            'anti_frequency': 0.30,     # Números fríos (mayor peso)
            'disruption': 0.20          # Anti-patrones
        }
        
        # 6. Superposición cuántica
        superposition, norm_weights = self.quantum_superposition(models, weights)
        
        # 7. Colapso a combinaciones
        combinations = self.quantum_collapse(
            superposition, 
            entanglement,
            n_combinations=n_combinations
        )
        
        # 8. Análisis de probabilidades
        print("\n📊 ANÁLISIS DE PROBABILIDADES")
        probs = superposition ** 2
        top_numbers = np.argsort(probs)[::-1][:15] + self.min_num
        
        print("\n🔝 Top 15 números por probabilidad cuántica:")
        for i, num in enumerate(top_numbers, 1):
            idx = num - self.min_num
            prob = probs[idx] * 100
            bar = "█" * int(prob * 5)
            print(f"   {i:2d}. #{num:2d}: {prob:5.2f}% {bar}")
        
        print("\n" + "="*60)
        print("🎯 COMBINACIONES PREDICHAS")
        print("="*60)
        
        for i, combo in enumerate(combinations, 1):
            print(f"\nCombinación #{i}: {combo}")
            
            # Calcular score de entrelazamiento
            entangle_score = 0
            for n1, n2 in itertools.combinations(combo, 2):
                idx1, idx2 = n1 - self.min_num, n2 - self.min_num
                entangle_score += entanglement[idx1, idx2]
            entangle_score = entangle_score / (self.combination_size * (self.combination_size - 1) / 2)
            
            print(f"   Entrelazamiento: {entangle_score:.4f}")
        
        # Guardar predicción
        self.save_prediction(draw_date, combinations, superposition, norm_weights)
        
        return combinations
    
    def save_prediction(self, draw_date: str, combinations: List[List[int]], 
                       superposition: np.ndarray, weights: Dict):
        """Guarda predicción con metadata cuántica"""
        
        output_dir = Path("data/predictions/quantum")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        probs = (superposition ** 2).tolist()
        
        prediction_data = {
            "metadata": {
                "system_version": "Quantum V7.0",
                "draw_date": draw_date,
                "prediction_timestamp": datetime.now().isoformat(),
                "methodology": "Quantum-Probabilistic Superposition"
            },
            "quantum_state": {
                "wave_function_norm": float(np.linalg.norm(superposition)),
                "model_weights": weights,
                "top_probabilities": {
                    str(i + self.min_num): float(probs[i]) 
                    for i in np.argsort(probs)[::-1][:20]
                }
            },
            "predictions": {
                "primary": combinations[0],
                "alternative_1": combinations[1] if len(combinations) > 1 else None,
                "alternative_2": combinations[2] if len(combinations) > 2 else None
            }
        }
        
        output_file = output_dir / f"quantum_prediction_{draw_date}.json"
        with open(output_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        print(f"\n✅ Predicción guardada: {output_file}")


def main():
    """Ejemplo de uso"""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python quantum_lottery_v7.py YYYYMMDD")
        print("Ejemplo: python quantum_lottery_v7.py 20251125")
        sys.exit(1)
    
    draw_date = sys.argv[1]
    
    # Crear sistema
    system = QuantumLotterySystem(min_num=1, max_num=56, combination_size=6)
    
    # Predecir
    combinations = system.predict(draw_date, n_combinations=3)
    
    print("\n" + "="*60)
    print("✅ PREDICCIÓN COMPLETADA")
    print("="*60)
    print("\n🎯 Combinaciones finales:")
    for i, combo in enumerate(combinations, 1):
        print(f"   {i}. {combo}")
    print()


if __name__ == "__main__":
    main()