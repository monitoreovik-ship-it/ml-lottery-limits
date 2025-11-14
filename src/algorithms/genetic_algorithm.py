"""
Algoritmo #14: Genetic Algorithm
Optimización evolutiva: selección natural sobre combinaciones de números
"""

import numpy as np
import os
from collections import Counter


class GeneticAlgorithmLottery:
    """
    Algoritmo Genético para predicción de lotería.
    
    Teoría:
    - Evoluciona población de "individuos" (combinaciones de 6 números)
    - Fitness: qué tan bien predicen sorteos históricos
    - Operadores: selección, crossover, mutación
    
    Proceso:
    1. Población inicial: 100 combinaciones aleatorias
    2. Evaluar fitness en historial
    3. Seleccionar mejores (elitismo)
    4. Crossover: combinar dos padres → hijo
    5. Mutación: cambiar números aleatoriamente
    6. Repetir 50 generaciones
    
    Hipótesis:
    - Si hay combinaciones "ganadoras", GA las encontrará
    - Esperamos: overfitting → performance histórica alta, prospectiva baja
    """
    
    def __init__(self, population_size=100, generations=50, mutation_rate=0.1):
        """
        Args:
            population_size: Tamaño de población
            generations: Número de generaciones
            mutation_rate: Probabilidad de mutación (0-1)
        """
        self.name = f"Genetic Algorithm (pop={population_size}, gen={generations})"
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.fitness_history = []
        
    def _create_individual(self):
        """
        Crea un individuo: combinación de 6 números únicos (1-56)
        """
        return sorted(np.random.choice(56, size=6, replace=False) + 1)
    
    def _initialize_population(self):
        """
        Crea población inicial aleatoria
        """
        return [self._create_individual() for _ in range(self.population_size)]
    
    def _fitness(self, individual, history):
        """
        Calcula fitness: cuántos aciertos tendría en promedio
        
        Fitness = promedio de aciertos en todos los sorteos históricos
        """
        total_matches = 0
        
        for draw in history:
            matches = len(set(individual) & set(draw['numbers']))
            total_matches += matches
        
        return total_matches / len(history)
    
    def _selection(self, population, fitnesses):
        """
        Selección por torneo: escoge 2 aleatorios, retorna el mejor
        """
        indices = np.random.choice(len(population), size=2, replace=False)
        
        if fitnesses[indices[0]] > fitnesses[indices[1]]:
            return population[indices[0]]
        else:
            return population[indices[1]]
    
    def _crossover(self, parent1, parent2):
        """
        Crossover de un punto: combina dos padres
        
        Método:
        - Tomar primeros 3 números de parent1
        - Completar con números de parent2 que no estén repetidos
        - Si faltan, agregar aleatorios
        """
        child = list(parent1[:3])  # Primeros 3 del padre 1
        
        # Agregar del padre 2 (sin repetir)
        for num in parent2:
            if num not in child and len(child) < 6:
                child.append(num)
        
        # Si faltan, completar con aleatorios
        while len(child) < 6:
            random_num = np.random.randint(1, 57)
            if random_num not in child:
                child.append(random_num)
        
        return sorted(child)
    
    def _mutate(self, individual):
        """
        Mutación: cambiar 1-2 números aleatoriamente
        """
        if np.random.rand() < self.mutation_rate:
            # Seleccionar posición a mutar
            pos = np.random.randint(0, 6)
            
            # Nuevo número (evitar repetidos)
            new_num = np.random.randint(1, 57)
            while new_num in individual:
                new_num = np.random.randint(1, 57)
            
            individual = list(individual)
            individual[pos] = new_num
            return sorted(individual)
        
        return individual
    
    def fit(self, history):
        """
        Evoluciona población durante N generaciones
        """
        if len(history) < 5:
            raise ValueError("❌ Historial insuficiente.")
        
        print(f"✅ {self.name}: Iniciando evolución...")
        print(f"   Población: {self.population_size}")
        print(f"   Generaciones: {self.generations}")
        print(f"   Tasa mutación: {self.mutation_rate}")
        
        # Inicializar población
        population = self._initialize_population()
        
        # Evolucionar
        for gen in range(self.generations):
            # Evaluar fitness
            fitnesses = [self._fitness(ind, history) for ind in population]
            
            # Guardar mejor fitness
            best_fitness = max(fitnesses)
            self.fitness_history.append(best_fitness)
            
            # Guardar mejor individuo
            best_idx = np.argmax(fitnesses)
            self.best_individual = population[best_idx].copy()
            
            # Mostrar progreso
            if (gen + 1) % 10 == 0:
                print(f"   Gen {gen+1}/{self.generations}: "
                      f"Best fitness = {best_fitness:.3f}")
            
            # Crear nueva generación
            new_population = []
            
            # Elitismo: mantener 10% mejores
            elite_count = self.population_size // 10
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generar resto por crossover + mutación
            while len(new_population) < self.population_size:
                parent1 = self._selection(population, fitnesses)
                parent2 = self._selection(population, fitnesses)
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        print(f"✅ Evolución completada")
        print(f"   Mejor fitness histórico: {max(self.fitness_history):.3f}")
        print(f"   Mejor individuo: {self.best_individual}")
        
        return self
    
    def predict(self, history):
        """
        Retorna el mejor individuo encontrado
        """
        if self.best_individual is None:
            raise ValueError("❌ Modelo no entrenado.")
        
        return self.best_individual.copy()
    
    def plot_evolution(self):
        """
        Visualiza curva de fitness a lo largo de generaciones
        """
        if not self.fitness_history:
            print("❌ No hay historial de fitness.")
            return
        
        print(f"\n📈 Evolución del Fitness:")
        print(f"   Gen 1:   {self.fitness_history[0]:.3f}")
        print(f"   Gen {len(self.fitness_history)}: {self.fitness_history[-1]:.3f}")
        print(f"   Mejora:  {self.fitness_history[-1] - self.fitness_history[0]:.3f}")
        
        # ASCII plot simplificado
        print("\n   Fitness por generación:")
        max_fit = max(self.fitness_history)
        for i in range(0, len(self.fitness_history), 10):
            fit = self.fitness_history[i]
            bar_len = int(fit / max_fit * 40)
            print(f"   {i+1:3d} |{'█' * bar_len} {fit:.3f}")


# ==================== TEST ====================
if __name__ == "__main__":
    print("🧬 ALGORITMO GENETIC - TEST")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    from src.data.collector import MelateCollector
    
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    print(f"\n📊 Dataset: {len(history)} sorteos históricos")
    
    # Test 1: Evolución
    print("\n" + "="*60)
    print("🔮 TEST 1: Evolución Genética")
    print("="*60)
    
    ga_model = GeneticAlgorithmLottery(
        population_size=50,
        generations=30,
        mutation_rate=0.15
    )
    ga_model.fit(history)
    
    # Test 2: Visualizar evolución
    ga_model.plot_evolution()
    
    # Test 3: Predicción
    print("\n" + "="*60)
    print("🔮 TEST 2: Predicción")
    print("="*60)
    
    prediction = ga_model.predict(history)
    print(f"\n🎯 Mejor combinación evolucionada: {prediction}")
    
    # Test 4: Validación
    print("\n" + "="*60)
    print("📊 VALIDACIÓN (último sorteo)")
    print("="*60)
    
    if len(history) >= 10:
        train_data = history[:-1]
        test_result = history[-1]['numbers']
        
        ga_val = GeneticAlgorithmLottery(population_size=50, generations=20)
        ga_val.fit(train_data)
        pred_val = ga_val.predict(train_data)
        
        matches = len(set(pred_val) & set(test_result))
        
        print(f"\nSorteo {history[-1]['date']}:")
        print(f"   Predicción:  {pred_val}")
        print(f"   Real:        {test_result}")
        print(f"   ✅ Aciertos: {matches}/6")
        
        # Fitness histórico vs performance real
        historical_fitness = ga_val._fitness(pred_val, train_data)
        print(f"\n   Fitness histórico: {historical_fitness:.3f}")
        print(f"   Performance real:  {matches/6:.3f}")
        print(f"   Ratio: {(matches/6) / historical_fitness:.2f}x")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    
    print("\n💡 Interpretación:")
    print("   - GA optimiza para fitness histórico")
    print("   - Encuentra combinación con mejor performance PASADA")
    print("   - Pero futuro es independiente del pasado")
    print("   - Performance real << fitness histórico (overfitting)")
    print("   - Conclusión: Evolución optimiza ruido, no señal")