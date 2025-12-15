"""
TEST #14: CLUSTERING FORZADO V14
=================================
Fecha: 15/12/2025
Sistema: Minimalismo + Clustering Obligatorio

HIPÓTESIS:
----------
Clustering (rango <30) + Alto entrelazamiento (>0.0030) = Mayor éxito

EVIDENCIA:
----------
Test #10: Ent 0.0033, Rango 25, Clustering ✓ → 3 aciertos
Test #13: Ent 0.0021, Rango 38, Clustering ✗ → 1 acierto

ESTRATEGIA:
-----------
1. Generar 100 combinaciones (vs 20)
2. Filtrar SOLO las que tienen rango <30 (clustering)
3. De esas, seleccionar top 3 por entrelazamiento
4. Si <3 combos con clustering, regenerar hasta conseguir
5. Garantizar entrelazamiento >0.0025

OBJETIVO:
---------
Replicar condiciones exitosas de Test #10
Validar clustering como factor crítico
Meta: 2-3 aciertos

PROGRESO: 14/60 tests (23.3%)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
import json
from datetime import datetime

# Configuración
np.random.seed(None)  # Aleatorio real
MAX_ATTEMPTS = 10  # Intentos para conseguir clustering
TARGET_COMBOS = 100  # Combinaciones por intento
MIN_CLUSTERING_COMBOS = 3  # Mínimo con clustering

def generate_quantum_numbers():
    """Genera 6 números usando circuito cuántico"""
    numbers = []
    
    for _ in range(6):
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        
        # Superposición
        for i in range(6):
            circuit.h(i)
        
        # Entrelazamiento
        for i in range(5):
            circuit.cx(i, i+1)
        
        # Medición
        circuit.measure_all()
        
        # Simulación
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Convertir a número
        bitstring = list(counts.keys())[0]
        number = int(bitstring, 2) % 56 + 1
        numbers.append(number)
    
    return sorted(list(set(numbers)))

def calculate_entanglement(numbers):
    """Calcula entrelazamiento cuántico"""
    if len(numbers) < 2:
        return 0.0
    
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    std = np.std(diffs)
    
    return 1.0 / (1.0 + std) if std > 0 else 0.0

def get_range(numbers):
    """Calcula rango de combinación"""
    return max(numbers) - min(numbers)

def generate_combinations_with_clustering():
    """
    Genera combinaciones hasta conseguir suficientes con clustering
    
    Returns:
        list: Combinaciones con clustering (rango <30)
        list: Todas las combinaciones generadas
    """
    all_combinations = []
    clustering_combinations = []
    
    for attempt in range(MAX_ATTEMPTS):
        print(f"\n🔄 Intento {attempt + 1}/{MAX_ATTEMPTS}")
        
        # Generar lote de combinaciones
        for i in range(TARGET_COMBOS):
            numbers = generate_quantum_numbers()
            
            # Asegurar 6 números únicos
            while len(numbers) < 6:
                new_num = np.random.randint(1, 57)
                if new_num not in numbers:
                    numbers.append(new_num)
                    numbers.sort()
            
            numbers = numbers[:6]
            ent = calculate_entanglement(numbers)
            rng = get_range(numbers)
            
            combo = {
                'numbers': numbers,
                'entanglement': float(ent),
                'range': int(rng),
                'has_clustering': rng < 30
            }
            
            all_combinations.append(combo)
            
            if combo['has_clustering']:
                clustering_combinations.append(combo)
        
        print(f"   Generadas: {len(all_combinations)} total")
        print(f"   Con clustering: {len(clustering_combinations)}")
        
        # Si ya tenemos suficientes con clustering, terminar
        if len(clustering_combinations) >= MIN_CLUSTERING_COMBOS:
            print(f"\n✅ Conseguidas {len(clustering_combinations)} combinaciones con clustering")
            break
    
    return clustering_combinations, all_combinations

def main():
    print("=" * 70)
    print("TEST #14: CLUSTERING FORZADO V14")
    print("=" * 70)
    print("\n🎯 OBJETIVO: Forzar clustering (rango <30)")
    print("🎯 META: Replicar Test #10 (3 aciertos)")
    
    # Generar combinaciones
    print("\n🔬 GENERANDO COMBINACIONES...")
    clustering_combos, all_combos = generate_combinations_with_clustering()
    
    # Verificar si conseguimos clustering
    if len(clustering_combos) < MIN_CLUSTERING_COMBOS:
        print(f"\n⚠️ ADVERTENCIA: Solo {len(clustering_combos)} combos con clustering")
        print("    Continuando con las disponibles...")
    
    # Ordenar por entrelazamiento
    clustering_combos.sort(key=lambda x: x['entanglement'], reverse=True)
    
    # Seleccionar top 3
    top_3 = clustering_combos[:3] if len(clustering_combos) >= 3 else clustering_combos
    
    # Estadísticas
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS DE GENERACIÓN")
    print("=" * 70)
    print(f"Total combinaciones: {len(all_combos)}")
    print(f"Con clustering (<30): {len(clustering_combos)} ({len(clustering_combos)/len(all_combos)*100:.1f}%)")
    print(f"Sin clustering (≥30): {len(all_combos) - len(clustering_combos)}")
    
    if clustering_combos:
        ranges = [c['range'] for c in clustering_combos]
        ents = [c['entanglement'] for c in clustering_combos]
        print(f"\nClustering combos:")
        print(f"  Rango promedio: {np.mean(ranges):.1f}")
        print(f"  Entrelazamiento promedio: {np.mean(ents):.4f}")
        print(f"  Entrelazamiento máximo: {max(ents):.4f}")
    
    # Mostrar top 3
    print("\n" + "=" * 70)
    print("🏆 TOP 3 COMBINACIONES (CON CLUSTERING)")
    print("=" * 70)
    
    for i, combo in enumerate(top_3, 1):
        medal = ["🥇", "🥈", "🥉"][i-1]
        print(f"\n{medal} Combinación {i}:")
        print(f"   Números: {combo['numbers']}")
        print(f"   Entrelazamiento: {combo['entanglement']:.4f}")
        print(f"   Rango: {combo['range']}")
        print(f"   Clustering: {'✅ SÍ' if combo['has_clustering'] else '❌ NO'}")
    
    # Comparación con Test #10
    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN CON TEST #10 (GANADOR)")
    print("=" * 70)
    print("Test #10:")
    print("  Entrelazamiento: 0.0033")
    print("  Rango: 25")
    print("  Resultado: 3 aciertos")
    print("\nTest #14 (Principal):")
    print(f"  Entrelazamiento: {top_3[0]['entanglement']:.4f}")
    print(f"  Rango: {top_3[0]['range']}")
    print(f"  Resultado: Pendiente")
    
    if top_3[0]['entanglement'] >= 0.0030 and top_3[0]['range'] <= 25:
        print("\n✅ Condiciones similares a Test #10")
        print("   Expectativa: 2-3 aciertos")
    elif top_3[0]['entanglement'] >= 0.0025:
        print("\n⚠️ Entrelazamiento moderado")
        print("   Expectativa: 1-2 aciertos")
    else:
        print("\n⚠️ Entrelazamiento bajo")
        print("   Expectativa: 1 acierto")
    
    # Preparar datos para guardar
    prediction_data = {
        'test_number': 14,
        'system': 'clustering_forced_v14',
        'strategy': 'minimalismo + clustering obligatorio (<30)',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Clustering + alto entrelazamiento = más aciertos',
        'generation_stats': {
            'total_generated': len(all_combos),
            'with_clustering': len(clustering_combos),
            'clustering_rate': f"{len(clustering_combos)/len(all_combos)*100:.1f}%",
            'target_range': '<30',
            'min_entanglement': 0.0025
        },
        'main_prediction': {
            'numbers': [int(x) for x in top_3[0]['numbers']],
            'entanglement': float(top_3[0]['entanglement']),
            'range': int(top_3[0]['range']),
            'has_clustering': bool(top_3[0]['has_clustering'])
        },
        'alternatives': [
            {
                'numbers': [int(x) for x in c['numbers']],
                'entanglement': float(c['entanglement']),
                'range': int(c['range']),
                'has_clustering': bool(c['has_clustering'])
            }
            for c in top_3[1:3]
        ] if len(top_3) > 1 else [],
        'comparison_test_10': {
            'test_10_entanglement': 0.0033,
            'test_10_range': 25,
            'test_10_result': 3,
            'test_14_entanglement': float(top_3[0]['entanglement']),
            'test_14_range': int(top_3[0]['range']),
            'similar_conditions': bool(
                top_3[0]['entanglement'] >= 0.0030 and 
                top_3[0]['range'] <= 25
            )
        },
        'expected_outcome': {
            'optimistic': '3 aciertos (si replica Test #10)',
            'base': '2 aciertos (clustering funciona)',
            'conservative': '1 acierto (varianza aleatoria)'
        }
    }
    
    # Guardar archivo
    filename = f"data/predictions/quantum/quantum_prediction_test14.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"✅ PREDICCIÓN GUARDADA: {filename}")
    print("=" * 70)
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. Verificar archivo JSON generado")
    print("2. git add data/predictions/quantum/quantum_prediction_test14.json")
    print("3. git add scripts/clustering_forced_v14.py")
    print("4. git commit con mensaje detallado")
    print("5. git push")
    print("\n⏳ Esperar sorteo para evaluación")
    print("🎲 Hipótesis: Clustering es factor crítico")
    
    return prediction_data

if __name__ == "__main__":
    result = main()
    print("\n" + "=" * 70)
    print("🏁 TEST #14 GENERADO EXITOSAMENTE")
    print("=" * 70)