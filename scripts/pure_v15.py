"""
TEST #15: MINIMALISMO PURO V15 - HITO 25%
==========================================
Fecha: 17/12/2025
Sistema: Minimalismo Puro (Retorno a lo Básico)

CONTEXTO - LECCIÓN DEL TEST #14:
---------------------------------
Test #14: Forzar clustering FALLÓ
- Predicción: Rango 28 (clustering)
- Sorteo real: Rango 48 (dispersión máxima)
- Resultado: 1 acierto (estrategia contraproducente)

CONCLUSIÓN: Forzar patrones es ERROR. Volver a minimalismo puro.

FILOSOFÍA TEST #15:
-------------------
"Dejemos que el sistema genere naturalmente.
 No forzar clustering, no forzar números, no agregar complejidad.
 Confiar en la generación cuántica pura."

ESTRATEGIA:
-----------
✅ Generación cuántica pura (NumPy simulación)
✅ Selección SOLO por entrelazamiento
✅ CERO restricciones de rango
✅ CERO números forzados
✅ 20 combinaciones generadas
✅ Top 3 por mayor entrelazamiento

HITO 25%:
---------
Este es el test #15 de 60 (25% completado)
Punto de evaluación estratégica intermedia
Promedio con n=15 es más confiable que n=5

PROGRESO: 15/60 tests (25.0%)
"""

import numpy as np
import json
from datetime import datetime

# Configuración
np.random.seed(None)  # Aleatorio real
TARGET_COMBOS = 20  # Igual que Tests #10, #11, #13 exitosos

def generate_quantum_like_numbers():
    """
    Genera 6 números simulando comportamiento cuántico con NumPy
    
    Simula:
    - Superposición: Distribución uniforme
    - Entrelazamiento: Correlación entre números
    """
    numbers = []
    
    # Primer número: completamente aleatorio
    numbers.append(np.random.randint(1, 57))
    
    # Números siguientes: con "entrelazamiento simulado"
    for _ in range(5):
        # 70% probabilidad de estar cerca del último número (entrelazamiento)
        # 30% probabilidad de ser completamente aleatorio
        if np.random.random() < 0.7 and len(numbers) > 0:
            # Cerca del último número (±15)
            base = numbers[-1]
            offset = np.random.randint(-15, 16)
            number = np.clip(base + offset, 1, 56)
        else:
            # Completamente aleatorio
            number = np.random.randint(1, 57)
        
        # Evitar duplicados
        attempts = 0
        while number in numbers and attempts < 50:
            number = np.random.randint(1, 57)
            attempts += 1
        
        numbers.append(number)
    
    # Asegurar únicos y ordenados
    numbers = sorted(list(set(numbers)))
    
    # Completar si faltan números
    while len(numbers) < 6:
        new_num = np.random.randint(1, 57)
        if new_num not in numbers:
            numbers.append(new_num)
    
    return sorted(numbers[:6])

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

def generate_combinations():
    """Genera combinaciones sin restricciones"""
    combinations = []
    
    print(f"🔬 Generando {TARGET_COMBOS} combinaciones...")
    
    for i in range(TARGET_COMBOS):
        numbers = generate_quantum_like_numbers()
        
        ent = calculate_entanglement(numbers)
        rng = get_range(numbers)
        
        combo = {
            'numbers': numbers,
            'entanglement': float(ent),
            'range': int(rng),
            'has_clustering': rng < 30
        }
        
        combinations.append(combo)
        
        if (i + 1) % 5 == 0:
            print(f"   Progreso: {i + 1}/{TARGET_COMBOS}")
    
    print(f"✅ {len(combinations)} combinaciones generadas")
    
    return combinations

def main():
    print("=" * 70)
    print("TEST #15: MINIMALISMO PURO V15 - HITO 25%")
    print("=" * 70)
    print("\n🎯 OBJETIVO: Generación natural sin restricciones")
    print("🎯 HITO: 25% del experimento completado (15/60)")
    print("🎯 LECCIÓN #14: NO forzar clustering ni patrones")
    
    # Generar combinaciones
    print("\n🔬 GENERANDO COMBINACIONES...")
    all_combos = generate_combinations()
    
    # Ordenar por entrelazamiento
    all_combos.sort(key=lambda x: x['entanglement'], reverse=True)
    
    # Seleccionar top 3
    top_3 = all_combos[:3]
    
    # Estadísticas generales
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS DE GENERACIÓN")
    print("=" * 70)
    print(f"Total combinaciones: {len(all_combos)}")
    
    clustering_count = len([c for c in all_combos if c['has_clustering']])
    print(f"Con clustering natural (<30): {clustering_count} ({clustering_count/len(all_combos)*100:.1f}%)")
    print(f"Sin clustering (≥30): {len(all_combos) - clustering_count}")
    
    ranges = [c['range'] for c in all_combos]
    ents = [c['entanglement'] for c in all_combos]
    print(f"\nTodas las combinaciones:")
    print(f"  Rango promedio: {np.mean(ranges):.1f}")
    print(f"  Rango mínimo: {min(ranges)}")
    print(f"  Rango máximo: {max(ranges)}")
    print(f"  Entrelazamiento promedio: {np.mean(ents):.4f}")
    print(f"  Entrelazamiento máximo: {max(ents):.4f}")
    
    # Mostrar top 3
    print("\n" + "=" * 70)
    print("🏆 TOP 3 COMBINACIONES")
    print("=" * 70)
    
    for i, combo in enumerate(top_3, 1):
        medal = ["🥇", "🥈", "🥉"][i-1]
        clustering_icon = "✅ SÍ" if combo['has_clustering'] else "❌ NO"
        print(f"\n{medal} Combinación {i}:")
        print(f"   Números: {combo['numbers']}")
        print(f"   Entrelazamiento: {combo['entanglement']:.4f}")
        print(f"   Rango: {combo['range']}")
        print(f"   Clustering: {clustering_icon}")
    
    # Comparación con tests anteriores
    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN CON TESTS ANTERIORES")
    print("=" * 70)
    print("\nTests Minimalistas (natural generation):")
    print("  Test #10: Ent 0.0033, Rng 25, Clustering ✅ → 3 aciertos")
    print("  Test #11: Ent 0.0025, Rng 45, Clustering ❌ → 1 acierto")
    print("  Test #13: Ent 0.0021, Rng 38, Clustering ❌ → 1 acierto")
    print("\nTest #14 (clustering forzado - FALLÓ):")
    print("  Predicción: Rng 28, Clustering ✅")
    print("  Sorteo real: Rng 48, Clustering ❌ → Opuestos!")
    print("  Resultado: 1 acierto")
    print("\nTest #15 (este test):")
    print(f"  Entrelazamiento: {top_3[0]['entanglement']:.4f}")
    print(f"  Rango: {top_3[0]['range']}")
    print(f"  Clustering: {'✅' if top_3[0]['has_clustering'] else '❌'}")
    print("  Estrategia: NATURAL (sin forzar)")
    
    # Análisis de expectativas
    print("\n" + "=" * 70)
    print("🎯 EXPECTATIVA Y ANÁLISIS")
    print("=" * 70)
    
    if top_3[0]['has_clustering']:
        print("✅ Clustering NATURAL presente")
        print("   Si sorteo también tiene clustering → 2-3 aciertos posibles")
        print("   Si sorteo es disperso → 1 acierto probable")
    else:
        print("❌ Sin clustering (generación natural)")
        print("   Si sorteo también es disperso → 1-2 aciertos probable")
        print("   Si sorteo tiene clustering → 1 acierto probable")
    
    print("\nPromedio esperado: 1.5-2.0 aciertos")
    print("Razón: Minimalismo puro promedio actual 1.60")
    
    # Hito 25%
    print("\n" + "=" * 70)
    print("🎉 HITO 25% - EVALUACIÓN INTERMEDIA")
    print("=" * 70)
    print("\nDespués de este test tendremos:")
    print("  • 15/60 tests completados (25%)")
    print("  • n=15 para promedio más confiable")
    print("  • Base estadística para decisión estratégica")
    print("\nDecisión post-Test #15:")
    print("  • Si promedio ≥1.5: Continuar minimalismo puro")
    print("  • Si promedio <1.5: Investigar otros factores")
    print("  • Análisis de varianza para entender fluctuaciones")
    
    # Preparar datos para guardar
    prediction_data = {
        'test_number': 15,
        'system': 'pure_minimalism_v15',
        'strategy': 'minimalismo puro - generación natural sin restricciones',
        'milestone': '25% (15/60 tests)',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'philosophy': 'Return to basics after Test #14 clustering forcing failed',
        'lesson_from_test14': 'Forcing patterns is counterproductive. Trust natural generation.',
        'generation_stats': {
            'total_generated': len(all_combos),
            'with_clustering': clustering_count,
            'natural_clustering_rate': f"{clustering_count/len(all_combos)*100:.1f}%",
            'avg_range': float(np.mean(ranges)),
            'min_range': int(min(ranges)),
            'max_range': int(max(ranges)),
            'avg_entanglement': float(np.mean(ents))
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
        ],
        'comparison_previous': {
            'test_10_natural': {'ent': 0.0033, 'range': 25, 'clustering': True, 'result': 3},
            'test_11_natural': {'ent': 0.0025, 'range': 45, 'clustering': False, 'result': 1},
            'test_13_natural': {'ent': 0.0021, 'range': 38, 'clustering': False, 'result': 1},
            'test_14_forced': {'ent': 0.2927, 'range': 28, 'clustering': True, 'result': 1, 'note': 'FAILED - forced clustering'},
            'test_15_natural': {'ent': float(top_3[0]['entanglement']), 'range': int(top_3[0]['range']), 'clustering': bool(top_3[0]['has_clustering']), 'note': 'Return to natural generation'}
        },
        'expected_outcome': {
            'optimistic': '2-3 aciertos',
            'base': '1-2 aciertos',
            'conservative': '1 acierto',
            'reasoning': 'Pure minimalism average is 1.60, expect similar performance'
        },
        'milestone_significance': {
            'progress': '25% (15/60)',
            'sample_size': 15,
            'reliability': 'moderate (n=15 more reliable than n=5)',
            'strategic_decision_point': True,
            'next_phase': 'Analyze 15-test results to decide strategy for tests 16-30'
        }
    }
    
    # Guardar archivo
    filename = f"data/predictions/quantum/quantum_prediction_test15.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"✅ PREDICCIÓN GUARDADA: {filename}")
    print("=" * 70)
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. Verificar archivo JSON generado")
    print("2. git add data/predictions/quantum/quantum_prediction_test15.json")
    print("3. git add scripts/pure_v15.py")
    print("4. git commit con mensaje detallado")
    print("5. git push")
    print("\n⏳ Esperar sorteo para evaluación")
    print("🎉 Luego: Análisis del HITO 25% con 15 tests")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📋 RESUMEN TEST #15")
    print("=" * 70)
    print(f"🥇 Predicción: {top_3[0]['numbers']}")
    print(f"📊 Entrelazamiento: {top_3[0]['entanglement']:.4f}")
    print(f"📏 Rango: {top_3[0]['range']}")
    print(f"🎯 Clustering: {'✅ SÍ (natural)' if top_3[0]['has_clustering'] else '❌ NO (natural)'}")
    print(f"🎉 HITO: 25% completado tras este test")
    
    return prediction_data

if __name__ == "__main__":
    result = main()
    print("\n" + "=" * 70)
    print("🏁 TEST #15 GENERADO EXITOSAMENTE")
    print("🎉 HITO 25% ALCANZADO")
    print("=" * 70)