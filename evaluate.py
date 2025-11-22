#!/usr/bin/env python3
"""
Script de Evaluación de Predicciones Prospectivas
Evalúa predicciones bloqueadas contra resultados oficiales de sorteos
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Set

def load_locked_prediction(draw_date: str) -> Dict:
    """Carga archivo de predicción bloqueado"""
    filename = f"data/predictions/locked/prediction_{draw_date}_locked.json"
    
    if not os.path.exists(filename):
        print(f"❌ ERROR: Archivo no encontrado: {filename}")
        print(f"\nArchivos disponibles:")
        locked_dir = "data/predictions/locked/"
        if os.path.exists(locked_dir):
            files = [f for f in os.listdir(locked_dir) if f.endswith('_locked.json')]
            for f in files:
                print(f"  - {f}")
        sys.exit(1)
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_official_numbers() -> List[int]:
    """Solicita los números oficiales del sorteo"""
    print("\n" + "="*60)
    print("📋 INGRESO DE NÚMEROS OFICIALES DEL SORTEO")
    print("="*60)
    
    while True:
        try:
            numbers_input = input("\nIngrese los 6 números ganadores (separados por comas o espacios):\n> ")
            
            # Limpiar y separar
            numbers_str = numbers_input.replace(',', ' ').split()
            numbers = [int(n.strip()) for n in numbers_str if n.strip()]
            
            # Validar
            if len(numbers) != 6:
                print(f"❌ Error: Debe ingresar exactamente 6 números (ingresó {len(numbers)})")
                continue
            
            if any(n < 1 or n > 56 for n in numbers):
                print("❌ Error: Los números deben estar entre 1 y 56")
                continue
            
            if len(set(numbers)) != 6:
                print("❌ Error: Los números no deben repetirse")
                continue
            
            # Confirmar
            print(f"\n✅ Números ingresados: {sorted(numbers)}")
            confirm = input("¿Son correctos? (s/n): ").lower()
            
            if confirm == 's':
                return sorted(numbers)
            
        except ValueError:
            print("❌ Error: Ingrese solo números válidos")
        except KeyboardInterrupt:
            print("\n\n❌ Evaluación cancelada por el usuario")
            sys.exit(0)

def calculate_matches(prediction: List[int], official: List[int]) -> Dict:
    """Calcula aciertos entre predicción y números oficiales"""
    pred_set = set(prediction)
    official_set = set(official)
    matches = pred_set & official_set
    
    return {
        "matches_count": len(matches),
        "matched_numbers": sorted(list(matches)),
        "prediction": sorted(prediction),
        "official": sorted(official)
    }

def evaluate_all_algorithms(predictions: Dict, official_numbers: List[int]) -> Dict:
    """Evalúa todos los algoritmos"""
    results = {}
    
    for algo_name, algo_data in predictions.items():
        if isinstance(algo_data, dict) and 'numbers' in algo_data:
            prediction = algo_data['numbers']
            match_data = calculate_matches(prediction, official_numbers)
            
            results[algo_name] = {
                "prediction": match_data["prediction"],
                "matches": match_data["matches_count"],
                "matched_numbers": match_data["matched_numbers"],
                "algorithm_type": algo_data.get("algorithm_type", "Unknown")
            }
    
    return results

def print_results(results: Dict, official_numbers: List[int]):
    """Imprime resultados formateados"""
    print("\n" + "="*60)
    print("🎯 RESULTADOS DE EVALUACIÓN")
    print("="*60)
    
    print(f"\n📊 Números oficiales: {official_numbers}")
    print(f"\n{'Algoritmo':<25} {'Aciertos':<10} {'Números Acertados'}")
    print("-"*60)
    
    # Ordenar por número de aciertos (descendente)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['matches'], reverse=True)
    
    for algo_name, data in sorted_results:
        matches = data['matches']
        matched = data['matched_numbers']
        
        # Formato visual
        check = "✅" * matches if matches > 0 else "❌"
        matched_str = str(matched) if matches > 0 else "-"
        
        print(f"{algo_name:<25} {matches:<10} {matched_str} {check}")
    
    # Estadísticas
    print("\n" + "="*60)
    print("📈 ESTADÍSTICAS")
    print("="*60)
    
    all_matches = [r['matches'] for r in results.values()]
    avg_matches = sum(all_matches) / len(all_matches) if all_matches else 0
    max_matches = max(all_matches) if all_matches else 0
    min_matches = min(all_matches) if all_matches else 0
    
    baseline_expected = 6 * 6 / 56  # ~0.64
    
    print(f"\n📊 Promedio de aciertos: {avg_matches:.2f}")
    print(f"📊 Máximo de aciertos: {max_matches}")
    print(f"📊 Mínimo de aciertos: {min_matches}")
    print(f"📊 Baseline esperado: {baseline_expected:.2f}")
    
    if avg_matches > 0:
        improvement = (avg_matches / baseline_expected - 1) * 100
        print(f"📊 Mejora sobre baseline: {improvement:+.1f}%")
    
    # Mejores algoritmos
    best_algos = [name for name, data in sorted_results if data['matches'] == max_matches]
    print(f"\n🏆 Mejor(es) algoritmo(s): {', '.join(best_algos)} ({max_matches} aciertos)")
    
    # Peores algoritmos
    worst_algos = [name for name, data in sorted_results if data['matches'] == min_matches]
    if min_matches == 0:
        print(f"⚠️  Algoritmos sin aciertos: {len(worst_algos)}")

def save_results(draw_date: str, metadata: Dict, results: Dict, official_numbers: List[int]):
    """Guarda resultados de evaluación"""
    
    # Crear directorio si no existe
    eval_dir = "data/predictions/evaluated/"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Preparar datos
    output_data = {
        "metadata": metadata,
        "official_numbers": official_numbers,
        "evaluation_timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "average_matches": sum(r['matches'] for r in results.values()) / len(results),
            "max_matches": max(r['matches'] for r in results.values()),
            "min_matches": min(r['matches'] for r in results.values()),
            "baseline_expected": 6 * 6 / 56
        }
    }
    
    # Guardar archivo
    output_file = f"{eval_dir}prediction_{draw_date}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Resultados guardados en: {output_file}")
    
    return output_file

def main():
    """Función principal"""
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("❌ Uso: python evaluate.py YYYYMMDD")
        print("   Ejemplo: python evaluate.py 20251119")
        sys.exit(1)
    
    draw_date = sys.argv[1]
    
    # Validar formato de fecha
    try:
        datetime.strptime(draw_date, "%Y%m%d")
    except ValueError:
        print(f"❌ Error: Fecha inválida '{draw_date}'. Use formato YYYYMMDD")
        sys.exit(1)
    
    print("="*60)
    print("🎯 EVALUACIÓN DE PREDICCIÓN PROSPECTIVA")
    print("="*60)
    print(f"\n📅 Fecha del sorteo: {draw_date}")
    
    # Cargar predicción bloqueada
    print(f"\n📂 Cargando predicción bloqueada...")
    prediction_data = load_locked_prediction(draw_date)
    
    # Verificar hash
    if 'lock' in prediction_data and 'sha256' in prediction_data['lock']:
        print(f"🔐 Hash SHA-256 verificado: {prediction_data['lock']['sha256'][:12]}...")
        print(f"🔐 Bloqueado en: {prediction_data['lock']['locked_at']}")
    
    # Obtener números oficiales
    official_numbers = get_official_numbers()
    
    # Evaluar todos los algoritmos
    print(f"\n⚙️  Evaluando {len(prediction_data['predictions'])} algoritmos...")
    results = evaluate_all_algorithms(prediction_data['predictions'], official_numbers)
    
    # Mostrar resultados
    print_results(results, official_numbers)
    
    # Guardar resultados
    output_file = save_results(
        draw_date,
        prediction_data['metadata'],
        results,
        official_numbers
    )
    
    print("\n" + "="*60)
    print("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
    print("="*60)
    print(f"\n📋 Próximos pasos:")
    print(f"   1. Verificar archivo: {output_file}")
    print(f"   2. git add {output_file}")
    print(f"   3. git commit -m \"results: evaluación test sorteo {draw_date}\"")
    print(f"   4. git push")
    print(f"   5. Actualizar PROGRESS.md con los resultados")
    print()

if __name__ == "__main__":
    main()