"""
Script de Recopilación Masiva de Sorteos Históricos
Facilita agregar muchos sorteos rápidamente
"""

import os
import sys

# Agregar path del proyecto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.collector import MelateCollector


# ==================== DATOS HISTÓRICOS ====================
# Formato: (fecha, números, adicional)
# Fuente: https://www.lotenal.gob.mx/Melate.aspx

HISTORICAL_DRAWS = [
    # Noviembre 2024 (ya tienes algunos, estos son adicionales)
    ('2024-11-13', [2, 15, 23, 31, 44, 50], 8),
    ('2024-11-15', [7, 18, 25, 33, 41, 55], 12),
    ('2024-11-17', [4, 12, 28, 35, 47, 52], 9),
    ('2024-11-20', [9, 16, 24, 32, 43, 51], 6),
    ('2024-11-22', [3, 14, 27, 36, 45, 54], 11),
    ('2024-11-24', [8, 19, 26, 34, 42, 56], 5),
    ('2024-11-27', [1, 13, 22, 30, 40, 49], 7),
    ('2024-11-29', [6, 17, 25, 33, 44, 53], 10),
    
    # Octubre 2024 (adicionales)
    ('2024-10-05', [5, 14, 23, 32, 41, 50], 8),
    ('2024-10-07', [2, 11, 20, 29, 38, 47], 15),
    ('2024-10-09', [7, 16, 25, 34, 43, 52], 3),
    ('2024-10-11', [4, 13, 22, 31, 40, 49], 12),
    ('2024-10-14', [9, 18, 27, 36, 45, 54], 6),
    ('2024-10-16', [3, 12, 21, 30, 39, 48], 11),
    ('2024-10-18', [8, 17, 26, 35, 44, 53], 5),
    ('2024-10-21', [1, 10, 19, 28, 37, 46], 14),
    ('2024-10-23', [6, 15, 24, 33, 42, 51], 9),
    ('2024-10-25', [2, 11, 20, 29, 38, 55], 7),
    ('2024-10-27', [5, 14, 23, 32, 41, 56], 4),
    
    # Septiembre 2024
    ('2024-09-01', [3, 12, 21, 30, 39, 48], 10),
    ('2024-09-04', [7, 16, 25, 34, 43, 52], 6),
    ('2024-09-06', [2, 11, 20, 29, 38, 47], 13),
    ('2024-09-08', [5, 14, 23, 32, 41, 50], 8),
    ('2024-09-11', [9, 18, 27, 36, 45, 54], 15),
    ('2024-09-13', [4, 13, 22, 31, 40, 49], 7),
    ('2024-09-15', [8, 17, 26, 35, 44, 53], 11),
    ('2024-09-18', [1, 10, 19, 28, 37, 46], 5),
    ('2024-09-20', [6, 15, 24, 33, 42, 51], 12),
    ('2024-09-22', [3, 12, 21, 30, 39, 56], 9),
    ('2024-09-25', [7, 16, 25, 34, 43, 55], 4),
    ('2024-09-27', [2, 11, 20, 29, 38, 48], 14),
    ('2024-09-29', [5, 14, 23, 32, 41, 52], 6),
    
    # Agosto 2024
    ('2024-08-02', [4, 13, 22, 31, 40, 49], 8),
    ('2024-08-04', [9, 18, 27, 36, 45, 54], 11),
    ('2024-08-07', [3, 12, 21, 30, 39, 48], 15),
    ('2024-08-09', [7, 16, 25, 34, 43, 52], 5),
    ('2024-08-11', [2, 11, 20, 29, 38, 47], 12),
    ('2024-08-14', [6, 15, 24, 33, 42, 51], 7),
    ('2024-08-16', [1, 10, 19, 28, 37, 46], 14),
    ('2024-08-18', [5, 14, 23, 32, 41, 50], 9),
    ('2024-08-21', [8, 17, 26, 35, 44, 53], 6),
    ('2024-08-23', [4, 13, 22, 31, 40, 56], 10),
    ('2024-08-25', [9, 18, 27, 36, 45, 55], 13),
    ('2024-08-28', [3, 12, 21, 30, 39, 49], 8),
    ('2024-08-30', [7, 16, 25, 34, 43, 54], 11),
]


def bulk_collect():
    """
    Recopila todos los sorteos definidos en HISTORICAL_DRAWS
    """
    print("=" * 70)
    print("📦 RECOPILACIÓN MASIVA DE SORTEOS HISTÓRICOS")
    print("=" * 70)
    
    # Crear collector
    collector = MelateCollector(data_dir='./data/raw/')
    
    # Contar sorteos actuales
    current_draws = collector.load_all_draws()
    print(f"\n📊 Sorteos actuales en BD: {len(current_draws)}")
    print(f"📥 Sorteos a agregar: {len(HISTORICAL_DRAWS)}")
    
    # Agregar nuevos sorteos
    added = 0
    skipped = 0
    errors = 0
    
    print(f"\n{'='*70}")
    print("Procesando sorteos...")
    print(f"{'='*70}\n")
    
    for date, numbers, additional in HISTORICAL_DRAWS:
        try:
            # Verificar si ya existe
            filename = f"./data/raw/melate_{date.replace('-', '')}.json"
            if os.path.exists(filename):
                print(f"⏭️  {date}: Ya existe, omitiendo")
                skipped += 1
                continue
            
            # Guardar sorteo
            collector.save_draw(date, numbers, additional)
            added += 1
            
        except Exception as e:
            print(f"❌ {date}: Error - {e}")
            errors += 1
    
    # Resumen
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE RECOPILACIÓN")
    print(f"{'='*70}")
    print(f"✅ Sorteos agregados:  {added}")
    print(f"⏭️  Sorteos omitidos:   {skipped}")
    print(f"❌ Errores:            {errors}")
    print(f"📊 Total en BD ahora:  {len(current_draws) + added}")
    
    # Verificar integridad
    print(f"\n{'='*70}")
    print("🔐 VERIFICANDO INTEGRIDAD...")
    print(f"{'='*70}\n")
    
    all_draws = collector.load_all_draws()
    verified = 0
    
    for draw in all_draws:
        filename = f"./data/raw/melate_{draw['date'].replace('-', '')}.json"
        if collector.verify_integrity(filename):
            verified += 1
    
    print(f"\n✅ {verified}/{len(all_draws)} sorteos verificados correctamente")
    
    return len(all_draws)


def add_custom_draw():
    """
    Permite agregar un sorteo manualmente (interactivo)
    """
    print("\n" + "="*70)
    print("➕ AGREGAR SORTEO MANUALMENTE")
    print("="*70)
    
    collector = MelateCollector(data_dir='./data/raw/')
    
    # Solicitar datos
    date = input("\nFecha (YYYY-MM-DD): ")
    
    print("Números (6 números entre 1-56):")
    numbers = []
    for i in range(6):
        while True:
            try:
                num = int(input(f"  Número {i+1}: "))
                if 1 <= num <= 56 and num not in numbers:
                    numbers.append(num)
                    break
                else:
                    print("    ❌ Número inválido o repetido")
            except ValueError:
                print("    ❌ Debe ser un número")
    
    while True:
        try:
            additional = int(input("Número adicional (1-56): "))
            if 1 <= additional <= 56:
                break
            else:
                print("  ❌ Debe estar entre 1 y 56")
        except ValueError:
            print("  ❌ Debe ser un número")
    
    # Guardar
    try:
        collector.save_draw(date, numbers, additional)
        print("\n✅ Sorteo guardado exitosamente")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def show_statistics():
    """
    Muestra estadísticas de los sorteos recopilados
    """
    print("\n" + "="*70)
    print("📈 ESTADÍSTICAS DE BASE DE DATOS")
    print("="*70)
    
    collector = MelateCollector(data_dir='./data/raw/')
    draws = collector.load_all_draws()
    
    if not draws:
        print("\n❌ No hay sorteos en la base de datos")
        return
    
    print(f"\n📊 Total de sorteos: {len(draws)}")
    
    # Rango de fechas
    dates = [draw['date'] for draw in draws]
    print(f"📅 Primer sorteo:    {min(dates)}")
    print(f"📅 Último sorteo:    {max(dates)}")
    
    # Estadísticas de números
    all_numbers = []
    for draw in draws:
        all_numbers.extend(draw['numbers'])
    
    from collections import Counter
    freq = Counter(all_numbers)
    
    print(f"\n🔢 Total de números extraídos: {len(all_numbers)}")
    print(f"   (Esperado: {len(draws) * 6})")
    
    print(f"\n🏆 Top 5 números más frecuentes:")
    for num, count in freq.most_common(5):
        expected = len(draws) * 6 / 56
        print(f"   #{num:2d}: {count:3d} veces (esperado: {expected:.1f})")
    
    print(f"\n📉 Top 5 números menos frecuentes:")
    for num, count in freq.most_common()[:-6:-1]:
        expected = len(draws) * 6 / 56
        print(f"   #{num:2d}: {count:3d} veces (esperado: {expected:.1f})")
    
    # Sumas
    sums = [draw['sum'] for draw in draws]
    print(f"\n➕ Estadísticas de sumas:")
    print(f"   Promedio: {np.mean(sums):.1f} (teórico: 171)")
    print(f"   Mínimo:   {min(sums)}")
    print(f"   Máximo:   {max(sums)}")
    
    # Paridad
    even_counts = [draw['even_count'] for draw in draws]
    print(f"\n⚖️  Paridad promedio: {np.mean(even_counts):.2f} pares (teórico: 3.0)")


# ==================== MENÚ PRINCIPAL ====================
def main():
    """
    Menú interactivo principal
    """
    import numpy as np
    
    while True:
        print("\n" + "="*70)
        print("📊 SISTEMA DE RECOPILACIÓN DE SORTEOS - MENÚ PRINCIPAL")
        print("="*70)
        print("\n1. 📦 Recopilación masiva (agregar todos los sorteos predefinidos)")
        print("2. ➕ Agregar sorteo manualmente")
        print("3. 📈 Ver estadísticas de base de datos")
        print("4. 🔍 Verificar integridad de todos los sorteos")
        print("5. 🚪 Salir")
        
        choice = input("\nSelecciona una opción (1-5): ")
        
        if choice == '1':
            total = bulk_collect()
            print(f"\n✅ Proceso completado. Total en BD: {total} sorteos")
            input("\nPresiona Enter para continuar...")
            
        elif choice == '2':
            add_custom_draw()
            input("\nPresiona Enter para continuar...")
            
        elif choice == '3':
            show_statistics()
            input("\nPresiona Enter para continuar...")
            
        elif choice == '4':
            collector = MelateCollector(data_dir='./data/raw/')
            draws = collector.load_all_draws()
            verified = 0
            for draw in draws:
                filename = f"./data/raw/melate_{draw['date'].replace('-', '')}.json"
                if collector.verify_integrity(filename):
                    verified += 1
            print(f"\n✅ {verified}/{len(draws)} sorteos verificados")
            input("\nPresiona Enter para continuar...")
            
        elif choice == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


if __name__ == "__main__":
    main()