import json
from datetime import datetime
import hashlib
import os

class MelateCollector:
    """
    Recolector de sorteos Melate con verificación de integridad
    """
    
    def __init__(self, data_dir='./data/raw/'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        print(f"✅ Directorio de datos: {data_dir}")
    
    def save_draw(self, date_str, numbers, additional):
        """
        Guarda un sorteo con timestamp criptográfico
        
        Args:
            date_str: Fecha en formato 'YYYY-MM-DD'
            numbers: Lista de 6 números [int]
            additional: Número adicional [int]
        """
        # Validaciones
        if len(numbers) != 6:
            raise ValueError("❌ Deben ser exactamente 6 números")
        
        if not all(1 <= n <= 56 for n in numbers):
            raise ValueError("❌ Números deben estar entre 1 y 56")
        
        if not (1 <= additional <= 56):
            raise ValueError("❌ Adicional debe estar entre 1 y 56")
        
        # Crear registro
        record = {
            'date': date_str,
            'numbers': sorted(numbers),
            'additional': additional,
            'timestamp_collected': datetime.now().isoformat(),
            'source': 'manual_entry',
            # Metadatos
            'sum': sum(numbers),
            'even_count': sum(1 for n in numbers if n % 2 == 0),
            'odd_count': sum(1 for n in numbers if n % 2 == 1),
            'range': f"{min(numbers)}-{max(numbers)}"
        }
        
        # Hash SHA-256 para integridad
        record_copy = record.copy()
        record_str = json.dumps(record_copy, sort_keys=True)
        record['hash'] = hashlib.sha256(record_str.encode()).hexdigest()
        
        # Guardar archivo
        filename = os.path.join(self.data_dir, f"melate_{date_str.replace('-', '')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Sorteo guardado: {filename}")
        print(f"   Números: {sorted(numbers)}")
        print(f"   Adicional: {additional}")
        print(f"   Hash: {record['hash'][:16]}...")
        
        return filename
    
    def verify_integrity(self, filename):
        """
        Verifica que el archivo no fue modificado
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            stored_hash = record.pop('hash')
            record_str = json.dumps(record, sort_keys=True)
            computed_hash = hashlib.sha256(record_str.encode()).hexdigest()
            
            if stored_hash == computed_hash:
                print(f"✅ Integridad verificada: {filename}")
                return True
            else:
                print(f"⚠️ ALERTA: Archivo modificado: {filename}")
                return False
                
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {filename}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def load_all_draws(self):
        """
        Carga todos los sorteos guardados
        """
        draws = []
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
        
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                draw = json.load(f)
                draws.append(draw)
        
        print(f"📊 Sorteos cargados: {len(draws)}")
        return draws


# CARGA DE SORTEOS REALES (Sep–Nov 2025)
if __name__ == "__main__":
    print("🎯 MELATE COLLECTOR - CARGA DE SORTEOS REALES (Sep–Nov 2025)")
    print("=" * 60)
    
    # Crear collector
    collector = MelateCollector()
    
    # Lista de 30 sorteos reales (fecha, números, adicional)
    sorteos_reales = [
        ('2025-11-09', [3, 16, 23, 30, 38, 53], 49),
        ('2025-11-07', [24, 31, 35, 40, 45, 51], 12),
        ('2025-11-05', [1, 19, 24, 25, 31, 38], 9),
        ('2025-11-02', [6, 21, 30, 36, 48, 52], 11),
        ('2025-10-31', [4, 9, 13, 27, 29, 44], 32),
        ('2025-10-29', [6, 11, 12, 18, 30, 44], 43),
        ('2025-10-26', [14, 25, 26, 28, 38, 54], 13),
        ('2025-10-24', [17, 30, 31, 33, 36, 44], 50),
        ('2025-10-22', [1, 5, 7, 8, 32, 40], 36),
        ('2025-10-19', [2, 13, 20, 23, 39, 44], 25),
        ('2025-10-17', [3, 9, 17, 34, 41, 48], 2),
        ('2025-10-15', [10, 29, 30, 33, 45, 47], 24),
        ('2025-10-12', [6, 21, 30, 41, 47, 48], 29),
        ('2025-10-10', [15, 36, 40, 43, 44, 47], 48),
        ('2025-10-08', [18, 39, 41, 45, 46, 50], 25),
        ('2025-10-05', [1, 7, 12, 26, 27, 48], 29),
        ('2025-10-03', [15, 21, 25, 30, 32, 35], 20),
        ('2025-10-01', [9, 14, 19, 23, 34, 54], 32),
        ('2025-09-28', [6, 10, 35, 43, 44, 47], 56),
        ('2025-09-26', [23, 31, 32, 35, 47, 50], 13),
        ('2025-09-24', [31, 33, 43, 50, 53, 55], 56),
        ('2025-09-21', [4, 5, 13, 41, 46, 50], 24),
        ('2025-09-19', [2, 5, 13, 24, 25, 34], 29),
        ('2025-09-17', [3, 12, 15, 29, 36, 50], 27),
        ('2025-09-14', [3, 7, 12, 24, 42, 55], 19),
        ('2025-09-12', [10, 23, 26, 29, 36, 49], 24),
        ('2025-09-10', [8, 17, 20, 26, 28, 33], 3),
        ('2025-09-07', [1, 17, 28, 40, 48, 54], 14),
        ('2025-09-05', [19, 27, 33, 41, 42, 50], 56),
        ('2025-09-03', [9, 11, 17, 31, 52, 56], 18),
    ]
    
    # Registrar todos los sorteos
    for fecha, numeros, adicional in sorteos_reales:
        print(f"\n📝 Registrando sorteo {fecha}...")
        collector.save_draw(
            date_str=fecha,
            numbers=numeros,
            additional=adicional
        )
    
    # Verificar integridad de todos
    print("\n🔍 Verificando integridad de todos los archivos...")
    for fecha, _, _ in sorteos_reales:
        filename = os.path.join('./data/raw', f"melate_{fecha.replace('-', '')}.json")
        collector.verify_integrity(filename)
    
    # Cargar todos y mostrar conteo
    print("\n📊 Cargando todos los sorteos...")
    all_draws = collector.load_all_draws()
    
    print("\n✅ CARGA COMPLETADA")
    print(f"   Total sorteos en BD: {len(all_draws)}")