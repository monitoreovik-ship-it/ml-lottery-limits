import numpy as np
from collections import Counter
import sys
import os

# Añadir src/data al path para importar collector
src_data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
sys.path.insert(0, src_data_path)

try:
    from collector import MelateCollector
except ImportError as e:
    print("❌ Error: No se pudo importar MelateCollector.")
    print("Asegúrate de que 'collector.py' esté en src/data/")
    sys.exit(1)


class RandomBaseline:
    """Baseline: Selección completamente aleatoria"""
    def __init__(self):
        self.name = "Random Baseline"
    def predict(self, history):
        return sorted(int(x) for x in np.random.choice(56, size=6, replace=False) + 1)


class FrequencySimple:
    """Top 6 números más frecuentes en el historial"""
    def __init__(self):
        self.name = "Frequency Simple"
    def predict(self, history):
        if not history:
            return sorted(int(x) for x in np.random.choice(56, size=6, replace=False) + 1)
        freq = Counter()
        for draw in history:
            freq.update(draw['numbers'])
        top_numbers = [num for num, count in freq.most_common(6)]
        while len(top_numbers) < 6:
            random_num = np.random.randint(1, 57)
            if random_num not in top_numbers:
                top_numbers.append(random_num)
        return sorted(top_numbers[:6])


if __name__ == "__main__":
    print("🤖 ALGORITMOS BASELINE - USANDO DATOS REALES")
    print("=" * 50)
    
    # Ruta absoluta a data/raw desde la raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    
    collector = MelateCollector(data_dir=data_dir)
    history = collector.load_all_draws()
    
    if not history:
        print("⚠️  No se encontraron sorteos en data/raw/.")
        print("Ejecuta primero: python src/data/collector.py")
        sys.exit(1)
    
    print(f"📈 Historial cargado: {len(history)} sorteos")
    
    models = [RandomBaseline(), FrequencySimple()]
    
    for model in models:
        print(f"\n🔮 {model.name}:")
        pred = model.predict(history)
        print(f"   Predicción: {pred}")
    
    print("\n✅ TEST COMPLETADO")