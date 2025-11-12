"""
Script de Comparación: Todos los Algoritmos vs Último Sorteo Real
Genera tabla comparativa y gráficos de performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import json

# Configurar paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importar todos los algoritmos
from src.data.collector import MelateCollector
from src.algorithms.baseline import RandomBaseline, FrequencySimple
from src.algorithms.markov import MarkovChain, MarkovSecondOrder
from src.algorithms.knn import KNNLottery, KNNEnsemble

# XGBoost opcional (puede no estar instalado)
try:
    from src.algorithms.xgboost_model import XGBoostLottery
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible (opcional)")


class AlgorithmComparison:
    """
    Sistema de comparación de algoritmos con walk-forward validation
    """
    
    def __init__(self, data_dir='./data/raw/'):
        self.data_dir = data_dir
        self.collector = MelateCollector(data_dir=data_dir)
        self.history = None
        self.results = []
        
    def load_data(self):
        """Carga todos los sorteos disponibles"""
        self.history = self.collector.load_all_draws()
        print(f"📊 Sorteos cargados: {len(self.history)}")
        
        if len(self.history) < 5:
            raise ValueError("❌ Se necesitan al menos 5 sorteos para comparación")
        
        return self.history
    
    def initialize_algorithms(self):
        """
        Inicializa todos los algoritmos disponibles
        
        Returns:
            Lista de tuplas (nombre, modelo, requiere_fit)
        """
        algorithms = [
            ("Random Baseline", RandomBaseline(), False),
            ("Frequency Simple", FrequencySimple(), False),
            ("Markov 1st Order", MarkovChain(), True),
        ]
        
        # Markov 2nd Order (solo si hay suficientes datos)
        if len(self.history) >= 5:
            algorithms.append(("Markov 2nd Order", MarkovSecondOrder(), True))
        
        # KNN (solo si hay suficientes datos)
        if len(self.history) >= 10:
            algorithms.append(("KNN (k=5)", KNNLottery(k=5), True))
            algorithms.append(("KNN Ensemble", KNNEnsemble(k_values=[3, 5, 7]), True))
        
        # XGBoost (solo si está disponible y hay datos suficientes)
        if XGBOOST_AVAILABLE and len(self.history) >= 15:
            algorithms.append(("XGBoost", XGBoostLottery(n_lags=5, n_estimators=50), True))
        
        print(f"✅ Algoritmos inicializados: {len(algorithms)}")
        return algorithms
    
    def run_single_test(self, model, model_name, train_data, test_data, requires_fit):
        """
        Ejecuta un test individual de un algoritmo
        
        Returns:
            dict con resultados
        """
        try:
            # Entrenar si es necesario
            if requires_fit:
                model.fit(train_data)
            
            # Predecir
            prediction = model.predict(train_data)
            
            # Evaluar
            actual = test_data['numbers']
            matches = len(set(prediction) & set(actual))
            
            # Calcular métricas adicionales
            precision = matches / 6  # Proporción de predicciones correctas
            
            # Análisis de números predichos
            predicted_set = set(prediction)
            actual_set = set(actual)
            true_positives = len(predicted_set & actual_set)
            false_positives = len(predicted_set - actual_set)
            false_negatives = len(actual_set - predicted_set)
            
            return {
                'algorithm': model_name,
                'prediction': prediction,
                'actual': actual,
                'matches': matches,
                'precision': precision,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'algorithm': model_name,
                'prediction': None,
                'actual': test_data['numbers'],
                'matches': 0,
                'precision': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_walk_forward_validation(self, n_tests=3):
        """
        Ejecuta validación walk-forward en los últimos N sorteos
        
        Args:
            n_tests: Número de sorteos a testear
        
        Returns:
            DataFrame con resultados
        """
        if len(self.history) < n_tests + 5:
            n_tests = max(1, len(self.history) - 5)
            print(f"⚠️ Ajustando n_tests a {n_tests} por datos insuficientes")
        
        algorithms = self.initialize_algorithms()
        all_results = []
        
        print(f"\n🔄 Walk-Forward Validation: {n_tests} sorteos")
        print("=" * 70)
        
        # Para cada sorteo de test
        for test_idx in range(n_tests):
            # Índice del sorteo a testear (contando desde el final)
            actual_idx = -(n_tests - test_idx)
            
            # Separar train/test
            if actual_idx == -1:
                train_data = self.history[:-1]
                test_data = self.history[-1]
            else:
                train_data = self.history[:actual_idx]
                test_data = self.history[actual_idx]
            
            test_date = test_data['date']
            print(f"\n📅 Test {test_idx + 1}/{n_tests}: {test_date}")
            print(f"   Train: {len(train_data)} sorteos | Test: {test_data['numbers']}")
            print("-" * 70)
            
            # Ejecutar cada algoritmo
            for model_name, model, requires_fit in algorithms:
                result = self.run_single_test(
                    model, model_name, train_data, test_data, requires_fit
                )
                result['test_date'] = test_date
                result['test_idx'] = test_idx
                all_results.append(result)
                
                # Mostrar resultado
                if result['success']:
                    print(f"   {model_name:20s}: {result['matches']}/6 aciertos - {result['prediction']}")
                else:
                    print(f"   {model_name:20s}: ❌ ERROR - {result['error']}")
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_results)
        self.results = df
        
        return df
    
    def generate_summary_table(self):
        """
        Genera tabla resumen con estadísticas por algoritmo
        """
        if self.results is None or len(self.results) == 0:
            print("❌ No hay resultados. Ejecuta run_walk_forward_validation() primero.")
            return None
        
        # Filtrar solo resultados exitosos
        df = self.results[self.results['success'] == True].copy()
        
        if len(df) == 0:
            print("❌ No hay resultados exitosos para resumir.")
            return None
        
        # Agrupar por algoritmo
        summary = df.groupby('algorithm').agg({
            'matches': ['mean', 'std', 'min', 'max'],
            'precision': ['mean'],
            'test_idx': 'count'
        }).round(3)
        
        summary.columns = ['Avg Matches', 'Std Matches', 'Min Matches', 'Max Matches', 'Avg Precision', 'N Tests']
        summary = summary.reset_index()
        
        # Calcular Z-score vs azar (esperado: 0.64 ± 0.72)
        expected_mean = 0.64
        expected_std = 0.72
        summary['Z-score'] = ((summary['Avg Matches'] - expected_mean) / expected_std).round(2)
        
        # Significancia estadística (|Z| > 1.96 → p < 0.05)
        summary['Significant'] = summary['Z-score'].abs() > 1.96
        
        # Ordenar por Avg Matches (descendente)
        summary = summary.sort_values('Avg Matches', ascending=False)
        
        return summary
    
    def plot_results(self, save_path='./results/figures/comparison.png'):
        """
        Genera visualizaciones de resultados
        """
        if self.results is None or len(self.results) == 0:
            print("❌ No hay resultados para graficar.")
            return
        
        # Crear directorio de resultados si no existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Filtrar resultados exitosos
        df = self.results[self.results['success'] == True].copy()
        
        if len(df) == 0:
            print("❌ No hay resultados exitosos para graficar.")
            return
        
        # Configurar estilo
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparación de Algoritmos - Lotería Melate', fontsize=16, fontweight='bold')
        
        # 1. Boxplot de aciertos por algoritmo
        ax1 = axes[0, 0]
        df_sorted = df.groupby('algorithm')['matches'].mean().sort_values(ascending=False)
        order = df_sorted.index.tolist()
        
        sns.boxplot(data=df, y='algorithm', x='matches', order=order, ax=ax1, palette='Set2')
        ax1.axvline(x=0.64, color='red', linestyle='--', linewidth=2, label='Esperado (azar)')
        ax1.axvspan(0.64 - 0.72, 0.64 + 0.72, alpha=0.2, color='red', label='±1 std')
        ax1.set_xlabel('Aciertos (0-6)', fontsize=12)
        ax1.set_ylabel('Algoritmo', fontsize=12)
        ax1.set_title('Distribución de Aciertos por Algoritmo', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Barplot de aciertos promedio
        ax2 = axes[0, 1]
        summary = df.groupby('algorithm')['matches'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        bars = ax2.bar(range(len(summary)), summary['mean'], yerr=summary['std'], 
                       capsize=5, alpha=0.7, color=sns.color_palette('Set2', len(summary)))
        ax2.axhline(y=0.64, color='red', linestyle='--', linewidth=2, label='Esperado (azar)')
        ax2.fill_between(range(len(summary)), 0.64 - 0.72, 0.64 + 0.72, 
                         alpha=0.2, color='red', label='±1 std')
        ax2.set_xticks(range(len(summary)))
        ax2.set_xticklabels(summary.index, rotation=45, ha='right')
        ax2.set_ylabel('Aciertos Promedio', fontsize=12)
        ax2.set_title('Performance Promedio (con error estándar)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Heatmap de aciertos por test
        ax3 = axes[1, 0]
        pivot = df.pivot_table(values='matches', index='algorithm', columns='test_idx', aggfunc='mean')
        pivot = pivot.reindex(order)  # Mismo orden que boxplot
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0.64, 
                    vmin=0, vmax=6, ax=ax3, cbar_kws={'label': 'Aciertos'})
        ax3.set_xlabel('Test Index', fontsize=12)
        ax3.set_ylabel('Algoritmo', fontsize=12)
        ax3.set_title('Aciertos por Test (Walk-Forward)', fontsize=14, fontweight='bold')
        
        # 4. Scatter: Z-score vs Avg Matches
        ax4 = axes[1, 1]
        summary_full = df.groupby('algorithm').agg({
            'matches': ['mean', 'std', 'count']
        })
        summary_full.columns = ['mean', 'std', 'count']
        summary_full['z_score'] = (summary_full['mean'] - 0.64) / 0.72
        
        colors = ['green' if abs(z) > 1.96 else 'gray' for z in summary_full['z_score']]
        ax4.scatter(summary_full['mean'], summary_full['z_score'], 
                   s=summary_full['count'] * 100, alpha=0.6, c=colors)
        
        for idx, row in summary_full.iterrows():
            ax4.annotate(idx, (row['mean'], row['z_score']), 
                        fontsize=9, ha='right', va='bottom')
        
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Z=0 (azar)')
        ax4.axhline(y=1.96, color='orange', linestyle=':', linewidth=1, label='p=0.05')
        ax4.axhline(y=-1.96, color='orange', linestyle=':', linewidth=1)
        ax4.axvline(x=0.64, color='red', linestyle='--', linewidth=1)
        ax4.set_xlabel('Aciertos Promedio', fontsize=12)
        ax4.set_ylabel('Z-score', fontsize=12)
        ax4.set_title('Significancia Estadística', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
        
        return fig
    
    def save_results(self, filename='comparison_results.csv'):
        """Guarda resultados detallados en CSV"""
        if self.results is None:
            print("❌ No hay resultados para guardar.")
            return
        
        output_path = f'./results/tables/{filename}'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convertir listas a strings para CSV
        df_save = self.results.copy()
        df_save['prediction'] = df_save['prediction'].apply(lambda x: str(x) if x else None)
        df_save['actual'] = df_save['actual'].apply(str)
        
        df_save.to_csv(output_path, index=False)
        print(f"✅ Resultados guardados: {output_path}")
    
    def generate_report(self):
        """
        Genera reporte completo en texto
        """
        if self.results is None:
            print("❌ No hay resultados. Ejecuta run_walk_forward_validation() primero.")
            return
        
        summary = self.generate_summary_table()
        
        print("\n" + "=" * 80)
        print("📊 REPORTE DE COMPARACIÓN - ALGORITMOS ML PARA LOTERÍA MELATE")
        print("=" * 80)
        
        print(f"\n🗓️  Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Total de sorteos: {len(self.history)}")
        print(f"🧪 Tests realizados: {len(self.results) // len(summary)}")
        print(f"🤖 Algoritmos evaluados: {len(summary)}")
        
        print("\n" + "-" * 80)
        print("📈 TABLA RESUMEN: Performance por Algoritmo")
        print("-" * 80)
        print(summary.to_string(index=False))
        
        print("\n" + "-" * 80)
        print("🏆 RANKING DE ALGORITMOS (por aciertos promedio)")
        print("-" * 80)
        for idx, row in summary.iterrows():
            rank = idx + 1
            stars = "⭐" * int(row['Avg Matches'])
            significant = "🎉 Significativo!" if row['Significant'] else ""
            print(f"{rank}. {row['algorithm']:25s} | {row['Avg Matches']:.2f} ± {row['Std Matches']:.2f} {stars} {significant}")
        
        print("\n" + "-" * 80)
        print("💡 INTERPRETACIÓN")
        print("-" * 80)
        
        best = summary.iloc[0]
        worst = summary.iloc[-1]
        
        print(f"✅ Mejor algoritmo: {best['algorithm']} ({best['Avg Matches']:.2f} aciertos)")
        print(f"❌ Peor algoritmo: {worst['algorithm']} ({worst['Avg Matches']:.2f} aciertos)")
        print(f"📊 Referencia (azar): 0.64 ± 0.72 aciertos")
        
        significant = summary[summary['Significant'] == True]
        if len(significant) > 0:
            print(f"\n⚠️  ATENCIÓN: {len(significant)} algoritmo(s) con diferencia estadísticamente significativa:")
            for _, row in significant.iterrows():
                print(f"   - {row['algorithm']}: Z-score = {row['Z-score']}")
                if row['Z-score'] > 0:
                    print(f"     ⚠️ POSIBLE overfitting o casualidad (requiere más tests)")
                else:
                    print(f"     ⚠️ Performance bajo el azar (implementación incorrecta?)")
        else:
            print("\n✅ CONCLUSIÓN: Ningún algoritmo supera significativamente el azar (p > 0.05)")
            print("   → Los resultados son consistentes con un proceso aleatorio")
            print("   → No hay evidencia de patrones predictivos en la lotería")
        
        print("\n" + "=" * 80)


# ==================== MAIN ====================
if __name__ == "__main__":
    print("🧪 COMPARACIÓN COMPLETA DE ALGORITMOS")
    print("=" * 80)
    
    # Inicializar comparador
    comparator = AlgorithmComparison(data_dir='./data/raw/')
    
    # Cargar datos
    try:
        comparator.load_data()
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # Ejecutar validación walk-forward
    n_tests = min(3, len(comparator.history) - 5)  # Máximo 3 tests
    results_df = comparator.run_walk_forward_validation(n_tests=n_tests)
    
    # Generar reporte
    print("\n")
    comparator.generate_report()
    
    # Guardar resultados
    comparator.save_results()
    
    # Generar gráficos
    try:
        comparator.plot_results()
        print("\n✅ Visualizaciones generadas exitosamente")
    except Exception as e:
        print(f"\n⚠️ Error al generar gráficos: {e}")
        print("   (Puede ser que matplotlib no esté configurado correctamente)")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)
    print("\n📁 Archivos generados:")
    print("   - results/tables/comparison_results.csv")
    print("   - results/figures/comparison.png")
    print("\n💡 Próximo paso: Revisar gráficos y analizar resultados")