import json
import numpy as np
from datetime import datetime
import os

print("🔮 MINIMALISMO PURO V13 - VALIDACIÓN")
print("=" * 60)
print("Volviendo a estrategia más exitosa")
print("Sin reglas, sin forzar números, solo entrelazamiento")
print("=" * 60)
print()

def quantum_pure():
    quantum_state = np.random.rand(56)
    for _ in range(np.random.randint(8, 15)):
        idx1, idx2 = np.random.randint(0, 56, 2)
        corr = (quantum_state[idx1] + quantum_state[idx2]) / 2
        quantum_state[idx1] = quantum_state[idx2] = corr
    
    probs = quantum_state / quantum_state.sum()
    combo = []
    attempts = 0
    
    while len(combo) < 6 and attempts < 200:
        num = int(np.random.choice(range(1, 57), p=probs))
        if num not in combo:
            combo.append(num)
        attempts += 1
    
    while len(combo) < 6:
        num = int(np.random.randint(1, 57))
        if num not in combo:
            combo.append(num)
    
    return sorted(combo)

def calc_entanglement(combo):
    sc = sorted(combo)
    distances = [sc[i+1] - sc[i] for i in range(5)]
    variance = np.var(distances)
    total_range = sc[-1] - sc[0]
    return round(float(1 / (1 + variance + total_range/100) * 0.01), 4)

def analyze_cluster(combo):
    sc = sorted(combo)
    rng = sc[-1] - sc[0]
    return {
        "range": int(rng),
        "min": int(sc[0]),
        "max": int(sc[-1]),
        "std": round(float(np.std(combo)), 2),
        "clustered": rng < 30
    }

all_combos = []
print("Generando 20 combinaciones cuánticas puras...\n")

for run in range(4):
    print(f"🔄 Run {run + 1}/4...")
    for cid in range(5):
        combo = quantum_pure()
        ent = calc_entanglement(combo)
        clust = analyze_cluster(combo)
        
        all_combos.append({
            "run": run + 1,
            "combo_id": cid + 1,
            "numbers": combo,
            "entanglement": ent,
            "clustering": clust
        })
        
        cl_mark = "📍" if clust["clustered"] else ""
        print(f"  #{cid+1}: {combo} | Ent: {ent} | R: {clust['range']} {cl_mark}")

all_combos.sort(key=lambda x: -x["entanglement"])
top_3 = all_combos[:3]

print("\n" + "=" * 60)
print("🏆 TOP 3 POR ENTRELAZAMIENTO:")
print("=" * 60)

for i, c in enumerate(top_3, 1):
    cl = c["clustering"]
    print(f"\n#{i}: {c['numbers']}")
    print(f"    Entrelazamiento: {c['entanglement']}")
    print(f"    Rango: {cl['range']} ({cl['min']}-{cl['max']})")
    print(f"    Clustering: {'SÍ ✅' if cl['clustered'] else 'NO'}")

# Estadísticas
print("\n" + "=" * 60)
print("📊 ESTADÍSTICAS:")
print("=" * 60)

clustered = sum(1 for c in all_combos if c["clustering"]["clustered"])
ranges = [c["clustering"]["range"] for c in all_combos]

print(f"Combos clustering: {clustered}/20 ({clustered/20*100:.0f}%)")
print(f"Top 3 clustering: {sum(1 for c in top_3 if c['clustering']['clustered'])}/3")
print(f"Rango promedio: {np.mean(ranges):.1f}")
print(f"Rango Top 3: {np.mean([c['clustering']['range'] for c in top_3]):.1f}")

pred = {
    "test_number": 13,
    "date": datetime.now().isoformat(),
    "draw_date": "2025-12-15",
    "system": "minimalist_pure_v13_validation",
    "philosophy": "Back to basics - pure minimalism validation",
    "reason": "Test #12 showed forcing #38 did not improve results. Return to pure minimalism (like Tests #10-11) to validate 2.0 avg with n>=5",
    "all_combinations": all_combos,
    "top_3": top_3,
    "statistics": {
        "total_generated": 20,
        "clustered": int(clustered),
        "avg_range": round(float(np.mean(ranges)), 1),
        "top3_avg_range": round(float(np.mean([c["clustering"]["range"] for c in top_3])), 1)
    }
}

os.makedirs("data/predictions/quantum", exist_ok=True)
with open("data/predictions/quantum/quantum_prediction_20251215.json", "w", encoding="utf-8") as f:
    json.dump(pred, f, indent=2)

print("\n" + "=" * 60)
print("✅ PREDICCIÓN GUARDADA")
print("=" * 60)
print(f"\n🎯 COMBINACIÓN PRINCIPAL: {top_3[0]['numbers']}")
print(f"   Entrelazamiento: {top_3[0]['entanglement']}")
print(f"   Rango: {top_3[0]['clustering']['range']}")
print()
print("🔬 VALIDACIÓN TEST #13:")
print("   Confirmar que minimalismo puro mantiene 2.0 avg")
print("   Sin complejidad adicional (n≥5 para validación)")
