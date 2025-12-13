import json
import numpy as np
from datetime import datetime
import os

print("🔮 SISTEMA V12: MINIMALISMO + #38 FORZADO")
print("=" * 60)
print("ESTRATEGIA:")
print("✅ Generación cuántica pura")
print("✅ Forzar #38 en 50% de combinaciones")
print("✅ Mix clustering + disperso")
print("=" * 60)
print()

def quantum_pure_with_38(force_38=False):
    """Generación cuántica con opción de forzar #38"""
    quantum_state = np.random.rand(56)
    
    # Entrelazamiento
    for _ in range(np.random.randint(8, 15)):
        idx1, idx2 = np.random.randint(0, 56, 2)
        corr = (quantum_state[idx1] + quantum_state[idx2]) / 2
        quantum_state[idx1] = quantum_state[idx2] = corr
    
    probs = quantum_state / quantum_state.sum()
    combo = []
    
    # Forzar #38 si se requiere
    if force_38:
        combo.append(38)
    
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
        "clustered": rng < 30,
        "has_38": 38 in combo
    }

all_combos = []
print("Generando 20 combinaciones...\n")

for run in range(4):
    print(f"🔄 Run {run + 1}/4...")
    for cid in range(5):
        # Alternar: forzar #38 en 50% (2-3 por run)
        force = (cid < 3)
        combo = quantum_pure_with_38(force_38=force)
        ent = calc_entanglement(combo)
        clust = analyze_cluster(combo)
        
        all_combos.append({
            "run": run + 1,
            "combo_id": cid + 1,
            "numbers": combo,
            "entanglement": ent,
            "clustering": clust
        })
        
        markers = []
        if clust["has_38"]: markers.append("🎯#38")
        if clust["clustered"]: markers.append("📍CLUSTER")
        marker_str = " ".join(markers)
        
        print(f"  #{cid+1}: {combo} | Ent: {ent} | R: {clust['range']} {marker_str}")

# Selección Top 3 balanceada
# Priorizar: al menos 1 con #38, mix de clustering
combos_with_38 = [c for c in all_combos if c["clustering"]["has_38"]]
combos_without_38 = [c for c in all_combos if not c["clustering"]["has_38"]]

# Ordenar ambos grupos por entrelazamiento
combos_with_38.sort(key=lambda x: -x["entanglement"])
combos_without_38.sort(key=lambda x: -x["entanglement"])

# Top 3: Al menos 2 con #38, 1 sin #38
top_3 = []
if len(combos_with_38) >= 2:
    top_3.append(combos_with_38[0])  # Mejor con #38
    top_3.append(combos_with_38[1])  # Segundo mejor con #38
if len(combos_without_38) > 0:
    top_3.append(combos_without_38[0])  # Mejor sin #38

# Si no hay suficientes, llenar por entrelazamiento
all_combos_sorted = sorted(all_combos, key=lambda x: -x["entanglement"])
while len(top_3) < 3:
    for c in all_combos_sorted:
        if c not in top_3:
            top_3.append(c)
            break

print("\n" + "=" * 60)
print("🏆 TOP 3 BALANCEADO:")
print("=" * 60)

for i, c in enumerate(top_3, 1):
    nums = c["numbers"]
    ent = c["entanglement"]
    cl = c["clustering"]
    
    markers = []
    if cl["has_38"]: markers.append("✅ #38")
    if cl["clustered"]: markers.append("📍 Cluster")
    
    print(f"\n#{i}: {nums}")
    print(f"    Entrelazamiento: {ent}")
    print(f"    Rango: {cl['range']} ({cl['min']}-{cl['max']})")
    print(f"    {' | '.join(markers) if markers else '⚠️ Sin #38'}")

# Estadísticas
print("\n" + "=" * 60)
print("📊 ANÁLISIS:")
print("=" * 60)

with_38 = sum(1 for c in all_combos if c["clustering"]["has_38"])
clustered = sum(1 for c in all_combos if c["clustering"]["clustered"])

print(f"Combos con #38: {with_38}/20 ({with_38/20*100:.0f}%)")
print(f"Combos clustering: {clustered}/20 ({clustered/20*100:.0f}%)")
print(f"\nTop 3 con #38: {sum(1 for c in top_3 if c['clustering']['has_38'])}/3")
print(f"Top 3 clustering: {sum(1 for c in top_3 if c['clustering']['clustered'])}/3")

ranges = [c["clustering"]["range"] for c in all_combos]
print(f"\nRango promedio: {np.mean(ranges):.1f}")
print(f"Rango Top 3: {np.mean([c['clustering']['range'] for c in top_3]):.1f}")

# Guardar
pred = {
    "test_number": 12,
    "date": datetime.now().isoformat(),
    "draw_date": "2025-12-14",
    "system": "minimalist_plus_38_v12.0",
    "strategy": {
        "description": "Minimalism + Force #38",
        "reason": "#38 has hit 4 times (most consistent number)",
        "approach": [
            "Generate 20 quantum combinations",
            "Force #38 in 50% of them",
            "Select Top 3 with at least 2 having #38",
            "Cover both clustering and dispersed scenarios"
        ]
    },
    "all_combinations": all_combos,
    "top_3": top_3,
    "statistics": {
        "total_generated": 20,
        "with_38": int(with_38),
        "with_38_percent": round(float(with_38/20*100), 1),
        "clustered": int(clustered),
        "top3_with_38": int(sum(1 for c in top_3 if c["clustering"]["has_38"])),
        "avg_range": round(float(np.mean(ranges)), 1)
    }
}

os.makedirs("data/predictions/quantum", exist_ok=True)
with open("data/predictions/quantum/quantum_prediction_20251214.json", "w", encoding="utf-8") as f:
    json.dump(pred, f, indent=2)

print("\n" + "=" * 60)
print("✅ PREDICCIÓN GUARDADA")
print("=" * 60)
print(f"\n🎯 COMBINACIONES RECOMENDADAS:")
for i, c in enumerate(top_3, 1):
    has38 = "✅ #38" if c["clustering"]["has_38"] else "⚠️ Sin #38"
    print(f"{i}. {c['numbers']} | {has38}")

print(f"\n🔬 ESTRATEGIA:")
print(f"   Al menos 2/3 con #38 (número más consistente)")
print(f"   Mix de rangos para cubrir clustering + disperso")
print(f"   Minimalismo puro sin reglas complejas")
