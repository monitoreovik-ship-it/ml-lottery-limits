import json
import numpy as np
from datetime import datetime
import os

print("🔮 ULTRA-PURO V11.0\n")

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
        "clustered": rng < 25
    }

all_combos = []
print("Generando 20 combinaciones...\n")

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

print("\n🏆 TOP 3:\n")
for i, c in enumerate(top_3, 1):
    cl = c["clustering"]
    print(f"#{i}: {c['numbers']}")
    print(f"    Ent: {c['entanglement']} | Rango: {cl['range']} ({cl['min']}-{cl['max']})")

pred = {
    "test_number": 11,
    "date": datetime.now().isoformat(),
    "draw_date": "2025-12-11",
    "system": "ultra_pure_v11.0",
    "all_combinations": all_combos,
    "top_3": top_3
}

os.makedirs("data/predictions/quantum", exist_ok=True)
with open("data/predictions/quantum/quantum_prediction_20251211.json", "w") as f:
    json.dump(pred, f, indent=2)

print(f"\n✅ Guardado")
print(f"\n🎯 PRINCIPAL: {top_3[0]['numbers']}")
print(f"   Entrelazamiento: {top_3[0]['entanglement']}")
print(f"\n🔬 HIPÓTESIS: Si Test #10 fue real, Test #11 debe tener 2+ aciertos")
