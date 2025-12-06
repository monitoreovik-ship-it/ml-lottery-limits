import json
import numpy as np
from datetime import datetime
import os

with open("data/historical/melate_completo.json", "r") as f:
    data = json.load(f)

def quantum_simulate_pure(run_id, combo_id, avoid_overvalued=True, overvalued=[30, 31, 44]):
    avoid = avoid_overvalued and (np.random.random() < 0.8)
    quantum_state = np.random.rand(56)
    
    for _ in range(10):
        idx1, idx2 = np.random.randint(0, 56, 2)
        correlation = (quantum_state[idx1] + quantum_state[idx2]) / 2
        quantum_state[idx1] = correlation
        quantum_state[idx2] = correlation
    
    probabilities = quantum_state / quantum_state.sum()
    combo = set()
    attempts = 0
    
    while len(combo) < 6 and attempts < 100:
        num = int(np.random.choice(range(1, 57), p=probabilities))
        if avoid and num in overvalued:
            attempts += 1
            continue
        if num not in combo:
            combo.add(num)
        attempts += 1
    
    while len(combo) < 6:
        num = int(np.random.randint(1, 57))
        if not (avoid and num in overvalued) and num not in combo:
            combo.add(num)
    
    return sorted(list(combo))

def calc_entanglement(combo):
    distances = []
    sorted_combo = sorted(combo)
    for i in range(len(sorted_combo) - 1):
        distances.append(sorted_combo[i+1] - sorted_combo[i])
    variance = np.var(distances)
    return round(float(1 / (1 + variance) * 0.01), 4)

overvalued = data["overvalued"]
all_combos = []

print("🔮 MINIMALISTA V10.0\n")

for run in range(5):
    print(f"🔄 Ejecución {run + 1}/5...")
    for cid in range(3):
        combo = quantum_simulate_pure(run, cid, True, overvalued)
        ent = calc_entanglement(combo)
        
        top_5 = data["top_5"]
        top_6_15 = data["top_6_15"]
        h = int(sum(1 for n in combo if n in top_5))
        n = int(sum(1 for n in combo if n in top_6_15))
        c = int(6 - h - n)
        has_ov = any(num in overvalued for num in combo)
        
        all_combos.append({
            "run": int(run+1),
            "combo_id": int(cid+1),
            "numbers": combo,
            "entanglement": ent,
            "hot": h,
            "neutral": n,
            "cold": c,
            "has_overvalued": bool(has_ov)
        })
        
        ov = "⚠️SV" if has_ov else ""
        print(f"  #{cid+1}: {combo} | Ent: {ent} | 🔥{h} 😐{n} ❄️{c} {ov}")

all_combos.sort(key=lambda x: -x["entanglement"])
top_3 = all_combos[:3]

print("\n🏆 TOP 3:\n")
for i, combo in enumerate(top_3, 1):
    ov = "⚠️SV" if combo["has_overvalued"] else "✅"
    print(f"#{i}: {combo['numbers']} | Ent: {combo['entanglement']} | 🔥{combo['hot']} 😐{combo['neutral']} ❄️{combo['cold']} | {ov}")

with_ov = sum(1 for c in all_combos if c["has_overvalued"])
print(f"\n📊 Con sobre-valorados: {with_ov}/15 ({with_ov/15*100:.0f}%)")
print(f"Objetivo: ~20% (regla: evitar 80%)")

balances = {}
for c in all_combos:
    key = f"{c['hot']}-{c['neutral']}-{c['cold']}"
    balances[key] = balances.get(key, 0) + 1

print(f"\nDistribución balance natural:")
for balance, count in sorted(balances.items(), key=lambda x: -x[1]):
    print(f"  {balance}: {count}/15 combos")

pred = {
    "test_number": 10,
    "date": datetime.now().isoformat(),
    "draw_date": "2025-12-08",
    "system": "minimalist_v10.0",
    "philosophy": "Less is more - Pure quantum with minimal rules",
    "all_combinations": all_combos,
    "top_3": top_3,
    "statistics": {
        "with_overvalued": int(with_ov),
        "with_overvalued_percent": round(with_ov/15*100, 1),
        "balance_distribution": balances
    }
}

os.makedirs("data/predictions/quantum", exist_ok=True)
with open("data/predictions/quantum/quantum_prediction_20251208.json", "w") as f:
    json.dump(pred, f, indent=2)

print(f"\n✅ Guardado: data/predictions/quantum/quantum_prediction_20251208.json")
print(f"\n🎯 COMBINACIÓN PRINCIPAL:")
print(f"   {top_3[0]['numbers']}")
print(f"   Balance natural: 🔥{top_3[0]['hot']} 😐{top_3[0]['neutral']} ❄️{top_3[0]['cold']}")
print(f"   Filosofía: Minimalismo - dejar que el cuántico hable")
