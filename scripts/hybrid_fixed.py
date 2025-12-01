import json
import numpy as np
from datetime import datetime
import os

with open("data/historical/melate_completo.json", "r") as f:
    data = json.load(f)

def generate_balanced_combo(top_5, top_6_15, overvalued, validated):
    combo = []
    
    # 1 caliente (evitar sobre-valorados)
    available_hot = [n for n in top_5 if n not in overvalued]
    if available_hot:
        combo.append(int(np.random.choice(available_hot)))
    
    # 3 neutrales
    available_neutral = [n for n in top_6_15 if n not in combo]
    if len(available_neutral) >= 3:
        neutrals = np.random.choice(available_neutral, size=3, replace=False)
        combo.extend([int(x) for x in neutrals])
    
    # 2 fríos
    all_nums = list(range(1, 57))
    cold_pool = [n for n in all_nums if n not in top_5 and n not in top_6_15 and n not in combo]
    remaining = 6 - len(combo)
    if len(cold_pool) >= remaining:
        colds = np.random.choice(cold_pool, size=remaining, replace=False)
        combo.extend([int(x) for x in colds])
    
    return sorted(combo[:6])

def calc_score(combo, top_5, top_6_15, overvalued, validated):
    score = 0
    hot = sum(1 for n in combo if n in top_5)
    neutral = sum(1 for n in combo if n in top_6_15)
    cold = 6 - hot - neutral
    
    if hot == 1: score += 15
    elif hot == 0: score += 5
    if neutral >= 3: score += 15
    if cold <= 2: score += 10
    score *= 0.4
    
    over = sum(1 for n in combo if n in overvalued)
    score += (30 - over * 10) * 0.3
    score += 20 * 0.2
    
    val = sum(1 for n in combo if n in validated)
    score += min(val * 2, 10) * 0.1
    
    return round(score, 2), hot, neutral, cold

top_5 = data["top_5"]
top_6_15 = data["top_6_15"]
overvalued = data["overvalued"]
validated = data["validated"]

all_combos = []
print("🔀 HÍBRIDO V8.0 CORREGIDO\n")

for run in range(5):
    print(f"🔄 Ejecución {run + 1}/5...")
    for cid in range(3):
        combo = generate_balanced_combo(top_5, top_6_15, overvalued, validated)
        score, h, n, c = calc_score(combo, top_5, top_6_15, overvalued, validated)
        all_combos.append({
            "run": run+1, 
            "combo_id": cid+1, 
            "numbers": combo, 
            "score": score, 
            "hot": h, 
            "neutral": n, 
            "cold": c
        })
        print(f"  #{cid+1}: {combo} | Score: {score} | 🔥{h} 😐{n} ❄️{c}")

all_combos.sort(key=lambda x: -x["score"])
top_3 = all_combos[:3]

print("\n🏆 TOP 3:\n")
for i, c in enumerate(top_3, 1):
    nums = c["numbers"]
    sc = c["score"]
    h = c["hot"]
    n = c["neutral"]
    co = c["cold"]
    print(f"#{i}: {nums} | Score: {sc} | 🔥{h} 😐{n} ❄️{co}")

pred = {
    "test_number": 8, 
    "date": datetime.now().isoformat(), 
    "draw_date": "2025-12-04", 
    "system": "hybrid_v8.0_fixed", 
    "all_combinations": all_combos, 
    "top_3": top_3
}

os.makedirs("data/predictions/quantum", exist_ok=True)
with open("data/predictions/quantum/quantum_prediction_20251204.json", "w") as f:
    json.dump(pred, f, indent=2)
    
print("\n✅ Guardado")
