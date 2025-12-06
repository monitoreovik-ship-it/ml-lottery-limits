import json
import numpy as np
from datetime import datetime
import os

with open("data/historical/melate_completo.json", "r") as f:
    data = json.load(f)

def generate_balanced_combo_v9(top_5, top_6_15, overvalued, validated, force_42=True, recent_winners=None):
    combo = []
    recent_winners = recent_winners or []
    
    if force_42 and np.random.random() < 0.7:
        combo.append(42)
        print("  🎯 #42 forzado")
    
    if 42 not in combo:
        available_hot = [n for n in top_5 if n not in overvalued and n not in combo]
        if available_hot:
            if 38 in available_hot:
                combo.append(38)
                print("  🎯 #38 seleccionado")
            else:
                combo.append(int(np.random.choice(available_hot)))
    
    available_neutral = [n for n in top_6_15 if n not in combo]
    recent_neutrals = [n for n in [6, 41, 56] if n in available_neutral]
    
    neutrals_to_add = 6 - len(combo) - 2
    if neutrals_to_add > 0:
        selected_neutrals = []
        for n in recent_neutrals:
            if len(selected_neutrals) < neutrals_to_add:
                selected_neutrals.append(n)
        
        remaining = neutrals_to_add - len(selected_neutrals)
        if remaining > 0:
            other_neutrals = [n for n in available_neutral if n not in selected_neutrals]
            if len(other_neutrals) >= remaining:
                extras = np.random.choice(other_neutrals, size=remaining, replace=False)
                selected_neutrals.extend([int(x) for x in extras])
        
        combo.extend(selected_neutrals)
    
    all_nums = list(range(1, 57))
    cold_pool = [n for n in all_nums if n not in top_5 and n not in top_6_15 and n not in combo]
    remaining = 6 - len(combo)
    
    if len(cold_pool) >= remaining and remaining > 0:
        colds = np.random.choice(cold_pool, size=remaining, replace=False)
        combo.extend([int(x) for x in colds])
    
    while len(combo) < 6:
        num = np.random.randint(1, 57)
        if num not in combo:
            combo.append(num)
    
    return sorted(combo[:6])

def calc_score_v9(combo, top_5, top_6_15, overvalued, validated, recent_winners):
    score = 0
    hot = sum(1 for n in combo if n in top_5)
    neutral = sum(1 for n in combo if n in top_6_15)
    cold = 6 - hot - neutral
    
    balance_score = 0
    if hot == 1: balance_score += 15
    elif hot == 0 and 42 in combo: balance_score += 10
    elif hot == 0: balance_score += 5
    if neutral >= 3: balance_score += 15
    if cold <= 2: balance_score += 10
    score += balance_score * 0.35
    
    over = sum(1 for n in combo if n in overvalued)
    score += (30 - over * 10) * 0.30
    score += 20 * 0.20
    
    val_bonus = 0
    if 42 in combo: val_bonus += 8
    if 38 in combo: val_bonus += 4
    for n in combo:
        if n in validated and n not in [42, 38]:
            val_bonus += 2
    score += min(val_bonus, 15) * 0.15
    
    recent_bonus = sum(2 for n in combo if n in recent_winners)
    score += min(recent_bonus, 5) * 0.05
    
    return round(score, 2), hot, neutral, cold

top_5 = data["top_5"]
top_6_15 = data["top_6_15"]
overvalued = data["overvalued"]
validated = data["validated"]
recent_winners = [1, 6, 29, 41, 42, 43, 56]

all_combos = []
print("🔀 HÍBRIDO V9.0 - REFINADO\n")

for run in range(5):
    print(f"🔄 Ejecución {run + 1}/5...")
    for cid in range(3):
        force_42 = (cid < 2)
        combo = generate_balanced_combo_v9(top_5, top_6_15, overvalued, validated, force_42, recent_winners)
        score, h, n, c = calc_score_v9(combo, top_5, top_6_15, overvalued, validated, recent_winners)
        
        all_combos.append({
            "run": run+1, "combo_id": cid+1, "numbers": combo, 
            "score": score, "hot": h, "neutral": n, "cold": c,
            "has_42": 42 in combo,
            "has_recent": len(set(combo).intersection(recent_winners))
        })
        
        ind = []
        if 42 in combo: ind.append("🎯#42")
        if 38 in combo: ind.append("🎯#38")
        rec = len(set(combo).intersection(recent_winners))
        if rec > 0: ind.append(f"📅{rec}rec")
        print(f"  #{cid+1}: {combo} | Score: {score} | 🔥{h} 😐{n} ❄️{c} {' '.join(ind)}")

all_combos.sort(key=lambda x: -x["score"])
top_3 = all_combos[:3]

print("\n🏆 TOP 3:\n")
for i, c in enumerate(top_3, 1):
    has_42 = "✅#42" if c["has_42"] else "⚠️Sin#42"
    print(f"#{i}: {c['numbers']} | Score: {c['score']} | {has_42}")

combos_42 = [c for c in all_combos if c["has_42"]]
print(f"\n📊 #42: {len(combos_42)}/15 ({len(combos_42)/15*100:.0f}%) | Top3: {sum(1 for c in top_3 if c['has_42'])}/3")

pred = {
    "test_number": 9, "date": datetime.now().isoformat(), "draw_date": "2025-12-07",
    "system": "hybrid_v9.0_refined", "all_combinations": all_combos, "top_3": top_3
}

os.makedirs("data/predictions/quantum", exist_ok=True)
with open("data/predictions/quantum/quantum_prediction_20251207.json", "w") as f:
    json.dump(pred, f, indent=2)

print("\n✅ Guardado: data/predictions/quantum/quantum_prediction_20251207.json")
