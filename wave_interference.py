# -*- coding: utf-8 -*-
# wave_interference_v2.py — interference with sham + syntactic controls (Gemini protocol)
import os, torch, random
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, result_of
from probe5 import BIG

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
STEER = r"E:\jspace\steer.pt"
SEED = 999

def get_trace(model, tok, proj, a, o, b, steps=48):
    p = prompt_of(a, o, b)
    T = _j_loop(model, tok, p, proj, steps, 0)[0]
    return T

def cosine_sim(T1, T2):
    """Среднее косинусное сходство по шагам траекторий."""
    min_len = min(T1.shape[0], T2.shape[0])
    T1, T2 = T1[:min_len].float(), T2[:min_len].float()
    # нормируем каждый шаг
    T1n = torch.nn.functional.normalize(T1, dim=1)
    T2n = torch.nn.functional.normalize(T2, dim=1)
    return (T1n * T2n).sum(dim=1).mean().item()

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)

    test_cases = [
        (13, 7, 25, 39),
        (83, 46, 96, 8),
        (15, 6, 58, 27),
    ]

    print(f"{'Case':>6s} | {'Sim_target':>10s} {'Sim_sham':>10s} {'Sim_base':>10s} | {'Δ_tgt-sham':>10s} | verdict")
    print("-" * 75)
    results = []
    for i, (a1, b1, a2, b2) in enumerate(test_cases):
        # Прямые траектории
        T_sum = get_trace(model, tok, proj, a1+a2, '+', b1+b2)   # (A1+A2)+(B1+B2)
        T_diff = get_trace(model, tok, proj, a1-a2, '-', b1-b2)  # (A1-A2)-(B1-B2)
        T_A1 = get_trace(model, tok, proj, a1, '+', b1)          # A1+B1 (целевая)

        # Интерференция
        min_len = min(T_sum.shape[0], T_diff.shape[0], T_A1.shape[0])
        T_sum, T_diff, T_A1 = T_sum[:min_len], T_diff[:min_len], T_A1[:min_len]
        T_interf = (T_sum + T_diff) / 2.0

        # Контроль 1: Sham (случайное C ≠ A1)
        random.seed(SEED + i)
        c_sham = random.choice([x for x in range(1, 100) if x != a1 and x != a1+a2 and x != a1-a2])
        T_sham = get_trace(model, tok, proj, c_sham, '+', b1)

        # Контроль 2: Baseline (синтаксический шум)
        # Насколько T_sum изначально близок к T_A1 без интерференции
        sim_target = cosine_sim(T_interf, T_A1)
        sim_sham = cosine_sim(T_interf, T_sham)
        sim_base = cosine_sim(T_sum, T_A1)

        delta = sim_target - sim_sham
        verdict = "✅ интерференция специфична" if delta > 0.1 else ("⚠️ слабая" if delta > 0 else "❌ нет")
        print(f"{i+1:>6d} | {sim_target:>10.3f} {sim_sham:>10.3f} {sim_base:>10.3f} | {delta:>+10.3f} | {verdict}")
        results.append((sim_target, sim_sham, sim_base, delta))

    avg_delta = sum(r[3] for r in results) / len(results)
    avg_target = sum(r[0] for r in results) / len(results)
    avg_sham = sum(r[1] for r in results) / len(results)
    print(f"\nСредние: target={avg_target:.3f} sham={avg_sham:.3f} Δ={avg_delta:+.3f}")
    if avg_delta > 0.1 and avg_target > 0.7:
        print("Вердикт: ✅ латентные траектории интерферируют как волны (специфично)")
    elif avg_delta > 0:
        print("Вердикт: ⚠️ частичная интерференция (нелинейность/шум операции)")
    else:
        print("Вердикт: ❌ интерференция не специфична (артефакт общего контекста)")

if __name__ == "__main__":
    main()