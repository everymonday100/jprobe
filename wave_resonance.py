# -*- coding: utf-8 -*-
# wave_resonance.py — activation steering resonance (causal, prompt-only, no leakage)
import os, re, torch
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, result_of
from probe5 import TEST

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")

def get_trace(model, tok, proj, a, o, b, steps=48):
    p = prompt_of(a, o, b)
    T = _j_loop(model, tok, p, proj, steps, 0)[0]
    return T

def extract_diff_vector(model, tok, proj, a, b):
    """Дифф. вектор: trace(a+b) - trace(a-b), усреднённый по шагам.
       Пара (a,b) НЕ должна входить в TEST."""
    T_sum = get_trace(model, tok, proj, a+b, '+', 0)   # (a+b) + 0
    T_diff = get_trace(model, tok, proj, a-b, '-', 0)  # (a-b) - 0
    min_len = min(T_sum.shape[0], T_diff.shape[0])
    diff = (T_sum[:min_len] - T_diff[:min_len]).mean(dim=0)
    return diff

def gen_with_steering(model, tok, prompt, steer_vec, alpha, layer_idx=18):
    ids = tok.apply_chat_template([{"role":"user","content":prompt}],
                                  add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # Стиринг ТОЛЬКО на промпте (h.shape[1] > 1), не на авторегрессии
        if h.shape[1] > 1:
            h_steered = h + alpha * steer_vec.to(h.device, h.dtype).unsqueeze(0).unsqueeze(0)
        else:
            h_steered = h
        return (h_steered,) + output[1:] if isinstance(output, tuple) else h_steered

    layers = model.model.layers
    if layer_idx >= len(layers):
        layer_idx = len(layers) // 2
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.findall(r"=\s*(-?\d+)", txt) or re.findall(r"-?\d+", txt)
        return int(m[-1]) if m else None
    finally:
        handle.remove()

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)

    # Проверка утечки: пара (SRC_A, SRC_B) не должна быть в TEST
    SRC_A, SRC_B = 42, 19
    test_pairs = {(a, b) for a, _, b in TEST}
    assert (SRC_A, SRC_B) not in test_pairs and (SRC_A+SRC_B, 0) not in {(a+b, 0) for a,_,b in TEST}, \
        "Источник дифф. вектора пересекается с TEST!"
    print(f"Извлечение дифф. вектора из ({SRC_A}+{SRC_B}) vs ({SRC_A}-{SRC_B}) [вне TEST]...")
    diff_vec = extract_diff_vector(model, tok, proj, SRC_A, SRC_B)
    diff_vec = diff_vec / diff_vec.norm()

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    print(f"\n{'Prompt':28s} {'exp':>4s} | " + " ".join(f"a={a:<4.1f}" for a in alphas))
    print("-" * 75)

    results = {a: [] for a in alphas}
    for a, o, b in TEST:
        p = prompt_of(a, o, b); exp = result_of(a, o, b)
        row = []
        for alpha in alphas:
            got = gen_with_steering(model, tok, p, diff_vec, alpha, layer_idx=18)
            results[alpha].append(got == exp); row.append(got)
        print(f"{p:28s} {exp:>4d} | " + " ".join(f"{str(v):>6s}" for v in row))

    print(f"\nAccuracy по альфа:")
    base_acc = sum(results[0]) / len(results[0])
    for alpha in alphas:
        acc = sum(results[alpha]) / len(results[alpha])
        marker = "🔥 резонанс" if (alpha > 0 and acc > base_acc + 0.15) else ""
        print(f"  alpha={alpha:<4.1f}: {acc:.3f} {marker}")

    best_alpha = max(alphas[1:], key=lambda a: sum(results[a])/len(results[a]))
    best_acc = sum(results[best_alpha]) / len(results[best_alpha])
    print(f"\nΔ (best − base): {best_acc - base_acc:+.3f} при alpha={best_alpha}")
    if best_acc - base_acc > 0.15:
        print("Вердикт: ✅ резонанс при стиринге (дифф. вектор каузально влияет, OOD)")
    elif abs(best_acc - base_acc) < 0.05:
        print("Вердикт: ❌ нет резонанса (стиринг не влияет)")
    else:
        print("Вердикт: ⚠️ слабое влияние (нелинейный отклик)")

if __name__ == "__main__":
    main()