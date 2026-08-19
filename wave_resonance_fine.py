# -*- coding: utf-8 -*-
# wave_resonance_fine.py — fine-grained alpha sweep + sham control
import os, re, torch, random
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, result_of
from probe5 import TEST

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
SEED_SHAM = 12345

def get_trace(model, tok, proj, a, o, b, steps=48):
    p = prompt_of(a, o, b)
    return _j_loop(model, tok, p, proj, steps, 0)[0]

def extract_diff_vector(model, tok, proj, a, b):
    T_sum = get_trace(model, tok, proj, a+b, '+', 0)
    T_diff = get_trace(model, tok, proj, a-b, '-', 0)
    min_len = min(T_sum.shape[0], T_diff.shape[0])
    diff = (T_sum[:min_len] - T_diff[:min_len]).mean(dim=0)
    return diff / diff.norm()

def gen_with_steering(model, tok, prompt, steer_vec, alpha, layer_idx=18):
    ids = tok.apply_chat_template([{"role":"user","content":prompt}],
                                  add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:  # только промпт
            h = h + alpha * steer_vec.to(h.device, h.dtype).unsqueeze(0).unsqueeze(0)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    layers = model.model.layers
    handle = layers[min(layer_idx, len(layers)-1)].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.findall(r"=\s*(-?\d+)", txt) or re.findall(r"-?\d+", txt)
        return int(m[-1]) if m else None
    finally:
        handle.remove()

def run_sweep(model, tok, proj, vec, label):
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"\n=== {label} ===")
    print(f"{'Prompt':28s} {'exp':>4s} | " + " ".join(f"a={a:<4.1f}" for a in alphas))
    print("-" * 70)
    all_answers = {a: [] for a in alphas}
    for a, o, b in TEST:
        p = prompt_of(a, o, b); exp = result_of(a, o, b)
        row = []
        for alpha in alphas:
            got = gen_with_steering(model, tok, p, vec, alpha)
            all_answers[alpha].append(got)
            row.append(got)
        print(f"{p:28s} {exp:>4d} | " + " ".join(f"{str(v):>6s}" for v in row))
    return alphas, all_answers

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)

    # REAL: дифф. вектор из (42,19) вне TEST
    SRC_A, SRC_B = 42, 19
    print(f"Извлечение REAL дифф. вектора ({SRC_A}±{SRC_B})...")
    real_vec = extract_diff_vector(model, tok, proj, SRC_A, SRC_B)

    # SHAM: случайный вектор той же нормы
    random.seed(SEED_SHAM); torch.manual_seed(SEED_SHAM)
    sham_vec = torch.randn_like(real_vec)
    sham_vec = sham_vec / sham_vec.norm()

    alphas_r, ans_r = run_sweep(model, tok, proj, real_vec, "REAL (diff vector)")
    alphas_s, ans_s = run_sweep(model, tok, proj, sham_vec, "SHAM (random vector)")

    # Анализ: плавность сдвига для 58+27 (ключевой пример)
    print("\n=== Анализ ключевого примера 58+27 (exp=85) ===")
    idx_5827 = next(i for i,(a,_,b) in enumerate(TEST) if a==58 and b==27)
    print(f"  REAL: {[ans_r[a][idx_5827] for a in alphas_r]}")
    print(f"  SHAM: {[ans_s[a][idx_5827] for a in alphas_s]}")

    # Монотонность/плавность: считаем число уникальных ответов по альфа
    real_unique = len(set(ans_r[a][idx_5827] for a in alphas_r))
    sham_unique = len(set(ans_s[a][idx_5827] for a in alphas_s))
    print(f"\n  Уникальных ответов (из 5 α): REAL={real_unique}, SHAM={sham_unique}")
    if real_unique >= 3 and sham_unique <= 2:
        print("  Вердикт: ✅ REAL даёт плавный сдвиг, SHAM — нет (специфично)")
    elif real_unique <= 2:
        print("  Вердикт: ❌ REAL тоже дискретен (нет плавного сдвига)")
    else:
        print("  Вердикт: ⚠️ оба дают вариацию (нужен дополнительный контроль)")

if __name__ == "__main__":
    main()