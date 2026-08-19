# -*- coding: utf-8 -*-
# probe_circular_magnitude.py — циркулярный зонд по ВЕЛИЧИНЕ (не по наличию цифры)
# Проверяет гипотезу: число кодируется как фаза на окружности mod base
import os, torch, math
from jspace import load, DEVICE, DTYPE
from probe import prompt_of, result_of
from probe5 import TEST, BIG

CIRC_MAG = r"E:\jspace\circ_magnitude.pt"
BASE = 10  # модуль окружности; можно менять (10, 20, 100)

def fit_circle_magnitude(X, values):
    """Для каждого остатка mod BASE: пара (cos, sin) через ridge по фазовому лейблу."""
    X = X.float(); n, dim = X.shape
    phases = 2 * math.pi * (values % BASE) / BASE   # непрерывная фаза
    cos_target = torch.cos(phases)
    sin_target = torch.sin(phases)
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
    I = torch.eye(dim + 1, device=X.device) * 1e-2
    G = Xb.T @ Xb + I
    wc = torch.linalg.solve(G, Xb.T @ cos_target)
    ws = torch.linalg.solve(G, Xb.T @ sin_target)
    return wc[:-1], ws[:-1], wc[-1], ws[-1]

def build(model, tok):
    Xs, vals = [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        # каждый шаг трейса несёт величину A (константа на трейс)
        Xs.append(hs); vals += [float(a)] * hs.shape[0]
    X = torch.cat(Xs); V = torch.tensor(vals, device=DEVICE)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    cw, sw, cb, sb = fit_circle_magnitude((X - mu) / sd, V)
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), cw=cw.cpu(), sw=sw.cpu(),
                    cb=cb.cpu(), sb=sb.cpu(), base=BASE), CIRC_MAG)
    return mu, sd, cw, sw, cb, sb

def readout(h, mu, sd, cw, sw, cb, sb):
    hn = ((h - mu) / sd).unsqueeze(0).float()
    c = (hn @ cw.unsqueeze(0).T).squeeze() + cb
    s = (hn @ sw.unsqueeze(0).T).squeeze() + sb
    amp = torch.sqrt(c*c + s*s)
    phase = torch.atan2(s, c)                       # [-pi, pi]
    decoded_value = (phase / (2*math.pi)) * BASE     # восстановленное значение mod BASE
    return amp.item(), phase.item(), decoded_value.item()

def main():
    model, tok = load()
    if os.path.exists(CIRC_MAG):
        g = torch.load(CIRC_MAG, weights_only=True)
        mu, sd = g["mu"].to(DEVICE), g["sd"].to(DEVICE)
        cw, sw, cb, sb = g["cw"].to(DEVICE), g["sw"].to(DEVICE), g["cb"].to(DEVICE), g["sb"].to(DEVICE)
        base = g["base"]
    else:
        print(f"Обучаем циркулярный зонд по величине mod {BASE} (32 промта, ~1.5 мин)...")
        mu, sd, cw, sw, cb, sb = build(model, tok)
        base = BASE

    print(f"\n=== Циркулярный readout по величине mod {base} ===")
    print(f"{'prompt':28s} {'true_A':>6s} {'true_A%base':>11s} | {'step':>4s} {'amp':>5s} {'phase':>7s} {'decoded':>7s}")
    print("-" * 80)
    amps_all, phases_all = [], []
    for a, o, b in TEST[:4]:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        cot = out[0][ids.shape[1]:]
        full = torch.cat([ids[0], cot]).unsqueeze(0)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        pl = ids.shape[1]
        true_mod = a % base
        print(f"{p:28s} {a:>6d} {true_mod:>11d} |")
        with torch.no_grad():
            for i, tid in enumerate(cot[::max(1,len(cot)//8)]):  # 8 сэмплов на трейс
                h = hs[pl + i]
                amp, phase, dec = readout(h, mu, sd, cw, sw, cb, sb)
                amps_all.append(amp); phases_all.append(phase)
                print(f"{'':>28s} {'':>6s} {'':>11s} | {i:>4d} {amp:>5.2f} {phase:>+7.2f} {dec:>+7.1f}")

    # Диагностика: амплитуды должны быть > шума, фазы — НЕ бинарными
    amps_t = torch.tensor(amps_all)
    phases_t = torch.tensor(phases_all)
    print(f"\nДиагностика:")
    print(f"  amp: mean={amps_t.mean():.3f} std={amps_t.std():.3f} min={amps_t.min():.3f}")
    print(f"  phase range: [{phases_t.min():+.2f}, {phases_t.max():+.2f}] "
          f"(должно быть широко, не только ±π/0)")
    # мера «небинарности»: сколько уникальных квантованных фаз (из 8 секторов)
    sectors = ((phases_t + math.pi) / (2*math.pi) * 8).long().clamp(0,7)
    unique_sectors = sectors.unique().numel()
    print(f"  уникальных фазовых секторов (из 8): {unique_sectors} "
          f"({'✅ гладкая окружность' if unique_sectors >= 5 else '❌ схлопывание'})")

if __name__ == "__main__":
    main()