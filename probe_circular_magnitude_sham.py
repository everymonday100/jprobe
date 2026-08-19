# -*- coding: utf-8 -*-
# probe_circular_magnitude_sham.py — matched-sham control for circular magnitude probe
import os, torch, math, random
from jspace import load, DEVICE, DTYPE
from probe import prompt_of, result_of
from probe5 import TEST, BIG

CIRC_MAG = r"E:\jspace\circ_magnitude.pt"
CIRC_MAG_SHAM = r"E:\jspace\circ_magnitude_sham.pt"
BASE = 10
SEED = 777

def fit_circle_magnitude(X, values):
    X = X.float(); n, dim = X.shape
    phases = 2 * math.pi * (values % BASE) / BASE
    cos_target = torch.cos(phases); sin_target = torch.sin(phases)
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
    I = torch.eye(dim + 1, device=X.device) * 1e-2
    G = Xb.T @ Xb + I
    wc = torch.linalg.solve(G, Xb.T @ cos_target)
    ws = torch.linalg.solve(G, Xb.T @ sin_target)
    return wc[:-1], ws[:-1], wc[-1], ws[-1]

def collect(model, tok):
    Xs, vals = [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        Xs.append(hs); vals += [float(a)] * hs.shape[0]
    X = torch.cat(Xs); V = torch.tensor(vals, device=DEVICE)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    return (X - mu) / sd, V, mu, sd

def build_sham(Xn, V, mu, sd):
    random.seed(SEED); torch.manual_seed(SEED)
    perm = torch.randperm(V.shape[0], device=DEVICE)
    cw, sw, cb, sb = fit_circle_magnitude(Xn, V[perm])
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), cw=cw.cpu(), sw=sw.cpu(),
                    cb=cb.cpu(), sb=sb.cpu(), base=BASE), CIRC_MAG_SHAM)
    return cw, sw, cb, sb

def diagnose(model, tok, mu, sd, cw, sw, cb, sb, label):
    amps, phases = [], []
    for a, o, b in TEST[:4]:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        cot = out[0][ids.shape[1]:]
        full = torch.cat([ids[0], cot]).unsqueeze(0)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        pl = ids.shape[1]
        with torch.no_grad():
            for i in range(0, len(cot), max(1, len(cot)//8)):
                h = hs[pl + i]
                hn = ((h - mu) / sd).unsqueeze(0).float()
                c = (hn @ cw.unsqueeze(0).T).squeeze() + cb
                s = (hn @ sw.unsqueeze(0).T).squeeze() + sb
                amps.append(torch.sqrt(c*c + s*s).item())
                phases.append(torch.atan2(s, c).item())
    amps_t = torch.tensor(amps); phases_t = torch.tensor(phases)
    sectors = ((phases_t + math.pi) / (2*math.pi) * 8).long().clamp(0, 7)
    uniq = sectors.unique().numel()
    print(f"\n=== {label} ===")
    print(f"  amp: mean={amps_t.mean():.3f} std={amps_t.std():.3f}")
    print(f"  phase range: [{phases_t.min():+.2f}, {phases_t.max():+.2f}]")
    print(f"  уникальных секторов (из 8): {uniq} "
          f"({'✅ гладкая окружность' if uniq >= 5 else '❌ схлопывание/шум'})")
    return uniq, amps_t.mean().item()

def main():
    model, tok = load()
    print("Сбор teacher-трейсов (32 промта, ~1.5 мин)...")
    Xn, V, mu, sd = collect(model, tok)

    # REAL: из сохранённого circ_magnitude.pt (переобучим на тех же данных для честности)
    print("\nОбучаем REAL циркулярный зонд...")
    cw_r, sw_r, cb_r, sb_r = fit_circle_magnitude(Xn, V)
    uniq_r, amp_r = diagnose(model, tok, mu, sd, cw_r, sw_r, cb_r, sb_r, "REAL (величина mod 10)")

    # SHAM: перемешанные значения
    if os.path.exists(CIRC_MAG_SHAM):
        g = torch.load(CIRC_MAG_SHAM, weights_only=True)
        cw_s, sw_s, cb_s, sb_s = g["cw"].to(DEVICE), g["sw"].to(DEVICE), g["cb"].to(DEVICE), g["sb"].to(DEVICE)
        mu_s, sd_s = g["mu"].to(DEVICE), g["sd"].to(DEVICE)
    else:
        print("\nСтроим SHAM циркулярный зонд (перемешанные величины)...")
        cw_s, sw_s, cb_s, sb_s = build_sham(Xn, V, mu, sd)
        mu_s, sd_s = mu, sd
    uniq_s, amp_s = diagnose(model, tok, mu_s, sd_s, cw_s, sw_s, cb_s, sb_s, "SHAM (перемешанные величины)")

    print(f"\nΔ уникальных секторов: {uniq_r - uniq_s:+d}")
    spec = (uniq_r >= 5) and (uniq_r - uniq_s >= 2)
    print(f"Вердикт: {'✅ геометрия специфична (окружность реальна)' if spec else '❌ окружность не специфична'}")

if __name__ == "__main__":
    main()