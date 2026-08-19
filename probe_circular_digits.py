# -*- coding: utf-8 -*-
# probe_circular_digits.py — фазовый (циркулярный) readout цифр по Sun et al.
import os, torch, math
from jspace import load, DEVICE, DTYPE
from probe import prompt_of, result_of
from probe5 import TEST, BIG

CIRC = r"E:\jspace\circ_digits.pt"

def fit_circle(X, y):
    """Для каждой цифры d: пара (cos_d, sin_d) + bias из ridge по one-hot присутствия."""
    X = X.float(); n, dim = X.shape
    cos_w, sin_w, cos_b, sin_b = [], [], [], []
    for d in range(10):
        target = y[:, d].float()                       # присутствие цифры d
        Xd = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
        # две независимые ridge-регрессии -> cos/sin компоненты
        I = torch.eye(dim + 1, device=X.device) * 1e-2
        G = Xd.T @ Xd + I
        wc = torch.linalg.solve(G, Xd.T @ torch.cos(target * math.pi))
        ws = torch.linalg.solve(G, Xd.T @ torch.sin(target * math.pi))
        cos_w.append(wc[:-1]); sin_w.append(ws[:-1])
        cos_b.append(wc[-1]); sin_b.append(ws[-1])
    return (torch.stack(cos_w), torch.stack(sin_w),
            torch.tensor(cos_b, device=X.device), torch.tensor(sin_b, device=X.device))

def build(model, tok):
    Xs, Ys = [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b); r = result_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        present = torch.tensor([float(str(d) in f"{a}{b}{r}") for d in range(10)], device=DEVICE)
        Xs.append(hs); Ys.append(present.unsqueeze(0).expand(hs.shape[0], -1))
    X = torch.cat(Xs); Y = torch.cat(Ys)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    cw, sw, cb, sb = fit_circle((X - mu) / sd, Y)
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), cw=cw.cpu(), sw=sw.cpu(),
                    cb=cb.cpu(), sb=sb.cpu()), CIRC)
    return mu, sd, cw, sw, cb, sb

def readout(h, mu, sd, cw, sw, cb, sb):
    hn = ((h - mu) / sd).unsqueeze(0).float()      # [1, dim], float32
    c = (hn @ cw.T).squeeze() + cb                  # [10]
    s = (hn @ sw.T).squeeze() + sb                  # [10]
    amp = torch.sqrt(c*c + s*s)                     # уверенность
    phase = torch.atan2(s, c)                       # фаза на окружности
    return amp, phase

def main():
    model, tok = load()
    if os.path.exists(CIRC):
        g = torch.load(CIRC, weights_only=True)
        mu, sd = g["mu"].to(DEVICE), g["sd"].to(DEVICE)
        cw, sw = g["cw"].to(DEVICE), g["sw"].to(DEVICE)
        cb, sb = g["cb"].to(DEVICE), g["sb"].to(DEVICE)
    else:
        print("Обучаем циркулярные digit-зонды (32 промта, ~1.5 мин)...")
        mu, sd, cw, sw, cb, sb = build(model, tok)

    for a, o, b in TEST[:2]:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        cot = out[0][ids.shape[1]:]
        full = torch.cat([ids[0], cot]).unsqueeze(0)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        pl = ids.shape[1]
        truth = sorted(set(int(d) for d in f"{a}{b}{result_of(a,o,b)}"))
        print(f"\n=== {p} (truth digits {truth}) ===")
        with torch.no_grad():
            for i, tid in enumerate(cot):
                txt = tok.decode([tid]).strip()
                if not txt or not any(ch.isdigit() for ch in txt): continue
                h = hs[pl + i]
                amp, phase = readout(h, mu, sd, cw, sw, cb, sb)
                top = amp.topk(4).indices.tolist()
                active = [d for d in range(10) if amp[d] > amp.mean() + amp.std()]
                print(f"  tok {i:3d} {txt!r:6s} top_amp={top} active={active} "
                      f"phase={[f'{phase[d]:+.2f}' for d in top]}")

if __name__ == "__main__":
    main()