# -*- coding: utf-8 -*-
# temporal_sham.py — matched-sham control: temporal precedence on label-shuffled op-probe
import os, torch, random
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, OPS
from probe5 import TEST, BIG

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
STEER = r"E:\jspace\steer.pt"
SHAM = r"E:\jspace\steer_sham.pt"
SEED = 123

def ridge(X, Y, lam=1e-2):
    X, Y = X.float(), Y.float()
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

def build_sham(model, tok):
    """Обучаем op-зонд на перемешанных лейблах операций."""
    random.seed(SEED); torch.manual_seed(SEED)
    Xs, ys = [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        Xs.append(hs); ys += [OPS.index(o)] * hs.shape[0]
    X = torch.cat(Xs); y = torch.tensor(ys, device=DEVICE)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    Xn = (X - mu) / sd
    # one-hot лейблы -> перемешиваем по строкам
    oh = torch.eye(len(OPS), device=DEVICE)[y]
    perm = torch.randperm(oh.shape[0], device=DEVICE)
    oh_sham = oh[perm]
    W_op = ridge(Xn, oh_sham)
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), W_op=W_op.cpu()), SHAM)
    return mu, sd, W_op

def op_margin(h, muS, sdS, W_op):
    hn = ((h - muS) / sdS).unsqueeze(0).float()
    logits = (hn @ W_op.float()).squeeze()
    logits = logits - logits.mean()
    order = logits.argsort(descending=True)
    return (logits[order[0]] - logits[order[1:]].mean()).item(), OPS[order[0].item()]

def first_op_token(tok, cot_ids, target_op):
    markers = {"*": ["*", "умнож", "умн"], "+": ["+", "слож", "прибав", "плюс"],
               "-": ["-", "выч", "минус"], "/": ["/", "дел", "раздел"]}
    mk = markers.get(target_op, [])
    for i, tid in enumerate(cot_ids):
        t = tok.decode([tid]).lower()
        if any(m in t for m in mk):
            return i
    return len(cot_ids)

def run_lead(model, tok, proj, muS, sdS, W_op):
    leads = []
    for a, o, b in TEST:
        p = prompt_of(a, o, b)
        T = _j_loop(model, tok, p, proj, 48, 0)[0]
        margins, preds = [], []
        with torch.no_grad():
            for t in range(T.shape[0]):
                m, pr = op_margin(T[t].to(DEVICE), muS, sdS, W_op)
                margins.append(m); preds.append(pr)
        mx = max(margins) if max(margins) > 1e-9 else 1.0
        plateau = next((t for t, m in enumerate(margins) if m >= 0.8 * mx and preds[t] == o), None)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        verb = first_op_token(tok, out[0][ids.shape[1]:], o)
        leads.append((verb - plateau) if plateau is not None else None)
    return leads

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)
    d = torch.load(STEER, weights_only=True)

    print("=== REAL op-probe (из steer.pt) ===")
    real = run_lead(model, tok, proj, d["mu"].to(DEVICE), d["sd"].to(DEVICE), d["W_op"].to(DEVICE))
    for (a,o,b), l in zip(TEST, real):
        print(f"  {prompt_of(a,o,b):28s} lead={l}")
    rv = [l for l in real if l is not None]
    print(f"  средний lead REAL: {sum(rv)/len(rv):+.1f} ({sum(1 for l in rv if l>0)}/{len(rv)} precedes)")

    print("\n=== SHAM op-probe (перемешанные лейблы) ===")
    if os.path.exists(SHAM):
        g = torch.load(SHAM, weights_only=True)
        muS, sdS, Ws = g["mu"].to(DEVICE), g["sd"].to(DEVICE), g["W_op"].to(DEVICE)
    else:
        print("  строим sham-зонд (32 промта, ~1.5 мин)...")
        muS, sdS, Ws = build_sham(model, tok)
    sham = run_lead(model, tok, proj, muS, sdS, Ws)
    for (a,o,b), l in zip(TEST, sham):
        print(f"  {prompt_of(a,o,b):28s} lead={l}")
    sv = [l for l in sham if l is not None]
    print(f"  средний lead SHAM: {sum(sv)/len(sv):+.1f} ({sum(1 for l in sv if l>0)}/{len(sv)} precedes)")

    print(f"\nΔ (REAL − SHAM): {(sum(rv)/len(rv) - sum(sv)/len(sv)):+.1f} шагов")

if __name__ == "__main__":
    main()