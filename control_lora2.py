# -*- coding: utf-8 -*-
# control_lora2.py — контроль согласования latent↔surface на выровненной LoRA-модели,
# зонды обучены на её собственных трейсах
import torch
from jspace import load, DEVICE, DTYPE
from finetune_math import inject_lora, last_number, LORA_PATH
from probe import prompt_of, result_of, corr, OPS
from probe5 import TEST, BIG, BINS

def ridge_fit_device(X, Y, lam=1e-2):
    Y = Y.to(X.device)
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

def one_hot_device(y, k):
    return torch.eye(k, dtype=torch.float32, device=y.device)[y.long()]

def apply_lora(model):
    params = inject_lora(model)
    saved = torch.load(LORA_PATH, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s.to(DEVICE))

def trace(model, tok, text):
    full = tok.encode(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model.model(full, output_hidden_states=True).hidden_states[-1][0]

def gen_answer(model, tok, p, max_new=160):
    ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def main():
    model, tok = load()
    apply_lora(model)

    # ---------- обучаем зонды на трейсах LoRA-модели (BIG) ----------
    Xs, yop, yA, yB = [], [], [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ans = gen_answer(model, tok, p)
        hs = trace(model, tok, p + "\n" + ans)
        n = hs.shape[0]
        Xs.append(hs)
        yop += [OPS.index(o)] * n; yA += [float(a)] * n; yB += [float(b)] * n
    X = torch.cat(Xs)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    Xn = ((X - mu) / sd).float()
    yo = torch.tensor(yop, dtype=torch.long).to(DEVICE)
    ya = torch.tensor(yA, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(yB, dtype=torch.float32).to(DEVICE)
    W_op = ridge_fit_device(Xn, one_hot_device(yo, len(OPS)))
    W_A  = ridge_fit_device(Xn, (ya - ya.mean()).unsqueeze(1))
    W_B  = ridge_fit_device(Xn, (yb - yb.mean()).unsqueeze(1))

    # ---------- eval на TEST (LoRA) ----------
    Xs, Ss, yop, yA, yB, surf_ok = [], [], [], [], [], 0
    for a, o, b in TEST:
        p = prompt_of(a, o, b)
        ans = gen_answer(model, tok, p)
        surf_ok += (last_number(ans) == result_of(a, o, b))
        hs = trace(model, tok, p + "\n" + ans)
        n = hs.shape[0]
        Xs.append(((hs - mu) / sd).float())
        Ss.append(torch.arange(n))
        yop += [OPS.index(o)] * n; yA += [float(a)] * n; yB += [float(b)] * n
    X = torch.cat(Xs); S = torch.cat(Ss)
    yo = torch.tensor(yop, dtype=torch.long).to(DEVICE)
    ya = torch.tensor(yA, dtype=torch.float32).to(DEVICE)
    yb = torch.tensor(yB, dtype=torch.float32).to(DEVICE)

    print(f"surface acc (TEST): {surf_ok/len(TEST):.2f}")
    print(f"{'bin':>8s} {'op':>5s} {'A':>6s} {'B':>6s}")
    for lo, hi in BINS:
        m = (S >= lo) & (S < hi)
        if m.sum() < 5: continue
        Xb = X[m]
        opacc = ((Xb @ W_op).argmax(1) == one_hot_device(yo[m], len(OPS)).argmax(1)).float().mean().item()
        print(f"{lo:3d}-{hi:3d}  {opacc:.2f} {corr((Xb @ W_A).squeeze(), ya[m]):+.2f} {corr((Xb @ W_B).squeeze(), yb[m]):+.2f}")

if __name__ == "__main__":
    main()