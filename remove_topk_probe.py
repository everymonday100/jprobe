# -*- coding: utf-8 -*-
# remove_topk_probe.py — вычесть доминирующие компоненты, мерить читаемость операнда
import torch, numpy as np
from sklearn.model_selection import KFold
from jspace import load, DEVICE
from probe import prompt_of, corr
from probe5 import BIG

def ridge(X, Y, lam=1.0):
    X, Y = X.float(), Y.float()
    return torch.linalg.solve(X.T @ X + lam*torch.eye(X.shape[1], device=X.device), X.T @ Y)

def remove_topk(X, k):
    """Занулить топ-k сингулярных компонент, реконструировать остаток."""
    if k == 0:
        return X
    Xc = X - X.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    S_f = S.clone(); S_f[:k] = 0.0          # вычитаем доминирующие
    return U @ torch.diag(S_f) @ V + X.mean(0, keepdim=True)

def collect(model, tok, layers):
    data = {l: [] for l in layers}
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        cur = {}
        def mk(li):
            def hk(m,i,o):
                h = o[0] if isinstance(o, tuple) else o
                cur[li] = h[0,-1,:].detach().cpu().float()
            return hk
        hs = [model.model.layers[l].register_forward_hook(mk(l)) for l in layers]
        try:
            with torch.no_grad(): model(ids)
        finally:
            for h in hs: h.remove()
        for l in layers: data[l].append(cur[l])
    return data

def oos_corr(X, A, lam=1.0, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    vals = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]; Atr, Ate = A[tr], A[te]
        m, s = Xtr.mean(0), Xtr.std(0).clamp_min(1e-3)
        W = ridge((Xtr-m)/s, (Atr-Atr.mean()).unsqueeze(1), lam)
        vals.append(corr(((Xte-m)/s) @ W).squeeze() if False else corr((((Xte-m)/s) @ W).squeeze(), Ate))
    return float(np.mean(vals))

def main():
    model, tok = load()
    layers = list(range(14, 28))
    print(f"Сбор активаций {len(BIG)} промптов на слоях {layers[0]}–{layers[-1]}...")
    data = collect(model, tok, layers)
    A = torch.tensor([float(a) for a,o,b in BIG])

    ks = [0, 4, 8, 17, 32, 64]
    print(f"\n{'Layer':>5s} | " + " ".join(f"k={k:<3d}" for k in ks))
    print("-"*60)
    mean_by_k = {k: [] for k in ks}
    for L in layers:
        X = torch.stack(data[L])
        row = []
        for k in ks:
            Xr = remove_topk(X, k)
            c = oos_corr(Xr, A)
            mean_by_k[k].append(c)
            row.append(c)
        print(f"{L:>5d} | " + " ".join(f"{c:>5.3f}" for c in row))

    print(f"\n{'Средн':>5s} | " + " ".join(f"{np.mean(mean_by_k[k]):>5.3f}" for k in ks))
    base = np.mean(mean_by_k[0])
    best_k = max(ks, key=lambda k: np.mean(mean_by_k[k]))
    best = np.mean(mean_by_k[best_k])
    print(f"\n📊 Базовый (k=0): {base:.3f}   Лучший: k={best_k} → {best:.3f}   Δ={best-base:+.3f}")
    if best - base > 0.02:
        print(f"✅ Удаление топ-{best_k} УСИЛИВАЕТ читаемость операнда (синтаксис маскировал)")
    elif abs(best - base) < 0.02:
        print("⚠️ Синтаксис не мешает — удаление топ-k нейтрально")
    else:
        print("❌ Удаление топ-k ВРЕДИТ — доминирующие компоненты несут и сигнал тоже")

if __name__ == "__main__":
    main()