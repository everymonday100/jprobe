# -*- coding: utf-8 -*-
# rmt_filter_probe.py — adaptive RMT filter + honest out-of-sample evaluation
import torch, numpy as np
from sklearn.model_selection import KFold
from jspace import load, DEVICE
from probe import prompt_of, corr
from probe5 import BIG

def ridge(X, Y, lam=1.0):
    X, Y = X.float(), Y.float()
    return torch.linalg.solve(X.T @ X + lam*torch.eye(X.shape[1], device=X.device), X.T @ Y)

class RMTActivationFilter:
    """Адаптивный MP-фильтр: Gram N×N, порог = медиана спектра × k."""
    def __init__(self, mult: float = 3.0, top_k_fallback: int = 4):
        self.mult = mult
        self.top_k = top_k_fallback

    def fit_transform(self, X: torch.Tensor):
        X = X.float()
        Xc = X - X.mean(0, keepdim=True)
        N, P = Xc.shape

        # Грам N×N (стабильнее, чем P×P при N<<P)
        Gram = (Xc @ Xc.T) / P
        eig, _ = torch.linalg.eigh(Gram)
        eig = torch.flip(eig, dims=[0])              # по убыванию
        adaptive_threshold = torch.median(eig) * self.mult

        # SVD исходной центрированной матрицы (шкала eigenvalues = S²/P, как у Gram)
        U, S, V = torch.linalg.svd(Xc, full_matrices=False)
        eig_svd = (S ** 2) / P                        # согласовано с Gram
        signal_mask = eig_svd > adaptive_threshold

        n_sig = int(signal_mask.sum())
        if not signal_mask.any():
            signal_mask[:self.top_k] = True           # GWT fallback топ-4
            n_sig = self.top_k

        S_f = S.clone(); S_f[~signal_mask] = 0.0
        X_rec = U @ torch.diag(S_f) @ V
        return X_rec + X.mean(0, keepdim=True), n_sig

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
    """Честная out-of-sample корреляция через k-fold CV."""
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    vals = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]; Atr, Ate = A[tr], A[te]
        m, s = Xtr.mean(0), Xtr.std(0).clamp_min(1e-3)
        W = ridge((Xtr-m)/s, (Atr-Atr.mean()).unsqueeze(1), lam)
        pred = ((Xte-m)/s) @ W
        vals.append(corr(pred.squeeze(), Ate))
    return float(np.mean(vals))

def main():
    model, tok = load()
    layers = list(range(14, 28))
    print(f"Сбор активаций {len(BIG)} промптов (последний токен) на слоях {layers[0]}–{layers[-1]}...")
    data = collect(model, tok, layers)
    A = torch.tensor([float(a) for a,o,b in BIG])
    rmt = RMTActivationFilter(mult=3.0, top_k_fallback=4)

    print(f"\n{'Layer':>5s} {'OOS_pre':>8s} {'OOS_post':>9s} {'Δ':>7s} {'n_sig':>6s} | verdict")
    print("-"*50)
    deltas, sigs = [], []
    for L in layers:
        X = torch.stack(data[L])
        pre  = oos_corr(X, A)
        Xf, n_sig = rmt.fit_transform(X)
        post = oos_corr(Xf, A)
        d = post - pre
        deltas.append(d); sigs.append(n_sig)
        mk = "✅" if d > 0.02 else ("⚠️" if d > -0.02 else "❌")
        print(f"{L:>5d} {pre:>+8.3f} {post:>+9.3f} {d:>+7.3f} {n_sig:>6d} | {mk}")

    ad, asig = np.mean(deltas), np.mean(sigs)
    print(f"\n📊 Средний OOS Δ: {ad:+.3f}")
    print(f"📊 Среднее число сигнальных компонент: {asig:.1f}")
    if asig > 0:
        print(f"🔗 n_signal≈{asig:.0f} vs ~4 chunks GWT: {'✅ совпадает' if abs(asig-4)<=2 else '⚠️ расходится'}")
    print("\n✅ RMT помогает обобщению" if ad > 0.02 else
          ("⚠️ RMT нейтрален" if abs(ad) < 0.02 else "❌ RMT вредит"))

if __name__ == "__main__":
    main()