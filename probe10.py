# -*- coding: utf-8 -*-
# probe10.py — probe-guided weighted voting по steering-роллаутам (ускоренный)
import os, hashlib, torch, torch.nn as nn
import numpy as np
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop, SF_DIR
from probe import OPS, prompt_of, result_of, one_hot
from probe5 import BIG, TEST

PROJ_PATH = os.environ.get("PROJ", r"E:\jspace\proj_mlp.pt")
STEER = r"E:\jspace\steer.pt"
CACHE = r"E:\jspace\trace_cache"; os.makedirs(CACHE, exist_ok=True)
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

def ridge_fit_device(X, Y, lam=1e-2):
    Y = Y.to(X.device)
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

def win(T):
    W = T[8:32]
    return W if W.shape[0] >= 4 else T

def _ckey(prompt, a):
    try: mt = str(os.path.getmtime(PROJ_PATH))
    except OSError: mt = "0"
    tag = os.path.basename(os.path.normpath(SF_DIR)) + os.path.basename(PROJ_PATH) + mt
    return hashlib.sha256((tag + prompt + str(a)).encode()).hexdigest()[:16]

def cached_trace(model, tok, proj, prompt, gate, u, a, max_steps=40):
    p = os.path.join(CACHE, _ckey(prompt, a) + ".npy")
    if os.path.exists(p): return torch.from_numpy(np.load(p))
    T = _j_loop(model, tok, prompt, proj, max_steps, 0,
                steer_u=u, steer_a=a, gate=gate if a > 0 else None)[0]
    np.save(p, T.numpy()); return T

class Readout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, len(OPS) + 2))
    def forward(self, x): return self.net(x)

def roll(model, tok, proj, prompt, gate, u, a):
    T = cached_trace(model, tok, proj, prompt, gate, u, a)
    return ((win(T).to(DEVICE) - gate["mu"]) / gate["sd"]).mean(0)

def calc_of(op, A, B):
    return {"*": A*B, "+": A+B, "-": A-B, "/": A/B if B else float("inf")}[op]

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)
    d = torch.load(STEER, weights_only=True)
    u = d["u"].to(DEVICE)
    gate = dict(mu=d["mu"].to(DEVICE), sd=d["sd"].to(DEVICE), W_abs=d["W_abs"].to(DEVICE), thr=0.20)

    seen = [(roll(model, tok, proj, prompt_of(a, o, b), gate, u, al), a, o, b)
            for a, o, b in BIG for al in ALPHAS]
    X = torch.stack([s[0] for s in seen])
    yop = torch.tensor([OPS.index(s[2]) for s in seen]).to(DEVICE)
    yAB = torch.tensor([[float(s[1]), float(s[3])] for s in seen]).to(DEVICE)
    ro = Readout(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(ro.parameters(), lr=1e-3)
    for ep in range(200):
        opt.zero_grad(); p = ro(X)
        loss = nn.functional.cross_entropy(p[:, :4], yop) + nn.functional.mse_loss(p[:, 4:], yAB)
        loss.backward(); opt.step()

    with torch.no_grad():
        P = ro(X)
        calcs = torch.tensor([calc_of(OPS[int(q[:4].argmax())], q[4].item(), q[5].item()) for q in P]).to(DEVICE)
        true = torch.tensor([result_of(s[1], s[2], s[3]) for s in seen]).float().to(DEVICE)
        lab = ((calcs - true).abs() <= 0.2 * true.abs().clamp_min(1)).long()
    W_c = ridge_fit_device(X, one_hot(lab, 2))

    print(f"{'Prompt':36s} {'Exp':>5s} {'vote':>8s} | rollouts (op,calc,weight)")
    print("-" * 110)
    for a, o, b in TEST:
        ws, cs, ops = [], [], []
        for al in ALPHAS:
            h = roll(model, tok, proj, prompt_of(a, o, b), gate, u, al)
            with torch.no_grad():
                p = ro(h.unsqueeze(0))[0]
                w = torch.softmax((h @ W_c).unsqueeze(0), -1)[0, 1].item()
            ops.append(OPS[int(p[:4].argmax())])
            cs.append(calc_of(ops[-1], p[4].item(), p[5].item())); ws.append(w)
        vote = sum(w*c for w, c in zip(ws, cs)) / max(sum(ws), 1e-9)
        print(f"{prompt_of(a,o,b):36s} {result_of(a,o,b):5d} {vote:8.1f} | "
              + " ".join(f"{q}{c:.0f}({w:.2f})" for q, c, w in zip(ops, cs, ws)))

if __name__ == "__main__":
    main()