# -*- coding: utf-8 -*-
import os, hashlib, torch, torch.nn as nn
import numpy as np
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop, SF_DIR
from probe import OPS, prompt_of, result_of, one_hot
from probe5 import BIG, TEST

PROJ_PATH = os.environ.get("PROJ", r"E:\jspace\proj_mlp.pt")
STEER = r"E:\jspace\steer.pt"
CACHE = r"E:\jspace\trace_cache"; os.makedirs(CACHE, exist_ok=True)

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

def states(model, tok, proj, items, gate, u):
    out = []
    for a, o, b in items:
        T = cached_trace(model, tok, proj, prompt_of(a, o, b), gate, u, 0.5)
        h = ((win(T).to(DEVICE) - gate["mu"]) / gate["sd"]).mean(0)
        out.append((h, a, o, b, result_of(a, o, b)))
    return out

def digits_of(v):
    v = int(abs(v)); return [v // 100, (v // 10) % 10, v % 10]

class CircularProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d) * 0.01)
        self.w2 = nn.Parameter(torch.randn(d) * 0.01)
    def forward(self, h):
        return torch.atan2(h @ self.w1, h @ self.w2) % (2 * torch.pi)
    def predict(self, h):
        return (self.forward(h) * 10 / (2 * torch.pi)).round().long() % 10

def train_circular(X, y_digit, epochs=500, lr=1e-3):
    probe = CircularProbe(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    target = y_digit.float() * 2 * torch.pi / 10
    for _ in range(epochs):
        opt.zero_grad()
        angles = probe(X)
        diff = (angles - target + torch.pi) % (2 * torch.pi) - torch.pi
        loss = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff))
        loss.backward(); opt.step()
    return probe

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)
    d = torch.load(STEER, weights_only=True)
    u = d["u"].to(DEVICE)
    gate = dict(mu=d["mu"].to(DEVICE), sd=d["sd"].to(DEVICE), W_abs=d["W_abs"].to(DEVICE), thr=0.20)

    seen = states(model, tok, proj, BIG, gate, u)
    X_seen = torch.stack([s[0] for s in seen])
    digits_seen = torch.tensor([digits_of(s[4]) for s in seen]).to(DEVICE)

    probes_c, probes_l = [], []
    for pos in range(3):
        y = digits_seen[:, pos]
        probes_c.append(train_circular(X_seen, y))
        probes_l.append(ridge_fit_device(X_seen, one_hot(y, 10)))

    def eval_group(title, items):
        data = states(model, tok, proj, items, gate, u)
        print(f"\n--- {title} ---")
        print(f"{'Prompt':38s} {'Exp':>5s} | circ [H,T,U] | lin [H,T,U] | совп")
        print("-" * 95)
        for h, a, o, b, expected in data:
            hb = h.unsqueeze(0)
            c = [probes_c[p].predict(hb).item() for p in range(3)]
            l = [int((hb @ probes_l[p]).argmax(-1).item()) for p in range(3)]
            c_num, l_num = c[0]*100 + c[1]*10 + c[2], l[0]*100 + l[1]*10 + l[2]
            match = "C" if c_num == expected else ("L" if l_num == expected else "-")
            print(f"{prompt_of(a,o,b):38s} {expected:5d} | [{c[0]:d},{c[1]:d},{c[2]:d}]      | "
                  f"[{l[0]:d},{l[1]:d},{l[2]:d}]      | {match}")

    eval_group("SEEN", BIG[:6])
    eval_group("TEST", TEST)

if __name__ == "__main__":
    main()