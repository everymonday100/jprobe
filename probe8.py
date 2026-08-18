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

class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)
    d = torch.load(STEER, weights_only=True)
    u = d["u"].to(DEVICE)
    gate = dict(mu=d["mu"].to(DEVICE), sd=d["sd"].to(DEVICE), W_abs=d["W_abs"].to(DEVICE), thr=0.20)

    def states(items):
        out = []
        for aa, o, bb in items:
            T = cached_trace(model, tok, proj, prompt_of(aa, o, bb), gate, u, 0.5)
            out.append((win(T).to(DEVICE), aa, o, bb))
        return out

    seen = states(BIG)
    X = torch.cat([s[0] for s in seen], 0)
    y_class = torch.cat([torch.full((s[0].shape[0],), OPS.index(s[2])) for s in seen]).long().to(DEVICE)
    y_reg = torch.cat([torch.tensor([[float(s[1]), float(s[3])]] * s[0].shape[0], device=DEVICE) for s in seen], 0)

    W_op = ridge_fit_device(X, one_hot(y_class, len(OPS)))
    pA = (y_reg[:, 0] - y_reg[:, 0].mean()) / y_reg[:, 0].std()
    pB = (y_reg[:, 1] - y_reg[:, 1].mean()) / y_reg[:, 1].std()
    W_A = ridge_fit_device(X, pA.unsqueeze(1))
    W_B = ridge_fit_device(X, pB.unsqueeze(1))

    mlp = MLPProbe(X.shape[1], len(OPS) + 2).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(200):
        opt.zero_grad(); pred = mlp(X)
        loss = (nn.functional.cross_entropy(pred[:, :len(OPS)], y_class)
                + nn.functional.mse_loss(pred[:, len(OPS)], y_reg[:, 0])
                + nn.functional.mse_loss(pred[:, len(OPS)+1], y_reg[:, 1]))
        loss.backward(); opt.step()

    A_mu, A_sd = y_reg[:, 0].mean().item(), y_reg[:, 0].std().item()
    B_mu, B_sd = y_reg[:, 1].mean().item(), y_reg[:, 1].std().item()
    def calc(op, A, B): return {"*":A*B, "+":A+B, "-":A-B, "/":A/B if B else float('inf')}[op]

    def eval_group(title, items):
        data = states(items)
        print(f"\n--- {title} ---")
        print(f"{'Prompt':38s} {'Exp':>5s} {'op_lin':>6s} {'A_lin':>7s} {'B_lin':>7s} {'Calc_lin':>8s} | "
              f"{'op_mlp':>6s} {'A_mlp':>7s} {'B_mlp':>7s} {'Calc_mlp':>8s}")
        print("-" * 140)
        for (aa, o, bb), (Wfull, _, _, _) in zip(items, data):
            h = Wfull.mean(0, keepdim=True)
            op_l = OPS[int((h @ W_op).argmax(-1).item())]
            A_l = float((h @ W_A).squeeze().item()) * A_sd + A_mu
            B_l = float((h @ W_B).squeeze().item()) * B_sd + B_mu
            with torch.no_grad():
                p = mlp(h)
                op_m = OPS[int(p[:, :len(OPS)].argmax(-1).item())]
                A_m, B_m = p[0, len(OPS)].item(), p[0, len(OPS)+1].item()
            print(f"{prompt_of(aa,o,bb):38s} {result_of(aa,o,bb):5d} "
                  f"{op_l:>6s} {A_l:7.1f} {B_l:7.1f} {calc(op_l,A_l,B_l):8.1f} | "
                  f"{op_m:>6s} {A_m:7.1f} {B_m:7.1f} {calc(op_m,A_m,B_m):8.1f}")

    eval_group("SEEN (контроль)", BIG[:6])
    eval_group("TEST (обобщение)", TEST)

if __name__ == "__main__":
    main()