# -*- coding: utf-8 -*-
import os, hashlib, torch
import numpy as np
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop, SF_DIR
from probe import OPS, prompt_of, result_of, one_hot, acc, corr
from probe2 import teacher_trace_labeled

PROJ_PATH = os.environ.get("PROJ", r"E:\jspace\proj_mlp.pt")
STEER = r"E:\jspace\steer.pt"
CACHE = r"E:\jspace\trace_cache"; os.makedirs(CACHE, exist_ok=True)
BINS = [(0, 8), (8, 16), (16, 32), (32, 999)]

BIG = [(17,"*",23),(12,"*",8),(7,"*",8),(14,"*",6),
       (45,"+",67),(28,"+",35),(56,"+",19),(74,"+",18),
       (96,"-",58),(73,"-",26),(85,"-",47),(64,"-",38),
       (144,"/",12),(96,"/",6),(81,"/",9),(72,"/",8)]
# после дособора корпуса раскомментируйте:
BIG += [(18,"*",5),(22,"*",4),(35,"+",48),(66,"+",27),
        (91,"-",37),(77,"-",29),(132,"/",11),(98,"/",7),
        (16,"*",9),(27,"*",3),(49,"+",52),(84,"+",19),
        (95,"-",68),(88,"-",39),(126,"/",9),(63,"/",7)]
BIG += [(19,"*",4),(23,"*",6),(31,"*",7),(26,"*",5),
        (17,"*",8),(24,"*",7),(35,"*",6),(42,"*",9),
        (57,"+",36),(68,"+",25),(47,"+",46),(79,"+",14),
        (86,"+",38),(53,"+",29),(92,"+",47),(61,"+",58),
        (94,"-",37),(82,"-",56),(75,"-",48),(63,"-",25),
        (97,"-",69),(84,"-",57),(71,"-",34),(90,"-",46),
        (112,"/",8),(135,"/",9),(108,"/",12),(91,"/",7),
        (114,"/",6),(128,"/",8),(117,"/",9),(144,"/",16)]
TEST = [(13,"*",7),(25,"+",39),(83,"-",46),(96,"/",8),(15,"*",6),(58,"+",27)]

def ridge_fit_device(X, Y, lam=1e-2):
    Y = Y.to(X.device)
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

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

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)

    tr = []
    for a, o, b in BIG:
        print("teacher:", prompt_of(a, o, b))
        T, Ab = teacher_trace_labeled(model, tok, prompt_of(a, o, b))
        tr.append(dict(X=T, op=OPS.index(o), A=float(a), B=float(b), abs=Ab))
    Xraw = torch.cat([d["X"] for d in tr], 0)
    mu, sd = Xraw.mean(0), Xraw.std(0).clamp_min(1e-3)
    def norm(T): return (T - mu) / sd
    Xtr = norm(Xraw)
    yop  = torch.cat([torch.full((d["X"].shape[0],), d["op"]) for d in tr]).long()
    yA   = torch.cat([d["X"].new_full((d["X"].shape[0],), d["A"]) for d in tr])
    yB   = torch.cat([d["X"].new_full((d["X"].shape[0],), d["B"]) for d in tr])
    yabs = torch.cat([d["abs"] for d in tr], 0).long()
    W_op  = ridge_fit_device(Xtr, one_hot(yop, 4))
    W_A   = ridge_fit_device(Xtr, (yA - yA.mean()).unsqueeze(1))
    W_B   = ridge_fit_device(Xtr, (yB - yB.mean()).unsqueeze(1))
    W_abs = ridge_fit_device(Xtr, one_hot(yabs, 2))
    M = torch.cat([W_A, W_B, W_op], 1)
    u = W_abs[:, 1] - W_abs[:, 0]
    u_orth = (u - M @ torch.linalg.lstsq(M, u).solution)
    u_orth = (u_orth / u_orth.norm()).to(DEVICE)
    torch.save(dict(u=u_orth.cpu(), mu=mu, sd=sd, W_op=W_op.cpu(), W_A=W_A.cpu(),
                    W_B=W_B.cpu(), W_abs=W_abs.cpu()), STEER)
    gate = dict(mu=mu.to(DEVICE), sd=sd.to(DEVICE), W_abs=W_abs.to(DEVICE), thr=0.20)

    def eval_group(title, items, su, sa):
        Xs, Ss, Gs = [], [], []
        for aa, o, bb in items:
            T = cached_trace(model, tok, proj, prompt_of(aa, o, bb), gate, su, sa, max_steps=128)
            n = T.shape[0]
            Xs.append(norm(T)); Ss.append(torch.arange(n))
            P = (norm(T) @ W_abs).argmax(1)
            Gs.append(int((P == 1).nonzero(as_tuple=True)[0][0]) if (P == 1).any() else n)
        X = torch.cat(Xs, 0); S = torch.cat(Ss, 0)
        P = (X @ W_abs).argmax(1)
        yop = torch.cat([t.new_full((t.shape[0],), OPS.index(o)) for t, (aa, o, bb) in zip(Xs, items)]).long()
        yA  = torch.cat([t.new_full((t.shape[0],), float(aa)) for t, (aa, o, bb) in zip(Xs, items)])
        yB  = torch.cat([t.new_full((t.shape[0],), float(bb)) for t, (aa, o, bb) in zip(Xs, items)])
        print(f"--- {title}  mean GFL={sum(Gs)/len(Gs):.0f}")
        for lo, hi in BINS:
            m = (S >= lo) & (S < hi)
            if m.sum() < 5: continue
            Xb = X[m]
            print(f"  {lo:3d}-{hi:3d} n={m.sum():3d} op={acc(W_op, Xb, one_hot(yop[m], 4)):.2f} "
                  f"A={corr((Xb @ W_A).squeeze(), yA[m]):.2f} B={corr((Xb @ W_B).squeeze(), yB[m]):.2f} "
                  f"abstract={P[m].float().mean().item():.2f}")

    eval_group("seen(BIG) adapt-orth", BIG[:6], u_orth, 0.5)
    eval_group("TEST none           ", TEST, None, 0.0)
    eval_group("TEST adapt-orth     ", TEST, u_orth, 0.5)

if __name__ == "__main__":
    main()