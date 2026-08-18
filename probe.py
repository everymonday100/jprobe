# -*- coding: utf-8 -*-
# probe.py — линейные зонды (ridge) по латентным трейсам J-space
import torch
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
import os
PROJ_PATH = os.environ.get("PROJ", r"E:\jspace\proj_mlp.pt")

OPS = ["*", "+", "-", "/"]
SEEN   = [(17,"*",23),(12,"*",8),(45,"+",67),(96,"-",58),(7,"*",8),(144,"/",12)]
UNSEEN = [(13,"*",7),(25,"+",39),(83,"-",46),(96,"/",8),(15,"*",6),(58,"+",27)]
PROJ_PATH = r"E:\jspace\proj_mlp.pt"

def prompt_of(a,o,b): return f"Сколько будет {a}{o}{b}? Рассуждай пошагово."
def result_of(a,o,b): return {"*":a*b, "+":a+b, "-":a-b, "/":a//b}[o]

# ---------- teacher-трейс (настоящие токены) ----------
def teacher_trace(model, tok, prompt, max_steps=128):
    base, head, embed = model.model, model.lm_head, model.get_input_embeddings()
    with torch.no_grad():
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        prompt_emb = embed(ids["input_ids"].to(DEVICE))
        suffix, Hs, text = [], [], ""
        for i in range(max_steps):
            emb = prompt_emb if not suffix else \
                  torch.cat([prompt_emb, torch.cat(suffix, 1)], 1)
            out = base(inputs_embeds=emb.to(DTYPE))
            h = out.last_hidden_state[:, -1:, :].detach().float()
            nt = int(head(h.to(DTYPE)).argmax(-1).item())
            text += tok.decode([nt])
            Hs.append(h.squeeze(0).squeeze(0).cpu())
            if nt == tok.eos_token_id or "｜User｜" in text: break
            suffix.append(embed(torch.tensor([[nt]], device=DEVICE)).detach())
    return torch.stack(Hs)

# ---------- ridge-зонды ----------
def one_hot(idx, K):
    y = torch.zeros(len(idx), K)
    y[torch.arange(len(idx)), idx] = 1.0
    return y

def ridge_fit(X, Y, lam=1e-2):
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ Y)

def acc(W, X, Y):
    return ((X @ W).argmax(1) == Y.argmax(1)).float().mean().item()

def corr(p, y):
    p = p - p.mean(); y = y - y.mean()
    return ((p * y).sum() / (p.norm() * y.norm() + 1e-9)).item()

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)
    data = []
    for split, items in (("seen", SEEN), ("unseen", UNSEEN)):
        for a, o, b in items:
            p = prompt_of(a, o, b)
            print("collect:", split, p)
            Tt = teacher_trace(model, tok, p)
            Tl = _j_loop(model, tok, p, proj, 128, 0)[0]
            for kind, T in (("teacher", Tt), ("latent", Tl)):
                n = T.shape[0]
                data.append(dict(split=split, kind=kind, X=T,
                                 op=OPS.index(o), res=result_of(a, o, b),
                                 step=torch.arange(n) / max(n - 1, 1)))

    def get(s, k): return [d for d in data if d["split"] == s and d["kind"] == k]
    X_tr_raw = torch.cat([d["X"] for d in get("seen", "teacher")], 0)
    mu, sd = X_tr_raw.mean(0), X_tr_raw.std(0).clamp_min(1e-3)
    def norm(T): return (T - mu) / sd

    res_vals = sorted({result_of(a, o, b) for a, o, b in SEEN + UNSEEN})
    ridx = {v: i for i, v in enumerate(res_vals)}
    K_op, K_res = len(OPS), len(res_vals)

    def XY(s, k):
        ds = get(s, k)
        X = norm(torch.cat([d["X"] for d in ds], 0))
        yop = torch.cat([torch.full((d["X"].shape[0],), d["op"]) for d in ds]).long()
        yres = torch.cat([torch.full((d["X"].shape[0],), ridx[d["res"]]) for d in ds]).long()
        yst = torch.cat([d["step"] for d in ds], 0)
        return X, one_hot(yop, K_op), one_hot(yres, K_res), yst

    Xtr, Yop_tr, Yres_tr, Yst_tr = XY("seen", "teacher")
    W_op  = ridge_fit(Xtr, Yop_tr)
    W_res = ridge_fit(Xtr, Yres_tr)
    W_st  = ridge_fit(Xtr, Yst_tr.unsqueeze(1))

    print(f"\n=== зонды, обучены на teacher-seen ===")
    print(f"шанс: op={1/K_op:.3f}, res={1/K_res:.3f}")
    for name, (s, k) in {"teacher-unseen": ("unseen", "teacher"),
                         "latent-seen":    ("seen",    "latent"),
                         "latent-unseen":  ("unseen",  "latent")}.items():
        X, Yop, Yres, Yst = XY(s, k)
        print(f"{name:15s} op_acc={acc(W_op, X, Yop):.3f} "
              f"res_acc={acc(W_res, X, Yres):.3f} step_corr={corr((X @ W_st).squeeze(), Yst):.3f}")

    Xl, Yop_l, _, _ = XY("seen", "latent")
    W_op_l = ridge_fit(Xl, Yop_l)
    Xlu, Yop_lu, _, _ = XY("unseen", "latent")
    print(f"latent-seen -> latent-unseen op_acc={acc(W_op_l, Xlu, Yop_lu):.3f}")

if __name__ == "__main__":
    main()