# -*- coding: utf-8 -*-
# probe2.py — step-resolved зонды: числа, чётность, абстрактность
import torch
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import (OPS, SEEN, UNSEEN, PROJ_PATH, prompt_of, result_of,
                   ridge_fit, one_hot, acc, corr)

MARK = "｜User｜"
BINS = [(0, 8), (8, 16), (16, 32), (32, 999)]

def teacher_trace_labeled(model, tok, prompt, max_steps=160):
    """teacher-трейс БЕЗ обрезки + метка abstract=1 после первого маркера лупа"""
    base, head, embed = model.model, model.lm_head, model.get_input_embeddings()
    with torch.no_grad():
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        prompt_emb = embed(ids["input_ids"].to(DEVICE))
        suffix, Hs, Abs, text, seen = [], [], [], "", False
        for i in range(max_steps):
            emb = prompt_emb if not suffix else \
                  torch.cat([prompt_emb, torch.cat(suffix, 1)], 1)
            out = base(inputs_embeds=emb.to(DTYPE))
            h = out.last_hidden_state[:, -1:, :].detach().float()
            nt = int(head(h.to(DTYPE)).argmax(-1).item())
            text += tok.decode([nt])
            if MARK in text: seen = True
            Hs.append(h.squeeze(0).squeeze(0).cpu())
            Abs.append(1.0 if seen else 0.0)
            if nt == tok.eos_token_id: break
            suffix.append(embed(torch.tensor([[nt]], device=DEVICE)).detach())
    return torch.stack(Hs), torch.tensor(Abs)

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)
    data = []
    for split, items in (("seen", SEEN), ("unseen", UNSEEN)):
        for a, o, b in items:
            p = prompt_of(a, o, b)
            print("collect:", split, p)
            Tt, Ab = teacher_trace_labeled(model, tok, p)
            Tl = _j_loop(model, tok, p, proj, 128, 0)[0]
            n = Tt.shape[0]
            data.append(dict(split=split, kind="teacher", X=Tt, step=torch.arange(n),
                             op=OPS.index(o), A=float(a), B=float(b),
                             par=result_of(a, o, b) % 2, abs=Ab))
            m = Tl.shape[0]
            data.append(dict(split=split, kind="latent", X=Tl, step=torch.arange(m),
                             op=OPS.index(o), A=float(a), B=float(b),
                             par=result_of(a, o, b) % 2, abs=None))

    def get(s, k): return [d for d in data if d["split"] == s and d["kind"] == k]
    Xtr_raw = torch.cat([d["X"] for d in get("seen", "teacher")], 0)
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0).clamp_min(1e-3)
    def norm(T): return (T - mu) / sd

    def pack(ds):
        X = norm(torch.cat([d["X"] for d in ds], 0))
        st = torch.cat([d["step"] for d in ds], 0)
        yop = torch.cat([torch.full((d["X"].shape[0],), d["op"]) for d in ds]).long()
        ypar = torch.cat([torch.full((d["X"].shape[0],), d["par"]) for d in ds]).long()
        yA = torch.cat([d["X"].new_full((d["X"].shape[0],), d["A"]) for d in ds])
        yB = torch.cat([d["X"].new_full((d["X"].shape[0],), d["B"]) for d in ds])
        return X, st, yop, ypar, yA, yB

    dtr = get("seen", "teacher")
    Xtr, _, Yop_tr, Ypar_tr, YA_tr, YB_tr = pack(dtr)
    W_op  = ridge_fit(Xtr, one_hot(Yop_tr, len(OPS)))
    W_par = ridge_fit(Xtr, one_hot(Ypar_tr, 2))
    W_A   = ridge_fit(Xtr, (YA_tr - YA_tr.mean()).unsqueeze(1))
    W_B   = ridge_fit(Xtr, (YB_tr - YB_tr.mean()).unsqueeze(1))
    Yabs_tr = torch.cat([d["abs"] for d in dtr], 0)
    W_abs = ridge_fit(Xtr, one_hot(Yabs_tr.long(), 2))

    for name, (s, k) in {"teacher-unseen": ("unseen", "teacher"),
                         "latent-seen":    ("seen",    "latent"),
                         "latent-unseen":  ("unseen",  "latent")}.items():
        ds = get(s, k)
        X, st, yop, ypar, yA, yB = pack(ds)
        print(f"\n=== {name} ===  (шанс: op=.25 par=.50)")
        for lo, hi in BINS:
            m = (st >= lo) & (st < hi)
            if m.sum() < 5: continue
            Xb = X[m]
            pabs = (Xb @ W_abs).argmax(1).float().mean().item()
            print(f"  шагов {lo:3d}-{hi:3d}  n={m.sum():3d}  "
                  f"op={acc(W_op, Xb, one_hot(yop[m], 4)):.2f} "
                  f"par={acc(W_par, Xb, one_hot(ypar[m], 2)):.2f} "
                  f"A={corr((Xb @ W_A).squeeze(), yA[m]):.2f} "
                  f"B={corr((Xb @ W_B).squeeze(), yB[m]):.2f} "
                  f"abstract={pabs:.2f}")

if __name__ == "__main__":
    main()