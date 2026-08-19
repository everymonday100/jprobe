# -*- coding: utf-8 -*-
import os, hashlib, torch
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, result_of, OPS
from probe5 import TEST, BIG

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
STEER = r"E:\jspace\steer.pt"
DIG = r"E:\jspace\digits.pt"

def ridge(X, Y, lam=1e-2):
    Y = Y.to(X.device)
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

def build_digits(model, tok):
    Xs, Ys = [], []
    for a, o, b in BIG:                      # полный BIG (32 промта)
        p = prompt_of(a, o, b); r = result_of(a, o, b)
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)  # 160 токенов
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        present = torch.tensor([float(str(d) in f"{a}{b}{r}") for d in range(10)], device=DEVICE)
        n = hs.shape[0]
        Xs.append(hs); Ys.append(present.unsqueeze(0).expand(n, -1))
    X = torch.cat(Xs).float()   # dtype fix: fp16 -> float32
    Y = torch.cat(Ys).float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    W = ridge((X - mu) / sd, Y)
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), W=W.cpu()), DIG)
    return mu, sd, W

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)
    d = torch.load(STEER, weights_only=True)
    if os.path.exists(DIG):
        g = torch.load(DIG, weights_only=True); mu, sd, W = g["mu"], g["sd"], g["W"]
    else:
        print("Строим digit-зонды (32 промта x 160 токенов, ~1.5 мин)...")
        mu, sd, W = build_digits(model, tok)
    mu, sd, W = mu.to(DEVICE), sd.to(DEVICE), W.to(DEVICE)
    W_op, muS, sdS = d["W_op"].to(DEVICE), d["mu"].to(DEVICE), d["sd"].to(DEVICE)

    for a, o, b in TEST[:2]:
        p = prompt_of(a, o, b)
        T = _j_loop(model, tok, p, proj, 48, 0)[0]
        idx = hashlib.sha256(T.numpy().tobytes()).hexdigest()[:16]
        print(f"\n=== {p} | idx={idx} (truth {a}{o}{b}) ===")
        with torch.no_grad():
            for t in range(0, T.shape[0], 6):
                h = T[t].to(DEVICE)
                op = OPS[(((h - muS) / sdS).unsqueeze(0) @ W_op).argmax().item()]
                dg = torch.sigmoid(((h - mu) / sd).unsqueeze(0) @ W).squeeze()
                top = dg.topk(3).indices.tolist()
                th = "".join(tok.decode([i]) for i in torch.topk(model.lm_head(h.unsqueeze(0).to(DTYPE)).float(), 3).indices[0].tolist())
                print(f"  step {t:2d}: op={op} digits={top} | {th!r}")

if __name__ == "__main__":
    main()