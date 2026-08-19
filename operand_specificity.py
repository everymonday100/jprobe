# -*- coding: utf-8 -*-
# operand_specificity.py — matched-sham control for operand (A/B) correlation
import os, torch, random
from jspace import load, DEVICE, DTYPE
from probe import prompt_of, result_of, corr
from probe5 import BIG

STEER = r"E:\jspace\steer.pt"
SHAM_AB = r"E:\jspace\steer_sham_ab.pt"
SEED = 321

def ridge(X, Y, lam=1e-2):
    X, Y = X.float(), Y.float()
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

def collect(model, tok):
    """Собираем (hidden states, A, B) по teacher-трейсам BIG."""
    Xs, As, Bs = [], [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        n = hs.shape[0]
        Xs.append(hs); As += [float(a)] * n; Bs += [float(b)] * n
    X = torch.cat(Xs).float()
    A = torch.tensor(As, device=DEVICE); B = torch.tensor(Bs, device=DEVICE)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    return (X - mu) / sd, A, B, mu, sd

def build_sham(Xn, A, B, mu, sd):
    """Зонды A/B на перемешанных операндах (matched control)."""
    random.seed(SEED); torch.manual_seed(SEED)
    perm = torch.randperm(A.shape[0], device=DEVICE)
    W_A = ridge(Xn, (A[perm] - A[perm].mean()).unsqueeze(1))
    W_B = ridge(Xn, (B[perm] - B[perm].mean()).unsqueeze(1))
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), W_A=W_A.cpu(), W_B=W_B.cpu()), SHAM_AB)
    return W_A, W_B

def eval_corr(W_A, W_B, Xn, A, B):
    predA = (Xn @ W_A.float()).squeeze()
    predB = (Xn @ W_B.float()).squeeze()
    return corr(predA, A), corr(predB, B)

def main():
    model, tok = load()
    print("Сбор teacher-трейсов (32 промта, ~1.5 мин)...")
    Xn, A, B, mu, sd = collect(model, tok)

    # REAL: зонды из steer.pt
    d = torch.load(STEER, weights_only=True)
    W_A_r, W_B_r = d["W_A"].to(DEVICE), d["W_B"].to(DEVICE)
    # steer.pt обучен на том же BIG? нормируем теми же mu/sd для честности
    Xn_real = ((torch.cat([torch.zeros(1, Xn.shape[1], device=DEVICE)])) )  # placeholder
    # используем те же Xn, но с mu/sd из steer для real-зонда
    mu_r, sd_r = d["mu"].to(DEVICE), d["sd"].to(DEVICE)
    # пересоберём Xn под steer-нормировку
    Xraw = Xn * sd + mu  # денормализуем
    Xn_r = (Xraw - mu_r) / sd_r
    rA, rB = eval_corr(W_A_r, W_B_r, Xn_r, A, B)
    print(f"\n=== REAL A/B-зонды (steer.pt) ===")
    print(f"  corr(A) = {rA:+.3f}   corr(B) = {rB:+.3f}")

    # SHAM: перемешанные операнды
    if os.path.exists(SHAM_AB):
        g = torch.load(SHAM_AB, weights_only=True)
        W_A_s, W_B_s = g["W_A"].to(DEVICE), g["W_B"].to(DEVICE)
        mu_s, sd_s = g["mu"].to(DEVICE), g["sd"].to(DEVICE)
        Xn_s = (Xraw - mu_s) / sd_s
    else:
        print("  строим sham A/B-зонды...")
        W_A_s, W_B_s = build_sham(Xn, A, B, mu, sd)
        Xn_s = Xn
    sA, sB = eval_corr(W_A_s, W_B_s, Xn_s, A, B)
    print(f"\n=== SHAM A/B-зонды (перемешанные операнды) ===")
    print(f"  corr(A) = {sA:+.3f}   corr(B) = {sB:+.3f}")

    print(f"\nΔ corr(A) = {rA - sA:+.3f}   Δ corr(B) = {rB - sB:+.3f}")
    spec = (rA - sA > 0.2) and (rB - sB > 0.2)
    print(f"\nВердикт: {'✅ операнды читаются специфично' if spec else '❌ corr не специфична (артефакт)'}")

if __name__ == "__main__":
    main()