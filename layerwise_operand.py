# -*- coding: utf-8 -*-
# layerwise_operand.py — layer-wise structure of operand encoding
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

def collect_all_layers(model, tok):
    """Собираем hidden states со ВСЕХ слоёв + (A, B)."""
    Xs_layers, As, Bs = [], [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_h = model.model(full, output_hidden_states=True)
            # hidden_states: tuple of (num_layers+1) tensors [batch, seq, dim]
            hs_all = out_h.hidden_states  # (layer0_emb, layer1, ..., layerN)
        n = hs_all[0].shape[1]
        for layer_idx in range(len(hs_all)):
            if layer_idx >= len(Xs_layers):
                Xs_layers.append([])
            Xs_layers[layer_idx].append(hs_all[layer_idx][0])  # [seq, dim]
        As += [float(a)] * n; Bs += [float(b)] * n
    # стек по слоям
    X_layers = [torch.cat(Xs_layers[i]).float() for i in range(len(Xs_layers))]
    A = torch.tensor(As, device=DEVICE); B = torch.tensor(Bs, device=DEVICE)
    return X_layers, A, B

def eval_layer(X, A, B, sham=False):
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    Xn = (X - mu) / sd
    if sham:
        random.seed(SEED); torch.manual_seed(SEED)
        perm = torch.randperm(A.shape[0], device=DEVICE)
        W_A = ridge(Xn, (A[perm] - A[perm].mean()).unsqueeze(1))
        W_B = ridge(Xn, (B[perm] - B[perm].mean()).unsqueeze(1))
    else:
        W_A = ridge(Xn, (A - A.mean()).unsqueeze(1))
        W_B = ridge(Xn, (B - B.mean()).unsqueeze(1))
    predA = (Xn @ W_A.float()).squeeze()
    predB = (Xn @ W_B.float()).squeeze()
    return corr(predA, A), corr(predB, B)   # без .item()

def main():
    model, tok = load()
    print("Сбор teacher-трейсов со всех слоёв (32 промта, ~2-3 мин)...")
    X_layers, A, B = collect_all_layers(model, tok)
    num_layers = len(X_layers)
    print(f"Всего слоёв (включая embedding): {num_layers}")

    print(f"\n{'Layer':>6s} {'corr(A)':>8s} {'corr(B)':>8s} | {'sham_A':>7s} {'sham_B':>7s} | {'Δ_A':>6s} {'Δ_B':>6s} | specific?")
    print("-" * 80)
    specific_layers = []
    for L in range(num_layers):
        rA, rB = eval_layer(X_layers[L], A, B, sham=False)
        sA, sB = eval_layer(X_layers[L], A, B, sham=True)
        dA, dB = rA - sA, rB - sB
        spec = (dA > 0.2) and (dB > 0.2)
        if spec:
            specific_layers.append(L)
        marker = "✅" if spec else "  "
        print(f"{L:>6d} {rA:>+8.3f} {rB:>+8.3f} | {sA:>+7.3f} {sB:>+7.3f} | {dA:>+6.3f} {dB:>+6.3f} | {marker}")

    print(f"\nСпецифичные слои (Δ > 0.2 по обоим): {specific_layers}")
    if specific_layers:
        early = [l for l in specific_layers if l < num_layers // 3]
        mid = [l for l in specific_layers if num_layers // 3 <= l < 2 * num_layers // 3]
        late = [l for l in specific_layers if l >= 2 * num_layers // 3]
        print(f"  ранние (0-{num_layers//3-1}): {early}")
        print(f"  средние ({num_layers//3}-{2*num_layers//3-1}): {mid}")
        print(f"  поздние ({2*num_layers//3}-{num_layers-1}): {late}")

if __name__ == "__main__":
    main()